import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.wrappers._anthropic_utils import Wrapper, extract_anthropic_usage, finalize_anthropic_tokens
from braintrust.wrappers.claude_agent_sdk._constants import (
    ANTHROPIC_MESSAGES_CREATE_SPAN_NAME,
    CLAUDE_AGENT_TASK_SPAN_NAME,
    DEFAULT_TOOL_NAME,
    MCP_TOOL_METADATA,
    MCP_TOOL_NAME_DELIMITER,
    MCP_TOOL_PREFIX,
    SERIALIZED_CONTENT_TYPE_BY_BLOCK_CLASS,
    SYSTEM_MESSAGE_TYPES,
    TOOL_METADATA,
    BlockClassName,
    MessageClassName,
    SerializedContentType,
)

log = logging.getLogger(__name__)
_thread_local = threading.local()


@dataclasses.dataclass(frozen=True)
class ParsedToolName:
    raw_name: str
    display_name: str
    is_mcp: bool = False
    mcp_server: str | None = None


@dataclasses.dataclass
class _ActiveToolSpan:
    span: Any
    raw_name: str
    display_name: str
    input: Any
    handler_active: bool = False

    @property
    def has_span(self) -> bool:
        return True

    def activate(self) -> None:
        self.handler_active = True
        self.span.set_current()

    def log_error(self, exc: Exception) -> None:
        self.span.log(error=str(exc))

    def release(self) -> None:
        if not self.handler_active:
            return

        self.handler_active = False
        self.span.unset_current()


class _NoopActiveToolSpan:
    @property
    def has_span(self) -> bool:
        return False

    def log_error(self, exc: Exception) -> None:
        del exc

    def release(self) -> None:
        return


_NOOP_ACTIVE_TOOL_SPAN = _NoopActiveToolSpan()


def _log_tracing_warning(exc: Exception) -> None:
    log.warning("Error in tracing code", exc_info=exc)

def _parse_tool_name(tool_name: Any) -> ParsedToolName:
    raw_name = str(tool_name) if tool_name is not None else DEFAULT_TOOL_NAME

    if not raw_name.startswith(MCP_TOOL_PREFIX):
        return ParsedToolName(raw_name=raw_name, display_name=raw_name)

    remainder = raw_name[len(MCP_TOOL_PREFIX) :]
    if not remainder:
        return ParsedToolName(raw_name=raw_name, display_name=raw_name)

    server_and_tool = remainder.rsplit(MCP_TOOL_NAME_DELIMITER, 1)
    if len(server_and_tool) != 2:
        return ParsedToolName(raw_name=raw_name, display_name=raw_name)

    server_name, tool_display_name = server_and_tool
    if not server_name or not tool_display_name:
        return ParsedToolName(raw_name=raw_name, display_name=raw_name)

    return ParsedToolName(
        raw_name=raw_name,
        display_name=tool_display_name,
        is_mcp=True,
        mcp_server=server_name,
    )


def _serialize_tool_result_content(content: Any) -> Any:
    if dataclasses.is_dataclass(content):
        serialized_content = _serialize_content_blocks([content])
        return serialized_content[0] if serialized_content else None

    if not isinstance(content, list):
        return content

    serialized_content = _serialize_content_blocks(content)
    if (
        isinstance(serialized_content, list)
        and len(serialized_content) == 1
        and isinstance(serialized_content[0], dict)
        and serialized_content[0].get("type") == SerializedContentType.TEXT
        and SerializedContentType.TEXT in serialized_content[0]
    ):
        return serialized_content[0][SerializedContentType.TEXT]

    return serialized_content


def _serialize_tool_result_output(tool_result_block: Any) -> dict[str, Any]:
    output = {"content": _serialize_tool_result_content(getattr(tool_result_block, "content", None))}

    if getattr(tool_result_block, "is_error", None) is True:
        output["is_error"] = True

    return output

def _serialize_system_message(message: Any) -> dict[str, Any]:
    serialized = {"subtype": getattr(message, "subtype", None)}

    for field_name in (
        "task_id",
        "description",
        "uuid",
        "session_id",
        "tool_use_id",
        "task_type",
        "status",
        "output_file",
        "summary",
        "last_tool_name",
        "usage",
    ):
        value = getattr(message, field_name, None)
        if value is not None:
            serialized[field_name] = value

    if len(serialized) == 1:
        data = getattr(message, "data", None)
        if data:
            serialized["data"] = data

    return serialized


def _create_tool_wrapper_class(original_tool_class: Any) -> Any:
    """Creates a wrapper class for SdkMcpTool that re-enters active TOOL spans."""

    class WrappedSdkMcpTool(original_tool_class):  # type: ignore[valid-type,misc]
        def __init__(
            self,
            name: Any,
            description: Any,
            input_schema: Any,
            handler: Any,
            **kwargs: Any,
        ):
            wrapped_handler = _wrap_tool_handler(handler, name)
            super().__init__(name, description, input_schema, wrapped_handler, **kwargs)  # type: ignore[call-arg]

        __class_getitem__ = classmethod(lambda cls, params: cls)  # type: ignore[assignment]

    return WrappedSdkMcpTool


def _wrap_tool_factory(tool_fn: Any) -> Any:
    """Wrap the tool() factory so decorated handlers inherit the active TOOL span."""

    def wrapped_tool(*args: Any, **kwargs: Any) -> Any:
        result = tool_fn(*args, **kwargs)
        if not callable(result):
            return result

        def wrapped_decorator(handler_fn: Any) -> Any:
            tool_def = result(handler_fn)
            if tool_def and hasattr(tool_def, "handler"):
                tool_name = getattr(tool_def, "name", DEFAULT_TOOL_NAME)
                tool_def.handler = _wrap_tool_handler(tool_def.handler, tool_name)
            return tool_def

        return wrapped_decorator

    return wrapped_tool


def _wrap_tool_handler(handler: Any, tool_name: Any) -> Any:
    """Wrap a tool handler so nested spans execute under the stream-based TOOL span."""
    if hasattr(handler, "_braintrust_wrapped"):
        return handler

    async def wrapped_handler(args: Any) -> Any:
        active_tool_span = _activate_tool_span_for_handler(tool_name, args)
        if not active_tool_span.has_span:
            with start_span(
                name=str(tool_name),
                span_attributes={"type": SpanTypeAttribute.TOOL},
                input=args,
            ) as span:
                result = await handler(args)
                span.log(output=result)
                return result

        try:
            return await handler(args)
        except Exception as exc:
            active_tool_span.log_error(exc)
            raise
        finally:
            active_tool_span.release()

    wrapped_handler._braintrust_wrapped = True  # type: ignore[attr-defined]
    return wrapped_handler


class ToolSpanTracker:
    def __init__(self):
        self._active_spans: dict[str, _ActiveToolSpan] = {}
        self._pending_task_link_tool_use_ids: set[str] = set()

    def start_tool_spans(self, message: Any, llm_span_export: str | None) -> None:
        if llm_span_export is None or not hasattr(message, "content"):
            return

        for block in message.content:
            if type(block).__name__ != BlockClassName.TOOL_USE:
                continue

            tool_use_id = getattr(block, "id", None)
            if not tool_use_id:
                continue

            tool_use_id = str(tool_use_id)
            if tool_use_id in self._active_spans:
                self._end_tool_span(tool_use_id)

            parsed_tool_name = _parse_tool_name(getattr(block, "name", None))
            metadata = {
                TOOL_METADATA.tool_name: parsed_tool_name.display_name,
                TOOL_METADATA.tool_call_id: tool_use_id,
            }
            if parsed_tool_name.raw_name != parsed_tool_name.display_name:
                metadata[TOOL_METADATA.raw_tool_name] = parsed_tool_name.raw_name
            if parsed_tool_name.is_mcp:
                metadata[TOOL_METADATA.operation_name] = MCP_TOOL_METADATA.operation_name
                metadata[TOOL_METADATA.mcp_method_name] = MCP_TOOL_METADATA.method_name
                if parsed_tool_name.mcp_server:
                    metadata[TOOL_METADATA.mcp_server] = parsed_tool_name.mcp_server

            tool_span = start_span(
                name=parsed_tool_name.display_name,
                span_attributes={"type": SpanTypeAttribute.TOOL},
                input=getattr(block, "input", None),
                metadata=metadata,
                parent=llm_span_export,
            )
            self._active_spans[tool_use_id] = _ActiveToolSpan(
                span=tool_span,
                raw_name=parsed_tool_name.raw_name,
                display_name=parsed_tool_name.display_name,
                input=getattr(block, "input", None),
            )
            if parsed_tool_name.display_name == "Agent":
                self._pending_task_link_tool_use_ids.add(tool_use_id)

    def finish_tool_spans(self, message: Any) -> None:
        if not hasattr(message, "content"):
            return

        for block in message.content:
            if type(block).__name__ != BlockClassName.TOOL_RESULT:
                continue

            tool_use_id = getattr(block, "tool_use_id", None)
            if tool_use_id is None:
                continue

            self._end_tool_span(str(tool_use_id), tool_result_block=block)

    def cleanup(self, end_time: float | None = None, exclude_tool_use_ids: frozenset[str] | None = None) -> None:
        for tool_use_id in list(self._active_spans):
            if exclude_tool_use_ids and tool_use_id in exclude_tool_use_ids:
                continue
            self._end_tool_span(tool_use_id, end_time=end_time)

    @property
    def has_active_spans(self) -> bool:
        return bool(self._active_spans)

    @property
    def pending_task_link_tool_use_ids(self) -> frozenset[str]:
        return frozenset(self._pending_task_link_tool_use_ids)

    def mark_task_started(self, tool_use_id: Any) -> None:
        if tool_use_id is None:
            return

        self._pending_task_link_tool_use_ids.discard(str(tool_use_id))

    def acquire_span_for_handler(self, tool_name: Any, args: Any) -> _ActiveToolSpan | None:
        parsed_tool_name = _parse_tool_name(tool_name)
        candidate_names = list(dict.fromkeys((parsed_tool_name.raw_name, parsed_tool_name.display_name, str(tool_name))))

        candidates = [
            active_tool_span
            for active_tool_span in self._active_spans.values()
            if not active_tool_span.handler_active
            and (active_tool_span.raw_name in candidate_names or active_tool_span.display_name in candidate_names)
        ]

        matched_span = _match_tool_span_for_handler(candidates, args)
        if matched_span is None:
            return None

        matched_span.activate()
        return matched_span

    def _end_tool_span(self, tool_use_id: str, tool_result_block: Any | None = None, end_time: float | None = None) -> None:
        active_tool_span = self._active_spans.pop(tool_use_id, None)
        self._pending_task_link_tool_use_ids.discard(tool_use_id)
        if active_tool_span is None:
            return

        if tool_result_block is None:
            active_tool_span.span.end(end_time=end_time)
            return

        output = _serialize_tool_result_output(tool_result_block)
        log_event: dict[str, Any] = {"output": output}
        if getattr(tool_result_block, "is_error", None) is True:
            log_event["error"] = str(output["content"])
        active_tool_span.span.log(**log_event)
        active_tool_span.span.end(end_time=end_time)

    def get_span_export(self, tool_use_id: Any) -> str | None:
        if tool_use_id is None:
            return None

        active_tool_span = self._active_spans.get(str(tool_use_id))
        if active_tool_span is None:
            return None

        return active_tool_span.span.export()


def _match_tool_span_for_handler(candidates: list[_ActiveToolSpan], args: Any) -> _ActiveToolSpan | None:
    if not candidates:
        return None

    exact_input_matches = [candidate for candidate in candidates if candidate.input == args]
    if exact_input_matches:
        return exact_input_matches[0]

    if len(candidates) == 1:
        return candidates[0]

    for active_tool_span in candidates:
        if active_tool_span.input is None:
            return active_tool_span

    return candidates[0]


def _activate_tool_span_for_handler(tool_name: Any, args: Any) -> _ActiveToolSpan | _NoopActiveToolSpan:
    tool_span_tracker = getattr(_thread_local, "tool_span_tracker", None)
    if tool_span_tracker is None:
        return _NOOP_ACTIVE_TOOL_SPAN

    return tool_span_tracker.acquire_span_for_handler(tool_name, args) or _NOOP_ACTIVE_TOOL_SPAN


class LLMSpanTracker:
    """Manages LLM span lifecycle for Claude Agent SDK message streams.

    Message flow per turn:
    1. UserMessage (tool results) -> mark the time when next LLM will start
    2. AssistantMessage - LLM response arrives -> create span with the marked start time, ending previous span
    3. ResultMessage - usage metrics -> log to span

    We end the previous span when the next AssistantMessage arrives, using the marked
    start time to ensure sequential spans (no overlapping LLM spans).
    """

    def __init__(self, query_start_time: float | None = None):
        self.current_span: Any | None = None
        self.current_span_export: str | None = None
        self.current_parent_export: str | None = None
        self.current_output: list[dict[str, Any]] | None = None
        self.next_start_time: float | None = query_start_time

    def get_next_start_time(self) -> float:
        return self.next_start_time if self.next_start_time is not None else time.time()

    def start_llm_span(
        self,
        message: Any,
        prompt: Any,
        conversation_history: list[dict[str, Any]],
        parent_export: str | None = None,
        start_time: float | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Start a new LLM span, ending the previous one if it exists."""
        current_message = _serialize_assistant_message(message)

        if (
            self.current_span
            and self.next_start_time is None
            and self.current_parent_export == parent_export
            and current_message is not None
        ):
            merged_message = _merge_assistant_messages(
                self.current_output[0] if self.current_output else None,
                current_message,
            )
            if merged_message is not None:
                self.current_output = [merged_message]
                self.current_span.log(output=self.current_output)
            return merged_message, True

        resolved_start_time = start_time if start_time is not None else self.get_next_start_time()
        first_token_time = time.time()

        if self.current_span:
            self.current_span.end(end_time=resolved_start_time)

        final_content, span = _create_llm_span_for_messages(
            [message],
            prompt,
            conversation_history,
            parent=parent_export,
            start_time=resolved_start_time,
        )
        if span is not None:
            span.log(metrics={"time_to_first_token": max(0.0, first_token_time - resolved_start_time)})
        self.current_span = span
        self.current_span_export = span.export() if span else None
        self.current_parent_export = parent_export
        self.current_output = [final_content] if final_content is not None else None
        self.next_start_time = None
        return final_content, False

    def mark_next_llm_start(self) -> None:
        """Mark when the next LLM call will start (after tool results)."""
        self.next_start_time = time.time()

    def log_usage(self, usage_metrics: dict[str, float]) -> None:
        """Log usage metrics to the current LLM span."""
        if self.current_span and usage_metrics:
            self.current_span.log(metrics=usage_metrics)

    def cleanup(self) -> None:
        """End any unclosed spans."""
        if self.current_span:
            self.current_span.end()
            self.current_span = None
            self.current_span_export = None
            self.current_parent_export = None
            self.current_output = None


class TaskEventSpanTracker:
    def __init__(self, root_span_export: str, tool_tracker: ToolSpanTracker):
        self._root_span_export = root_span_export
        self._tool_tracker = tool_tracker
        self._active_spans: dict[str, Any] = {}
        self._task_span_by_tool_use_id: dict[str, Any] = {}
        self._active_task_order: list[str] = []

    def process(self, message: Any) -> None:
        task_id = getattr(message, "task_id", None)
        if task_id is None:
            return

        task_id = str(task_id)
        message_type = type(message).__name__
        task_span = self._active_spans.get(task_id)

        if task_span is None:
            task_span = start_span(
                name=self._span_name(message, task_id),
                span_attributes={"type": SpanTypeAttribute.TASK},
                metadata=self._metadata(message),
                parent=self._parent_export(message),
            )
            self._active_spans[task_id] = task_span
            self._active_task_order.append(task_id)
            tool_use_id = getattr(message, "tool_use_id", None)
            if tool_use_id is not None:
                tool_use_id = str(tool_use_id)
                self._task_span_by_tool_use_id[tool_use_id] = task_span
                self._tool_tracker.mark_task_started(tool_use_id)
        else:
            update: dict[str, Any] = {}
            metadata = self._metadata(message)
            if metadata:
                update["metadata"] = metadata

            output = self._output(message)
            if output is not None:
                update["output"] = output

            if update:
                task_span.log(**update)

        if self._should_end(message_type):
            tool_use_id = getattr(message, "tool_use_id", None)
            if tool_use_id is not None:
                self._task_span_by_tool_use_id.pop(str(tool_use_id), None)
            task_span.end()
            del self._active_spans[task_id]
            self._active_task_order = [active_task_id for active_task_id in self._active_task_order if active_task_id != task_id]

    @property
    def active_tool_use_ids(self) -> frozenset[str]:
        return frozenset(self._task_span_by_tool_use_id.keys())

    def cleanup(self) -> None:
        for task_id, span in list(self._active_spans.items()):
            span.end()
            del self._active_spans[task_id]
        self._task_span_by_tool_use_id.clear()
        self._active_task_order.clear()

    def parent_export_for_message(self, message: Any, fallback_export: str) -> str:
        parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
        if parent_tool_use_id is None:
            if _message_starts_subagent_tool(message):
                return fallback_export
            active_task_export = self._latest_active_task_export()
            return active_task_export or fallback_export

        task_span = self._task_span_by_tool_use_id.get(str(parent_tool_use_id))
        if task_span is not None:
            return task_span.export()

        active_task_export = self._latest_active_task_export()
        return active_task_export or fallback_export

    def _latest_active_task_export(self) -> str | None:
        for task_id in reversed(self._active_task_order):
            task_span = self._active_spans.get(task_id)
            if task_span is not None:
                return task_span.export()

        return None

    def _parent_export(self, message: Any) -> str:
        return self._tool_tracker.get_span_export(getattr(message, "tool_use_id", None)) or self._root_span_export

    def _span_name(self, message: Any, task_id: str) -> str:
        return (
            getattr(message, "description", None)
            or getattr(message, "task_type", None)
            or f"Task {task_id}"
        )

    def _metadata(self, message: Any) -> dict[str, Any]:
        metadata = {
            k: v
            for k, v in {
                "task_id": getattr(message, "task_id", None),
                "session_id": getattr(message, "session_id", None),
                "tool_use_id": getattr(message, "tool_use_id", None),
                "task_type": getattr(message, "task_type", None),
                "status": getattr(message, "status", None),
                "last_tool_name": getattr(message, "last_tool_name", None),
                "usage": getattr(message, "usage", None),
            }.items()
            if v is not None
        }
        return metadata

    def _output(self, message: Any) -> dict[str, Any] | None:
        summary = getattr(message, "summary", None)
        output_file = getattr(message, "output_file", None)

        if summary is None and output_file is None:
            return None

        return {
            k: v
            for k, v in {
                "summary": summary,
                "output_file": output_file,
            }.items()
            if v is not None
        }

    def _should_end(self, message_type: str) -> bool:
        return message_type == MessageClassName.TASK_NOTIFICATION


def _message_starts_subagent_tool(message: Any) -> bool:
    if not hasattr(message, "content"):
        return False

    for block in message.content:
        if type(block).__name__ != BlockClassName.TOOL_USE:
            continue
        if getattr(block, "name", None) == "Agent":
            return True

    return False


def _create_client_wrapper_class(original_client_class: Any) -> Any:
    """Creates a wrapper class for ClaudeSDKClient that wraps query and receive_response."""

    class WrappedClaudeSDKClient(Wrapper):
        def __init__(self, *args: Any, **kwargs: Any):
            # Create the original client instance
            client = original_client_class(*args, **kwargs)
            super().__init__(client)
            self.__client = client
            self.__last_prompt: str | None = None
            self.__query_start_time: float | None = None
            self.__captured_messages: list[dict[str, Any]] | None = None

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            """Wrap query to capture the prompt and start time for tracing."""
            # Capture the time when query is called (when LLM call starts)
            self.__query_start_time = time.time()
            self.__captured_messages = None

            # Capture the prompt for use in receive_response
            prompt = args[0] if args else kwargs.get("prompt")

            if prompt is not None:
                if isinstance(prompt, str):
                    self.__last_prompt = prompt
                elif isinstance(prompt, AsyncIterable):
                    # AsyncIterable[dict] - wrap it to capture messages as they're yielded
                    captured: list[dict[str, Any]] = []
                    self.__captured_messages = captured
                    self.__last_prompt = None  # Will be set after messages are captured

                    async def capturing_wrapper() -> AsyncGenerator[dict[str, Any], None]:
                        async for msg in prompt:
                            captured.append(msg)
                            yield msg

                    # Replace the prompt with our capturing wrapper
                    if args:
                        args = (capturing_wrapper(),) + args[1:]
                    else:
                        kwargs["prompt"] = capturing_wrapper()
                else:
                    self.__last_prompt = str(prompt)

            return await self.__client.query(*args, **kwargs)

        async def receive_response(self) -> AsyncGenerator[Any, None]:
            """Wrap receive_response to add tracing.

            Uses start_span context manager which automatically:
            - Handles exceptions and logs them as errors
            - Sets the span as current so tool calls automatically nest under it
            - Manages span lifecycle (start/end)
            """
            generator = self.__client.receive_response()

            # Determine the initial input - may be updated later if using async generator
            initial_input = self.__last_prompt if self.__last_prompt else None

            with start_span(
                name=CLAUDE_AGENT_TASK_SPAN_NAME,
                span_attributes={"type": SpanTypeAttribute.TASK},
                input=initial_input,
            ) as span:
                # If we're capturing async messages, we'll update input after they're consumed
                input_needs_update = self.__captured_messages is not None

                final_results: list[dict[str, Any]] = []
                task_events: list[dict[str, Any]] = []
                llm_tracker = LLMSpanTracker(query_start_time=self.__query_start_time)
                tool_tracker = ToolSpanTracker()
                task_event_span_tracker = TaskEventSpanTracker(span.export(), tool_tracker)
                _thread_local.tool_span_tracker = tool_tracker

                try:
                    async for message in generator:
                        # Update input from captured async messages (once, after they're consumed)
                        if input_needs_update:
                            captured_input = self.__captured_messages if self.__captured_messages else []
                            if captured_input:
                                span.log(input=captured_input)
                            input_needs_update = False

                        message_type = type(message).__name__

                        if message_type == MessageClassName.ASSISTANT:
                            if llm_tracker.current_span and tool_tracker.has_active_spans:
                                active_subagent_tool_use_ids = (
                                    task_event_span_tracker.active_tool_use_ids | tool_tracker.pending_task_link_tool_use_ids
                                )
                                tool_tracker.cleanup(
                                    end_time=llm_tracker.get_next_start_time(),
                                    exclude_tool_use_ids=active_subagent_tool_use_ids,
                                )
                            llm_parent_export = task_event_span_tracker.parent_export_for_message(
                                message,
                                span.export(),
                            )
                            final_content, extended_existing_span = llm_tracker.start_llm_span(
                                message,
                                self.__last_prompt,
                                final_results,
                                parent_export=llm_parent_export,
                            )
                            tool_tracker.start_tool_spans(message, llm_tracker.current_span_export)
                            if final_content:
                                if extended_existing_span and final_results and final_results[-1].get("role") == "assistant":
                                    final_results[-1] = final_content
                                else:
                                    final_results.append(final_content)
                        elif message_type == MessageClassName.USER:
                            tool_tracker.finish_tool_spans(message)
                            has_tool_results = False
                            if hasattr(message, "content"):
                                has_tool_results = any(
                                    type(block).__name__ == BlockClassName.TOOL_RESULT
                                    for block in message.content
                                )
                                content = _serialize_content_blocks(message.content)
                                final_results.append({"content": content, "role": "user"})
                            if has_tool_results:
                                llm_tracker.mark_next_llm_start()
                        elif message_type == MessageClassName.RESULT:
                            if hasattr(message, "usage"):
                                usage_metrics = _extract_usage_from_result_message(message)
                                llm_tracker.log_usage(usage_metrics)

                            result_metadata = {
                                k: v
                                for k, v in {
                                    "num_turns": getattr(message, "num_turns", None),
                                    "session_id": getattr(message, "session_id", None),
                                }.items()
                                if v is not None
                            }
                            span.log(metadata=result_metadata)
                        elif message_type in SYSTEM_MESSAGE_TYPES:
                            task_event_span_tracker.process(message)
                            task_events.append(_serialize_system_message(message))

                        yield message
                except asyncio.CancelledError:
                    # The CancelledError may come from the subprocess transport
                    # (e.g., anyio internal cleanup when subagents complete) rather
                    # than a genuine external cancellation. We suppress it here so
                    # the response stream ends cleanly. If the caller genuinely
                    # cancelled the task, they still have pending cancellation
                    # requests that will fire at their next await point.
                    if final_results:
                        span.log(output=final_results[-1])
                else:
                    if final_results:
                        span.log(output=final_results[-1])
                finally:
                    if task_events:
                        span.log(metadata={"task_events": task_events})
                    task_event_span_tracker.cleanup()
                    tool_tracker.cleanup()
                    llm_tracker.cleanup()
                    if hasattr(_thread_local, "tool_span_tracker"):
                        delattr(_thread_local, "tool_span_tracker")

        async def __aenter__(self) -> "WrappedClaudeSDKClient":
            await self.__client.__aenter__()
            return self

        async def __aexit__(self, *args: Any) -> None:
            await self.__client.__aexit__(*args)

    return WrappedClaudeSDKClient


def _create_llm_span_for_messages(
    messages: list[Any],  # List of AssistantMessage objects
    prompt: Any,
    conversation_history: list[dict[str, Any]],
    parent: str | None = None,
    start_time: float | None = None,
) -> tuple[dict[str, Any] | None, Any | None]:
    """Creates an LLM span for a group of AssistantMessage objects.

    Returns a tuple of (final_content, span):
    - final_content: The final message content to add to conversation history
    - span: The LLM span object (for logging metrics later)

    Automatically nests under the current span (TASK span from receive_response).

    Note: This is called from within a catch_exceptions block, so errors won't break user code.
    """
    if not messages:
        return None, None

    last_message = messages[-1]
    if type(last_message).__name__ != MessageClassName.ASSISTANT:
        return None, None
    model = getattr(last_message, "model", None)
    input_messages = _build_llm_input(prompt, conversation_history)

    outputs: list[dict[str, Any]] = []
    for msg in messages:
        if hasattr(msg, "content"):
            content = _serialize_content_blocks(msg.content)
            outputs.append({"content": content, "role": "assistant"})

    llm_span = start_span(
        name=ANTHROPIC_MESSAGES_CREATE_SPAN_NAME,
        span_attributes={"type": SpanTypeAttribute.LLM},
        input=input_messages,
        output=outputs,
        metadata={"model": model} if model else None,
        parent=parent,
        start_time=start_time,
    )

    # Return final message content for conversation history and the span
    if hasattr(last_message, "content"):
        content = _serialize_content_blocks(last_message.content)
        return {"content": content, "role": "assistant"}, llm_span

    return None, llm_span


def _serialize_assistant_message(message: Any) -> dict[str, Any] | None:
    if not hasattr(message, "content"):
        return None

    return {"content": _serialize_content_blocks(message.content), "role": "assistant"}


def _merge_assistant_messages(existing_message: dict[str, Any] | None, new_message: dict[str, Any]) -> dict[str, Any]:
    if existing_message is None:
        return new_message

    existing_content = existing_message.get("content")
    new_content = new_message.get("content")
    if isinstance(existing_content, list) and isinstance(new_content, list):
        return {
            "role": "assistant",
            "content": [*existing_content, *new_content],
        }

    return new_message


def _serialize_content_blocks(content: Any) -> Any:
    """Converts content blocks to a serializable format with proper type fields.

    Claude Agent SDK uses dataclasses for content blocks, so we use dataclasses.asdict()
    for serialization and add the 'type' field based on the class name.
    """
    if isinstance(content, list):
        result = []
        for block in content:
            if dataclasses.is_dataclass(block):
                serialized = dataclasses.asdict(block)

                block_type = type(block).__name__
                serialized_type = SERIALIZED_CONTENT_TYPE_BY_BLOCK_CLASS.get(block_type)
                if serialized_type is not None:
                    serialized["type"] = serialized_type

                if block_type == BlockClassName.TOOL_RESULT:
                    content_value = serialized.get("content")
                    if isinstance(content_value, list) and len(content_value) == 1:
                        item = content_value[0]
                        if (
                            isinstance(item, dict)
                            and item.get("type") == SerializedContentType.TEXT
                            and SerializedContentType.TEXT in item
                        ):
                            serialized["content"] = item[SerializedContentType.TEXT]

                    if "is_error" in serialized and serialized["is_error"] is None:
                        del serialized["is_error"]
            else:
                serialized = block

            result.append(serialized)
        return result
    return content


def _extract_usage_from_result_message(result_message: Any) -> dict[str, float]:
    """Extracts and normalizes usage metrics from a ResultMessage.

    Uses shared Anthropic utilities for consistent metric extraction.
    """
    if not hasattr(result_message, "usage"):
        return {}

    usage = result_message.usage
    if not usage:
        return {}

    metrics = extract_anthropic_usage(usage)
    if metrics:
        metrics = finalize_anthropic_tokens(metrics)

    return metrics


def _build_llm_input(prompt: Any, conversation_history: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Builds the input array for an LLM span from the initial prompt and conversation history.

    Formats input to match Anthropic messages API format for proper UI rendering.
    """
    if isinstance(prompt, str):
        if len(conversation_history) == 0:
            return [{"content": prompt, "role": "user"}]
        else:
            return [{"content": prompt, "role": "user"}] + conversation_history

    return conversation_history if conversation_history else None
