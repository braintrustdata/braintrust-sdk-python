import asyncio
import collections
import dataclasses
import json
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from braintrust.integrations.anthropic._utils import Wrapper, extract_anthropic_usage
from braintrust.integrations.claude_agent_sdk._constants import (
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
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute


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
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None
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


_HOOK_REDACTED_FIELDS = frozenset({"cwd", "transcript_path", "agent_transcript_path"})


def _serialize_hook_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _serialize_hook_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _serialize_hook_value(item)
            for key, item in value.items()
            if str(key) not in _HOOK_REDACTED_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_hook_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hook_event_name(input_data: Any) -> str:
    if isinstance(input_data, dict) and input_data.get("hook_event_name"):
        return str(input_data["hook_event_name"])
    return "Hook"


def _hook_callback_name(callback: Any) -> str:
    name = getattr(callback, "__name__", None) or getattr(callback, "__qualname__", None)
    return name if isinstance(name, str) and name else "hook_callback"


def _hook_metadata(callback: Any, input_data: Any, tool_use_id: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "claude_agent_sdk.hook.event_name": _hook_event_name(input_data),
        "claude_agent_sdk.hook.callback_name": _hook_callback_name(callback),
    }

    if tool_use_id is not None:
        metadata["claude_agent_sdk.hook.tool_use_id"] = str(tool_use_id)

    if isinstance(input_data, dict):
        for key in ("tool_name", "session_id", "agent_id", "agent_type"):
            value = input_data.get(key)
            if value is not None:
                metadata[f"claude_agent_sdk.hook.{key}"] = value

    return metadata


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


def _make_dispatch_key(tool_name: str, tool_input: Any) -> tuple[str, str]:
    """Create a hashable key for dispatch queue lookup from tool name and input."""
    try:
        input_sig = json.dumps(tool_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        input_sig = repr(tool_input)
    return (tool_name, input_sig)


class ToolSpanTracker:
    def __init__(self):
        self._active_spans: dict[str, _ActiveToolSpan] = {}
        self._completed_span_exports: dict[str, str] = {}
        # Per-(tool_name, input_signature) FIFO queue of tool_use_ids.
        # Used by acquire_span_for_handler to disambiguate identical concurrent
        # tool calls (same name + same input) from sibling subagents.
        self._dispatch_queues: dict[tuple[str, str], collections.deque[str]] = {}

    def start_tool_spans(self, message: Any, llm_span_export: str | None) -> None:
        if llm_span_export is None or not hasattr(message, "content"):
            return

        message_parent_tool_use_id = getattr(message, "parent_tool_use_id", None)

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
            tool_input = getattr(block, "input", None)
            self._active_spans[tool_use_id] = _ActiveToolSpan(
                span=tool_span,
                raw_name=parsed_tool_name.raw_name,
                display_name=parsed_tool_name.display_name,
                input=tool_input,
                tool_use_id=tool_use_id,
                parent_tool_use_id=message_parent_tool_use_id,
            )
            dispatch_key = _make_dispatch_key(parsed_tool_name.raw_name, tool_input)
            self._dispatch_queues.setdefault(dispatch_key, collections.deque()).append(tool_use_id)

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

    def cleanup_context(
        self,
        parent_tool_use_id: str | None,
        *,
        end_time: float | None = None,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> None:
        """Close tool spans belonging to one subagent context.

        Skips any span whose tool_use_id is in exclude_ids (live Agent spans).
        Called before starting a new LLM span for that context.
        """
        for tool_use_id in list(self._active_spans):
            if tool_use_id in exclude_ids:
                continue
            if self._active_spans[tool_use_id].parent_tool_use_id != parent_tool_use_id:
                continue
            self._end_tool_span(tool_use_id, end_time=end_time)

    def cleanup_all(self, end_time: float | None = None) -> None:
        """Close all remaining active spans. Called at end-of-stream."""
        for tool_use_id in list(self._active_spans):
            self._end_tool_span(tool_use_id, end_time=end_time)

    @property
    def has_active_spans(self) -> bool:
        return bool(self._active_spans)

    def acquire_span_for_handler(self, tool_name: Any, args: Any) -> _ActiveToolSpan | None:
        parsed_tool_name = _parse_tool_name(tool_name)
        candidate_names = list(
            dict.fromkeys((parsed_tool_name.raw_name, parsed_tool_name.display_name, str(tool_name)))
        )

        candidates = [
            active_tool_span
            for active_tool_span in self._active_spans.values()
            if not active_tool_span.handler_active
            and (active_tool_span.raw_name in candidate_names or active_tool_span.display_name in candidate_names)
        ]

        matched_span = self._match_via_dispatch_queue(parsed_tool_name.raw_name, args, candidates)
        if matched_span is None:
            matched_span = _match_tool_span_for_handler(candidates, args)
        if matched_span is None:
            return None

        matched_span.activate()
        return matched_span

    def _match_via_dispatch_queue(
        self, raw_name: str, args: Any, candidates: list[_ActiveToolSpan]
    ) -> _ActiveToolSpan | None:
        """Use the dispatch queue to match by tool_use_id when multiple identical
        candidates exist (same name + same input from different subagents)."""
        dispatch_key = _make_dispatch_key(raw_name, args)
        queue = self._dispatch_queues.get(dispatch_key)
        if not queue:
            return None

        # Pop tool_use_ids until we find one that corresponds to an available
        # (non-handler_active) candidate, skipping stale entries.
        candidate_ids = {c.tool_use_id for c in candidates}
        while queue:
            tool_use_id = queue.popleft()
            if tool_use_id in candidate_ids:
                for candidate in candidates:
                    if candidate.tool_use_id == tool_use_id:
                        return candidate

        return None

    def _end_tool_span(
        self, tool_use_id: str, tool_result_block: Any | None = None, end_time: float | None = None
    ) -> None:
        active_tool_span = self._active_spans.pop(tool_use_id, None)
        if active_tool_span is None:
            return

        self._completed_span_exports[tool_use_id] = active_tool_span.span.export()

        # Remove from dispatch queue so stale entries don't accumulate.
        dispatch_key = _make_dispatch_key(active_tool_span.raw_name, active_tool_span.input)
        queue = self._dispatch_queues.get(dispatch_key)
        if queue:
            try:
                queue.remove(tool_use_id)
            except ValueError:
                pass
            if not queue:
                del self._dispatch_queues[dispatch_key]

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
            return self._completed_span_exports.get(str(tool_use_id))

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


def _msg_field(message: Any, field: str) -> Any:
    """Read a field from a system message, falling back to message.data for older SDK versions.

    SDK >= 0.1.11 exposes TaskStartedMessage / TaskProgressMessage /
    TaskNotificationMessage with fields as top-level attributes.
    SDK 0.1.10 uses a flat SystemMessage(subtype, data=<full raw payload dict>)
    where task fields live directly in data (e.g. data["task_id"]).
    """
    value = getattr(message, field, None)
    if value is not None:
        return value
    # Older SDK: message.data is the full raw payload dict with task fields at its top level.
    data = getattr(message, "data", None)
    if isinstance(data, dict):
        return data.get(field)
    return None


def _task_span_name(message: Any, task_id: str) -> str:
    return _msg_field(message, "description") or _msg_field(message, "task_type") or f"Task {task_id}"


def _task_metadata(message: Any) -> dict[str, Any]:
    return {
        k: v
        for k, v in {
            "task_id": _msg_field(message, "task_id"),
            "session_id": _msg_field(message, "session_id"),
            "tool_use_id": _msg_field(message, "tool_use_id"),
            "task_type": _msg_field(message, "task_type"),
            "status": _msg_field(message, "status"),
            "last_tool_name": _msg_field(message, "last_tool_name"),
            "usage": _msg_field(message, "usage"),
        }.items()
        if v is not None
    }


def _task_output(message: Any) -> dict[str, Any] | None:
    summary = _msg_field(message, "summary")
    output_file = _msg_field(message, "output_file")

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


def _message_starts_subagent_tool(message: Any) -> bool:
    if not hasattr(message, "content"):
        return False

    for block in message.content:
        if type(block).__name__ != BlockClassName.TOOL_USE:
            continue
        if getattr(block, "name", None) == "Agent":
            return True

    return False


@dataclasses.dataclass
class _AgentContext:
    """Per-subagent-context state, keyed by parent_tool_use_id (None = orchestrator)."""

    llm_span: Any | None = None
    llm_parent_export: str | None = None
    llm_output: list[dict[str, Any]] | None = None
    next_llm_start: float | None = None
    task_span: Any | None = None
    task_confirmed: bool = False


class ContextTracker:
    """Single consumer of the raw SDK message stream. Maintains state and spans for the root agent context and any number of nested subagent contexts."""

    def __init__(
        self,
        root_span: Any,
        prompt: Any,
        query_start_time: float | None = None,
        captured_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._root_span = root_span
        self._root_span_export = root_span.export()
        self._prompt = prompt
        self._captured_messages = captured_messages  # logged to root span on first add()

        self._tool_tracker = ToolSpanTracker()
        self._contexts: dict[str | None, _AgentContext] = {None: _AgentContext(next_llm_start=query_start_time)}
        self._active_key: str | None = None
        self._task_order: list[str | None] = []

        self._final_results: list[dict[str, Any]] = []
        self._result_output: Any | None = None
        self._task_events: list[dict[str, Any]] = []

        _thread_local.tool_span_tracker = self._tool_tracker

    # -- public API --

    def add(self, message: Any) -> None:
        """Consume one SDK message and update spans accordingly."""
        if self._captured_messages is not None:
            if self._captured_messages:
                self._root_span.log(input=self._captured_messages)
            self._captured_messages = None

        message_type = type(message).__name__
        if message_type == MessageClassName.ASSISTANT:
            self._handle_assistant(message)
        elif message_type == MessageClassName.USER:
            self._handle_user(message)
        elif message_type == MessageClassName.RESULT:
            self._handle_result(message)
        elif message_type in SYSTEM_MESSAGE_TYPES:
            self._handle_system(message)

    def log_output(self) -> None:
        """Log the canonical root span output for the request."""
        if self._result_output is not None:
            self._root_span.log(output=self._result_output)
            return

        if self._final_results:
            self._root_span.log(output=self._final_results[-1])

    def log_tasks(self) -> None:
        """Flush accumulated task events to the root span metadata."""
        if self._task_events:
            self._root_span.log(metadata={"task_events": self._task_events})

    def cleanup(self) -> None:
        """End all open LLM spans, TASK spans, and TOOL spans; clear thread-local."""
        for ctx in self._contexts.values():
            if ctx.llm_span:
                ctx.llm_span.end()
                ctx.llm_span = None
            if ctx.task_span:
                ctx.task_span.end()
                ctx.task_span = None
        self._task_order.clear()
        self._tool_tracker.cleanup_all()
        if hasattr(_thread_local, "tool_span_tracker"):
            delattr(_thread_local, "tool_span_tracker")

    def get_tool_span_export(self, tool_use_id: str | None) -> str | None:
        return self._tool_tracker.get_span_export(tool_use_id)

    def current_llm_export(self) -> str | None:
        active_ctx = self._contexts.get(self._active_key)
        if active_ctx is not None and active_ctx.llm_span is not None:
            return active_ctx.llm_span.export()

        root_ctx = self._contexts.get(None)
        if root_ctx is not None and root_ctx.llm_span is not None:
            return root_ctx.llm_span.export()

        for key in reversed(self._task_order):
            ctx = self._contexts.get(key)
            if ctx is not None and ctx.llm_span is not None:
                return ctx.llm_span.export()

        for ctx in self._contexts.values():
            if ctx.llm_span is not None:
                return ctx.llm_span.export()

        return None

    def current_task_export(self) -> str | None:
        active_ctx = self._contexts.get(self._active_key)
        if active_ctx is not None and active_ctx.task_span is not None:
            return active_ctx.task_span.export()

        for key in reversed(self._task_order):
            ctx = self._contexts.get(key)
            if ctx is not None and ctx.task_span is not None:
                return ctx.task_span.export()

        return None

    # -- internal handlers --

    def _handle_assistant(self, message: Any) -> None:
        incoming_parent = getattr(message, "parent_tool_use_id", None)
        self._active_key = incoming_parent
        ctx = self._get_context(incoming_parent)

        # Close dangling tool spans from the previous turn in this context.
        if ctx.llm_span and self._tool_tracker.has_active_spans:
            self._tool_tracker.cleanup_context(
                incoming_parent,
                end_time=ctx.next_llm_start or time.time(),
                exclude_ids=self._live_agent_tool_use_ids(),
            )

        parent_export = self._llm_parent_for_message(message)
        final_content, extended = self._start_or_merge_llm_span(message, parent_export, ctx)

        llm_export = ctx.llm_span.export() if ctx.llm_span else None
        self._tool_tracker.start_tool_spans(message, llm_export)

        self._register_pending_agent_contexts(message)

        if final_content:
            if extended and self._final_results and self._final_results[-1].get("role") == "assistant":
                self._final_results[-1] = final_content
            else:
                self._final_results.append(final_content)

    def _handle_user(self, message: Any) -> None:
        self._tool_tracker.finish_tool_spans(message)
        has_tool_results = False
        if hasattr(message, "content"):
            has_tool_results = any(type(b).__name__ == BlockClassName.TOOL_RESULT for b in message.content)
            content = _serialize_content_blocks(message.content)
            self._final_results.append({"content": content, "role": "user"})
        if has_tool_results:
            user_parent = getattr(message, "parent_tool_use_id", None)
            resolved_key = user_parent if user_parent is not None else self._active_key
            self._get_context(resolved_key).next_llm_start = time.time()

    def _handle_result(self, message: Any) -> None:
        self._active_key = None
        if hasattr(message, "usage"):
            usage_metrics, usage_metadata = extract_anthropic_usage(message.usage)
            ctx = self._get_context(None)
            if ctx.llm_span and (usage_metrics or usage_metadata):
                ctx.llm_span.log(metrics=usage_metrics or None, metadata=usage_metadata or None)

        result_value = getattr(message, "result", None)
        if result_value is not None:
            self._result_output = result_value

        result_metadata = {
            k: v
            for k, v in {
                "num_turns": getattr(message, "num_turns", None),
                "session_id": getattr(message, "session_id", None),
            }.items()
            if v is not None
        }
        if result_metadata:
            self._root_span.log(metadata=result_metadata)

    def _handle_system(self, message: Any) -> None:
        agent_span_export = self._tool_tracker.get_span_export(_msg_field(message, "tool_use_id"))
        self._process_task_event(message, agent_span_export)
        self._task_events.append(_serialize_system_message(message))

    # -- internal helpers --

    def _get_context(self, key: str | None) -> _AgentContext:
        ctx = self._contexts.get(key)
        if ctx is None:
            ctx = _AgentContext()
            self._contexts[key] = ctx
        return ctx

    def _register_pending_agent_contexts(self, message: Any) -> None:
        """Pre-create _AgentContext for Agent tool calls (task_confirmed=False)."""
        if not hasattr(message, "content"):
            return
        for block in message.content:
            if type(block).__name__ == BlockClassName.TOOL_USE and getattr(block, "name", None) == "Agent":
                tool_use_id = getattr(block, "id", None)
                if tool_use_id:
                    self._get_context(str(tool_use_id))

    def _live_agent_tool_use_ids(self) -> frozenset[str]:
        """Return tool_use_ids of Agent spans that must not be closed yet."""
        result: set[str] = set()
        for key, ctx in self._contexts.items():
            if key is None:
                continue
            if not ctx.task_confirmed or ctx.task_span is not None:
                result.add(key)
        return frozenset(result)

    def _llm_parent_for_message(self, message: Any) -> str:
        """Determine the parent span export for an incoming AssistantMessage."""
        parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
        if parent_tool_use_id is not None:
            ctx = self._contexts.get(str(parent_tool_use_id))
            if ctx is not None and ctx.task_span is not None:
                return ctx.task_span.export()

        if _message_starts_subagent_tool(message):
            return self._root_span_export

        for key in reversed(self._task_order):
            ctx = self._contexts.get(key)
            if ctx is not None and ctx.task_span is not None:
                return ctx.task_span.export()

        return self._root_span_export

    def _start_or_merge_llm_span(
        self,
        message: Any,
        parent_export: str | None,
        ctx: _AgentContext,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Start a new LLM span or extend the existing one via merge."""
        current_message = _serialize_assistant_message(message)

        # Merge path.
        if (
            ctx.llm_span
            and ctx.next_llm_start is None
            and ctx.llm_parent_export == parent_export
            and current_message is not None
        ):
            merged = _merge_assistant_messages(
                ctx.llm_output[0] if ctx.llm_output else None,
                current_message,
            )
            if merged is not None:
                ctx.llm_output = [merged]
                ctx.llm_span.log(output=ctx.llm_output)
            return merged, True

        # New span path.
        resolved_start = ctx.next_llm_start or time.time()
        first_token_time = time.time()

        if ctx.llm_span:
            ctx.llm_span.end(end_time=resolved_start)

        final_content, span = _create_llm_span_for_messages(
            [message],
            self._prompt,
            self._final_results,
            parent=parent_export,
            start_time=resolved_start,
        )
        if span is not None:
            span.log(metrics={"time_to_first_token": max(0.0, first_token_time - resolved_start)})
        ctx.llm_span = span
        ctx.llm_parent_export = parent_export
        ctx.llm_output = [final_content] if final_content is not None else None
        ctx.next_llm_start = None
        return final_content, False

    def _process_task_event(self, message: Any, agent_span_export: str | None) -> None:
        """Handle TaskStarted / TaskProgress / TaskNotification system messages."""
        task_id = _msg_field(message, "task_id")
        if task_id is None:
            return
        task_id = str(task_id)
        tool_use_id = _msg_field(message, "tool_use_id")
        tool_use_id_str = str(tool_use_id) if tool_use_id is not None else None
        ctx = self._get_context(tool_use_id_str)
        message_type = type(message).__name__

        if ctx.task_span is None:
            ctx.task_span = start_span(
                name=_task_span_name(message, task_id),
                span_attributes={"type": SpanTypeAttribute.TASK},
                metadata=_task_metadata(message),
                parent=agent_span_export or self._root_span_export,
            )
            ctx.task_confirmed = True
            self._task_order.append(tool_use_id_str)
        else:
            update: dict[str, Any] = {}
            metadata = _task_metadata(message)
            if metadata:
                update["metadata"] = metadata
            output = _task_output(message)
            if output is not None:
                update["output"] = output
            if update:
                ctx.task_span.log(**update)

        if message_type == MessageClassName.TASK_NOTIFICATION:
            ctx.task_span.end()
            ctx.task_span = None
            self._task_order = [k for k in self._task_order if k != tool_use_id_str]


class RequestTracker:
    """Request-scoped tracker for hook callbacks and message-stream tracing."""

    def __init__(
        self,
        *,
        prompt: Any,
        query_start_time: float | None = None,
        captured_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._root_span = start_span(
            name=CLAUDE_AGENT_TASK_SPAN_NAME,
            span_attributes={"type": SpanTypeAttribute.TASK},
            input=prompt or None,
            start_time=query_start_time,
        )
        self._context_tracker = ContextTracker(
            root_span=self._root_span,
            prompt=prompt,
            query_start_time=query_start_time,
            captured_messages=captured_messages,
        )
        self._finished = False

    def add_message(self, message: Any) -> None:
        self._context_tracker.add(message)

    def log_error(self, exc: Exception) -> None:
        self._root_span.log(error=str(exc))

    async def trace_hook_callback(
        self,
        callback: Any,
        input_data: Any,
        tool_use_id: str | None,
        context: Any,
    ) -> Any:
        parent_export = self._hook_parent_export(tool_use_id)

        with start_span(
            name=f"{_hook_event_name(input_data)} hook",
            span_attributes={"type": SpanTypeAttribute.FUNCTION},
            input=_serialize_hook_value(input_data),
            metadata=_hook_metadata(callback, input_data, tool_use_id),
            parent=parent_export,
        ) as span:
            try:
                result = await callback(input_data, tool_use_id, context)
            except Exception as exc:
                span.log(error=str(exc))
                raise

            span.log(output=_serialize_hook_value(result))
            return result

    def finish(self, *, log_output: bool = False) -> None:
        if self._finished:
            return

        if log_output:
            self._context_tracker.log_output()
        self._context_tracker.log_tasks()
        self._context_tracker.cleanup()
        self._root_span.end()
        self._finished = True

    def _hook_parent_export(self, tool_use_id: str | None) -> str:
        tool_export = self._context_tracker.get_tool_span_export(tool_use_id)
        if tool_export is not None:
            return tool_export

        llm_export = self._context_tracker.current_llm_export()
        if llm_export is not None:
            return llm_export

        task_export = self._context_tracker.current_task_export()
        if task_export is not None:
            return task_export

        return self._root_span.export()


def _prepare_prompt_for_tracing(prompt: Any) -> tuple[Any, str | None, list[dict[str, Any]] | None]:
    if prompt is None:
        return None, None, None

    if isinstance(prompt, str):
        return prompt, prompt, None

    if isinstance(prompt, AsyncIterable):
        captured: list[dict[str, Any]] = []

        async def capturing_wrapper() -> AsyncGenerator[dict[str, Any], None]:
            async for msg in prompt:
                captured.append(msg)
                yield msg

        return capturing_wrapper(), None, captured

    return prompt, str(prompt), None


async def _stream_messages_with_tracing(
    generator: AsyncIterable[Any],
    *,
    request_tracker: RequestTracker,
    finish_request_tracker: Any,
) -> AsyncGenerator[Any, None]:
    try:
        async for message in generator:
            request_tracker.add_message(message)
            yield message
    except asyncio.CancelledError:
        # The CancelledError may come from the subprocess transport
        # (e.g., anyio internal cleanup when subagents complete) rather
        # than a genuine external cancellation. We suppress it here so
        # the response stream ends cleanly. If the caller genuinely
        # cancelled the task, they still have pending cancellation
        # requests that will fire at their next await point.
        finish_request_tracker(log_output=True)
    else:
        finish_request_tracker(log_output=True)
    finally:
        finish_request_tracker()


def _create_query_wrapper_function(original_query: Any) -> Any:
    """Create a tracing wrapper for the exported one-shot ``query()`` helper."""

    async def wrapped_query(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        query_start_time = time.time()
        prompt = args[0] if args else kwargs.get("prompt")
        prompt, traced_prompt, captured_messages = _prepare_prompt_for_tracing(prompt)

        if args:
            args = (prompt,) + args[1:]
        else:
            kwargs = dict(kwargs)
            kwargs["prompt"] = prompt

        request_tracker = RequestTracker(
            prompt=traced_prompt,
            query_start_time=query_start_time,
            captured_messages=captured_messages,
        )

        async for message in _stream_messages_with_tracing(
            original_query(*args, **kwargs),
            request_tracker=request_tracker,
            finish_request_tracker=request_tracker.finish,
        ):
            yield message

    return wrapped_query


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
            self.__request_tracker: RequestTracker | None = None
            self.__instrumented_hook_callbacks: set[tuple[int, str]] = set()

        def __instrument_hook_callbacks(self) -> None:
            query = getattr(self.__client, "_query", None)
            hook_callbacks = getattr(query, "hook_callbacks", None)
            if not isinstance(hook_callbacks, dict):
                return

            for callback_id, callback in list(hook_callbacks.items()):
                marker = (id(query), str(callback_id))
                if marker in self.__instrumented_hook_callbacks:
                    continue

                async def wrapped_callback(
                    input_data: Any,
                    tool_use_id: str | None,
                    context: Any,
                    *,
                    _callback: Any = callback,
                ) -> Any:
                    request_tracker = self.__request_tracker
                    if request_tracker is None:
                        return await _callback(input_data, tool_use_id, context)
                    return await request_tracker.trace_hook_callback(
                        _callback,
                        input_data,
                        tool_use_id,
                        context,
                    )

                hook_callbacks[callback_id] = wrapped_callback
                self.__instrumented_hook_callbacks.add(marker)

        def __start_request_tracker(self) -> RequestTracker:
            if self.__request_tracker is not None:
                self.__finish_request_tracker()

            self.__request_tracker = RequestTracker(
                prompt=self.__last_prompt,
                query_start_time=self.__query_start_time,
                captured_messages=self.__captured_messages,
            )
            return self.__request_tracker

        def __finish_request_tracker(self, *, log_output: bool = False) -> None:
            request_tracker = self.__request_tracker
            if request_tracker is None:
                return

            request_tracker.finish(log_output=log_output)
            self.__request_tracker = None

        async def connect(self, *args: Any, **kwargs: Any) -> Any:
            result = await self.__client.connect(*args, **kwargs)
            self.__instrument_hook_callbacks()
            return result

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            """Wrap query to capture the prompt and start time for tracing."""
            # Capture the time when query is called (when LLM call starts)
            self.__query_start_time = time.time()

            # Capture the prompt for use in receive_response
            prompt = args[0] if args else kwargs.get("prompt")
            prompt, self.__last_prompt, self.__captured_messages = _prepare_prompt_for_tracing(prompt)

            if args:
                args = (prompt,) + args[1:]
            else:
                kwargs["prompt"] = prompt

            self.__instrument_hook_callbacks()
            self.__start_request_tracker()

            try:
                return await self.__client.query(*args, **kwargs)
            except Exception as exc:
                if self.__request_tracker is not None:
                    self.__request_tracker.log_error(exc)
                self.__finish_request_tracker()
                raise

        async def receive_response(self) -> AsyncGenerator[Any, None]:
            """Wrap receive_response to add tracing via ContextTracker."""
            generator = self.__client.receive_response()
            request_tracker = self.__request_tracker or self.__start_request_tracker()

            async for message in _stream_messages_with_tracing(
                generator,
                request_tracker=request_tracker,
                finish_request_tracker=self.__finish_request_tracker,
            ):
                yield message

        async def __aenter__(self) -> "WrappedClaudeSDKClient":
            await self.__client.__aenter__()
            self.__instrument_hook_callbacks()
            return self

        async def disconnect(self) -> None:
            self.__finish_request_tracker()
            await self.__client.disconnect()

        async def __aexit__(self, *args: Any) -> None:
            self.__finish_request_tracker()
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

    Called by ContextTracker._start_or_merge_llm_span with an explicit parent export.
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
