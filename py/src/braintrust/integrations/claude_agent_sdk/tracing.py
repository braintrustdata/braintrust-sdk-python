import asyncio
import collections
import contextvars
import dataclasses
import json
import math
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from braintrust.integrations.anthropic._utils import Wrapper, extract_anthropic_usage
from braintrust.integrations.claude_agent_sdk._constants import (
    ANTHROPIC_MESSAGES_CREATE_SPAN_NAME,
    CLAUDE_AGENT_RUN_FAILED_ERROR,
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
from braintrust.logger import start_span as _bt_start_span


_INSTRUMENTATION = "claude-agent-sdk-auto"


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


from braintrust.span_types import SpanTypeAttribute


_thread_local = threading.local()
_request_tracker_context: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "braintrust_claude_agent_sdk_request_tracker",
    default=None,
)
_tool_span_tracker_context: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "braintrust_claude_agent_sdk_tool_span_tracker",
    default=None,
)


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


# Top-level allowlist for hook payloads (both `input_data` handed to the hook
# and whatever the hook returns). Anything outside this set — notably
# `cwd`, `transcript_path`, and `agent_transcript_path` — is dropped. Applied
# only at the top level so nested values (e.g. arbitrary keys inside
# `tool_input` or `hookSpecificOutput`) pass through and reach Braintrust's
# log-time serializer intact.
_HOOK_ALLOWED_FIELDS = frozenset(
    {
        # Common input fields
        "session_id",
        "hook_event_name",
        # Event-specific input fields
        "tool_name",
        "tool_input",
        "tool_response",
        "prompt",
        "message",
        "stop_hook_active",
        "trigger",
        "custom_instructions",
        "source",
        "reason",
        "agent_id",
        "agent_type",
        # Hook return fields (Claude Code hook protocol)
        "continue",
        "stopReason",
        "suppressOutput",
        "decision",
        "hookSpecificOutput",
        "systemMessage",
    }
)


def _serialize_hook_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if str(key) in _HOOK_ALLOWED_FIELDS}
    return value


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
        tool_span_tracker = _tool_span_tracker_context.get()
        if tool_span_tracker is None:
            tool_span_tracker = getattr(_thread_local, "tool_span_tracker", None)

        active_tool_span = (
            tool_span_tracker.acquire_span_for_handler(tool_name, args) if tool_span_tracker is not None else None
        )
        if active_tool_span is None:
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
        # Per-(display_name, input_signature) FIFO queue of tool_use_ids.
        # SDK MCP handlers receive the bare display name while stream messages
        # use names like ``mcp__server__tool``, so both sides key by display name.
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
            dispatch_key = _make_dispatch_key(parsed_tool_name.display_name, tool_input)
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

        matched_span = self._match_via_dispatch_queue(parsed_tool_name.display_name, args, candidates)
        if matched_span is None:
            matched_span = _match_tool_span_for_handler(candidates, args)
        if matched_span is None:
            return None

        matched_span.activate()
        return matched_span

    def _match_via_dispatch_queue(
        self, display_name: str, args: Any, candidates: list[_ActiveToolSpan]
    ) -> _ActiveToolSpan | None:
        """Use the dispatch queue to match by tool_use_id when multiple identical
        candidates exist (same name + same input from different subagents)."""
        dispatch_key = _make_dispatch_key(display_name, args)
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
        dispatch_key = _make_dispatch_key(active_tool_span.display_name, active_tool_span.input)
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


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or int(value) != value:
        return None
    return int(value)


def _copy_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if not isinstance(usage, dict):
        try:
            usage = vars(usage)
        except TypeError:
            return None

    copied: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        value = _token_count(usage.get(key))
        if value is not None:
            copied[key] = value

    cache_creation = usage.get("cache_creation")
    if cache_creation is not None and not isinstance(cache_creation, dict):
        try:
            cache_creation = vars(cache_creation)
        except TypeError:
            cache_creation = None
    if isinstance(cache_creation, dict):
        copied_cache_creation = {}
        for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
            value = _token_count(cache_creation.get(key))
            if value is not None:
                copied_cache_creation[key] = value
        if copied_cache_creation:
            copied["cache_creation"] = copied_cache_creation

    return copied or None


def _merge_usage(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    if base is None or override is None:
        return override or base
    merged = {**base, **override}
    base_cache = base.get("cache_creation")
    override_cache = override.get("cache_creation")
    if isinstance(base_cache, dict) or isinstance(override_cache, dict):
        merged["cache_creation"] = {
            **(base_cache if isinstance(base_cache, dict) else {}),
            **(override_cache if isinstance(override_cache, dict) else {}),
        }
    return merged


def _aggregate_model_usage(model_usage: Any) -> dict[str, int] | None:
    """Aggregate the SDK's all-agent model usage into Anthropic usage fields."""
    if not isinstance(model_usage, dict):
        return None

    field_names = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "cacheReadInputTokens": "cache_read_input_tokens",
        "cacheCreationInputTokens": "cache_creation_input_tokens",
    }
    totals: dict[str, int] = {}
    for raw_usage in model_usage.values():
        if not isinstance(raw_usage, dict):
            try:
                raw_usage = vars(raw_usage)
            except TypeError:
                continue
        for source_name, target_name in field_names.items():
            value = _token_count(raw_usage.get(source_name))
            if value is not None:
                totals[target_name] = totals.get(target_name, 0) + value
    return totals or None


@dataclasses.dataclass
class _AgentContext:
    """Per-subagent-context state, keyed by parent_tool_use_id (None = orchestrator)."""

    llm_span: Any | None = None
    llm_parent_export: str | None = None
    llm_output: list[dict[str, Any]] | None = None
    llm_message_id: str | None = None
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
        include_partial_messages: bool = False,
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
        self._include_partial_messages = include_partial_messages
        self._active_partial_message_id_by_parent: dict[str | None, str] = {}
        self._latest_partial_message_id_by_parent: dict[str | None, str] = {}
        self._usage_by_message_id: dict[str, dict[str, Any]] = {}
        self._final_output_usage_message_ids: set[str] = set()

        _thread_local.tool_span_tracker = self._tool_tracker

    # -- public API --

    def add(self, message: Any) -> None:
        """Consume one SDK message and update spans accordingly."""
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
        elif message_type == "StreamEvent":
            self._handle_stream_event(message)
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
        self._active_partial_message_id_by_parent.clear()
        self._latest_partial_message_id_by_parent.clear()
        self._usage_by_message_id.clear()
        self._final_output_usage_message_ids.clear()
        if getattr(_thread_local, "tool_span_tracker", None) is self._tool_tracker:
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
        #
        # Only run cleanup when a user tool_result has advanced the clock
        # (``ctx.next_llm_start`` is set), which marks a real turn boundary.
        # Starting in claude-agent-sdk 0.1.61, a subagent may emit multiple
        # ``tool_use`` AssistantMessages back-to-back (merged into one LLM
        # turn) before any ``tool_result`` arrives. In that case the earlier
        # tool spans are still awaiting their results and must not be closed.
        if ctx.llm_span and ctx.next_llm_start is not None and self._tool_tracker.has_active_spans:
            self._tool_tracker.cleanup_context(
                incoming_parent,
                end_time=ctx.next_llm_start,
                exclude_ids=self._live_agent_tool_use_ids(),
            )

        parent_export = self._llm_parent_for_message(message)
        final_content, extended = self._start_or_merge_llm_span(message, parent_export, ctx)

        message_error = getattr(message, "error", None)
        if message_error and ctx.llm_span is not None:
            ctx.llm_span.log(error=str(message_error))

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

    def _handle_stream_event(self, message: Any) -> None:
        event = getattr(message, "event", None)
        if not isinstance(event, dict):
            return
        parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
        event_type = event.get("type")
        if event_type == "message_start":
            raw_message = event.get("message")
            if not isinstance(raw_message, dict):
                return
            message_id = raw_message.get("id")
            if not isinstance(message_id, str):
                return
            self._active_partial_message_id_by_parent[parent_tool_use_id] = message_id
            self._latest_partial_message_id_by_parent[parent_tool_use_id] = message_id
            usage = _copy_usage(raw_message.get("usage"))
            if usage:
                self._usage_by_message_id[message_id] = usage
            return

        message_id = self._active_partial_message_id_by_parent.get(parent_tool_use_id)
        if message_id is None:
            return
        if event_type == "message_delta":
            update = _copy_usage(event.get("usage"))
            if update:
                usage = _merge_usage(self._usage_by_message_id.get(message_id), update) or {}
                self._usage_by_message_id[message_id] = usage
                if "output_tokens" in update:
                    self._final_output_usage_message_ids.add(message_id)
                    ctx = self._contexts.get(parent_tool_use_id)
                    if ctx is not None and ctx.llm_span is not None and ctx.llm_message_id == message_id:
                        metrics, _ = extract_anthropic_usage(usage)
                        ctx.llm_span.log(metrics=metrics or None)
        elif event_type == "message_stop":
            self._active_partial_message_id_by_parent.pop(parent_tool_use_id, None)

    def _handle_result(self, message: Any) -> None:
        self._active_key = None
        result_value = getattr(message, "result", None)
        if result_value is not None:
            self._result_output = result_value

        result_metadata = {
            k: v
            for k, v in {
                "num_turns": getattr(message, "num_turns", None),
                "session_id": getattr(message, "session_id", None),
                "stop_reason": getattr(message, "stop_reason", None),
                "total_cost_usd": getattr(message, "total_cost_usd", None),
                "duration_ms": getattr(message, "duration_ms", None),
                "duration_api_ms": getattr(message, "duration_api_ms", None),
            }.items()
            if v is not None
        }
        result_metrics: dict[str, float] = {}
        if not self._include_partial_messages:
            raw_usage = getattr(message, "usage", None)
            _, usage_metadata = extract_anthropic_usage(raw_usage)
            result_metadata.update(usage_metadata)
            aggregate_usage = _aggregate_model_usage(getattr(message, "model_usage", None))
            usage = aggregate_usage or _copy_usage(raw_usage)
            result_metrics, _ = extract_anthropic_usage(usage)
        if result_metadata or result_metrics:
            self._root_span.log(metadata=result_metadata or None, metrics=result_metrics or None)

        if getattr(message, "is_error", None) is True:
            error_text = (
                result_value if isinstance(result_value, str) and result_value else CLAUDE_AGENT_RUN_FAILED_ERROR
            )
            self._root_span.log(error=error_text)

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
        """Start one LLM span per provider message ID and merge its snapshots."""
        current_message = _serialize_assistant_message(message)
        message_id = getattr(message, "message_id", None)
        if not isinstance(message_id, str):
            message_id = self._latest_partial_message_id_by_parent.get(getattr(message, "parent_tool_use_id", None))
        same_provider_message = (
            isinstance(message_id, str)
            and message_id == ctx.llm_message_id
            or message_id is None
            and ctx.llm_message_id is None
            and ctx.next_llm_start is None
        )

        if (
            ctx.llm_span
            and same_provider_message
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
            self._log_assistant_usage(message, ctx, message_id)
            return merged, True

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
        ctx.llm_message_id = message_id if isinstance(message_id, str) else None
        ctx.next_llm_start = None
        self._log_assistant_usage(message, ctx, ctx.llm_message_id)
        return final_content, False

    def _log_assistant_usage(
        self,
        message: Any,
        ctx: _AgentContext,
        message_id: str | None,
    ) -> None:
        if ctx.llm_span is None or not self._include_partial_messages:
            return
        raw_message_usage = getattr(message, "usage", None)
        usage = _merge_usage(
            _copy_usage(raw_message_usage),
            self._usage_by_message_id.get(message_id) if message_id else None,
        )
        if not usage:
            return
        has_final_output = message_id in self._final_output_usage_message_ids if message_id else False
        metrics, _ = extract_anthropic_usage(usage, include_output=has_final_output)
        _, metadata = extract_anthropic_usage(raw_message_usage, include_output=False)
        ctx.llm_span.log(metrics=metrics or None, metadata=metadata or None)

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
        include_partial_messages: bool = False,
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
            include_partial_messages=include_partial_messages,
        )
        self._pretraced_message_types: collections.deque[str] = collections.deque()
        self._finished = False

    def add_message(self, message: Any) -> None:
        message_type = type(message).__name__
        if self._pretraced_message_types and self._pretraced_message_types[0] == message_type:
            self._pretraced_message_types.popleft()
            return
        self._context_tracker.add(message)

    def add_raw_message(self, data: Any) -> None:
        """Trace one raw SDK message before it is queued for application consumption."""
        if self._finished or not isinstance(data, dict):
            return

        _tool_span_tracker_context.set(self._context_tracker._tool_tracker)

        try:
            from claude_agent_sdk._internal.message_parser import parse_message

            message = parse_message(data)
            if message is None:
                return
            self._context_tracker.add(message)
        except Exception:
            # Provider message parsing and instrumentation must not interrupt
            # the SDK's reader task.
            return

        self._pretraced_message_types.append(type(message).__name__)

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
        self._pretraced_message_types.clear()

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


class _TracingMessageSendStream:
    """Observe raw SDK messages before the provider queues them for consumers."""

    def __init__(self, send_stream: Any, query: Any) -> None:
        self._send_stream = send_stream
        self._query = query
        self._braintrust_wrapped = True

    async def send(self, value: Any) -> None:
        request_tracker = getattr(self._query, "_braintrust_request_tracker", None)
        if request_tracker is not None:
            request_tracker.add_raw_message(value)
        await self._send_stream.send(value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._send_stream, name)


def _install_query_message_tracing(query: Any) -> None:
    if query is None:
        return
    send_stream = getattr(query, "_message_send", None)
    if send_stream is None or getattr(send_stream, "_braintrust_wrapped", False):
        return
    query._message_send = _TracingMessageSendStream(send_stream, query)


def _wrap_query_read_messages(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """Install request-context-aware raw message observation on this SDK reader."""
    request_tracker = _request_tracker_context.get()
    if request_tracker is not None:
        instance._braintrust_request_tracker = request_tracker
    _install_query_message_tracing(instance)
    return wrapped(*args, **kwargs)


async def _bind_request_tracker_to_query(
    generator: AsyncIterable[Any], request_tracker: RequestTracker
) -> AsyncGenerator[Any, None]:
    """Bind the query reader during startup without retaining a ContextVar token across yields."""
    iterator = generator.__aiter__()
    token = _request_tracker_context.set(request_tracker)
    try:
        try:
            first_message = await anext(iterator)
        except StopAsyncIteration:
            return
    finally:
        _request_tracker_context.reset(token)

    yield first_message
    async for message in iterator:
        yield message


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


def _include_partial_messages(options: Any) -> bool:
    return getattr(options, "include_partial_messages", False) is True


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

        options = kwargs.get("options")
        if options is None and len(args) > 1:
            options = args[1]

        request_tracker = RequestTracker(
            prompt=traced_prompt,
            query_start_time=query_start_time,
            captured_messages=captured_messages,
            include_partial_messages=_include_partial_messages(options),
        )
        generator = _bind_request_tracker_to_query(original_query(*args, **kwargs), request_tracker)

        async for message in _stream_messages_with_tracing(
            generator,
            request_tracker=request_tracker,
            finish_request_tracker=request_tracker.finish,
        ):
            yield message

    return wrapped_query


def _create_client_wrapper_class(original_client_class: Any) -> Any:
    """Creates a wrapper class for ClaudeSDKClient that wraps query and receive_response."""

    class WrappedClaudeSDKClient(Wrapper):
        def __init__(self, *args: Any, **kwargs: Any):
            options = args[0] if args else kwargs.get("options")
            self.__include_partial_messages = _include_partial_messages(options)
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

        def __prepare_request_prompt(
            self, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            self.__query_start_time = time.time()
            prompt = args[0] if args else kwargs.get("prompt")
            prompt, self.__last_prompt, self.__captured_messages = _prepare_prompt_for_tracing(prompt)
            if args:
                return (prompt, *args[1:]), kwargs
            prepared_kwargs = dict(kwargs)
            prepared_kwargs["prompt"] = prompt
            return args, prepared_kwargs

        def __start_request_tracker(self) -> RequestTracker:
            if self.__request_tracker is not None:
                self.__finish_request_tracker()

            self.__request_tracker = RequestTracker(
                prompt=self.__last_prompt,
                query_start_time=self.__query_start_time,
                captured_messages=self.__captured_messages,
                include_partial_messages=self.__include_partial_messages,
            )
            query = getattr(self.__client, "_query", None)
            _install_query_message_tracing(query)
            if query is not None:
                query._braintrust_request_tracker = self.__request_tracker
            return self.__request_tracker

        def __finish_request_tracker(self, *, log_output: bool = False) -> None:
            request_tracker = self.__request_tracker
            if request_tracker is None:
                return

            query = getattr(self.__client, "_query", None)
            if query is not None and getattr(query, "_braintrust_request_tracker", None) is request_tracker:
                delattr(query, "_braintrust_request_tracker")
            request_tracker.finish(log_output=log_output)
            self.__request_tracker = None

        async def connect(self, *args: Any, **kwargs: Any) -> Any:
            prompt = args[0] if args else kwargs.get("prompt")
            if prompt is not None:
                args, kwargs = self.__prepare_request_prompt(args, kwargs)
                self.__start_request_tracker()
            try:
                result = await self.__client.connect(*args, **kwargs)
            except Exception as exc:
                if self.__request_tracker is not None:
                    self.__request_tracker.log_error(exc)
                self.__finish_request_tracker()
                raise

            query = getattr(self.__client, "_query", None)
            _install_query_message_tracing(query)
            if query is not None and self.__request_tracker is not None:
                query._braintrust_request_tracker = self.__request_tracker
            self.__instrument_hook_callbacks()
            return result

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            """Wrap query to capture the prompt and start time for tracing."""
            args, kwargs = self.__prepare_request_prompt(args, kwargs)
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
            _install_query_message_tracing(getattr(self.__client, "_query", None))
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

    metadata: dict[str, Any] = {"provider": "anthropic"}
    if model:
        metadata["model"] = model

    llm_span = start_span(
        name=ANTHROPIC_MESSAGES_CREATE_SPAN_NAME,
        span_attributes={"type": SpanTypeAttribute.LLM},
        input=input_messages,
        output=outputs,
        metadata=metadata,
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
    """Convert Claude Agent SDK content-block dataclasses into readable dicts.

    Shallow field copy only. Braintrust serializes nested values (including
    ToolUseBlock.input payloads and any nested content-block dataclasses) at
    log time via bt_json, so we do not recursively walk here. We only add the
    provider-shaped ``type`` discriminator that bt_json cannot infer, and
    apply the ToolResult flatten/prune rules that require field-level checks.
    """
    if not isinstance(content, list):
        return content
    return [_serialize_block(block) for block in content]


def _serialize_block(block: Any) -> Any:
    if not dataclasses.is_dataclass(block):
        return block

    serialized = {f.name: getattr(block, f.name) for f in dataclasses.fields(block)}

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

    return serialized


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
