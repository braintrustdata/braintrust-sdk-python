"""Strands Agents tracing helpers."""

import uuid
import weakref
from typing import Any

from braintrust.integrations.utils import _is_supported_metric_value
from braintrust.logger import Span
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute


_INSTRUMENTATION = "strands-auto"

_SPANS_BY_OTEL_SPAN: "weakref.WeakKeyDictionary[Any, Span]" = weakref.WeakKeyDictionary()
_SPANS_BY_INVALID_OTEL_KEY: dict[uuid.UUID, Span] = {}

_STRANDS_USAGE_KEYS = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "totalTokens": "total_tokens",
    "cacheReadInputTokens": "cache_read_input_tokens",
    "cacheCreationInputTokens": "cache_creation_input_tokens",
    "cacheWriteInputTokens": "cache_write_input_tokens",
}

_STRANDS_METRIC_KEYS: dict[str, str] = {"latencyMs": "latencyMs", "timeToFirstByteMs": "timeToFirstByteMs"}


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


class _InvalidOtelSpanKey:
    """Per-call key for OTEL's shared INVALID_SPAN singleton."""

    def __init__(self, otel_span: Any):
        self.key = uuid.uuid4()
        self._otel_span = otel_span

    def __getattr__(self, name: str) -> Any:
        return getattr(self._otel_span, name)

    def __eq__(self, other: Any) -> bool:
        return self._otel_span == other

    def __hash__(self) -> int:
        return hash(self.key)

    def __bool__(self) -> bool:
        return bool(self._otel_span)


def _arg(args: Any, kwargs: dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _pick_numeric(source: Any, keys: dict[str, str]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    result: dict[str, Any] = {}
    for src_key, dst_key in keys.items():
        value = source.get(src_key)
        if _is_supported_metric_value(value):
            result[dst_key] = value
    return result


def _agent_metadata_from_result(result: Any) -> dict[str, Any]:
    metrics_obj = getattr(result, "metrics", None)
    if metrics_obj is None:
        return {}
    metadata: dict[str, Any] = {}
    cycle_count = getattr(metrics_obj, "cycle_count", None)
    if _is_supported_metric_value(cycle_count):
        metadata["cycle_count"] = cycle_count
    strands_usage = _pick_numeric(getattr(metrics_obj, "accumulated_usage", None), _STRANDS_USAGE_KEYS)
    if strands_usage:
        metadata["strands_usage"] = strands_usage
    strands_metrics = _pick_numeric(getattr(metrics_obj, "accumulated_metrics", None), _STRANDS_METRIC_KEYS)
    if strands_metrics:
        metadata["strands_metrics"] = strands_metrics
    return metadata


def _is_valid_otel_span(otel_span: Any) -> bool:
    span_context = getattr(otel_span, "get_span_context", lambda: None)()
    return bool(getattr(span_context, "is_valid", False))


def _span_for_otel(otel_span: Any) -> Span | None:
    if otel_span is None:
        return None
    if isinstance(otel_span, _InvalidOtelSpanKey):
        return _SPANS_BY_INVALID_OTEL_KEY.get(otel_span.key)
    return _SPANS_BY_OTEL_SPAN.get(otel_span)


def _start_span_for_otel(
    otel_span: Any,
    *,
    name: str,
    span_type: str,
    input: Any = None,
    metadata: Any = None,
    parent_otel_span: Any = None,
) -> Any:
    if otel_span is None:
        return otel_span
    span_key = otel_span if _is_valid_otel_span(otel_span) else _InvalidOtelSpanKey(otel_span)
    parent = _span_for_otel(parent_otel_span)
    start = parent.start_span if parent is not None else start_span
    span = start(
        name=name,
        type=span_type,
        input=input,
        metadata=metadata,
        internal={"instrumentation": _INSTRUMENTATION},
    )
    if isinstance(span_key, _InvalidOtelSpanKey):
        _SPANS_BY_INVALID_OTEL_KEY[span_key.key] = span
    else:
        _SPANS_BY_OTEL_SPAN[span_key] = span
    return span_key


def _end_span_for_otel(
    otel_span: Any,
    *,
    output: Any = None,
    metadata: Any = None,
    error: BaseException | None = None,
) -> None:
    span = (
        _SPANS_BY_INVALID_OTEL_KEY.pop(otel_span.key, None)
        if isinstance(otel_span, _InvalidOtelSpanKey)
        else _SPANS_BY_OTEL_SPAN.pop(otel_span, None)
    )
    if span is None:
        return
    if error is not None:
        span.log(error=repr(error))
    span.log(output=output, metadata=metadata)
    span.end()


def _start_agent_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    otel_span = wrapped(*args, **kwargs)
    messages = _arg(args, kwargs, 0, "messages")
    agent_name = _arg(args, kwargs, 1, "agent_name")
    model_id = _arg(args, kwargs, 2, "model_id")
    metadata = {
        "agent_name": agent_name,
        "model": model_id,
        "tools": _arg(args, kwargs, 3, "tools"),
        "trace_attributes": _arg(args, kwargs, 4, "custom_trace_attributes"),
        "tools_config": _arg(args, kwargs, 5, "tools_config"),
    }
    return _start_span_for_otel(
        otel_span,
        name=f"{agent_name or 'Agent'}.invoke",
        span_type=SpanTypeAttribute.TASK,
        input={"messages": messages},
        metadata=metadata,
    )


def _end_agent_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    span = _arg(args, kwargs, 0, "span")
    response = _arg(args, kwargs, 1, "response")
    error = _arg(args, kwargs, 2, "error")
    try:
        return wrapped(*args, **kwargs)
    finally:
        output = (
            {
                "stop_reason": getattr(response, "stop_reason", None),
                "message": getattr(response, "message", None),
                "structured_output": getattr(response, "structured_output", None),
            }
            if response is not None
            else None
        )
        metadata = _agent_metadata_from_result(response)
        _end_span_for_otel(span, output=output, metadata=metadata, error=error)


def _start_event_loop_cycle_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    otel_span = wrapped(*args, **kwargs)
    invocation_state = _arg(args, kwargs, 0, "invocation_state")
    messages = _arg(args, kwargs, 1, "messages")
    parent_span = _arg(args, kwargs, 2, "parent_span")
    event_loop_cycle_id = None
    if isinstance(invocation_state, dict):
        if parent_span is None:
            parent_span = invocation_state.get("event_loop_parent_span")
        event_loop_cycle_id = invocation_state.get("event_loop_cycle_id")
    metadata = {
        "event_loop_cycle_id": str(event_loop_cycle_id) if event_loop_cycle_id is not None else None,
    }
    return _start_span_for_otel(
        otel_span,
        name="event_loop.cycle",
        span_type=SpanTypeAttribute.TASK,
        input={"messages": messages},
        metadata=metadata,
        parent_otel_span=parent_span,
    )


def _end_event_loop_cycle_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    span = _arg(args, kwargs, 0, "span")
    message = _arg(args, kwargs, 1, "message")
    tool_result_message = _arg(args, kwargs, 2, "tool_result_message")
    try:
        return wrapped(*args, **kwargs)
    finally:
        _end_span_for_otel(span, output={"message": message, "tool_result_message": tool_result_message})


def _start_model_invoke_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    otel_span = wrapped(*args, **kwargs)
    messages = _arg(args, kwargs, 0, "messages")
    parent_span = _arg(args, kwargs, 1, "parent_span")
    model_id = _arg(args, kwargs, 2, "model_id")
    metadata = {
        "model": model_id,
        "trace_attributes": _arg(args, kwargs, 3, "custom_trace_attributes"),
        "system_prompt": _arg(args, kwargs, 4, "system_prompt"),
        "system_prompt_content": _arg(args, kwargs, 5, "system_prompt_content"),
    }
    return _start_span_for_otel(
        otel_span,
        name=f"{model_id or 'Model'}.chat",
        span_type=SpanTypeAttribute.LLM,
        input={"messages": messages},
        metadata=metadata,
        parent_otel_span=parent_span,
    )


def _end_model_invoke_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    span = _arg(args, kwargs, 0, "span")
    message = _arg(args, kwargs, 1, "message")
    usage = _arg(args, kwargs, 2, "usage")
    metrics = _arg(args, kwargs, 3, "metrics")
    stop_reason = _arg(args, kwargs, 4, "stop_reason")
    try:
        return wrapped(*args, **kwargs)
    finally:
        metadata: dict[str, Any] = {
            "stop_reason": stop_reason,
            "strands_usage": _pick_numeric(usage, _STRANDS_USAGE_KEYS),
        }
        strands_metrics = _pick_numeric(metrics, _STRANDS_METRIC_KEYS)
        if strands_metrics:
            metadata["strands_metrics"] = strands_metrics
        _end_span_for_otel(span, output={"message": message}, metadata=metadata)


def _start_tool_call_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    otel_span = wrapped(*args, **kwargs)
    tool = _arg(args, kwargs, 0, "tool")
    parent_span = _arg(args, kwargs, 1, "parent_span")
    name = tool.get("name") if isinstance(tool, dict) else None
    tool_use_id = tool.get("toolUseId") if isinstance(tool, dict) else None
    return _start_span_for_otel(
        otel_span,
        name=f"{name or 'tool'}.execute",
        span_type=SpanTypeAttribute.TOOL,
        input=tool,
        metadata={"tool_name": name, "tool_use_id": tool_use_id},
        parent_otel_span=parent_span,
    )


def _end_tool_call_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    span = _arg(args, kwargs, 0, "span")
    tool_result = _arg(args, kwargs, 1, "tool_result")
    error = _arg(args, kwargs, 2, "error")
    try:
        return wrapped(*args, **kwargs)
    finally:
        _end_span_for_otel(
            span,
            output=tool_result,
            metadata={"status": tool_result.get("status") if isinstance(tool_result, dict) else None},
            error=error,
        )
