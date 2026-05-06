"""Strands Agents tracing helpers."""

import weakref
from typing import Any

from braintrust.integrations.utils import _is_supported_metric_value
from braintrust.logger import Span, start_span
from braintrust.span_types import SpanTypeAttribute


_SPANS_BY_OTEL_SPAN: "weakref.WeakKeyDictionary[Any, Span]" = weakref.WeakKeyDictionary()


def _arg(args: Any, kwargs: dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _strands_usage_from_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    strands_usage: dict[str, Any] = {}
    mapping = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "totalTokens": "total_tokens",
        "cacheReadInputTokens": "cache_read_input_tokens",
        "cacheCreationInputTokens": "cache_creation_input_tokens",
    }
    for source, target in mapping.items():
        value = usage.get(source)
        if _is_supported_metric_value(value):
            strands_usage[target] = value
    return strands_usage


def _agent_metrics_and_metadata(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_obj = getattr(result, "metrics", None)
    metrics = metrics_obj
    if not isinstance(metrics, dict):
        return {}, {}

    bt_metrics: dict[str, Any] = {}
    cycles = metrics.get("cycle_count") or metrics.get("cycleCount")
    if _is_supported_metric_value(cycles):
        bt_metrics["cycles"] = cycles

    usage = metrics.get("accumulated_usage") or metrics.get("usage") or metrics.get("accumulatedUsage")
    metadata = {"strands_usage": _strands_usage_from_usage(usage)}
    return bt_metrics, metadata


def _span_for_otel(otel_span: Any) -> Span | None:
    if otel_span is None:
        return None
    return _SPANS_BY_OTEL_SPAN.get(otel_span)


def _start_span_for_otel(otel_span: Any, *, name: str, span_type: str, input: Any = None, metadata: Any = None) -> Any:
    if otel_span is None:
        return otel_span
    parent = None
    # Strands passes parent OTEL spans into child start methods. If present, nest under the mirrored BT span.
    if isinstance(metadata, dict):
        parent_otel = metadata.pop("_bt_parent_otel_span", None)
        parent = _span_for_otel(parent_otel)
    span = (parent.start_span if parent is not None else start_span)(
        name=name, type=span_type, input=input, metadata=metadata
    )
    _SPANS_BY_OTEL_SPAN[otel_span] = span
    return otel_span


def _end_span_for_otel(
    otel_span: Any,
    *,
    output: Any = None,
    metadata: Any = None,
    metrics: Any = None,
    error: BaseException | None = None,
) -> None:
    span = _SPANS_BY_OTEL_SPAN.pop(otel_span, None)
    if span is None:
        return
    if error is not None:
        span.log(error=repr(error))
    span.log(output=output, metadata=metadata, metrics=metrics)
    span.end()


def _start_agent_span_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    otel_span = wrapped(*args, **kwargs)
    messages = _arg(args, kwargs, 0, "messages")
    agent_name = _arg(args, kwargs, 1, "agent_name")
    model_id = _arg(args, kwargs, 2, "model_id")
    metadata = {
        "agent_name": agent_name,
        "model": model_id,
        "tools": kwargs.get("tools"),
        "tools_config": kwargs.get("tools_config"),
        "trace_attributes": kwargs.get("custom_trace_attributes"),
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
        metrics, metadata = _agent_metrics_and_metadata(response)
        _end_span_for_otel(span, output=output, metadata=metadata, metrics=metrics, error=error)


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
        "_bt_parent_otel_span": parent_span,
    }
    return _start_span_for_otel(
        otel_span,
        name="event_loop.cycle",
        span_type=SpanTypeAttribute.TASK,
        input={"messages": messages},
        metadata=metadata,
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
        "system_prompt": kwargs.get("system_prompt"),
        "system_prompt_content": kwargs.get("system_prompt_content"),
        "trace_attributes": kwargs.get("custom_trace_attributes"),
        "_bt_parent_otel_span": parent_span,
    }
    return _start_span_for_otel(
        otel_span,
        name=f"{model_id or 'Model'}.chat",
        span_type=SpanTypeAttribute.LLM,
        input={"messages": messages},
        metadata=metadata,
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
        bt_metrics = {}
        if isinstance(metrics, dict):
            bt_metrics.update({k: v for k, v in metrics.items() if _is_supported_metric_value(v)})
        metadata = {"stop_reason": stop_reason, "strands_usage": _strands_usage_from_usage(usage)}
        _end_span_for_otel(span, output={"message": message}, metadata=metadata, metrics=bt_metrics)


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
        metadata={"tool_name": name, "tool_use_id": tool_use_id, "_bt_parent_otel_span": parent_span},
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
