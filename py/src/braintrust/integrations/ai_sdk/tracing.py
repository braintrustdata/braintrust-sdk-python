"""Translate Vercel AI SDK telemetry spans into Braintrust spans."""

import json
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from braintrust.integrations.utils import (
    _is_supported_metric_value,
    _materialize_attachment,
    _parse_openai_usage_metrics,
)
from braintrust.logger import Span
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import clean_nones


if TYPE_CHECKING:
    from ai.types.messages import (  # pylint: disable=import-error
        BuiltinToolCallPart,
        BuiltinToolReturnPart,
        FilePart,
        Part,
        ToolCallPart,
        ToolResultPart,
    )


_INSTRUMENTATION = "ai-sdk-auto"
# Parents are only consulted by children that start after the parent ended
# (durable replay, out-of-order pushes), so this holds finished spans just
# long enough for one replayed batch.
_MAX_ARCHIVED_PARENTS = 1_000
_CONTENT_PART_KINDS = frozenset({"text", "reasoning", "file", "hook"})
_TOOL_CALL_KINDS = frozenset({"tool_call", "builtin_tool_call"})
_TOOL_RESULT_KINDS = frozenset({"tool_result", "builtin_tool_return"})
_USAGE_NAME_MAP = {
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "total_tokens": "tokens",
    "reasoning_tokens": "completion_reasoning_tokens",
    "cache_read_tokens": "prompt_cached_tokens",
    "cache_write_tokens": "prompt_cache_creation_tokens",
}
_USAGE_PREFIX_MAP = {"input": "prompt", "output": "completion"}


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_int(value: Any) -> bool:
    """True for a real integer; ``bool`` never counts as one here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_file_part(part: "FilePart | Mapping[str, Any]") -> dict[str, Any]:
    data = _value(part, "data")
    media_type = _value(part, "media_type") or "application/octet-stream"
    filename = _value(part, "filename")
    resolved = _materialize_attachment(data, mime_type=media_type, filename=filename)

    if resolved is not None:
        return {"type": "image_url" if resolved.is_image else "file", **resolved.multimodal_part_payload}
    if str(media_type).startswith("image/"):
        return {"type": "image_url", "image_url": {"url": data}}
    return {"type": "file", "file": {"file_data": data, "filename": filename or "file"}}


def _content_part(part: "Part | Mapping[str, Any]") -> Any:
    kind = _value(part, "kind")
    if kind in ("text", "reasoning"):
        return {"type": kind, "text": _value(part, "text", "")}
    if kind == "file":
        return _shape_file_part(part)
    if kind == "hook":
        return {
            "type": "hook",
            "hook_id": _value(part, "hook_id"),
            "hook_type": _value(part, "hook_type"),
            "status": _value(part, "status"),
            "resolution": _value(part, "resolution"),
        }
    return part


def _tool_call(part: "ToolCallPart | BuiltinToolCallPart | Mapping[str, Any]") -> dict[str, Any]:
    return {
        "id": _value(part, "tool_call_id"),
        "type": "function",
        "function": {
            "name": _value(part, "tool_name"),
            "arguments": _value(part, "tool_args", ""),
        },
    }


def _shape_message(message: Any) -> dict[str, Any]:
    role = _value(message, "role", "assistant")
    text_chunks: list[str] = []
    content_source: list[Any] = []
    calls: list[dict[str, Any]] = []
    all_text = True
    for part in _value(message, "parts", []) or []:
        kind = _value(part, "kind")
        if kind in _TOOL_CALL_KINDS:
            calls.append(_tool_call(part))
        elif kind in _CONTENT_PART_KINDS:
            content_source.append(part)
            if kind == "text":
                text_chunks.append(_value(part, "text", ""))
            else:
                all_text = False

    if not content_source:
        content: Any = None
    elif all_text:
        content = "".join(text_chunks)
    else:
        content = [_content_part(part) for part in content_source]

    shaped: dict[str, Any] = {"role": role, "content": content}
    if calls:
        shaped["tool_calls"] = calls
    return shaped


def _tool_result_content(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    return json.dumps(result, separators=(",", ":"), default=str)


def _tool_result_model_input(part: "ToolResultPart | BuiltinToolReturnPart | Mapping[str, Any]") -> Any:
    get_model_input = getattr(part, "get_model_input", None)
    if callable(get_model_input):
        return get_model_input()
    if isinstance(part, Mapping) and "model_input" in part:
        return part["model_input"]
    return _value(part, "result")


def _shape_input_messages(messages: Any) -> list[dict[str, Any]]:
    shaped: list[dict[str, Any]] = []
    for message in messages or []:
        if _value(message, "role") == "tool":
            tool_results = [
                part for part in _value(message, "parts", []) or [] if _value(part, "kind") in _TOOL_RESULT_KINDS
            ]
            if tool_results:
                shaped.extend(
                    {
                        "role": "tool",
                        "tool_call_id": _value(part, "tool_call_id"),
                        "content": _tool_result_content(_tool_result_model_input(part)),
                    }
                    for part in tool_results
                )
                continue
        shaped.append(_shape_message(message))
    return shaped


def _normalize_finish_reason(reason: Any, shaped_message: Mapping[str, Any]) -> str:
    if reason in {"tool_call", "tool_calls"}:
        return "tool_calls"
    if isinstance(reason, str) and reason:
        return reason
    return "tool_calls" if shaped_message.get("tool_calls") else "stop"


def _shape_completion_output(message: Any, finish_reason: Any = None) -> list[dict[str, Any]] | None:
    if message is None:
        return None
    shaped = _shape_message(message)
    return [
        {
            "index": 0,
            "finish_reason": _normalize_finish_reason(finish_reason, shaped),
            "message": shaped,
        }
    ]


def _tool_definitions(names: Any) -> list[dict[str, Any]] | None:
    if not names:
        return None
    return [{"type": "function", "function": {"name": str(name)}} for name in names]


def _usage_metrics(usage: Any) -> dict[str, int | float]:
    parsed = _parse_openai_usage_metrics(usage, token_name_map=_USAGE_NAME_MAP, token_prefix_map=_USAGE_PREFIX_MAP)
    metrics = {name: value for name, value in parsed.items() if _is_supported_metric_value(value) and value >= 0}
    if "tokens" not in metrics and "prompt_tokens" in metrics and "completion_tokens" in metrics:
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    return metrics


def _time_to_first_token(span: Any) -> float | None:
    started_at = _value(span, "started_at")
    if started_at is None:
        return None
    for event in _value(span, "events", []) or []:
        if _value(event, "name") == "first_token":
            event_time = _value(event, "time_ns")
            if isinstance(event_time, int):
                return max(0.0, (event_time - started_at) / 1_000_000_000)
    return None


def _sampling_metadata(params: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    sampling = _value(params, "sampling")
    samplers = sampling.values() if isinstance(sampling, Mapping) else []
    for sampler in samplers:
        for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            value = _value(sampler, name)
            if _is_supported_metric_value(value):
                metadata[name] = value

    max_tokens = _value(_value(params, "output"), "max_tokens")
    if _is_int(max_tokens):
        metadata["max_tokens"] = max_tokens

    tool_calling = _value(params, "tool_calling")
    tool_choice = _value(tool_calling, "tool_choice")
    if tool_choice is not None:
        choice_string = str(_value(tool_choice, "value", tool_choice))
        if choice_string in {"auto", "none", "required"}:
            metadata["tool_choice"] = choice_string
        elif choice_string:
            metadata["tool_choice"] = {"type": "function", "function": {"name": choice_string}}
    parallel_tool_calls = _value(tool_calling, "parallel_tool_calls")
    if isinstance(parallel_tool_calls, bool):
        metadata["parallel_tool_calls"] = parallel_tool_calls
    max_tool_calls = _value(tool_calling, "max_tool_calls")
    if _is_int(max_tool_calls):
        metadata["max_tool_calls"] = max_tool_calls
    return metadata


def _request_metadata(data: Any) -> dict[str, Any]:
    """Metadata known when the call is issued."""
    optional = {
        "tools": _tool_definitions(_value(data, "tool_names")),
        "output_type": _value(data, "output_type"),
    }
    return {
        "model": _value(data, "model") or "unknown",
        "provider": _value(data, "provider") or "unknown",
        **_sampling_metadata(_value(data, "params")),
        **{name: value for name, value in optional.items() if value},
    }


def _response_metadata(data: Any) -> dict[str, Any]:
    """Metadata the provider only fills in once the call completes."""
    optional = {
        "model": _value(data, "response_model"),
        "response_id": _value(data, "response_id"),
        "finish_reason": _value(data, "finish_reason"),
    }
    return {name: value for name, value in optional.items() if value}


_FieldsBuilder = Callable[[Any, Any], dict[str, Any]]


def _no_fields(_upstream_span: Any, _data: Any) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class _SpanState:
    start: _FieldsBuilder
    end: _FieldsBuilder = _no_fields


@dataclass(frozen=True)
class _ActiveSpan:
    span: Span
    state: _SpanState


def _run_start(_upstream_span: Any, data: Any) -> dict[str, Any]:
    return {
        "name": "Agent.run",
        "type": SpanTypeAttribute.TASK,
        "input": _shape_input_messages(_value(data, "messages")),
        "metadata": {
            "agent": _value(data, "agent"),
            "model": _value(data, "model") or "unknown",
            "provider": _value(data, "provider") or "unknown",
            "tool_names": _value(data, "tool_names"),
            "output_type": _value(data, "output_type"),
        },
    }


def _run_end(_upstream_span: Any, data: Any) -> dict[str, Any]:
    final_message = _value(data, "final_message")
    return {
        "output": _shape_message(final_message) if final_message else None,
        "metadata": {"blocked": bool(_value(data, "blocked"))},
    }


def _loop_turn_start(_upstream_span: Any, _data: Any) -> dict[str, Any]:
    return {"name": "Loop Turn", "type": SpanTypeAttribute.TASK}


def _stream_start(_upstream_span: Any, data: Any) -> dict[str, Any]:
    # The AI SDK delegates transport to an instrumentable provider client.
    # Keep this framework-level span a task and let that provider own the
    # single LLM span for the request.
    return {
        "name": "ai.stream",
        "type": SpanTypeAttribute.TASK,
        "input": _shape_input_messages(_value(data, "messages")),
        "metadata": _request_metadata(data),
    }


def _stream_end(upstream_span: Any, data: Any) -> dict[str, Any]:
    event = {
        "output": _shape_completion_output(_value(data, "message"), _value(data, "finish_reason")),
        "metadata": _response_metadata(data),
    }
    first_token = _time_to_first_token(upstream_span)
    if first_token is not None:
        event["metrics"] = {"time_to_first_token": first_token}
    return event


def _generate_start(_upstream_span: Any, data: Any) -> dict[str, Any]:
    return {
        "name": "ai.generate",
        "type": SpanTypeAttribute.LLM,
        "input": _shape_input_messages(_value(data, "messages")),
        "metadata": _request_metadata(data),
    }


def _generate_end(_upstream_span: Any, data: Any) -> dict[str, Any]:
    return {
        "output": _shape_completion_output(_value(data, "message")),
        "metadata": _response_metadata(data),
        "metrics": _usage_metrics(_value(data, "usage")),
    }


def _tool_start(_upstream_span: Any, data: Any) -> dict[str, Any]:
    return {
        "name": _value(data, "tool_name") or "tool",
        "type": SpanTypeAttribute.TOOL,
        "input": _value(data, "args"),
        "metadata": clean_nones(
            {
                "tool_name": _value(data, "tool_name"),
                "tool_call_id": _value(data, "tool_call_id"),
                "description": _value(data, "tool_description"),
            }
        ),
    }


def _tool_end(upstream_span: Any, data: Any) -> dict[str, Any]:
    result = _value(data, "result")
    model_input = _value(data, "model_input")
    event = {"output": result}
    if model_input is not None:
        event["metadata"] = {"model_input": model_input}
    if _value(data, "is_error") and _value(upstream_span, "error") is None:
        event["error"] = str(result)
    return event


def _hook_start(_upstream_span: Any, data: Any) -> dict[str, Any]:
    return {
        "name": f"Hook: {_value(data, 'label')}",
        "type": SpanTypeAttribute.TASK,
        "input": _value(data, "metadata"),
        "metadata": {
            "hook_type": _value(data, "hook_type"),
            "tool_call_id": _value(data, "tool_call_id"),
            "status": _value(data, "status"),
        },
    }


def _hook_end(_upstream_span: Any, data: Any) -> dict[str, Any]:
    status = _value(data, "status")
    return {
        "output": {"status": status, "resolution": _value(data, "resolution")},
        "metadata": {"status": status},
    }


def _custom_start(upstream_span: Any, data: Any) -> dict[str, Any]:
    return {
        "name": _value(upstream_span, "name") or "custom",
        "metadata": _value(data, "attrs", {}),
    }


def _custom_end(_upstream_span: Any, data: Any) -> dict[str, Any]:
    return {"metadata": _value(data, "attrs", {})}


def _default_start(upstream_span: Any, data: Any) -> dict[str, Any]:
    kind = _value(data, "kind", _value(upstream_span, "name"))
    return {"name": _value(upstream_span, "name") or str(kind or "ai")}


_SPAN_STATES = {
    "run": _SpanState(_run_start, _run_end),
    "loop_turn": _SpanState(_loop_turn_start),
    "ai_stream": _SpanState(_stream_start, _stream_end),
    "ai_generate": _SpanState(_generate_start, _generate_end),
    "tool_execution": _SpanState(_tool_start, _tool_end),
    "hook": _SpanState(_hook_start, _hook_end),
    "custom": _SpanState(_custom_start, _custom_end),
}
_DEFAULT_STATE = _SpanState(_default_start)


def _state_for(upstream_span: Any) -> tuple[Any, _SpanState]:
    data = _value(upstream_span, "data")
    kind = _value(data, "kind", _value(upstream_span, "name"))
    state = _SPAN_STATES.get(kind) if isinstance(kind, str) else None
    return data, state or _DEFAULT_STATE


def _is_empty(value: Any) -> bool:
    """Empty for logging purposes: absent, or an empty container.

    Deliberately keeps falsy scalars, so a tool that returns ``0`` or ``""``
    still logs an output.
    """
    return value is None or (isinstance(value, (dict, list)) and not value)


def _end_fields(upstream_span: Any, data: Any, state: _SpanState) -> dict[str, Any]:
    event = state.end(upstream_span, data)
    error = _value(upstream_span, "error")
    if error is not None:
        # Upstream models the failure as data, not a live exception, so log the
        # text rather than fabricating an exception to be stringified.
        event["error"] = f"{_value(error, 'type', 'Error')}: {_value(error, 'message', str(error))}"
    return {key: value for key, value in event.items() if not _is_empty(value)}


class BraintrustAISDKAdapter:
    """Adapter for ``ai.experimental_telemetry``'s callback protocol."""

    def __init__(self) -> None:
        self._active: dict[str, _ActiveSpan] = {}
        self._archived_parents: OrderedDict[str, Span] = OrderedDict()

    def _remember_parent(self, upstream_id: str, span: Span) -> None:
        # ``Span.start_span`` only reads ids, which stay valid after ``end()``,
        # so a finished parent still links children losslessly.
        self._archived_parents[upstream_id] = span
        if len(self._archived_parents) > _MAX_ARCHIVED_PARENTS:
            self._archived_parents.popitem(last=False)

    async def on_span_start(self, upstream_span: Any) -> None:
        upstream_id = _value(upstream_span, "id")
        parent_id = _value(upstream_span, "parent_id")
        start_time_ns = _value(upstream_span, "started_at")
        set_as_current = bool(_value(upstream_span, "set_as_current", True))
        data, state = _state_for(upstream_span)
        active_parent = self._active.get(parent_id)
        parent_span = active_parent.span if active_parent is not None else self._archived_parents.get(parent_id)
        kwargs = {
            **state.start(upstream_span, data),
            "start_time": start_time_ns / 1_000_000_000 if isinstance(start_time_ns, int) else None,
            "set_current": set_as_current,
            "internal": {"instrumentation": _INSTRUMENTATION},
        }
        bt_span = parent_span.start_span(**kwargs) if parent_span is not None else start_span(**kwargs)
        if set_as_current:
            bt_span.set_current()
        self._active[upstream_id] = _ActiveSpan(bt_span, state)

    async def on_span_event(self, upstream_span: Any, event: Any) -> None:
        # Events remain on the upstream span snapshot. They are interpreted at
        # span end so durable/replayed spans and live spans share one code path.
        del upstream_span, event

    async def on_span_end(self, upstream_span: Any) -> None:
        upstream_id = _value(upstream_span, "id")
        active_span = self._active.pop(upstream_id, None)
        if active_span is None:
            return
        data = _value(upstream_span, "data")
        self._remember_parent(upstream_id, active_span.span)
        event = _end_fields(upstream_span, data, active_span.state)
        if event:
            active_span.span.log(**event)
        active_span.span.unset_current()
        ended_at = _value(upstream_span, "ended_at")
        active_span.span.end(end_time=ended_at / 1_000_000_000 if isinstance(ended_at, int) else None)
