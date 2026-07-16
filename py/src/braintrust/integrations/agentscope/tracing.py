"""AgentScope-specific span creation and stream aggregation."""

import contextlib
import inspect
import time
from contextlib import aclosing
from contextvars import ContextVar
from typing import Any

from braintrust.integrations.utils import (
    _is_supported_metric_value,
    _normalize_chat_messages,
    _parse_openai_usage_metrics,
)
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import clean_nones


_INSTRUMENTATION = "agentscope-auto"
_STREAM_METRICS: ContextVar[dict[str, float] | None] = ContextVar("agentscope_stream_metrics", default=None)


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


# Model class → canonical provider slug (matches JS SDK + pricing lookups).
_PROVIDER_BY_MODEL_CLASS = {
    "OpenAIChatModel": "openai",
    "AnthropicChatModel": "anthropic",
    "GeminiChatModel": "google",
    "DashScopeChatModel": "dashscope",
    "OllamaChatModel": "ollama",
    "TrinityChatModel": "trinity",
}

# Config kwargs safe to surface in metadata. Keeping this explicit avoids
# leaking credentials or other non-config kwargs some providers accept.
_METADATA_CONFIG_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_output_tokens",
        "stop",
        "stop_sequences",
        "n",
        "seed",
        "response_format",
        "reasoning_effort",
        "frequency_penalty",
        "presence_penalty",
        "stream",
    }
)

# Provider usage payloads that vary between chat- and responses-style APIs.
_USAGE_NAME_MAP = {
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "total_tokens": "tokens",
}
_USAGE_PREFIX_MAP = {
    "input": "prompt",
    "output": "completion",
}


def _kw_or_pos(kwargs: dict[str, Any], key: str, args: Any, index: int) -> Any:
    """Return ``kwargs[key]`` if set, otherwise ``args[index]`` if in range."""
    value = kwargs.get(key)
    if value is not None:
        return value
    return args[index] if len(args) > index else None


def _args_kwargs_input(args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return clean_nones(
        {
            "args": list(args) if args else None,
            "kwargs": kwargs if kwargs else None,
        }
    )


def _agent_name(instance: Any) -> str:
    return getattr(instance, "name", None) or instance.__class__.__name__


def _pipeline_metadata(args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    agents = kwargs.get("agents")
    if agents is None and args:
        agents = args[0]

    agent_names = None
    if agents:
        agent_names = [getattr(agent, "name", agent.__class__.__name__) for agent in agents]

    return clean_nones({"agent_names": agent_names})


def _extract_metrics(*candidates: Any) -> dict[str, float] | None:
    for candidate in candidates:
        usage = _field_value(candidate, "usage")
        if usage is None:
            continue
        parsed = _parse_openai_usage_metrics(
            usage,
            token_name_map=_USAGE_NAME_MAP,
            token_prefix_map=_USAGE_PREFIX_MAP,
        )
        metrics = {k: float(v) for k, v in parsed.items() if _is_supported_metric_value(v) and v >= 0}
        if "tokens" not in metrics and "prompt_tokens" in metrics and "completion_tokens" in metrics:
            metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
        if metrics:
            return metrics
    return None


def _model_provider_name(instance: Any) -> str:
    class_name = instance.__class__.__name__
    return _PROVIDER_BY_MODEL_CLASS.get(class_name, class_name)


def _model_metadata(instance: Any) -> dict[str, Any]:
    return clean_nones(
        {
            "model": getattr(instance, "model_name", None) or getattr(instance, "model", None),
            "provider": _model_provider_name(instance),
            "model_class": instance.__class__.__name__,
        }
    )


def _model_call_input(args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return clean_nones({"messages": _normalize_chat_messages(_kw_or_pos(kwargs, "messages", args, 0))})


def _model_call_metadata(instance: Any, args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    extra = {k: v for k, v in kwargs.items() if k in _METADATA_CONFIG_KEYS and v is not None}
    return clean_nones(
        {
            **_model_metadata(instance),
            "tools": _kw_or_pos(kwargs, "tools", args, 1),
            "tool_choice": _kw_or_pos(kwargs, "tool_choice", args, 2),
            "structured_model": _kw_or_pos(kwargs, "structured_model", args, 3),
            **extra,
        }
    )


def _model_call_output(result: Any) -> Any:
    if isinstance(result, dict):
        data = result
    elif _field_value(result, "content") is not None or _field_value(result, "metadata") is not None:
        data = {
            "content": _field_value(result, "content"),
            "metadata": _field_value(result, "metadata"),
        }
    else:
        return result

    normalized = clean_nones(
        {
            "role": "assistant" if data.get("content") is not None else None,
            "content": data.get("content"),
            "metadata": data.get("metadata"),
        }
    )
    return normalized or data


def _field_value(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        return data.get(key)
    return getattr(data, key, None)


def _tool_name(tool_call: Any) -> str:
    if isinstance(tool_call, str):
        return tool_call
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or "unknown_tool")
    return str(getattr(tool_call, "name", "unknown_tool"))


def _make_task_wrapper(
    *,
    name_fn: Any,
    metadata_fn: Any,
    input_fn: Any = _args_kwargs_input,
) -> Any:
    """Build a simple async wrapper that creates a TASK span and logs the result."""

    async def _wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
        with start_span(
            name=name_fn(instance, args, kwargs),
            type=SpanTypeAttribute.TASK,
            input=input_fn(args, kwargs),
            metadata=metadata_fn(instance, args, kwargs),
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)
                span.log(output=result)
                return result
            except Exception as exc:
                span.log(error=exc)
                raise

    return _wrapper


_agent_call_wrapper = _make_task_wrapper(
    name_fn=lambda instance, _a, _k: f"{_agent_name(instance)}.reply",
    metadata_fn=lambda instance, _a, _k: clean_nones({"agent_class": instance.__class__.__name__}),
)

_sequential_pipeline_wrapper = _make_task_wrapper(
    name_fn=lambda _i, _a, _k: "sequential_pipeline.run",
    metadata_fn=lambda _i, args, kwargs: _pipeline_metadata(args, kwargs),
)

_fanout_pipeline_wrapper = _make_task_wrapper(
    name_fn=lambda _i, _a, _k: "fanout_pipeline.run",
    metadata_fn=lambda _i, args, kwargs: _pipeline_metadata(args, kwargs),
)


def _is_async_iterator(value: Any) -> bool:
    try:
        return getattr(value, "__aiter__", None) is not None and getattr(value, "__anext__", None) is not None
    except Exception:
        return False


def _deferred_stream_trace(
    result: Any,
    span: Any,
    stack: contextlib.ExitStack,
    log_fn: Any,
    on_first_chunk: Any = None,
) -> Any:
    """Wrap an async iterator so the span stays open until the stream is consumed.

    ``log_fn(span, last_chunk)`` is invoked once at stream end. ``on_first_chunk``
    (optional) is invoked with no arguments the first time a chunk is yielded, e.g.
    to stamp ``time_to_first_token``.
    """
    deferred = stack.pop_all()

    async def _trace():
        with deferred:
            last_chunk = None
            first_seen = False
            async with aclosing(result) as agen:
                async for chunk in agen:
                    if not first_seen:
                        first_seen = True
                        if on_first_chunk is not None:
                            on_first_chunk()
                    last_chunk = chunk
                    yield chunk
            if last_chunk is not None:
                log_fn(span, last_chunk)

    return _trace()


async def _toolkit_call_tool_function_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    tool_call = args[0] if args else kwargs.get("tool_call")
    tool_name = _tool_name(tool_call)
    with contextlib.ExitStack() as stack:
        span = stack.enter_context(
            start_span(
                name=f"{tool_name}.execute",
                type=SpanTypeAttribute.TOOL,
                input=clean_nones(
                    {
                        "tool_name": tool_name,
                        "tool_call": tool_call,
                    }
                ),
                metadata=clean_nones({"toolkit_class": instance.__class__.__name__}),
            )
        )
        try:
            result = wrapped(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            if _is_async_iterator(result):
                return _deferred_stream_trace(
                    result,
                    span,
                    stack,
                    lambda s, chunk: s.log(output=chunk),
                )

            span.log(output=result)
            return result
        except Exception as exc:
            span.log(error=exc)
            raise


def _capture_openai_stream_usage_wrapper(
    wrapped: Any,
    _instance: Any,
    args: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Capture AgentScope 1.x usage-only OpenAI chunks without changing its stream."""
    metrics = _STREAM_METRICS.get()
    if metrics is None:
        return wrapped(*args, **kwargs)

    positional = list(args)
    response = positional[1] if len(positional) > 1 else kwargs.get("response")
    if response is None:
        return wrapped(*args, **kwargs)

    async def _capture_usage():
        async for chunk in response:
            # OpenAI only sets `usage` on the terminal chunk when
            # `stream_options.include_usage` is set; skip parsing every chunk.
            if _field_value(chunk, "usage") is not None:
                chunk_metrics = _extract_metrics(chunk)
                if chunk_metrics:
                    metrics.update(chunk_metrics)
            yield chunk

    if len(positional) > 1:
        positional[1] = _capture_usage()
    else:
        kwargs = {**kwargs, "response": _capture_usage()}
    return wrapped(*positional, **kwargs)


async def _model_call_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    with contextlib.ExitStack() as stack:
        span = stack.enter_context(
            start_span(
                name=f"{_model_provider_name(instance)}.call",
                type=SpanTypeAttribute.LLM,
                input=_model_call_input(args, kwargs),
                metadata=_model_call_metadata(instance, args, kwargs),
            )
        )
        try:
            request_start_time = time.time()
            captured_stream_metrics: dict[str, float] = {}
            time_to_first_token: list[float | None] = [None]
            stream_metrics_token = _STREAM_METRICS.set(captured_stream_metrics)
            try:
                result = await wrapped(*args, **kwargs)
            finally:
                _STREAM_METRICS.reset(stream_metrics_token)
            if _is_async_iterator(result):

                def _stamp_ttft() -> None:
                    time_to_first_token[0] = time.time() - request_start_time

                def _log_final(s: Any, chunk: Any) -> None:
                    metrics = {**captured_stream_metrics, **(_extract_metrics(chunk) or {})}
                    if time_to_first_token[0] is not None:
                        metrics["time_to_first_token"] = time_to_first_token[0]
                    s.log(output=_model_call_output(chunk), metrics=metrics or None)

                return _deferred_stream_trace(
                    result,
                    span,
                    stack,
                    _log_final,
                    on_first_chunk=_stamp_ttft,
                )

            span.log(output=_model_call_output(result), metrics=_extract_metrics(result))
            return result
        except Exception as exc:
            span.log(error=exc)
            raise
