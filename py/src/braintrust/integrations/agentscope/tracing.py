"""AgentScope-specific span creation and stream aggregation."""

import contextlib
import inspect
import time
from contextlib import aclosing
from contextvars import ContextVar
from typing import Any

from braintrust.integrations.utils import _normalize_chat_messages
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
    key_map = {
        "prompt_tokens": "prompt_tokens",
        "input_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "output_tokens": "completion_tokens",
        "total_tokens": "tokens",
        "tokens": "tokens",
    }

    for candidate in candidates:
        data = _field_value(candidate, "usage") or candidate

        metrics = {}
        for source_key, target_key in key_map.items():
            value = _field_value(data, source_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                metrics[target_key] = float(value)

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
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]

    return clean_nones({"messages": _normalize_chat_messages(messages)})


def _model_call_metadata(instance: Any, args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    tools = kwargs.get("tools")
    if tools is None and len(args) > 1:
        tools = args[1]

    tool_choice = kwargs.get("tool_choice")
    if tool_choice is None and len(args) > 2:
        tool_choice = args[2]

    structured_model = kwargs.get("structured_model")
    if structured_model is None and len(args) > 3:
        structured_model = args[3]

    extra = {key: kwargs[key] for key in _METADATA_CONFIG_KEYS if key in kwargs and kwargs[key] is not None}

    return clean_nones(
        {
            **_model_metadata(instance),
            "tools": tools,
            "tool_choice": tool_choice,
            "structured_model": structured_model,
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
    try:
        return getattr(data, key, None)
    except Exception:
        return None


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
    request_start_time: float | None = None,
) -> Any:
    """Wrap an async iterator so the span stays open until the stream is consumed."""
    deferred = stack.pop_all()

    async def _trace():
        with deferred:
            last_chunk = None
            time_to_first_token: float | None = None
            async with aclosing(result) as agen:
                async for chunk in agen:
                    if time_to_first_token is None and request_start_time is not None:
                        time_to_first_token = time.time() - request_start_time
                    last_chunk = chunk
                    yield chunk
            if last_chunk is not None:
                log_fn(span, last_chunk, time_to_first_token)

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
                    lambda s, chunk, _ttft: s.log(output=chunk),
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
            chunk_metrics = _extract_metrics(chunk)
            if chunk_metrics:
                metrics.update(chunk_metrics)
            yield chunk

    if len(positional) > 1:
        positional[1] = _capture_usage()
    else:
        kwargs = {**kwargs, "response": _capture_usage()}
    return wrapped(*positional, **kwargs)


def _log_model_stream_chunk(
    span: Any,
    chunk: Any,
    time_to_first_token: float | None,
    captured_metrics: dict[str, float],
) -> None:
    metrics = {**captured_metrics, **(_extract_metrics(chunk) or {})}
    if time_to_first_token is not None:
        metrics["time_to_first_token"] = time_to_first_token
    span.log(output=_model_call_output(chunk), metrics=metrics or None)


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
            stream_metrics_token = _STREAM_METRICS.set(captured_stream_metrics)
            try:
                result = await wrapped(*args, **kwargs)
            finally:
                _STREAM_METRICS.reset(stream_metrics_token)
            if _is_async_iterator(result):
                return _deferred_stream_trace(
                    result,
                    span,
                    stack,
                    lambda s, chunk, ttft: _log_model_stream_chunk(
                        s,
                        chunk,
                        ttft,
                        captured_stream_metrics,
                    ),
                    request_start_time=request_start_time,
                )

            span.log(output=_model_call_output(result), metrics=_extract_metrics(result))
            return result
        except Exception as exc:
            span.log(error=exc)
            raise
