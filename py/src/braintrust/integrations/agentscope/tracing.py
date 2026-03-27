"""AgentScope-specific span creation and stream aggregation."""

import contextlib
from contextlib import aclosing
from typing import Any

from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute


def _clean(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def _args_kwargs_input(args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return _clean(
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

    return _clean({"agent_names": agent_names})


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
            if isinstance(value, (int, float)):
                metrics[target_key] = float(value)
        if metrics:
            return metrics

    return None


def _model_provider_name(instance: Any) -> str:
    class_name = instance.__class__.__name__
    if class_name.endswith("Model"):
        return class_name[: -len("Model")]
    return class_name


def _model_metadata(instance: Any) -> dict[str, Any]:
    return _clean(
        {
            "model": getattr(instance, "model_name", None),
            "provider": _model_provider_name(instance),
            "model_class": instance.__class__.__name__,
        }
    )


def _model_call_input(args: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]

    tools = kwargs.get("tools")
    if tools is None and len(args) > 1:
        tools = args[1]

    tool_choice = kwargs.get("tool_choice")
    if tool_choice is None and len(args) > 2:
        tool_choice = args[2]

    structured_model = kwargs.get("structured_model")
    if structured_model is None and len(args) > 3:
        structured_model = args[3]

    return _clean(
        {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "structured_model": structured_model,
        }
    )


def _model_call_metadata(instance: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    extra_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"messages", "tools", "tool_choice", "structured_model"} and value is not None
    }
    return {**_model_metadata(instance), **extra_kwargs}


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

    normalized = _clean(
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
                span.log(error=str(exc))
                raise

    return _wrapper


_agent_call_wrapper = _make_task_wrapper(
    name_fn=lambda instance, _a, _k: f"{_agent_name(instance)}.reply",
    metadata_fn=lambda instance, _a, _k: _clean({"agent_class": instance.__class__.__name__}),
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


def _deferred_stream_trace(result: Any, span: Any, stack: contextlib.ExitStack, log_fn: Any) -> Any:
    """Wrap an async iterator so the span stays open until the stream is consumed."""
    deferred = stack.pop_all()

    async def _trace():
        with deferred:
            last_chunk = None
            async with aclosing(result) as agen:
                async for chunk in agen:
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
                input=_clean(
                    {
                        "tool_name": tool_name,
                        "tool_call": tool_call,
                    }
                ),
                metadata=_clean({"toolkit_class": instance.__class__.__name__}),
            )
        )
        try:
            result = await wrapped(*args, **kwargs)
            if _is_async_iterator(result):
                return _deferred_stream_trace(result, span, stack, lambda s, chunk: s.log(output=chunk))

            span.log(output=result)
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _model_call_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    with contextlib.ExitStack() as stack:
        span = stack.enter_context(
            start_span(
                name=f"{_model_provider_name(instance)}.call",
                type=SpanTypeAttribute.LLM,
                input=_model_call_input(args, kwargs),
                metadata=_model_call_metadata(instance, kwargs),
            )
        )
        try:
            result = await wrapped(*args, **kwargs)
            if _is_async_iterator(result):
                return _deferred_stream_trace(
                    result,
                    span,
                    stack,
                    lambda s, chunk: s.log(output=_model_call_output(chunk), metrics=_extract_metrics(chunk)),
                )

            span.log(output=_model_call_output(result), metrics=_extract_metrics(result))
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise
