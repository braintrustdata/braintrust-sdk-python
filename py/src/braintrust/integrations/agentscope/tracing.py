"""AgentScope-specific span creation and stream aggregation."""

from contextlib import aclosing
from contextvars import ContextVar
from typing import Any

from braintrust.logger import start_span
from braintrust.span_types import SpanPurpose, SpanTypeAttribute


_SUPPRESS_TASK_EVALUATE_SPAN: ContextVar[bool] = ContextVar("_SUPPRESS_TASK_EVALUATE_SPAN", default=False)


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


def _call_arg(args: Any, kwargs: dict[str, Any], index: int, key: str) -> Any:
    if key in kwargs:
        return kwargs[key]
    return args[index] if len(args) > index else None


def _maybe_awaitable_name(value: Any) -> str | None:
    return getattr(value, "__name__", None) or getattr(value, "__qualname__", None)


def _metric_name(metric: Any) -> str:
    return str(getattr(metric, "name", None) or metric.__class__.__name__)


def _task_id(task: Any) -> str:
    return str(_field_value(task, "id") or _field_value(task, "name") or task.__class__.__name__)


def _task_input(task: Any) -> Any:
    for key in ("input", "input_data", "question", "prompt"):
        value = _field_value(task, key)
        if value is not None:
            return value
    return None


def _task_expected(task: Any) -> Any:
    for key in ("ground_truth", "expected", "reference", "answer"):
        value = _field_value(task, key)
        if value is not None:
            return value
    return None


def _task_tags(task: Any) -> Any:
    tags = _field_value(task, "tags")
    if isinstance(tags, dict):
        return [f"{key}:{value}" for key, value in sorted(tags.items())]
    return tags


def _task_metric_names(task: Any) -> list[str] | None:
    metrics = _field_value(task, "metrics")
    if not metrics:
        return None
    return [_metric_name(metric) for metric in metrics]


def _task_metadata(task: Any) -> dict[str, Any]:
    metadata = _field_value(task, "metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _solution_output_summary(solution_output: Any) -> Any:
    if solution_output is None:
        return None
    if isinstance(solution_output, dict):
        return solution_output

    summary = _clean(
        {
            "output": _field_value(solution_output, "output"),
            "success": _field_value(solution_output, "success"),
            "trajectory": _field_value(solution_output, "trajectory"),
            "meta": _field_value(solution_output, "meta") or _field_value(solution_output, "metadata"),
            "message": _field_value(solution_output, "message"),
        }
    )
    return summary or solution_output


def _metric_result_summary(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, dict):
        return result

    summary = _clean(
        {
            "result": _field_value(result, "result"),
            "message": _field_value(result, "message"),
            "detail": _field_value(result, "detail"),
            "metadata": _field_value(result, "metadata") or _field_value(result, "meta"),
        }
    )
    return summary or result


def _metric_score(metric: Any, result: Any) -> dict[str, float] | None:
    value = _field_value(result, "result")
    if isinstance(value, bool):
        return {_metric_name(metric): 1.0 if value else 0.0}
    if isinstance(value, (int, float)):
        return {_metric_name(metric): float(value)}
    return None


def _evaluator_metadata(instance: Any, solution: Any = None) -> dict[str, Any]:
    benchmark = getattr(instance, "benchmark", None)
    task_count = len(benchmark) if benchmark is not None and hasattr(benchmark, "__len__") else None
    return _clean(
        {
            "evaluator_class": instance.__class__.__name__,
            "evaluator_name": getattr(instance, "name", None),
            "benchmark_name": _field_value(benchmark, "name"),
            "benchmark_description": _field_value(benchmark, "description"),
            "task_count": task_count,
            "n_repeat": getattr(instance, "n_repeat", None),
            "n_workers": getattr(instance, "n_workers", None),
            "storage_class": getattr(getattr(instance, "storage", None), "__class__", type(None)).__name__,
            "solution_name": _maybe_awaitable_name(solution),
        }
    )


def _task_span_metadata(task: Any, repeat_id: str | None = None, **extra: Any) -> dict[str, Any]:
    raw_tags = _field_value(task, "tags")
    return _clean(
        {
            **_task_metadata(task),
            "task_id": _task_id(task),
            "repeat_id": repeat_id,
            "metric_names": _task_metric_names(task),
            "task_tags": raw_tags if isinstance(raw_tags, dict) else None,
            **extra,
        }
    )


def _storage_get(storage: Any, method_name: str, *args: Any) -> Any:
    method = getattr(storage, method_name, None)
    if method is None:
        return None
    try:
        return method(*args)
    except Exception:
        return None


def _stored_solution_output(instance: Any, task: Any, repeat_id: str) -> Any:
    storage = getattr(instance, "storage", None)
    if storage is None:
        return None
    return _storage_get(storage, "get_solution_result", _task_id(task), repeat_id)


def _stored_evaluation_results(instance: Any, task: Any, repeat_id: str) -> list[Any] | None:
    storage = getattr(instance, "storage", None)
    metrics = _field_value(task, "metrics") or []
    if storage is None or not metrics:
        return None

    results = []
    for metric in metrics:
        result = _storage_get(storage, "get_evaluation_result", _task_id(task), repeat_id, _metric_name(metric))
        if result is None:
            return None
        results.append(result)
    return results


def _log_metric_span(parent_span: Any, metric: Any, solution_output: Any, result: Any) -> None:
    with parent_span.start_span(
        name=_metric_name(metric),
        type=SpanTypeAttribute.SCORE,
        span_attributes={"purpose": SpanPurpose.SCORER.value},
        input=_solution_output_summary(solution_output),
        metadata=_clean({"metric_class": metric.__class__.__name__}),
    ) as metric_span:
        metric_span.log(
            output=_metric_result_summary(result),
            metadata=_field_value(result, "metadata") or _field_value(result, "meta"),
            scores=_metric_score(metric, result),
        )


async def _agent_call_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    with start_span(
        name=f"{_agent_name(instance)}.reply",
        type=SpanTypeAttribute.TASK,
        input=_args_kwargs_input(args, kwargs),
        metadata=_clean({"agent_class": instance.__class__.__name__}),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            span.log(output=result)
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _general_evaluator_run_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    solution = _call_arg(args, kwargs, 0, "solution")
    with start_span(
        name="agentscope.evaluate.run",
        type=SpanTypeAttribute.EVAL,
        input=_clean(
            {
                "benchmark_name": _field_value(getattr(instance, "benchmark", None), "name"),
                "n_repeat": getattr(instance, "n_repeat", None),
                "n_workers": getattr(instance, "n_workers", None),
            }
        ),
        metadata=_evaluator_metadata(instance, solution),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            span.log(output={"status": "completed"})
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _ray_evaluator_run_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    solution = _call_arg(args, kwargs, 0, "solution")
    with start_span(
        name="agentscope.evaluate.run",
        type=SpanTypeAttribute.EVAL,
        input=_clean(
            {
                "benchmark_name": _field_value(getattr(instance, "benchmark", None), "name"),
                "n_repeat": getattr(instance, "n_repeat", None),
                "n_workers": getattr(instance, "n_workers", None),
            }
        ),
        metadata={**_evaluator_metadata(instance, solution), "distributed": True},
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            span.log(output={"status": "completed"})
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _general_evaluator_run_solution_wrapper(
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: dict[str, Any],
) -> Any:
    repeat_id = str(_call_arg(args, kwargs, 0, "repeat_id"))
    task = _call_arg(args, kwargs, 1, "task")
    storage = getattr(instance, "storage", None)
    was_cached = False
    if storage is not None and task is not None:
        exists = getattr(storage, "solution_result_exists", None)
        if exists is not None:
            try:
                was_cached = bool(exists(_task_id(task), repeat_id))
            except Exception:
                was_cached = False

    with start_span(
        name=f"{_task_id(task)}.solution",
        type=SpanTypeAttribute.TASK,
        input=_task_input(task),
        expected=_task_expected(task),
        tags=_task_tags(task),
        metadata=_task_span_metadata(task, repeat_id, cached=was_cached),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            solution_output = _stored_solution_output(instance, task, repeat_id)
            span.log(output=_solution_output_summary(solution_output))
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _general_evaluator_run_evaluation_wrapper(
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: dict[str, Any],
) -> Any:
    task = _call_arg(args, kwargs, 0, "task")
    repeat_id = str(_call_arg(args, kwargs, 1, "repeat_id"))
    solution_output = _call_arg(args, kwargs, 2, "solution_output")

    with start_span(
        name=f"{_task_id(task)}.evaluate",
        type=SpanTypeAttribute.EVAL,
        input=_solution_output_summary(solution_output),
        metadata=_task_span_metadata(task, repeat_id),
    ) as span:
        token = _SUPPRESS_TASK_EVALUATE_SPAN.set(True)
        try:
            result = await wrapped(*args, **kwargs)
            evaluation_results = _stored_evaluation_results(instance, task, repeat_id)
            if evaluation_results is not None:
                metrics = _field_value(task, "metrics") or []
                for metric, evaluation_result in zip(metrics, evaluation_results):
                    _log_metric_span(span, metric, solution_output, evaluation_result)
                span.log(output=[_metric_result_summary(item) for item in evaluation_results])
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise
        finally:
            _SUPPRESS_TASK_EVALUATE_SPAN.reset(token)


async def _task_evaluate_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    if _SUPPRESS_TASK_EVALUATE_SPAN.get():
        return await wrapped(*args, **kwargs)

    solution_output = _call_arg(args, kwargs, 0, "solution_output")
    with start_span(
        name=f"{_task_id(instance)}.evaluate",
        type=SpanTypeAttribute.EVAL,
        input=_solution_output_summary(solution_output),
        metadata=_task_span_metadata(instance),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            span.log(output=[_metric_result_summary(item) for item in result] if result is not None else None)
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _metric_call_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    if _SUPPRESS_TASK_EVALUATE_SPAN.get():
        return await wrapped(*args, **kwargs)

    solution_output = _call_arg(args, kwargs, 0, "solution_output")
    with start_span(
        name=_metric_name(instance),
        type=SpanTypeAttribute.SCORE,
        span_attributes={"purpose": SpanPurpose.SCORER.value},
        input=_solution_output_summary(solution_output),
        metadata=_clean({"metric_class": instance.__class__.__name__}),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            span.log(
                output=_metric_result_summary(result),
                metadata=_field_value(result, "metadata") or _field_value(result, "meta"),
                scores=_metric_score(instance, result),
            )
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _sequential_pipeline_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    with start_span(
        name="sequential_pipeline.run",
        type=SpanTypeAttribute.TASK,
        input=_args_kwargs_input(args, kwargs),
        metadata=_pipeline_metadata(args, kwargs),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            span.log(output=result)
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _fanout_pipeline_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    with start_span(
        name="fanout_pipeline.run",
        type=SpanTypeAttribute.TASK,
        input=_args_kwargs_input(args, kwargs),
        metadata=_pipeline_metadata(args, kwargs),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            span.log(output=result)
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


async def _toolkit_call_tool_function_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    tool_call = args[0] if args else kwargs.get("tool_call")
    tool_name = _tool_name(tool_call)
    with start_span(
        name=f"{tool_name}.execute",
        type=SpanTypeAttribute.TOOL,
        input=_clean(
            {
                "tool_name": tool_name,
                "tool_call": tool_call,
            }
        ),
        metadata=_clean({"toolkit_class": instance.__class__.__name__}),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            if _is_async_iterator(result):

                async def _trace():
                    last_chunk = None
                    async with aclosing(result) as agen:
                        async for chunk in agen:
                            last_chunk = chunk
                            yield chunk
                    if last_chunk is not None:
                        span.log(output=last_chunk)

                return _trace()

            span.log(output=result)
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise


def _is_async_iterator(value: Any) -> bool:
    try:
        return getattr(value, "__aiter__", None) is not None and getattr(value, "__anext__", None) is not None
    except Exception:
        return False


async def _model_call_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: dict[str, Any]) -> Any:
    with start_span(
        name=f"{_model_provider_name(instance)}.call",
        type=SpanTypeAttribute.LLM,
        input=_model_call_input(args, kwargs),
        metadata=_model_call_metadata(instance, kwargs),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
            if _is_async_iterator(result):

                async def _trace():
                    last_chunk = None
                    async with aclosing(result) as agen:
                        async for chunk in agen:
                            last_chunk = chunk
                            yield chunk
                    if last_chunk is not None:
                        span.log(output=_model_call_output(last_chunk), metrics=_extract_metrics(last_chunk))

                return _trace()

            span.log(output=_model_call_output(result), metrics=_extract_metrics(result))
            return result
        except Exception as exc:
            span.log(error=str(exc))
            raise
