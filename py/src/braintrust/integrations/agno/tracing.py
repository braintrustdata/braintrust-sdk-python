import time
from inspect import isawaitable
from typing import Any

from braintrust.integrations.utils import _try_to_dict
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import is_numeric


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def omit(obj: dict[str, Any], keys: list[str]):
    return {k: v for k, v in obj.items() if k not in keys}


def clean(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if v is not None}


def get_args_kwargs(args: list[str], kwargs: dict[str, Any], keys: list[str]):
    return {k: args[i] if args else kwargs.get(k) for i, k in enumerate(keys)}, omit(kwargs, keys)


def is_sync_iterator(result: Any) -> bool:
    return hasattr(result, "__iter__") and hasattr(result, "__next__")


def is_async_iterator(result: Any) -> bool:
    return hasattr(result, "__aiter__") and hasattr(result, "__anext__")


# ---------------------------------------------------------------------------
# Metrics mapping & extraction
# ---------------------------------------------------------------------------

AGNO_METRICS_MAP = {
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "total_tokens": "tokens",
    "reasoning_tokens": "completion_reasoning_tokens",
    "audio_input_tokens": "prompt_audio_tokens",
    "audio_output_tokens": "completion_audio_tokens",
    "cache_read_tokens": "prompt_cached_tokens",
    "cache_write_tokens": "prompt_cache_creation_tokens",
    "duration": "duration",
    "time_to_first_token": "time_to_first_token",
}


def extract_metadata(instance: Any, component: str) -> dict[str, Any]:
    """Extract metadata from any component (model, agent, team)."""
    metadata = {"component": component}

    if component == "model":
        if hasattr(instance, "id") and instance.id:
            metadata["model"] = instance.id
            metadata["model_id"] = instance.id
        if hasattr(instance, "provider") and instance.provider:
            metadata["provider"] = instance.provider
        if hasattr(instance, "name") and instance.name:
            metadata["model_name"] = instance.name
        if hasattr(instance, "__class__"):
            metadata["model_class"] = instance.__class__.__name__
    elif component == "agent":
        metadata["agent_name"] = getattr(instance, "name", None)
        model = getattr(instance, "model", None)
        if model:
            metadata["model"] = getattr(model, "id", None) or model.__class__.__name__
    elif component == "team":
        metadata["team_name"] = getattr(instance, "name", None)
        model = getattr(instance, "model", None)
        if model:
            metadata["model"] = getattr(model, "id", None) or model.__class__.__name__
    elif component == "workflow":
        metadata["workflow_id"] = getattr(instance, "id", None)
        metadata["workflow_name"] = getattr(instance, "name", None)
        steps = getattr(instance, "steps", None)
        if steps:
            metadata["steps_count"] = len(steps)

    return metadata


def parse_metrics_from_agno(usage: Any) -> dict[str, Any]:
    """Parse metrics from Agno usage object, following OpenAI wrapper pattern."""
    metrics = {}
    if not usage:
        return metrics
    usage_dict = _try_to_dict(usage)
    if not isinstance(usage_dict, dict):
        return metrics
    for agno_name, value in usage_dict.items():
        if agno_name in AGNO_METRICS_MAP and is_numeric(value) and value != 0:
            braintrust_name = AGNO_METRICS_MAP[agno_name]
            metrics[braintrust_name] = value
    return metrics


def extract_metrics(result: Any, messages: list | None = None) -> dict[str, Any]:
    """Unified metrics extraction for all components."""
    if hasattr(result, "response_usage") and result.response_usage:
        return parse_metrics_from_agno(result.response_usage)
    if hasattr(result, "metrics") and result.metrics:
        metrics = parse_metrics_from_agno(result.metrics)
        return metrics if metrics else None
    if messages:
        for msg in messages:
            if hasattr(msg, "role") and msg.role == "assistant" and hasattr(msg, "metrics") and msg.metrics:
                return parse_metrics_from_agno(msg.metrics)
    return {}


def extract_streaming_metrics(aggregated: dict[str, Any], start_time: float) -> dict[str, Any] | None:
    """Extract metrics from aggregated streaming response."""
    metrics = {}
    if aggregated.get("metrics") and isinstance(aggregated["metrics"], dict):
        metrics.update(aggregated["metrics"])
    elif aggregated.get("metrics"):
        parsed_metrics = parse_metrics_from_agno(aggregated["metrics"])
        if parsed_metrics:
            metrics.update(parsed_metrics)
    elif aggregated.get("response_usage"):
        response_metrics = parse_metrics_from_agno(aggregated["response_usage"])
        if response_metrics:
            metrics.update(response_metrics)
    metrics["duration"] = time.time() - start_time
    return metrics if metrics else None


# ---------------------------------------------------------------------------
# Chunk aggregation
# ---------------------------------------------------------------------------


def _aggregate_metrics(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Aggregate metrics from source into target dict."""
    for key, value in source.items():
        if is_numeric(value):
            if key in target:
                if "time" in key.lower() or "duration" in key.lower():
                    target[key] = value
                elif "token" in key.lower() or key == "tokens":
                    target[key] = (target.get(key, 0) or 0) + value
                else:
                    target[key] = value
            else:
                target[key] = value


def _aggregate_model_chunks(chunks: list[Any]) -> dict[str, Any]:
    """Aggregate ModelResponse chunks from invoke_stream into a complete response."""
    aggregated = {
        "content": "",
        "reasoning_content": "",
        "tool_calls": [],
        "role": None,
        "audio": None,
        "images": [],
        "videos": [],
        "files": [],
        "citations": None,
        "metrics": {},
    }

    for chunk in chunks:
        if hasattr(chunk, "content") and chunk.content:
            aggregated["content"] += str(chunk.content)
        if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
            aggregated["reasoning_content"] += chunk.reasoning_content
        if hasattr(chunk, "role") and chunk.role and not aggregated["role"]:
            aggregated["role"] = chunk.role
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            aggregated["tool_calls"].extend(chunk.tool_calls)
        if hasattr(chunk, "audio") and chunk.audio:
            aggregated["audio"] = chunk.audio
        if hasattr(chunk, "images") and chunk.images:
            aggregated["images"].extend(chunk.images)
        if hasattr(chunk, "videos") and chunk.videos:
            aggregated["videos"].extend(chunk.videos)
        if hasattr(chunk, "files") and chunk.files:
            aggregated["files"].extend(chunk.files)
        if hasattr(chunk, "citations") and chunk.citations:
            aggregated["citations"] = chunk.citations
        if hasattr(chunk, "response_usage") and chunk.response_usage:
            chunk_metrics = parse_metrics_from_agno(chunk.response_usage)
            if chunk_metrics:
                _aggregate_metrics(aggregated["metrics"], chunk_metrics)

    if aggregated["metrics"]:
        aggregated["response_usage"] = aggregated["metrics"]
    else:
        aggregated["metrics"] = None

    return aggregated


def _aggregate_response_stream_chunks(chunks: list[Any]) -> dict[str, Any]:
    """Aggregate chunks from response_stream (ModelResponse, RunOutputEvent, etc.)."""
    aggregated = {
        "content": "",
        "reasoning_content": "",
        "tool_calls": [],
        "role": None,
        "audio": None,
        "images": [],
        "videos": [],
        "files": [],
        "citations": None,
        "metrics": {},
    }

    for chunk in chunks:
        if hasattr(chunk, "__class__") and "ModelResponse" in chunk.__class__.__name__:
            if hasattr(chunk, "content") and chunk.content:
                aggregated["content"] += str(chunk.content)
            if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
                aggregated["reasoning_content"] += chunk.reasoning_content
            if hasattr(chunk, "role") and chunk.role and not aggregated["role"]:
                aggregated["role"] = chunk.role
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                aggregated["tool_calls"].extend(chunk.tool_calls)
            if hasattr(chunk, "audio") and chunk.audio:
                aggregated["audio"] = chunk.audio
            if hasattr(chunk, "images") and chunk.images:
                aggregated["images"].extend(chunk.images)
            if hasattr(chunk, "videos") and chunk.videos:
                aggregated["videos"].extend(chunk.videos)
            if hasattr(chunk, "files") and chunk.files:
                aggregated["files"].extend(chunk.files)
            if hasattr(chunk, "citations") and chunk.citations:
                aggregated["citations"] = chunk.citations
            if hasattr(chunk, "response_usage") and chunk.response_usage:
                chunk_metrics = parse_metrics_from_agno(chunk.response_usage)
                if chunk_metrics:
                    _aggregate_metrics(aggregated["metrics"], chunk_metrics)
            elif hasattr(chunk, "metrics") and chunk.metrics:
                chunk_metrics = parse_metrics_from_agno(chunk.metrics)
                if chunk_metrics:
                    _aggregate_metrics(aggregated["metrics"], chunk_metrics)
        elif hasattr(chunk, "content"):
            if chunk.content:
                aggregated["content"] += str(chunk.content)

        if hasattr(chunk, "metrics") and chunk.metrics and "metrics" not in str(type(chunk)):
            chunk_metrics = parse_metrics_from_agno(chunk.metrics)
            if chunk_metrics:
                _aggregate_metrics(aggregated["metrics"], chunk_metrics)

    if aggregated["metrics"]:
        aggregated["response_usage"] = aggregated["metrics"]
    else:
        aggregated["metrics"] = None

    return aggregated


def _aggregate_agent_chunks(chunks: list[Any]) -> dict[str, Any]:
    """Aggregate BaseAgentRunEvent/BaseTeamRunEvent chunks into a complete response."""
    aggregated = {
        "content": "",
        "reasoning_content": "",
        "model": "",
        "model_provider": "",
        "tool_calls": [],
        "citations": None,
        "references": None,
        "metrics": None,
        "finish_reason": None,
    }

    for chunk in chunks:
        event = getattr(chunk, "event", None)

        if event == "RunStarted":
            if hasattr(chunk, "model"):
                aggregated["model"] = chunk.model
            if hasattr(chunk, "model_provider"):
                aggregated["model_provider"] = chunk.model_provider
        elif event == "RunContent":
            if hasattr(chunk, "content") and chunk.content:
                aggregated["content"] += str(chunk.content)  # type: ignore
            if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
                aggregated["reasoning_content"] += chunk.reasoning_content
            if hasattr(chunk, "citations"):
                aggregated["citations"] = chunk.citations
            if hasattr(chunk, "references"):
                aggregated["references"] = chunk.references
        elif event == "RunCompleted":
            if hasattr(chunk, "metrics"):
                parsed_metrics = parse_metrics_from_agno(chunk.metrics)
                aggregated["metrics"] = parsed_metrics if parsed_metrics else chunk.metrics
            aggregated["finish_reason"] = "stop"
        elif event == "RunError":
            aggregated["finish_reason"] = "error"
        elif event == "ToolCallStarted":
            if hasattr(chunk, "tool_call"):
                aggregated["tool_calls"].append(  # type:ignore
                    {
                        "id": getattr(chunk.tool_call, "id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(chunk.tool_call, "name", None),
                            "arguments": getattr(chunk.tool_call, "arguments", ""),
                        },
                    }
                )

    return {k: v for k, v in aggregated.items() if v not in (None, "")}


def _aggregate_workflow_chunks(chunks: list[Any], workflow_run_response: Any | None = None) -> dict[str, Any]:
    """Aggregate workflow/step events into a final workflow-style response."""
    aggregated = {
        "content": "",
        "status": None,
        "metrics": None,
    }
    final_workflow_content = None

    for chunk in chunks:
        event = getattr(chunk, "event", None)

        if hasattr(chunk, "content") and chunk.content:
            if event == "WorkflowCompleted":
                final_workflow_content = str(chunk.content)
            elif final_workflow_content is None:
                aggregated["content"] += str(chunk.content)

        if hasattr(chunk, "status") and chunk.status:
            aggregated["status"] = chunk.status

        if hasattr(chunk, "metrics") and chunk.metrics:
            parsed_metrics = parse_metrics_from_agno(chunk.metrics)
            aggregated["metrics"] = parsed_metrics if parsed_metrics else chunk.metrics

    if final_workflow_content is not None:
        accumulated_content = aggregated["content"]
        if not accumulated_content:
            aggregated["content"] = final_workflow_content
        elif accumulated_content.endswith(final_workflow_content):
            aggregated["content"] = accumulated_content
        else:
            aggregated["content"] = f"{accumulated_content}{final_workflow_content}"

    if workflow_run_response is not None:
        if not aggregated["content"] and hasattr(workflow_run_response, "content") and workflow_run_response.content:
            aggregated["content"] = str(workflow_run_response.content)
        if not aggregated["status"] and hasattr(workflow_run_response, "status") and workflow_run_response.status:
            aggregated["status"] = workflow_run_response.status
        if not aggregated["metrics"] and hasattr(workflow_run_response, "metrics") and workflow_run_response.metrics:
            parsed_metrics = parse_metrics_from_agno(workflow_run_response.metrics)
            aggregated["metrics"] = parsed_metrics if parsed_metrics else workflow_run_response.metrics

    return {k: v for k, v in aggregated.items() if v not in (None, "")}


# ---------------------------------------------------------------------------
# Stream tracing helpers
# ---------------------------------------------------------------------------


def _trace_sync_stream(result: Any, span: Any, start: float):
    def _inner():
        should_unset = True
        try:
            first = True
            all_chunks = []
            for chunk in result:
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                all_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_agent_chunks(all_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
        except GeneratorExit:
            should_unset = False
            raise
        except Exception as e:
            span.log(error=str(e))
            raise
        finally:
            if should_unset:
                span.unset_current()
            span.end()

    return _inner()


def _trace_async_stream(result: Any, span: Any, start: float):
    async def _inner():
        should_unset = True
        try:
            first = True
            all_chunks = []
            async for chunk in result:
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                all_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_agent_chunks(all_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
        except GeneratorExit:
            should_unset = False
            raise
        except Exception as e:
            span.log(error=str(e))
            raise
        finally:
            if should_unset:
                span.unset_current()
            span.end()

    return _inner()


# ===========================================================================
# Raw wrapt wrapper functions — used by FunctionWrapperPatcher in patchers.py
# ===========================================================================


# ---------------------------------------------------------------------------
# Agent / Team private wrappers
# ---------------------------------------------------------------------------


def _agent_run_private_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for Agent._run(run_response, run_messages)."""
    run_response = args[0] if len(args) > 0 else kwargs.get("run_response")
    run_messages = args[1] if len(args) > 1 else kwargs.get("run_messages")
    input_data = {"run_response": run_response, "run_messages": run_messages}
    agent_name = getattr(instance, "name", None) or "Agent"
    with start_span(
        name=f"{agent_name}.run",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata={**omit(kwargs, list(input_data.keys())), **extract_metadata(instance, "agent")},
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result))
        return result


async def _agent_arun_private_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for Agent._arun(run_response, input)."""
    run_response = args[0] if len(args) > 0 else kwargs.get("run_response")
    input_arg = args[1] if len(args) > 1 else kwargs.get("input")
    input_data = {"run_response": run_response, "input": input_arg}
    agent_name = getattr(instance, "name", None) or "Agent"
    with start_span(
        name=f"{agent_name}.arun",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata={**omit(kwargs, list(input_data.keys())), **extract_metadata(instance, "agent")},
    ) as span:
        result = await wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result))
        return result


def _agent_run_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for Agent._run_stream."""
    agent_name = getattr(instance, "name", None) or "Agent"
    run_response = args[0] if args else kwargs.get("run_response")
    run_messages = args[1] if args else kwargs.get("run_messages")

    def _trace_stream():
        start = time.time()
        span = start_span(
            name=f"{agent_name}.run_stream",
            type=SpanTypeAttribute.TASK,
            input={"run_response": run_response, "run_messages": run_messages},
            metadata={**omit(kwargs, ["run_response", "run_messages"]), **extract_metadata(instance, "agent")},
        )
        span.set_current()
        should_unset = True
        try:
            first = True
            all_chunks = []
            for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                all_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_agent_chunks(all_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
        except GeneratorExit:
            should_unset = False
            raise
        except Exception as e:
            span.log(error=str(e))
            raise
        finally:
            if should_unset:
                span.unset_current()
            span.end()

    return _trace_stream()


def _agent_arun_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for Agent._arun_stream."""
    agent_name = getattr(instance, "name", None) or "Agent"
    run_response = args[0] if args else kwargs.get("run_response")
    input = args[2] if args else kwargs.get("input")

    async def _trace_stream():
        start = time.time()
        span = start_span(
            name=f"{agent_name}.arun_stream",
            type=SpanTypeAttribute.TASK,
            input={"run_response": run_response, "input": input},
            metadata={**omit(kwargs, ["run_response", "input"]), **extract_metadata(instance, "agent")},
        )
        span.set_current()
        should_unset = True
        try:
            first = True
            all_chunks = []
            async for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                all_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_agent_chunks(all_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
        except GeneratorExit:
            should_unset = False
            raise
        except Exception as e:
            span.log(error=str(e))
            raise
        finally:
            if should_unset:
                span.unset_current()
            span.end()

    return _trace_stream()


def _team_run_private_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for Team._run(run_response, run_messages)."""
    run_response = args[0] if len(args) > 0 else kwargs.get("run_response")
    run_messages = args[1] if len(args) > 1 else kwargs.get("run_messages")
    input_data = {"run_response": run_response, "run_messages": run_messages}
    team_name = getattr(instance, "name", None) or "Team"
    with start_span(
        name=f"{team_name}.run",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata={**omit(kwargs, list(input_data.keys())), **extract_metadata(instance, "team")},
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result))
        return result


async def _team_arun_private_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for Team._arun(run_response, input)."""
    run_response = args[0] if len(args) > 0 else kwargs.get("run_response")
    input_arg = args[1] if len(args) > 1 else kwargs.get("input")
    input_data = {"run_response": run_response, "input": input_arg}
    team_name = getattr(instance, "name", None) or "Team"
    with start_span(
        name=f"{team_name}.arun",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata={**omit(kwargs, list(input_data.keys())), **extract_metadata(instance, "team")},
    ) as span:
        result = await wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result))
        return result


def _team_run_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for Team._run_stream."""
    team_name = getattr(instance, "name", None) or "Team"
    run_response = args[0] if args else kwargs.get("run_response")
    run_messages = args[1] if args else kwargs.get("run_messages")

    def _trace_stream():
        start = time.time()
        span = start_span(
            name=f"{team_name}.run_stream",
            type=SpanTypeAttribute.TASK,
            input={"run_response": run_response, "run_messages": run_messages},
            metadata={**omit(kwargs, ["run_response", "run_messages"]), **extract_metadata(instance, "team")},
        )
        span.set_current()
        should_unset = True
        try:
            first = True
            all_chunks = []
            for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                all_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_agent_chunks(all_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
        except GeneratorExit:
            should_unset = False
            raise
        except Exception as e:
            span.log(error=str(e))
            raise
        finally:
            if should_unset:
                span.unset_current()
            span.end()

    return _trace_stream()


def _team_arun_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for Team._arun_stream."""
    team_name = getattr(instance, "name", None) or "Team"
    run_response = args[0] if args else kwargs.get("run_response")
    input = args[2] if args else kwargs.get("input")

    async def _trace_stream():
        start = time.time()
        span = start_span(
            name=f"{team_name}.arun_stream",
            type=SpanTypeAttribute.TASK,
            input={"run_response": run_response, "input": input},
            metadata={**omit(kwargs, ["run_response", "input"]), **extract_metadata(instance, "team")},
        )
        span.set_current()
        should_unset = True
        try:
            first = True
            all_chunks = []
            async for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                all_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_agent_chunks(all_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
        except GeneratorExit:
            should_unset = False
            raise
        except Exception as e:
            span.log(error=str(e))
            raise
        finally:
            if should_unset:
                span.unset_current()
            span.end()

    return _trace_stream()


# ---------------------------------------------------------------------------
# Agent / Team public dispatch wrappers (Agno >= 2.5)
# ---------------------------------------------------------------------------


def _run_public_dispatch_wrapper(
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: Any,
    *,
    default_name: str,
    metadata_component: str,
) -> Any:
    """Trace a public synchronous `run(...)` dispatch method."""
    component_name = getattr(instance, "name", None) or default_name
    input_arg = args[0] if len(args) > 0 else kwargs.get("input")
    input_data = {"input": input_arg}
    metadata = {**omit(kwargs, ["input"]), **extract_metadata(instance, metadata_component)}

    span = start_span(
        name=f"{component_name}.run",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata=metadata,
    )
    span.set_current()
    start = time.time()
    try:
        result = wrapped(*args, **kwargs)
        if is_sync_iterator(result):
            return _trace_sync_stream(result, span, start)
        span.log(output=result, metrics=extract_metrics(result))
        span.unset_current()
        span.end()
        return result
    except Exception as e:
        span.log(error=str(e))
        span.unset_current()
        span.end()
        raise


def _arun_public_dispatch_wrapper(
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: Any,
    *,
    default_name: str,
    metadata_component: str,
) -> Any:
    """Trace a public `arun(...)` dispatch method across async return contracts."""
    component_name = getattr(instance, "name", None) or default_name
    input_arg = args[0] if len(args) > 0 else kwargs.get("input")
    input_data = {"input": input_arg}
    metadata = {**omit(kwargs, ["input"]), **extract_metadata(instance, metadata_component)}

    span = start_span(
        name=f"{component_name}.arun",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata=metadata,
    )
    span.set_current()
    start = time.time()
    try:
        result = wrapped(*args, **kwargs)

        if isawaitable(result):

            async def _trace_awaitable():
                should_end_span = True
                try:
                    awaited = await result
                    if is_async_iterator(awaited):
                        should_end_span = False
                        return _trace_async_stream(awaited, span, start)
                    span.log(output=awaited, metrics=extract_metrics(awaited))
                    return awaited
                except Exception as e:
                    span.log(error=str(e))
                    raise
                finally:
                    if should_end_span:
                        span.unset_current()
                        span.end()

            return _trace_awaitable()

        if is_async_iterator(result):
            return _trace_async_stream(result, span, start)

        span.log(output=result, metrics=extract_metrics(result))
        span.unset_current()
        span.end()
        return result
    except Exception as e:
        span.log(error=str(e))
        span.unset_current()
        span.end()
        raise


def _agent_run_public_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    return _run_public_dispatch_wrapper(
        wrapped, instance, args, kwargs, default_name="Agent", metadata_component="agent"
    )


def _agent_arun_public_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    return _arun_public_dispatch_wrapper(
        wrapped, instance, args, kwargs, default_name="Agent", metadata_component="agent"
    )


def _team_run_public_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    return _run_public_dispatch_wrapper(
        wrapped, instance, args, kwargs, default_name="Team", metadata_component="team"
    )


def _team_arun_public_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    return _arun_public_dispatch_wrapper(
        wrapped, instance, args, kwargs, default_name="Team", metadata_component="team"
    )


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


def _get_model_name(instance: Any) -> str:
    provider = getattr(instance, "provider", None)
    if provider:
        return str(provider)
    if hasattr(instance, "get_provider") and callable(instance.get_provider):
        return str(instance.get_provider())
    return getattr(instance.__class__, "__name__", "Model")


def _model_invoke_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, clean_kwargs = get_args_kwargs(
        args, kwargs, ["assistant_message", "messages", "response_format", "tools", "tool_choice"]
    )
    with start_span(
        name=f"{model_name}.invoke",
        type=SpanTypeAttribute.LLM,
        input=input,
        metadata={**clean_kwargs, **extract_metadata(instance, "model")},
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result, kwargs.get("messages", [])))
        return result


async def _model_ainvoke_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, clean_kwargs = get_args_kwargs(
        args, kwargs, ["messages", "assistant_message", "response_format", "tools", "tool_choice"]
    )
    with start_span(
        name=f"{model_name}.ainvoke",
        type=SpanTypeAttribute.LLM,
        input=input,
        metadata={**clean_kwargs, **extract_metadata(instance, "model")},
    ) as span:
        result = await wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result, kwargs.get("messages", [])))
        return result


def _model_invoke_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, clean_kwargs = get_args_kwargs(
        args, kwargs, ["messages", "assistant_messages", "response_format", "tools", "tool_choice"]
    )

    def _trace_stream():
        start = time.time()
        with start_span(
            name=f"{model_name}.invoke_stream",
            type=SpanTypeAttribute.LLM,
            input=input,
            metadata={**clean_kwargs, **extract_metadata(instance, "model")},
        ) as span:
            first = True
            collected_chunks = []
            for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                collected_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_model_chunks(collected_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))

    return _trace_stream()


def _model_ainvoke_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, clean_kwargs = get_args_kwargs(
        args, kwargs, ["messages", "assistant_messages", "response_format", "tools", "tool_choice"]
    )

    async def _trace_astream():
        start = time.time()
        with start_span(
            name=f"{model_name}.ainvoke_stream",
            type=SpanTypeAttribute.LLM,
            input=input,
            metadata={**clean_kwargs, **extract_metadata(instance, "model")},
        ) as span:
            first = True
            collected_chunks = []
            async for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                collected_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_model_chunks(collected_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))

    return _trace_astream()


def _model_response_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, clean_kwargs = get_args_kwargs(
        args, kwargs, ["messages", "response_format", "tools", "functions", "tool_chocie", "tool_call_limit"]
    )
    with start_span(
        name=f"{model_name}.response",
        type=SpanTypeAttribute.LLM,
        input=input,
        metadata={**clean_kwargs, **extract_metadata(instance, "model")},
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result, kwargs.get("messages", [])))
        return result


async def _model_aresponse_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, clean_kwargs = get_args_kwargs(
        args, kwargs, ["messages", "response_format", "tools", "functions", "tool_chocie", "tool_call_limit"]
    )
    with start_span(
        name=f"{model_name}.aresponse",
        type=SpanTypeAttribute.LLM,
        input=input,
        metadata={**clean_kwargs, **extract_metadata(instance, "model")},
    ) as span:
        result = await wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result, kwargs.get("messages", [])))
        return result


def _model_response_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, clean_kwargs = get_args_kwargs(
        args, kwargs, ["messages", "response_format", "tools", "functions", "tool_chocie", "tool_call_limit"]
    )

    def _trace_stream():
        start = time.time()
        with start_span(
            name=f"{model_name}.response_stream",
            type=SpanTypeAttribute.LLM,
            input=input,
            metadata={**clean_kwargs, **extract_metadata(instance, "model")},
        ) as span:
            first = True
            collected_chunks = []
            for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                collected_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_response_stream_chunks(collected_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))

    return _trace_stream()


def _model_aresponse_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, clean_kwargs = get_args_kwargs(
        args, kwargs, ["messages", "response_format", "tools", "functions", "tool_chocie", "tool_call_limit"]
    )

    async def _trace_astream():
        start = time.time()
        with start_span(
            name=f"{model_name}.aresponse_stream",
            type=SpanTypeAttribute.LLM,
            input=input,
            metadata={**clean_kwargs, **extract_metadata(instance, "model")},
        ) as span:
            first = True
            collected_chunks = []
            async for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                collected_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_response_stream_chunks(collected_chunks)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))

    return _trace_astream()


# ---------------------------------------------------------------------------
# FunctionCall wrappers
# ---------------------------------------------------------------------------


def _get_function_name(instance) -> str:
    if hasattr(instance, "function") and hasattr(instance.function, "name"):
        return instance.function.name
    return "Unknown"


def _function_call_execute_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    function_name = _get_function_name(instance)
    entrypoint_args = instance._build_entrypoint_args()
    with start_span(
        name=f"{function_name}.execute",
        type=SpanTypeAttribute.TOOL,
        input=(instance.arguments or {}),
        metadata={
            "name": instance.function.name,
            "entrypoint": instance.function.entrypoint.__name__,
            **(entrypoint_args or {}),
        },
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=result)
        return result


async def _function_call_aexecute_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    function_name = _get_function_name(instance)
    entrypoint_args = instance._build_entrypoint_args()
    with start_span(
        name=f"{function_name}.aexecute",
        type=SpanTypeAttribute.TOOL,
        input=(instance.arguments or {}),
        metadata={
            "name": instance.function.name,
            "entrypoint": instance.function.entrypoint.__name__,
            **(entrypoint_args or {}),
        },
    ) as span:
        result = await wrapped(*args, **kwargs)
        span.log(output=result)
        return result


# ---------------------------------------------------------------------------
# Workflow wrappers
# ---------------------------------------------------------------------------


def _extract_workflow_input(
    args: Any,
    kwargs: Any,
    *,
    execution_input_index: int,
    workflow_run_response_index: int,
) -> dict[str, Any]:
    execution_input = (
        args[execution_input_index] if len(args) > execution_input_index else kwargs.get("execution_input")
    )
    workflow_run_response = (
        args[workflow_run_response_index]
        if len(args) > workflow_run_response_index
        else kwargs.get("workflow_run_response")
    )
    result: dict[str, Any] = {}
    if execution_input:
        if hasattr(execution_input, "input"):
            result["input"] = execution_input.input
        result["execution_input"] = _try_to_dict(execution_input)
    if workflow_run_response:
        result["run_response"] = _try_to_dict(workflow_run_response)
    return result


def _extract_workflow_agent_input(args: Any, kwargs: Any) -> dict[str, Any]:
    user_input = args[0] if len(args) > 0 else kwargs.get("user_input")
    execution_input = args[2] if len(args) > 2 else kwargs.get("execution_input")
    result: dict[str, Any] = {"input": user_input}
    if execution_input:
        result["execution_input"] = _try_to_dict(execution_input)
    return result


def _workflow_execute_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    workflow_name = getattr(instance, "name", None) or "Workflow"
    input_data = _extract_workflow_input(args, kwargs, execution_input_index=1, workflow_run_response_index=2)
    workflow_metadata = extract_metadata(instance, "workflow")
    with start_span(
        name=f"{workflow_name}.run",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata=workflow_metadata,
        propagated_event={"metadata": workflow_metadata},
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result))
        return result


def _workflow_execute_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    workflow_name = getattr(instance, "name", None) or "Workflow"
    input_data = _extract_workflow_input(args, kwargs, execution_input_index=1, workflow_run_response_index=2)
    workflow_metadata = extract_metadata(instance, "workflow")
    workflow_run_response = args[2] if len(args) > 2 else kwargs.get("workflow_run_response")

    def _trace_stream():
        start = time.time()
        span = start_span(
            name=f"{workflow_name}.run_stream",
            type=SpanTypeAttribute.TASK,
            input=input_data,
            metadata=workflow_metadata,
            propagated_event={"metadata": workflow_metadata},
        )
        span.set_current()
        should_unset = True
        try:
            first = True
            all_chunks = []
            for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                all_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_workflow_chunks(all_chunks, workflow_run_response)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
        except GeneratorExit:
            should_unset = False
            raise
        except Exception as e:
            span.log(error=str(e))
            raise
        finally:
            if should_unset:
                span.unset_current()
            span.end()

    return _trace_stream()


async def _workflow_aexecute_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    workflow_name = getattr(instance, "name", None) or "Workflow"
    input_data = _extract_workflow_input(args, kwargs, execution_input_index=2, workflow_run_response_index=3)
    workflow_metadata = extract_metadata(instance, "workflow")
    with start_span(
        name=f"{workflow_name}.arun",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata=workflow_metadata,
        propagated_event={"metadata": workflow_metadata},
    ) as span:
        result = await wrapped(*args, **kwargs)
        span.log(output=result, metrics=extract_metrics(result))
        return result


def _workflow_aexecute_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    workflow_name = getattr(instance, "name", None) or "Workflow"
    input_data = _extract_workflow_input(args, kwargs, execution_input_index=2, workflow_run_response_index=3)
    workflow_metadata = extract_metadata(instance, "workflow")
    workflow_run_response = args[3] if len(args) > 3 else kwargs.get("workflow_run_response")

    async def _trace_stream():
        start = time.time()
        span = start_span(
            name=f"{workflow_name}.arun_stream",
            type=SpanTypeAttribute.TASK,
            input=input_data,
            metadata=workflow_metadata,
            propagated_event={"metadata": workflow_metadata},
        )
        span.set_current()
        should_unset = True
        try:
            first = True
            all_chunks = []
            async for chunk in wrapped(*args, **kwargs):
                if first:
                    span.log(metrics={"time_to_first_token": time.time() - start})
                    first = False
                all_chunks.append(chunk)
                yield chunk
            aggregated = _aggregate_workflow_chunks(all_chunks, workflow_run_response)
            span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
        except GeneratorExit:
            should_unset = False
            raise
        except Exception as e:
            span.log(error=str(e))
            raise
        finally:
            if should_unset:
                span.unset_current()
            span.end()

    return _trace_stream()


def _workflow_execute_workflow_agent_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    workflow_name = getattr(instance, "name", None) or "Workflow"
    stream = kwargs.get("stream", False)
    span_suffix = "run_stream" if stream else "run"
    workflow_metadata = extract_metadata(instance, "workflow")
    input_data = _extract_workflow_agent_input(args, kwargs)

    span = start_span(
        name=f"{workflow_name}.{span_suffix}",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata=workflow_metadata,
        propagated_event={"metadata": workflow_metadata},
    )
    span.set_current()
    start = time.time()
    try:
        result = wrapped(*args, **kwargs)
        if stream and is_sync_iterator(result):

            def _trace_stream():
                should_unset = True
                try:
                    first = True
                    all_chunks = []
                    for chunk in result:
                        if first:
                            span.log(metrics={"time_to_first_token": time.time() - start})
                            first = False
                        all_chunks.append(chunk)
                        yield chunk
                    aggregated = _aggregate_workflow_chunks(all_chunks)
                    span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
                except GeneratorExit:
                    should_unset = False
                    raise
                except Exception as e:
                    span.log(error=str(e))
                    raise
                finally:
                    if should_unset:
                        span.unset_current()
                    span.end()

            return _trace_stream()

        span.log(output=result, metrics=extract_metrics(result))
        span.unset_current()
        span.end()
        return result
    except Exception as e:
        span.log(error=str(e))
        span.unset_current()
        span.end()
        raise


async def _workflow_aexecute_workflow_agent_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    workflow_name = getattr(instance, "name", None) or "Workflow"
    stream = kwargs.get("stream", False)
    span_suffix = "arun_stream" if stream else "arun"
    workflow_metadata = extract_metadata(instance, "workflow")
    input_data = _extract_workflow_agent_input(args, kwargs)

    span = start_span(
        name=f"{workflow_name}.{span_suffix}",
        type=SpanTypeAttribute.TASK,
        input=input_data,
        metadata=workflow_metadata,
        propagated_event={"metadata": workflow_metadata},
    )
    span.set_current()
    start = time.time()
    try:
        result = await wrapped(*args, **kwargs)
        if stream and is_async_iterator(result):

            async def _trace_stream():
                should_unset = True
                try:
                    first = True
                    all_chunks = []
                    async for chunk in result:
                        if first:
                            span.log(metrics={"time_to_first_token": time.time() - start})
                            first = False
                        all_chunks.append(chunk)
                        yield chunk
                    aggregated = _aggregate_workflow_chunks(all_chunks)
                    span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
                except GeneratorExit:
                    should_unset = False
                    raise
                except Exception as e:
                    span.log(error=str(e))
                    raise
                finally:
                    if should_unset:
                        span.unset_current()
                    span.end()

            return _trace_stream()

        span.log(output=result, metrics=extract_metrics(result))
        span.unset_current()
        span.end()
        return result
    except Exception as e:
        span.log(error=str(e))
        span.unset_current()
        span.end()
        raise
