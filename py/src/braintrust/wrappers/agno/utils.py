import time
from typing import Any

from braintrust.util import is_numeric


def omit(obj: dict[str, Any], keys: list[str]):
    return {k: v for k, v in obj.items() if k not in keys}


def is_patched(obj: Any) -> bool:
    return getattr(obj, "_braintrust_patched", False)


def mark_patched(obj: Any):
    setattr(obj, "_braintrust_patched", True)


def clean(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if v is not None}


def get_args_kwargs(args: list[str], kwargs: dict[str, Any], keys: list[str]):
    return {k: args[i] if args else kwargs.get(k) for i, k in enumerate(keys)}, omit(kwargs, keys)



def _try_to_dict(obj: Any) -> Any:
    """Convert object to dict, handling different object types like OpenAI wrapper."""
    if isinstance(obj, dict):
        return obj
    # convert a pydantic object to a dict
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # deprecated pydantic method, try model_dump first.
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    # handle objects with __dict__ (like Agno Metrics objects)
    if hasattr(obj, "__dict__"):
        try:
            return obj.__dict__.copy()
        except Exception:
            pass
    return obj


# Agno field names to canonical Braintrust field names (following OpenAI wrapper pattern)
AGNO_METRICS_MAP = {
    # Core token metrics - using OpenAI wrapper naming
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "total_tokens": "tokens",
    # Reasoning and audio tokens
    "reasoning_tokens": "completion_reasoning_tokens",
    "audio_input_tokens": "prompt_audio_tokens",
    "audio_output_tokens": "completion_audio_tokens",
    # Cache tokens
    "cache_read_tokens": "prompt_cached_tokens",
    "cache_write_tokens": "prompt_cache_creation_tokens",
    # Timing metrics
    "duration": "duration",
    "time_to_first_token": "time_to_first_token",
}


def extract_metadata(instance: Any, component: str) -> dict[str, Any]:
    """Extract metadata from any component (model, agent, team)."""
    metadata = {"component": component}

    # Component-specific name fields
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

    # Convert to dict like OpenAI wrapper
    usage_dict = _try_to_dict(usage)
    if not isinstance(usage_dict, dict):
        return metrics

    # Simple loop through Agno fields and map to Braintrust names
    for agno_name, value in usage_dict.items():
        if agno_name in AGNO_METRICS_MAP and is_numeric(value) and value != 0:
            braintrust_name = AGNO_METRICS_MAP[agno_name]
            metrics[braintrust_name] = value

    return metrics


def extract_metrics(result: Any, messages: list | None = None) -> dict[str, Any]:
    """
    Unified metrics extraction for all components.

    Handles:
    - Model responses with response_usage
    - Agent/Team responses with metrics
    - Messages with metrics (for model responses)
    """
    # For model responses with response_usage
    if hasattr(result, "response_usage") and result.response_usage:
        return parse_metrics_from_agno(result.response_usage)

    # For agent/team responses with metrics
    if hasattr(result, "metrics") and result.metrics:
        metrics = parse_metrics_from_agno(result.metrics)
        return metrics if metrics else None

    # If no metrics found and we have messages, look for metrics in assistant messages (model-specific)
    if messages:
        for msg in messages:
            # Look for assistant messages with metrics
            if hasattr(msg, "role") and msg.role == "assistant" and hasattr(msg, "metrics") and msg.metrics:
                return parse_metrics_from_agno(msg.metrics)

    return {}


def extract_streaming_metrics(aggregated: dict[str, Any], start_time: float) -> dict[str, Any] | None:
    """Extract metrics from aggregated streaming response."""
    metrics = {}

    # Extract metrics from aggregated data
    # The metrics are already in Braintrust format from _aggregate_model_chunks
    if aggregated.get("metrics") and isinstance(aggregated["metrics"], dict):
        # Merge the aggregated metrics
        metrics.update(aggregated["metrics"])
    # Handle object-like metrics payloads (e.g. RunCompletedEvent.metrics)
    elif aggregated.get("metrics"):
        parsed_metrics = parse_metrics_from_agno(aggregated["metrics"])
        if parsed_metrics:
            metrics.update(parsed_metrics)
    # Also check response_usage for backward compatibility
    elif aggregated.get("response_usage"):
        response_metrics = parse_metrics_from_agno(aggregated["response_usage"])
        if response_metrics:
            metrics.update(response_metrics)

    # Ensure we have the duration calculated from start_time
    metrics["duration"] = time.time() - start_time

    return metrics if metrics else None


def _aggregate_metrics(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Aggregate metrics from source into target dict."""
    for key, value in source.items():
        if is_numeric(value):
            if key in target:
                # For timing metrics, we keep the latest
                if "time" in key.lower() or "duration" in key.lower():
                    target[key] = value
                # For token counts, we sum them
                elif "token" in key.lower() or key == "tokens":
                    target[key] = (target.get(key, 0) or 0) + value
                # For other metrics, keep the latest
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
            # Parse and aggregate metrics from each chunk
            chunk_metrics = parse_metrics_from_agno(chunk.response_usage)
            if chunk_metrics:
                _aggregate_metrics(aggregated["metrics"], chunk_metrics)

    # Convert aggregated metrics dict to the response_usage format for backward compatibility
    if aggregated["metrics"]:
        aggregated["response_usage"] = aggregated["metrics"]
    else:
        aggregated["metrics"] = None

    return aggregated


def _aggregate_response_stream_chunks(chunks: list[Any]) -> dict[str, Any]:
    """
    Aggregate chunks from response_stream which can be ModelResponse, RunOutputEvent, or TeamRunOutputEvent.

    This is more robust than _aggregate_model_chunks as it handles different event types.
    """
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
        # Handle ModelResponse chunks
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
                # Parse and aggregate metrics from each chunk
                chunk_metrics = parse_metrics_from_agno(chunk.response_usage)
                if chunk_metrics:
                    _aggregate_metrics(aggregated["metrics"], chunk_metrics)

            # Also check for metrics attribute directly (for some response types)
            elif hasattr(chunk, "metrics") and chunk.metrics:
                chunk_metrics = parse_metrics_from_agno(chunk.metrics)
                if chunk_metrics:
                    _aggregate_metrics(aggregated["metrics"], chunk_metrics)

        # Handle RunOutputEvent/TeamRunOutputEvent chunks - these typically contain content
        elif hasattr(chunk, "content"):
            if chunk.content:
                aggregated["content"] += str(chunk.content)

        # Handle other event types that might have metrics
        if hasattr(chunk, "metrics") and chunk.metrics and "metrics" not in str(type(chunk)):
            chunk_metrics = parse_metrics_from_agno(chunk.metrics)
            if chunk_metrics:
                _aggregate_metrics(aggregated["metrics"], chunk_metrics)

    # Convert aggregated metrics dict to the response_usage format for backward compatibility
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


def is_sync_iterator(result: Any) -> bool:
    return hasattr(result, "__iter__") and hasattr(result, "__next__")


def is_async_iterator(result: Any) -> bool:
    return hasattr(result, "__aiter__") and hasattr(result, "__anext__")


def trace_sync_stream_result(result: Any, span: Any, start: float):
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

            aggregated = _aggregate_agent_chunks(all_chunks)
            span.log(
                output=aggregated,
                metrics=extract_streaming_metrics(aggregated, start),
            )
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


def trace_async_stream_result(result: Any, span: Any, start: float):
    async def _trace_astream():
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
            span.log(
                output=aggregated,
                metrics=extract_streaming_metrics(aggregated, start),
            )
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

    return _trace_astream()


# Legacy aliases for backward compatibility
_extract_run_metrics = extract_metrics
_extract_streaming_metrics = extract_streaming_metrics
_extract_model_metrics = extract_metrics
_parse_metrics_from_agno = parse_metrics_from_agno
