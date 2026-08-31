import contextvars
import time
from contextlib import contextmanager
from inspect import isawaitable
from typing import Any

from braintrust.integrations.utils import _materialize_attachment, _try_to_dict
from braintrust.logger import start_span as _bt_start_span


_INSTRUMENTATION = "agno-auto"

_SUPPRESSED: contextvars.ContextVar[bool] = contextvars.ContextVar("braintrust_agno_suppressed", default=False)


@contextmanager
def suppress_spans():
    """Skip agno instrumentation entirely for the duration of the block.

    ``PerformanceEval`` calls the measured function ``warmup_runs + num_iterations``
    times (60 by default). Tracing all of those would bury the eval's own row under
    dozens of identical child traces, so the agno patchers hand straight through to
    the wrapped method while this is set, building no payloads and retaining no
    stream chunks (see ``_AgnoFunctionWrapperPatcher``). Spans from other
    integrations (openai, anthropic, ...) are not agno's to suppress and still
    appear under the eval's row.
    """
    token = _SUPPRESSED.set(True)
    try:
        yield
    finally:
        _SUPPRESSED.reset(token)


def spans_suppressed() -> bool:
    """Whether agno instrumentation is currently suppressed."""
    return _SUPPRESSED.get()


def start_span(*args, parent_object: Any | None = None, **kwargs):
    """Start a span stamped as agno-instrumented.

    ``parent_object`` starts the span on an explicit parent (an experiment, say)
    rather than on whatever the ambient span/experiment/logger resolution picks, so
    both routes keep the instrumentation stamp.
    """
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    start = _bt_start_span if parent_object is None else parent_object.start_span
    return start(*args, **kwargs)


from braintrust.span_types import SpanTypeAttribute
from braintrust.util import is_numeric


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def omit(obj: dict[str, Any], keys: list[str]):
    return {k: v for k, v in obj.items() if k not in keys}


def bound_args(args: Any, kwargs: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Resolve a wrapped method's arguments by name, positional or keyword.

    Cheaper and more forgiving than binding the real signature per call, which the
    wrappers here deliberately avoid.
    """
    bound: dict[str, Any] = dict(zip(names, args))
    for name in names:
        if name in kwargs:
            bound[name] = kwargs[name]
    return bound


# Keys the SDK-integrations spec routes into metadata rather than the span input.
_MODEL_METADATA_KEYS = ("tools", "tool_choice", "functions", "tool_call_limit")


def _split_model_call(
    args: tuple, kwargs: dict[str, Any], positional_order: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind positional args to their names and split into (input, metadata_extras).

    Tools, tool_choice, function schemas and tool-call limits go to metadata;
    everything else that names a request field stays in input.
    """
    combined: dict[str, Any] = dict(kwargs)
    for i, key in enumerate(positional_order):
        if i < len(args):
            combined[key] = args[i]
    metadata_extras: dict[str, Any] = {}
    for k in _MODEL_METADATA_KEYS:
        if k in combined:
            metadata_extras[k] = combined.pop(k)
    return combined, metadata_extras


def _prepare_model_input(input_data: dict[str, Any]) -> dict[str, Any]:
    """Materialize inline media in an Agno request-input dict before logging.

    Fast path: leave `messages` as raw Agno objects when no message carries
    inline binary media — Braintrust's log-time serializer handles the rest.
    """
    messages = input_data.get("messages")
    if not isinstance(messages, list) or not any(_message_has_inline_media(m) for m in messages):
        return input_data
    return {**input_data, "messages": _materialize_agno_messages(messages)}


def _prepare_model_output(result: Any) -> Any:
    """Materialize inline media in an Agno model response before logging.

    Fast path: pass the raw SDK object through when it has no inline binary
    media — Braintrust's log-time serializer already converts dataclasses and
    Pydantic models. Only when we actually need to swap in an Attachment do we
    pay the dict conversion.
    """
    if not _result_has_inline_media(result):
        return result
    try:
        as_dict = _try_to_dict(result)
    except Exception:
        return result
    if not isinstance(as_dict, dict):
        return result
    return _materialize_agno_output_media(dict(as_dict))


def _agno_media_has_inline_bytes(media: Any) -> bool:
    """True when *media* carries a `content` byte payload or a `filepath`."""
    if media is None:
        return False
    content = getattr(media, "content", None) if not isinstance(media, dict) else media.get("content")
    if isinstance(content, (bytes, bytearray)) and content:
        return True
    if isinstance(content, str) and content:
        return True
    filepath = getattr(media, "filepath", None) if not isinstance(media, dict) else media.get("filepath")
    return isinstance(filepath, str) and bool(filepath)


def _iter_media_field(container: Any, field: str) -> Any:
    val = getattr(container, field, None) if not isinstance(container, dict) else container.get(field)
    if val is None:
        return ()
    return val if isinstance(val, list) else (val,)


def _message_has_inline_media(msg: Any) -> bool:
    for field, _ in _AGNO_MESSAGE_MEDIA_FIELDS:
        for item in _iter_media_field(msg, field):
            if _agno_media_has_inline_bytes(item):
                return True
    return False


def _result_has_inline_media(result: Any) -> bool:
    for field in ("images", "audio", "audios", "videos", "files"):
        for item in _iter_media_field(result, field):
            if _agno_media_has_inline_bytes(item):
                return True
    return False


def is_sync_iterator(result: Any) -> bool:
    return hasattr(result, "__iter__") and hasattr(result, "__next__")


def is_async_iterator(result: Any) -> bool:
    return hasattr(result, "__aiter__") and hasattr(result, "__anext__")


# ---------------------------------------------------------------------------
# Multimodal materialization
# ---------------------------------------------------------------------------
#
# Agno's Image / Audio / Video / File objects carry inline bytes on ``content``
# (plus optional ``url`` / ``filepath`` / ``mime_type`` / ``format`` / ``filename``).
# The integrations spec requires inline binary media to be converted to
# Braintrust ``Attachment`` objects at the leaf position; remote URLs stay as
# strings and unrecognized shapes pass through unchanged.


def _agno_media_mime_type(as_dict: dict[str, Any], media_kind: str) -> str | None:
    mime = as_dict.get("mime_type")
    if isinstance(mime, str) and mime:
        return mime
    fmt = as_dict.get("format")
    if isinstance(fmt, str) and fmt:
        return f"{media_kind}/{fmt}"
    return None


def _materialize_agno_media(value: Any, media_kind: str) -> Any:
    """Materialize a single Agno media object to a serializable dict.

    Returns the input unchanged when it is not a recognizable Agno media object
    or when materialization is not possible.
    """
    if value is None:
        return value

    as_dict = value if isinstance(value, dict) else _try_to_dict(value)
    if not isinstance(as_dict, dict):
        return value

    content = as_dict.get("content")
    filepath = as_dict.get("filepath")
    url = as_dict.get("url")
    filename = as_dict.get("filename")
    mime_type = _agno_media_mime_type(as_dict, media_kind)

    raw: Any = content if isinstance(content, (bytes, bytearray, str)) and content else None
    if raw is None and isinstance(filepath, str) and filepath:
        raw = filepath

    resolved = None
    if raw is not None:
        try:
            resolved = _materialize_attachment(
                raw,
                mime_type=mime_type,
                filename=filename if isinstance(filename, str) else None,
                label=media_kind,
                prefix=media_kind,
            )
        except Exception:
            resolved = None

    result: dict[str, Any] = {k: v for k, v in as_dict.items() if k not in ("content", "filepath")}
    if resolved is not None:
        result.update(resolved.multimodal_part_payload)
    elif isinstance(url, str) and url and "url" not in result:
        # Preserve remote URL references verbatim per the spec.
        result["url"] = url
    return result


def _materialize_agno_media_list(value: Any, media_kind: str) -> Any:
    if not isinstance(value, list):
        return value
    return [_materialize_agno_media(v, media_kind) for v in value]


_AGNO_MESSAGE_MEDIA_FIELDS = (
    ("images", "image"),
    ("image_output", "image"),
    ("audio", "audio"),
    ("audio_output", "audio"),
    ("videos", "video"),
    ("video_output", "video"),
    ("files", "file"),
    ("file_output", "file"),
)


def _materialize_agno_message(msg: Any) -> Any:
    """Return a dict form of an Agno Message with inline media replaced by attachments."""
    as_dict = msg if isinstance(msg, dict) else _try_to_dict(msg)
    if not isinstance(as_dict, dict):
        return msg
    result = dict(as_dict)
    for field, kind in _AGNO_MESSAGE_MEDIA_FIELDS:
        if field in result and result[field]:
            if isinstance(result[field], list):
                result[field] = [_materialize_agno_media(v, kind) for v in result[field]]
            else:
                result[field] = _materialize_agno_media(result[field], kind)
    return result


def _materialize_agno_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    return [_materialize_agno_message(m) for m in messages]


def _materialize_agno_output_media(aggregated: dict[str, Any]) -> dict[str, Any]:
    """Materialize model-response-style media fields in an aggregated dict in place."""
    for field, kind in (("images", "image"), ("videos", "video"), ("files", "file")):
        val = aggregated.get(field)
        if isinstance(val, list) and val:
            aggregated[field] = [_materialize_agno_media(v, kind) for v in val]
    audio_val = aggregated.get("audio")
    if audio_val is not None:
        if isinstance(audio_val, list):
            aggregated["audio"] = [_materialize_agno_media(v, "audio") for v in audio_val]
        else:
            aggregated["audio"] = _materialize_agno_media(audio_val, "audio")
    return aggregated


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

    return _materialize_agno_output_media(aggregated)


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

    return _materialize_agno_output_media(aggregated)


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
            span.log(error=e)
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
            span.log(error=e)
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
            span.log(error=e)
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
            span.log(error=e)
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
            span.log(error=e)
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
            span.log(error=e)
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
        span.log(error=e)
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
                    span.log(error=e)
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
        span.log(error=e)
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
    input, metadata_extras = _split_model_call(args, kwargs, ["assistant_message", "messages"])
    input = _prepare_model_input(input)
    with start_span(
        name=f"{model_name}.invoke",
        type=SpanTypeAttribute.LLM,
        input=input,
        metadata={**metadata_extras, **extract_metadata(instance, "model")},
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=_prepare_model_output(result), metrics=extract_metrics(result, kwargs.get("messages", [])))
        return result


async def _model_ainvoke_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, metadata_extras = _split_model_call(args, kwargs, ["messages", "assistant_message"])
    input = _prepare_model_input(input)
    with start_span(
        name=f"{model_name}.ainvoke",
        type=SpanTypeAttribute.LLM,
        input=input,
        metadata={**metadata_extras, **extract_metadata(instance, "model")},
    ) as span:
        result = await wrapped(*args, **kwargs)
        span.log(output=_prepare_model_output(result), metrics=extract_metrics(result, kwargs.get("messages", [])))
        return result


def _model_invoke_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, metadata_extras = _split_model_call(args, kwargs, ["messages", "assistant_messages"])
    input = _prepare_model_input(input)

    def _trace_stream():
        start = time.time()
        with start_span(
            name=f"{model_name}.invoke_stream",
            type=SpanTypeAttribute.LLM,
            input=input,
            metadata={**metadata_extras, **extract_metadata(instance, "model")},
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
    input, metadata_extras = _split_model_call(args, kwargs, ["messages", "assistant_messages"])
    input = _prepare_model_input(input)

    async def _trace_astream():
        start = time.time()
        with start_span(
            name=f"{model_name}.ainvoke_stream",
            type=SpanTypeAttribute.LLM,
            input=input,
            metadata={**metadata_extras, **extract_metadata(instance, "model")},
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
    input, metadata_extras = _split_model_call(args, kwargs, ["messages"])
    input = _prepare_model_input(input)
    with start_span(
        name=f"{model_name}.response",
        type=SpanTypeAttribute.LLM,
        input=input,
        metadata={**metadata_extras, **extract_metadata(instance, "model")},
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=_prepare_model_output(result), metrics=extract_metrics(result, kwargs.get("messages", [])))
        return result


async def _model_aresponse_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, metadata_extras = _split_model_call(args, kwargs, ["messages"])
    input = _prepare_model_input(input)
    with start_span(
        name=f"{model_name}.aresponse",
        type=SpanTypeAttribute.LLM,
        input=input,
        metadata={**metadata_extras, **extract_metadata(instance, "model")},
    ) as span:
        result = await wrapped(*args, **kwargs)
        span.log(output=_prepare_model_output(result), metrics=extract_metrics(result, kwargs.get("messages", [])))
        return result


def _model_response_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    model_name = _get_model_name(instance)
    input, metadata_extras = _split_model_call(args, kwargs, ["messages"])
    input = _prepare_model_input(input)

    def _trace_stream():
        start = time.time()
        with start_span(
            name=f"{model_name}.response_stream",
            type=SpanTypeAttribute.LLM,
            input=input,
            metadata={**metadata_extras, **extract_metadata(instance, "model")},
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
    input, metadata_extras = _split_model_call(args, kwargs, ["messages"])
    input = _prepare_model_input(input)

    async def _trace_astream():
        start = time.time()
        with start_span(
            name=f"{model_name}.aresponse_stream",
            type=SpanTypeAttribute.LLM,
            input=input,
            metadata={**metadata_extras, **extract_metadata(instance, "model")},
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


def _function_call_metadata(instance: Any) -> dict[str, Any]:
    """Best-effort metadata extraction for a FunctionCall. Contains instrumentation errors."""
    metadata: dict[str, Any] = {}
    try:
        metadata["name"] = instance.function.name
    except Exception:
        pass
    try:
        metadata["entrypoint"] = instance.function.entrypoint.__name__
    except Exception:
        pass
    try:
        entrypoint_args = instance._build_entrypoint_args()
        if entrypoint_args:
            metadata.update(entrypoint_args)
    except Exception:
        pass
    return metadata


def _function_call_execute_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    function_name = _get_function_name(instance)
    metadata = _function_call_metadata(instance)
    with start_span(
        name=f"{function_name}.execute",
        type=SpanTypeAttribute.TOOL,
        input=(getattr(instance, "arguments", None) or {}),
        metadata=metadata,
    ) as span:
        result = wrapped(*args, **kwargs)
        span.log(output=result)
        return result


async def _function_call_aexecute_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    function_name = _get_function_name(instance)
    metadata = _function_call_metadata(instance)
    with start_span(
        name=f"{function_name}.aexecute",
        type=SpanTypeAttribute.TOOL,
        input=(getattr(instance, "arguments", None) or {}),
        metadata=metadata,
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
            span.log(error=e)
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
            span.log(error=e)
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
                    span.log(error=e)
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
        span.log(error=e)
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
                    span.log(error=e)
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
        span.log(error=e)
        span.unset_current()
        span.end()
        raise
