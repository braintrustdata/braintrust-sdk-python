"""Google GenAI-specific span creation, metadata extraction, stream handling, and output normalization."""

import enum
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, get_args, get_origin

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.logger import Attachment, start_span
from braintrust.span_types import SpanTypeAttribute


if TYPE_CHECKING:
    from google.genai.types import (
        EmbedContentResponse,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_input(api_client: Any, input: dict[str, Any]) -> dict[str, Any]:
    config = bt_safe_deep_copy(input.get("config"))

    if config is not None:
        tools = _serialize_tools(api_client, input)

        if tools is not None:
            config["tools"] = tools

        input["config"] = config

    # Serialize contents to handle binary data (e.g., images)
    if "contents" in input:
        input["contents"] = _serialize_contents(input["contents"])

    return input


def _serialize_contents(contents: Any) -> Any:
    """Serialize contents, converting binary data into attachments."""
    if contents is None:
        return None

    if isinstance(contents, list):
        return [_serialize_content_item(item) for item in contents]

    return _serialize_content_item(contents)


def _serialize_content_item(item: Any) -> Any:
    if item is None or isinstance(item, (str, int, float, bool)):
        return item

    if _is_content_like(item):
        return _serialize_content(item)

    return _serialize_part(item)


def _is_content_like(item: Any) -> bool:
    if isinstance(item, dict):
        return "parts" in item
    return getattr(getattr(item, "__class__", None), "__name__", None) != "Part" and hasattr(item, "parts")


def _serialize_content(content: Any) -> Any:
    if isinstance(content, dict):
        result = {}
        for key, value in content.items():
            if key == "parts" and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                result[key] = [_serialize_part(part) for part in value]
            else:
                result[key] = bt_safe_deep_copy(value)
        return result

    serialized = _generic_serialize(content)
    result = dict(serialized) if isinstance(serialized, dict) else {}

    result["parts"] = [_serialize_part(part) for part in _ensure_list(_get_attr_or_key(content, "parts"))]

    role = _get_attr_or_key(content, "role")
    if role is not None:
        result["role"] = role

    return result


def _serialize_part(part: Any) -> Any:
    if part is None or isinstance(part, (str, int, float, bool)):
        return part

    if isinstance(part, dict):
        serialized_part = {}
        for key, value in part.items():
            if key in ("inline_data", "inlineData"):
                inline_data = _serialize_inline_data(value)
                if inline_data is not None:
                    serialized_part.update(inline_data)
                else:
                    serialized_part[key] = bt_safe_deep_copy(value)
            else:
                serialized_part[key] = bt_safe_deep_copy(value)
        return serialized_part

    if hasattr(part, "text") and part.text is not None:
        result = {"text": part.text}
        if hasattr(part, "thought") and part.thought:
            result["thought"] = part.thought
        return result

    inline_data = _serialize_inline_data(_get_attr_or_key(part, "inline_data", "inlineData"))
    if inline_data is not None:
        return inline_data

    return _generic_serialize(part)


def _serialize_inline_data(inline_data: Any) -> dict[str, Any] | None:
    if inline_data is None:
        return None

    data = _get_attr_or_key(inline_data, "data")
    mime_type = _get_attr_or_key(inline_data, "mime_type", "mimeType")
    if not isinstance(data, bytes) or not isinstance(mime_type, str):
        return None

    extension = mime_type.split("/")[1] if "/" in mime_type else "bin"
    attachment = Attachment(data=data, filename=f"file.{extension}", content_type=mime_type)
    return {"image_url": {"url": attachment}}


def _generic_serialize(item: Any) -> Any:
    if item is None or isinstance(item, (str, int, float, bool)):
        return item

    if hasattr(item, "model_dump") and callable(item.model_dump):
        return item.model_dump()
    if hasattr(item, "dump") and callable(item.dump):
        return item.dump()
    if hasattr(item, "to_dict") and callable(item.to_dict):
        return item.to_dict()

    return bt_safe_deep_copy(item)


def _serialize_tools(api_client: Any, input: Any | None) -> Any | None:
    try:
        return _serialize_tools_with_google(api_client, input)
    except Exception:
        backend = "vertex" if getattr(api_client, "vertexai", False) else "mldev"
        logger.debug("Failed to serialize tools via Google SDK for %s", backend, exc_info=True)
        return _serialize_tools_fallback(input)


def _serialize_tools_with_google(api_client: Any, input: Any | None) -> Any | None:
    from google.genai.models import (
        _GenerateContentParameters_to_mldev,  # pyright: ignore [reportPrivateUsage]
        _GenerateContentParameters_to_vertex,  # pyright: ignore [reportPrivateUsage]
    )

    if api_client.vertexai:
        serialized = _GenerateContentParameters_to_vertex(api_client, input)
    else:
        serialized = _GenerateContentParameters_to_mldev(api_client, input)

    return serialized.get("tools")


def _serialize_tools_fallback(input: Any | None) -> Any | None:
    config = _get_attr_or_key(input, "config")
    tools = _get_attr_or_key(config, "tools")
    if not tools:
        return None

    serialized_tools = [_serialize_tool(tool) for tool in _ensure_list(tools)]
    return serialized_tools or None


def _serialize_tool(tool: Any) -> Any:
    if callable(tool):
        return {"functionDeclarations": [_serialize_callable_function_declaration(tool)]}

    serialized = _generic_serialize(tool)
    if isinstance(serialized, dict):
        declarations = _get_attr_or_key(serialized, "functionDeclarations", "function_declarations")
        if declarations is not None:
            result = {
                key: value
                for key, value in serialized.items()
                if key not in ("functionDeclarations", "function_declarations") and value is not None
            }
            result["functionDeclarations"] = [
                _serialize_function_declaration(declaration) for declaration in _ensure_list(declarations)
            ]
            return result

        if _looks_like_function_declaration(serialized):
            return {"functionDeclarations": [_serialize_function_declaration(serialized)]}

        return serialized

    if _looks_like_function_declaration(tool):
        return {"functionDeclarations": [_serialize_function_declaration(tool)]}

    return bt_safe_deep_copy(tool)


def _looks_like_function_declaration(obj: Any) -> bool:
    if isinstance(obj, dict):
        return "name" in obj and any(
            key in obj for key in ("description", "parameters", "parameters_json_schema", "parametersJsonSchema")
        )

    return hasattr(obj, "name") and any(
        hasattr(obj, attr) for attr in ("description", "parameters", "parameters_json_schema", "parametersJsonSchema")
    )


def _serialize_function_declaration(declaration: Any) -> dict[str, Any]:
    serialized = declaration if isinstance(declaration, dict) else _generic_serialize(declaration)
    result = (
        {key: value for key, value in serialized.items() if value is not None} if isinstance(serialized, dict) else {}
    )

    name = _get_attr_or_key(declaration, "name")
    if name is None:
        name = _get_attr_or_key(serialized, "name")
    if name is not None:
        result["name"] = name

    description = _get_attr_or_key(declaration, "description")
    if description is None:
        description = _get_attr_or_key(serialized, "description")
    if description:
        result["description"] = description

    parameters = _get_attr_or_key(declaration, "parameters")
    if parameters is None:
        parameters = _get_attr_or_key(declaration, "parameters_json_schema", "parametersJsonSchema")
    if parameters is None:
        parameters = _get_attr_or_key(serialized, "parameters")
    if parameters is None:
        parameters = _get_attr_or_key(serialized, "parameters_json_schema", "parametersJsonSchema")
    if parameters is not None:
        result["parameters"] = bt_safe_deep_copy(parameters)

    result.pop("parameters_json_schema", None)
    result.pop("parametersJsonSchema", None)
    return result


def _serialize_callable_function_declaration(tool: Any) -> dict[str, Any]:
    declaration: dict[str, Any] = {"name": getattr(tool, "__name__", type(tool).__name__)}

    description = inspect.getdoc(tool)
    if description:
        declaration["description"] = description

    try:
        signature = inspect.signature(tool)
    except (TypeError, ValueError):
        return declaration

    try:
        type_hints = inspect.get_annotations(tool, eval_str=True)
    except Exception:
        type_hints = getattr(tool, "__annotations__", {})

    properties = {}
    required = []

    for param_name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param_name in ("self", "cls"):
            continue

        schema = _annotation_to_google_schema(type_hints.get(param_name, param.annotation), param.default)
        if param.default is not inspect.Signature.empty:
            schema["default"] = _serialize_schema_default(param.default)
        else:
            required.append(param_name)

        properties[param_name] = schema

    if properties:
        declaration["parameters"] = {"type": "OBJECT", "properties": properties}
        if required:
            declaration["parameters"]["required"] = required

    return declaration


def _annotation_to_google_schema(annotation: Any, default: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty:
        return _value_to_google_schema(default)

    origin = get_origin(annotation)
    if origin is not None:
        if str(origin) == "typing.Annotated":
            return _annotation_to_google_schema(get_args(annotation)[0], default)

        if origin in (list, tuple, set, frozenset, Sequence):
            schema = {"type": "ARRAY"}
            args = get_args(annotation)
            if args:
                items_schema = _annotation_to_google_schema(args[0], inspect.Signature.empty)
                if items_schema:
                    schema["items"] = items_schema
            return schema

        if origin in (dict, Mapping):
            return {"type": "OBJECT"}

        if str(origin) in ("typing.Union", "types.UnionType"):
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(args) == 1:
                return _annotation_to_google_schema(args[0], default)
            return _value_to_google_schema(default)

        if str(origin) == "typing.Literal":
            literal_values = list(get_args(annotation))
            schema = _value_to_google_schema(literal_values[0] if literal_values else default)
            schema["enum"] = literal_values
            return schema

    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        enum_values = [member.value for member in annotation]
        schema = _value_to_google_schema(enum_values[0] if enum_values else default)
        if enum_values:
            schema["enum"] = enum_values
        return schema

    if inspect.isclass(annotation):
        if hasattr(annotation, "model_json_schema") and callable(annotation.model_json_schema):
            return annotation.model_json_schema()
        if hasattr(annotation, "schema") and callable(annotation.schema):
            return annotation.schema()

    primitive_types = {
        str: "STRING",
        int: "INTEGER",
        float: "NUMBER",
        bool: "BOOLEAN",
    }
    if annotation in primitive_types:
        return {"type": primitive_types[annotation]}

    return _value_to_google_schema(default)


def _value_to_google_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "BOOLEAN"}
    if isinstance(value, int):
        return {"type": "INTEGER"}
    if isinstance(value, float):
        return {"type": "NUMBER"}
    if isinstance(value, str):
        return {"type": "STRING"}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {"type": "ARRAY"}
    if isinstance(value, dict):
        return {"type": "OBJECT"}
    return {}


def _serialize_schema_default(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    return bt_safe_deep_copy(value)


def _get_attr_or_key(obj: Any, *names: str) -> Any:
    if obj is None:
        return None

    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


# ---------------------------------------------------------------------------
# Argument extraction helpers
# ---------------------------------------------------------------------------


def _omit(obj: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in keys}


def _get_args_kwargs(
    args: list[str], kwargs: dict[str, Any], keys: Iterable[str], omit_keys: Iterable[str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    return {k: args[i] if args else kwargs.get(k) for i, k in enumerate(keys)}, _omit(kwargs, omit_keys or keys)


def _clean(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if v is not None}


def _prepare_traced_call(
    api_client: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    input, clean_kwargs = _get_args_kwargs(args, kwargs, ["model", "contents", "config"], ["contents", "config"])
    return _serialize_input(api_client, input), clean_kwargs


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------


def _extract_usage_metadata_metrics(
    usage_metadata: "GenerateContentResponseUsageMetadata", metrics: dict[str, Any]
) -> None:
    """Mutate metrics in-place with token counts from a usage_metadata object."""
    if hasattr(usage_metadata, "prompt_token_count"):
        metrics["prompt_tokens"] = usage_metadata.prompt_token_count
    if hasattr(usage_metadata, "candidates_token_count"):
        metrics["completion_tokens"] = usage_metadata.candidates_token_count
    if hasattr(usage_metadata, "total_token_count"):
        metrics["tokens"] = usage_metadata.total_token_count
    if hasattr(usage_metadata, "cached_content_token_count"):
        metrics["prompt_cached_tokens"] = usage_metadata.cached_content_token_count
    if hasattr(usage_metadata, "thoughts_token_count"):
        metrics["completion_reasoning_tokens"] = usage_metadata.thoughts_token_count


def _extract_generate_content_metrics(response: "GenerateContentResponse", start: float) -> dict[str, Any]:
    """Extract metrics from a non-streaming generate_content response."""
    end_time = time.time()
    metrics = dict(
        start=start,
        end=end_time,
        duration=end_time - start,
    )

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        _extract_usage_metadata_metrics(response.usage_metadata, metrics)

    return _clean(dict(metrics))


def _extract_embed_content_output(response: "EmbedContentResponse") -> dict[str, Any]:
    embeddings = getattr(response, "embeddings", None) or []
    first_embedding = embeddings[0] if embeddings else None
    first_values = getattr(first_embedding, "values", None) or []

    return _clean(
        {
            "embedding_length": len(first_values) if first_values else None,
            "embeddings_count": len(embeddings) if embeddings else None,
        }
    )


def _extract_embed_content_metrics(response: "EmbedContentResponse", start: float) -> dict[str, Any]:
    end_time = time.time()
    metrics = dict(
        start=start,
        end=end_time,
        duration=end_time - start,
    )

    embeddings = getattr(response, "embeddings", None) or []
    token_counts = []
    for embedding in embeddings:
        statistics = getattr(embedding, "statistics", None)
        token_count = getattr(statistics, "token_count", None)
        if token_count is not None:
            token_counts.append(token_count)

    if token_counts:
        metrics["prompt_tokens"] = sum(token_counts)
        metrics["tokens"] = metrics["prompt_tokens"]

    metadata = getattr(response, "metadata", None)
    billable_character_count = getattr(metadata, "billable_character_count", None)
    if billable_character_count is not None:
        metrics["billable_characters"] = billable_character_count

    return _clean(metrics)


# ---------------------------------------------------------------------------
# Result processing helpers
# ---------------------------------------------------------------------------


def _gc_process_result(result: "GenerateContentResponse", start: float) -> tuple[Any, dict[str, Any]]:
    return result, _extract_generate_content_metrics(result, start)


def _embed_process_result(result: "EmbedContentResponse", start: float) -> tuple[Any, dict[str, Any]]:
    return _extract_embed_content_output(result), _extract_embed_content_metrics(result, start)


# ---------------------------------------------------------------------------
# Stream aggregation
# ---------------------------------------------------------------------------


def _aggregate_generate_content_chunks(
    chunks: "list[GenerateContentResponse]", start: float, first_token_time: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate streaming chunks into a single response with metrics."""
    end_time = time.time()
    metrics = dict(
        start=start,
        end=end_time,
        duration=end_time - start,
    )

    # Add time_to_first_token if available
    if first_token_time is not None:
        metrics["time_to_first_token"] = first_token_time - start

    if not chunks:
        return {}, metrics

    # Accumulate text and metadata
    text = ""
    thought_text = ""
    other_parts = []
    usage_metadata = None
    last_response = None

    for chunk in chunks:
        last_response = chunk

        # Accumulate usage metadata
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            usage_metadata = chunk.usage_metadata

        # Process candidates and their parts
        if hasattr(chunk, "candidates") and chunk.candidates:
            for candidate in chunk.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        for part in candidate.content.parts:
                            # Handle text parts
                            if hasattr(part, "text") and part.text:
                                if hasattr(part, "thought") and part.thought:
                                    thought_text += part.text
                                else:
                                    text += part.text
                            # Collect non-text parts
                            elif hasattr(part, "function_call"):
                                other_parts.append({"function_call": part.function_call})
                            elif hasattr(part, "code_execution_result"):
                                other_parts.append({"code_execution_result": part.code_execution_result})
                            elif hasattr(part, "executable_code"):
                                other_parts.append({"executable_code": part.executable_code})

    # Build aggregated response
    aggregated = {}

    # Build parts list
    parts = []
    if thought_text:
        parts.append({"text": thought_text, "thought": True})
    if text:
        parts.append({"text": text})
    parts.extend(other_parts)

    # Build candidates
    if parts and last_response and hasattr(last_response, "candidates"):
        candidates = []
        for candidate in last_response.candidates:
            candidate_dict = {"content": {"parts": parts, "role": "model"}}

            # Add metadata from last candidate
            if hasattr(candidate, "finish_reason"):
                candidate_dict["finish_reason"] = candidate.finish_reason
            if hasattr(candidate, "safety_ratings"):
                candidate_dict["safety_ratings"] = candidate.safety_ratings

            candidates.append(candidate_dict)

        aggregated["candidates"] = candidates

    # Add usage metadata
    if usage_metadata:
        aggregated["usage_metadata"] = usage_metadata
        _extract_usage_metadata_metrics(usage_metadata, metrics)

    # Add convenience text property
    if text:
        aggregated["text"] = text

    clean_metrics = _clean(dict(metrics))

    return aggregated, clean_metrics


# ---------------------------------------------------------------------------
# Traced call orchestration
# ---------------------------------------------------------------------------


def _run_traced_call(
    api_client: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    name: str,
    invoke: Callable[[], Any],
    process_result: Callable[[Any, float], tuple[Any, dict[str, Any]]],
) -> Any:
    input, clean_kwargs = _prepare_traced_call(api_client, args, kwargs)

    start = time.time()
    with start_span(name=name, type=SpanTypeAttribute.LLM, input=input, metadata=clean_kwargs) as span:
        result = invoke()
        output, metrics = process_result(result, start)
        span.log(output=output, metrics=metrics)
        return result


async def _run_async_traced_call(
    api_client: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    name: str,
    invoke: Callable[[], Awaitable[Any]],
    process_result: Callable[[Any, float], tuple[Any, dict[str, Any]]],
) -> Any:
    input, clean_kwargs = _prepare_traced_call(api_client, args, kwargs)

    start = time.time()
    with start_span(name=name, type=SpanTypeAttribute.LLM, input=input, metadata=clean_kwargs) as span:
        result = await invoke()
        output, metrics = process_result(result, start)
        span.log(output=output, metrics=metrics)
        return result


def _run_stream_traced_call(
    api_client: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    name: str,
    invoke: Callable[[], Any],
    aggregate: Callable[[list[Any], float, float | None], tuple[Any, dict[str, Any]]],
) -> Any:
    input, clean_kwargs = _prepare_traced_call(api_client, args, kwargs)

    start = time.time()
    first_token_time = None
    with start_span(name=name, type=SpanTypeAttribute.LLM, input=input, metadata=clean_kwargs) as span:
        chunks = []
        for chunk in invoke():
            if first_token_time is None:
                first_token_time = time.time()
            chunks.append(chunk)
            yield chunk

        output, metrics = aggregate(chunks, start, first_token_time)
        span.log(output=output, metrics=metrics)
        return output


def _run_async_stream_traced_call(
    api_client: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    name: str,
    invoke: Callable[[], Awaitable[Any]],
    aggregate: Callable[[list[Any], float, float | None], tuple[Any, dict[str, Any]]],
) -> Any:
    input, clean_kwargs = _prepare_traced_call(api_client, args, kwargs)

    async def stream_generator():
        start = time.time()
        first_token_time = None
        with start_span(name=name, type=SpanTypeAttribute.LLM, input=input, metadata=clean_kwargs) as span:
            chunks = []
            async for chunk in await invoke():
                if first_token_time is None:
                    first_token_time = time.time()
                chunks.append(chunk)
                yield chunk

            output, metrics = aggregate(chunks, start, first_token_time)
            span.log(output=output, metrics=metrics)

    return stream_generator()


# ---------------------------------------------------------------------------
# wrapt wrapper functions (used by patchers)
# ---------------------------------------------------------------------------


def _generate_content_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_traced_call(
        instance._api_client,
        args,
        kwargs,
        name="generate_content",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_gc_process_result,
    )


def _generate_content_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_stream_traced_call(
        instance._api_client,
        args,
        kwargs,
        name="generate_content_stream",
        invoke=lambda: wrapped(*args, **kwargs),
        aggregate=_aggregate_generate_content_chunks,
    )


def _embed_content_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_traced_call(
        instance._api_client,
        args,
        kwargs,
        name="embed_content",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_embed_process_result,
    )


async def _async_generate_content_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return await _run_async_traced_call(
        instance._api_client,
        args,
        kwargs,
        name="generate_content",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_gc_process_result,
    )


async def _async_generate_content_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_async_stream_traced_call(
        instance._api_client,
        args,
        kwargs,
        name="generate_content_stream",
        invoke=lambda: wrapped(*args, **kwargs),
        aggregate=_aggregate_generate_content_chunks,
    )


async def _async_embed_content_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return await _run_async_traced_call(
        instance._api_client,
        args,
        kwargs,
        name="embed_content",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_embed_process_result,
    )
