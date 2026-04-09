"""Google GenAI-specific span creation, metadata extraction, stream handling, and output normalization."""

import base64
import binascii
import contextvars
import dataclasses
import enum
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, get_args, get_origin

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.logger import Attachment, start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import clean_nones


if TYPE_CHECKING:
    from google.genai._interactions.types.interaction import Interaction
    from google.genai.types import (
        EmbedContentResponse,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
    )

logger = logging.getLogger(__name__)

_MEDIA_CONTENT_TYPES = {"image", "audio", "video", "document"}
_TOOL_CALL_TYPES = {
    "function_call",
    "code_execution_call",
    "url_context_call",
    "google_search_call",
    "mcp_server_tool_call",
    "file_search_call",
}
_TOOL_RESULT_TYPES = {
    "function_result",
    "code_execution_result",
    "url_context_result",
    "google_search_result",
    "mcp_server_tool_result",
    "file_search_result",
}


@dataclasses.dataclass
class _ActiveInteractionToolSpan:
    span: Any
    is_current: bool = False


_interaction_tool_spans: contextvars.ContextVar[dict[str, _ActiveInteractionToolSpan] | None] = contextvars.ContextVar(
    "braintrust_google_genai_interaction_tool_spans", default=None
)


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
        return _serialize_tools_fallback(api_client, input)


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


def _serialize_tools_fallback(api_client: Any, input: Any | None) -> Any | None:
    config = _get_attr_or_key(input, "config")
    tools = _get_attr_or_key(config, "tools")
    if not tools:
        return None

    serialized_tools = []
    function_declarations = []

    for tool in _ensure_list(tools):
        serialized_tool = _serialize_tool(api_client, tool)
        if not serialized_tool:
            continue

        declarations = _get_attr_or_key(serialized_tool, "functionDeclarations")
        if declarations:
            function_declarations.extend(_ensure_list(declarations))
        else:
            serialized_tools.append(serialized_tool)

    if function_declarations:
        serialized_tools.append({"functionDeclarations": function_declarations})

    return serialized_tools or None


def _serialize_tool(api_client: Any, tool: Any) -> Any:
    if callable(tool):
        return {"functionDeclarations": [_serialize_callable_function_declaration(api_client, tool)]}

    declarations = _get_attr_or_key(tool, "function_declarations", "functionDeclarations")
    if declarations is None and _looks_like_function_declaration(tool):
        declarations = [tool]

    serialized = _generic_serialize(tool)
    if isinstance(serialized, dict):
        if declarations is None:
            declarations = _get_attr_or_key(serialized, "function_declarations", "functionDeclarations")
        if declarations is None and _looks_like_function_declaration(serialized):
            declarations = [serialized]

        result = {
            _camelize_tool_key(key): bt_safe_deep_copy(value)
            for key, value in serialized.items()
            if key not in ("functionDeclarations", "function_declarations") and value is not None
        }
        if declarations is not None:
            result["functionDeclarations"] = [
                _serialize_function_declaration(api_client, declaration) for declaration in _ensure_list(declarations)
            ]
        if result:
            return result

        return serialized

    if _looks_like_function_declaration(tool):
        return {"functionDeclarations": [_serialize_function_declaration(api_client, tool)]}

    return bt_safe_deep_copy(tool)


def _looks_like_function_declaration(obj: Any) -> bool:
    if isinstance(obj, dict):
        return "name" in obj and any(
            key in obj for key in ("description", "parameters", "parameters_json_schema", "parametersJsonSchema")
        )

    return hasattr(obj, "name") and any(
        hasattr(obj, attr) for attr in ("description", "parameters", "parameters_json_schema", "parametersJsonSchema")
    )


def _serialize_function_declaration(api_client: Any, declaration: Any) -> Any:
    if getattr(api_client, "vertexai", False):
        return _serialize_vertex_function_declaration(declaration)

    coerced = _coerce_function_declaration(declaration)
    if coerced is not None:
        return coerced

    return _serialize_function_declaration_dict(declaration)


def _coerce_function_declaration(declaration: Any) -> Any | None:
    try:
        from google.genai import types as google_types
    except Exception:
        return None

    if getattr(getattr(declaration, "__class__", None), "__name__", None) == "FunctionDeclaration":
        return declaration

    if isinstance(declaration, dict):
        try:
            return google_types.FunctionDeclaration.model_validate(declaration)
        except Exception:
            return None

    serialized = _generic_serialize(declaration)
    if isinstance(serialized, dict):
        try:
            return google_types.FunctionDeclaration.model_validate(serialized)
        except Exception:
            return None

    return None


def _serialize_function_declaration_dict(declaration: Any) -> dict[str, Any]:
    serialized = declaration if isinstance(declaration, dict) else _generic_serialize(declaration)
    if not isinstance(serialized, dict):
        return {}

    result = {}
    for key in (
        "description",
        "name",
        "parameters",
        "parameters_json_schema",
        "response",
        "response_json_schema",
        "behavior",
    ):
        value = _get_attr_or_key(declaration, key, _camelize_tool_key(key))
        if value is None:
            value = _get_attr_or_key(serialized, key, _camelize_tool_key(key))
        if value is not None:
            result[key] = bt_safe_deep_copy(value)

    return result


def _serialize_vertex_function_declaration(declaration: Any) -> dict[str, Any]:
    serialized = _serialize_function_declaration_dict(declaration)
    return {
        key: value
        for key, value in {
            "description": serialized.get("description"),
            "name": serialized.get("name"),
            "parameters": serialized.get("parameters"),
            "parametersJsonSchema": serialized.get("parameters_json_schema"),
            "response": serialized.get("response"),
            "responseJsonSchema": serialized.get("response_json_schema"),
            "behavior": serialized.get("behavior"),
        }.items()
        if value is not None
    }


def _serialize_callable_function_declaration(api_client: Any, tool: Any) -> Any:
    try:
        from google.genai import types as google_types

        return google_types.FunctionDeclaration.from_callable(client=api_client, callable=tool)
    except Exception:
        logger.debug("Failed to serialize callable tool via public Google types", exc_info=True)
        return _serialize_callable_function_declaration_fallback(tool)


def _serialize_callable_function_declaration_fallback(tool: Any) -> dict[str, Any]:
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


def _camelize_tool_key(key: str) -> str:
    if "_" not in key:
        return key

    first, *rest = key.split("_")
    return first + "".join(part[:1].upper() + part[1:] for part in rest)


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


def _attachment_from_base64_data(data: str, mime_type: str, *, label: str) -> Attachment | None:
    raw_data = data
    if raw_data.startswith("data:"):
        _, _, encoded = raw_data.partition(",")
        raw_data = encoded

    try:
        decoded = base64.b64decode(raw_data, validate=True)
    except (ValueError, binascii.Error):
        return None

    extension = mime_type.split("/")[1] if "/" in mime_type else "bin"
    return Attachment(data=decoded, filename=f"{label}.{extension}", content_type=mime_type)


def _serialize_interaction_content_dict(value: dict[str, Any]) -> dict[str, Any]:
    serialized = {key: _serialize_interaction_value(val) for key, val in value.items() if val is not None}

    content_type = serialized.get("type")
    data = serialized.get("data")
    mime_type = serialized.get("mime_type")
    if content_type in _MEDIA_CONTENT_TYPES and isinstance(data, str) and isinstance(mime_type, str):
        attachment = _attachment_from_base64_data(data, mime_type, label=content_type)
        if attachment is not None:
            serialized["data"] = attachment

    return serialized


def _serialize_interaction_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, Attachment)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_serialize_interaction_value(item) for item in value]
    if isinstance(value, dict):
        return _serialize_interaction_content_dict(value)
    if hasattr(value, "model_dump"):
        try:
            return _serialize_interaction_value(value.model_dump(exclude_none=True))
        except TypeError:
            return _serialize_interaction_value(value.model_dump())
    return value


# ---------------------------------------------------------------------------
# Argument extraction helpers
# ---------------------------------------------------------------------------


def _omit(obj: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in keys}


def _get_args_kwargs(
    args: list[str], kwargs: dict[str, Any], keys: Iterable[str], omit_keys: Iterable[str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    return {k: args[i] if args else kwargs.get(k) for i, k in enumerate(keys)}, _omit(kwargs, omit_keys or keys)


def _prepare_traced_call(
    api_client: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    input, clean_kwargs = _get_args_kwargs(args, kwargs, ["model", "contents", "config"], ["contents", "config"])
    return _serialize_input(api_client, input), clean_kwargs


def _prepare_generate_images_traced_call(
    api_client: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    del api_client
    input, clean_kwargs = _get_args_kwargs(args, kwargs, ["model", "prompt", "config"], ["prompt", "config"])
    if input.get("config") is not None:
        input["config"] = bt_safe_deep_copy(input["config"])
    return clean_nones(input), clean_kwargs


def _prepare_interaction_create_traced_call(
    api_client: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    del api_client, args

    input_data = clean_nones(
        {
            "model": kwargs.get("model"),
            "agent": kwargs.get("agent"),
            "input": _serialize_interaction_value(kwargs.get("input")),
            "background": kwargs.get("background"),
            "generation_config": _serialize_interaction_value(kwargs.get("generation_config")),
            "previous_interaction_id": kwargs.get("previous_interaction_id"),
            "response_format": _serialize_interaction_value(kwargs.get("response_format")),
            "response_mime_type": kwargs.get("response_mime_type"),
            "response_modalities": _serialize_interaction_value(kwargs.get("response_modalities")),
            "store": kwargs.get("store"),
            "stream": kwargs.get("stream"),
            "system_instruction": kwargs.get("system_instruction"),
            "tools": _serialize_interaction_value(kwargs.get("tools")),
            "agent_config": _serialize_interaction_value(kwargs.get("agent_config")),
        }
    )
    metadata = clean_nones(
        {
            "api_version": kwargs.get("api_version"),
            "model": kwargs.get("model"),
            "agent": kwargs.get("agent"),
            "previous_interaction_id": kwargs.get("previous_interaction_id"),
        }
    )
    return input_data, metadata


def _prepare_interaction_get_traced_call(
    api_client: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    del api_client

    interaction_id = args[0] if args else kwargs.get("id")
    input_data = clean_nones(
        {
            "id": interaction_id,
            "include_input": kwargs.get("include_input"),
            "last_event_id": kwargs.get("last_event_id"),
            "stream": kwargs.get("stream"),
        }
    )
    metadata = clean_nones({"api_version": kwargs.get("api_version")})
    return input_data, metadata


def _prepare_interaction_id_traced_call(
    api_client: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    del api_client

    interaction_id = args[0] if args else kwargs.get("id")
    input_data = clean_nones({"id": interaction_id})
    metadata = clean_nones({"api_version": kwargs.get("api_version")})
    return input_data, metadata


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

    return clean_nones(dict(metrics))


def _extract_embed_content_output(response: "EmbedContentResponse") -> dict[str, Any]:
    embeddings = getattr(response, "embeddings", None) or []
    first_embedding = embeddings[0] if embeddings else None
    first_values = getattr(first_embedding, "values", None) or []

    return clean_nones(
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

    return clean_nones(metrics)


def _extract_generate_images_output(response: Any) -> dict[str, Any]:
    generated_images = getattr(response, "generated_images", None) or []
    serialized_images = []

    for i, generated_image in enumerate(generated_images):
        image = getattr(generated_image, "image", None)
        image_bytes = getattr(image, "image_bytes", None)
        mime_type = getattr(image, "mime_type", None)
        safety_attributes = getattr(generated_image, "safety_attributes", None)

        image_entry: dict[str, Any] = clean_nones(
            {
                "mime_type": mime_type,
                "gcs_uri": getattr(image, "gcs_uri", None),
                "image_size_bytes": len(image_bytes) if image_bytes is not None else None,
                "rai_filtered_reason": getattr(generated_image, "rai_filtered_reason", None),
                "enhanced_prompt": getattr(generated_image, "enhanced_prompt", None),
                "safety_categories": getattr(safety_attributes, "categories", None),
                "safety_scores": getattr(safety_attributes, "scores", None),
                "safety_content_type": getattr(safety_attributes, "content_type", None),
            }
        )

        # Convert image bytes to an Attachment so the SDK uploads them to
        # object storage and the Braintrust UI can render the image.
        if isinstance(image_bytes, bytes) and mime_type:
            extension = mime_type.split("/")[1] if "/" in mime_type else "bin"
            filename = f"generated_image_{i}.{extension}"
            attachment = Attachment(data=image_bytes, filename=filename, content_type=mime_type)
            image_entry["image_url"] = {"url": attachment}

        serialized_images.append(image_entry)

    positive_prompt_safety_attributes = getattr(response, "positive_prompt_safety_attributes", None)
    positive_prompt_summary = None
    if positive_prompt_safety_attributes is not None:
        positive_prompt_summary = clean_nones(
            {
                "categories": getattr(positive_prompt_safety_attributes, "categories", None),
                "scores": getattr(positive_prompt_safety_attributes, "scores", None),
                "content_type": getattr(positive_prompt_safety_attributes, "content_type", None),
            }
        )

    return clean_nones(
        {
            "generated_images_count": len(generated_images),
            "generated_images": serialized_images,
            "has_positive_prompt_safety_attributes": positive_prompt_safety_attributes is not None,
            "positive_prompt_safety_attributes": positive_prompt_summary,
        }
    )


def _extract_generic_timing_metrics(start: float) -> dict[str, Any]:
    end_time = time.time()
    return clean_nones(
        {
            "start": start,
            "end": end_time,
            "duration": end_time - start,
        }
    )


def _extract_interaction_usage_metrics(usage: Any, metrics: dict[str, Any]) -> None:
    if usage is None:
        return

    if hasattr(usage, "total_input_tokens") and usage.total_input_tokens is not None:
        metrics["prompt_tokens"] = usage.total_input_tokens
    if hasattr(usage, "total_output_tokens") and usage.total_output_tokens is not None:
        metrics["completion_tokens"] = usage.total_output_tokens
    if hasattr(usage, "total_tokens") and usage.total_tokens is not None:
        metrics["tokens"] = usage.total_tokens
    if hasattr(usage, "total_cached_tokens") and usage.total_cached_tokens is not None:
        metrics["prompt_cached_tokens"] = usage.total_cached_tokens
    if hasattr(usage, "total_thought_tokens") and usage.total_thought_tokens is not None:
        metrics["completion_reasoning_tokens"] = usage.total_thought_tokens
    if hasattr(usage, "total_tool_use_tokens") and usage.total_tool_use_tokens is not None:
        metrics["tool_use_tokens"] = usage.total_tool_use_tokens


def _extract_interaction_text(outputs: list[dict[str, Any]]) -> str | None:
    text_parts = []
    for item in outputs:
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
    return "".join(text_parts) or None


def _serialize_interaction_outputs(response: "Interaction") -> list[dict[str, Any]]:
    outputs = _serialize_interaction_value(getattr(response, "outputs", None))
    return outputs if isinstance(outputs, list) else ([] if outputs is None else [outputs])


def _extract_interaction_output(
    response: "Interaction", serialized_outputs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    outputs_list = serialized_outputs if serialized_outputs is not None else _serialize_interaction_outputs(response)

    return clean_nones(
        {
            "status": getattr(response, "status", None),
            "outputs": outputs_list,
            "text": _extract_interaction_text(outputs_list),
        }
    )


def _extract_interaction_metadata(response: "Interaction") -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    usage_serialized = _serialize_interaction_value(usage)
    usage_by_modality = None
    if isinstance(usage_serialized, dict):
        usage_by_modality = clean_nones(
            {
                "input_tokens_by_modality": usage_serialized.get("input_tokens_by_modality"),
                "output_tokens_by_modality": usage_serialized.get("output_tokens_by_modality"),
                "cached_tokens_by_modality": usage_serialized.get("cached_tokens_by_modality"),
                "tool_use_tokens_by_modality": usage_serialized.get("tool_use_tokens_by_modality"),
            }
        )

    return clean_nones(
        {
            "interaction_id": getattr(response, "id", None),
            "previous_interaction_id": getattr(response, "previous_interaction_id", None),
            "role": getattr(response, "role", None),
            "response_mime_type": getattr(response, "response_mime_type", None),
            "response_modalities": _serialize_interaction_value(getattr(response, "response_modalities", None)),
            "usage_by_modality": usage_by_modality,
        }
    )


def _extract_interaction_metrics(response: "Interaction", start: float) -> dict[str, Any]:
    metrics = _extract_generic_timing_metrics(start)
    _extract_interaction_usage_metrics(getattr(response, "usage", None), metrics)
    return metrics


# ---------------------------------------------------------------------------
# Result processing helpers
# ---------------------------------------------------------------------------


def _gc_process_result(result: "GenerateContentResponse", start: float) -> tuple[Any, dict[str, Any]]:
    return result, _extract_generate_content_metrics(result, start)


def _embed_process_result(result: "EmbedContentResponse", start: float) -> tuple[Any, dict[str, Any]]:
    return _extract_embed_content_output(result), _extract_embed_content_metrics(result, start)


def _generate_images_process_result(result: Any, start: float) -> tuple[Any, dict[str, Any]]:
    return _extract_generate_images_output(result), _extract_generic_timing_metrics(start)


def _tool_span_name(call_item: dict[str, Any] | None, result_item: dict[str, Any] | None) -> str:
    item = call_item or result_item or {}
    if item.get("server_name") and item.get("name"):
        return f"{item['server_name']}.{item['name']}"
    if item.get("name"):
        return str(item["name"])
    return str(item.get("type") or "interaction_tool")


def _tool_span_input(call_item: dict[str, Any] | None) -> Any:
    if not call_item:
        return None
    if call_item.get("arguments") is not None:
        return call_item["arguments"]
    return (
        clean_nones(
            {
                key: value
                for key, value in call_item.items()
                if key not in {"id", "name", "type", "signature", "server_name"}
            }
        )
        or None
    )


def _tool_span_output(result_item: dict[str, Any] | None) -> Any:
    if not result_item:
        return None
    if result_item.get("result") is not None:
        return result_item["result"]
    return (
        clean_nones(
            {
                key: value
                for key, value in result_item.items()
                if key not in {"call_id", "name", "type", "signature", "server_name", "is_error"}
            }
        )
        or None
    )


def _interaction_process_result(
    result: "Interaction", start: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    outputs_list = _serialize_interaction_outputs(result)
    return (
        _extract_interaction_output(result, outputs_list),
        _extract_interaction_metrics(result, start),
        _extract_interaction_metadata(result),
    )


def _generic_process_result(result: Any, start: float) -> tuple[Any, dict[str, Any]]:
    return _serialize_interaction_value(result), _extract_generic_timing_metrics(start)


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
            if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                gm = candidate.grounding_metadata
                candidate_dict["grounding_metadata"] = (
                    gm.model_dump(exclude_none=True) if hasattr(gm, "model_dump") else gm
                )

            candidates.append(candidate_dict)

        aggregated["candidates"] = candidates

    # Add usage metadata
    if usage_metadata:
        aggregated["usage_metadata"] = usage_metadata
        _extract_usage_metadata_metrics(usage_metadata, metrics)

    # Add convenience text property
    if text:
        aggregated["text"] = text

    clean_metrics = clean_nones(dict(metrics))

    return aggregated, clean_metrics


def _is_interaction_content_event(event: Any) -> bool:
    return getattr(event, "event_type", None) in {"content.start", "content.delta"}


def _merge_interaction_content_delta(item: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    delta_type = delta.get("type")
    if item.get("type") is None:
        if delta_type == "thought_signature":
            item["type"] = "thought"
        elif delta_type == "thought_summary":
            item["type"] = "thought"
        elif delta_type is not None:
            item["type"] = delta_type

    for key, value in delta.items():
        if key == "type" or value is None:
            continue
        if (
            key in item
            and isinstance(item[key], str)
            and isinstance(value, str)
            and key in {"text", "data", "signature"}
        ):
            item[key] += value
        else:
            item[key] = value

    return item


def _reconstruct_interaction_outputs_from_events(events: list[Any]) -> list[dict[str, Any]]:
    outputs_by_index: dict[int, dict[str, Any]] = {}

    for event in events:
        event_type = getattr(event, "event_type", None)
        index = getattr(event, "index", None)
        if not isinstance(index, int):
            continue

        if event_type == "content.start":
            outputs_by_index[index] = _serialize_interaction_value(getattr(event, "content", None)) or {}
        elif event_type == "content.delta":
            item = outputs_by_index.setdefault(index, {})
            delta = _serialize_interaction_value(getattr(event, "delta", None)) or {}
            if isinstance(delta, dict):
                outputs_by_index[index] = _merge_interaction_content_delta(item, delta)

    return [outputs_by_index[index] for index in sorted(outputs_by_index)]


def _get_active_interaction_tool_spans() -> dict[str, _ActiveInteractionToolSpan]:
    active_tool_spans = _interaction_tool_spans.get()
    if active_tool_spans is None:
        active_tool_spans = {}
        _interaction_tool_spans.set(active_tool_spans)
    return active_tool_spans


def _tool_span_metadata(call_item: dict[str, Any] | None, result_item: dict[str, Any] | None) -> dict[str, Any] | None:
    return (
        clean_nones(
            {
                "tool_type": (call_item or result_item or {}).get("type"),
                "call_id": (call_item or {}).get("id") or (result_item or {}).get("call_id"),
                "server_name": (call_item or result_item or {}).get("server_name"),
                "signature": (call_item or result_item or {}).get("signature"),
            }
        )
        or None
    )


def _log_posthoc_interaction_tool_span(call_item: dict[str, Any] | None, result_item: dict[str, Any] | None) -> None:
    with start_span(
        name=_tool_span_name(call_item, result_item),
        type=SpanTypeAttribute.TOOL,
        input=_tool_span_input(call_item),
        metadata=_tool_span_metadata(call_item, result_item),
    ) as tool_span:
        if not result_item:
            return
        if result_item.get("is_error"):
            tool_span.log(error=_tool_span_output(result_item))
        else:
            tool_span.log(output=_tool_span_output(result_item))


def _cleanup_interaction_tool_span_state(active_tool_spans: dict[str, _ActiveInteractionToolSpan]) -> None:
    if active_tool_spans:
        return
    _interaction_tool_spans.set(None)


def _close_active_interaction_tool_span(
    call_id: str, result_item: dict[str, Any] | None = None, *, end_time: float | None = None
) -> bool:
    active_tool_spans = _get_active_interaction_tool_spans()
    active_tool_span = active_tool_spans.pop(call_id, None)
    if active_tool_span is None:
        return False

    if active_tool_span.is_current:
        active_tool_span.span.unset_current()

    if result_item is not None:
        if result_item.get("is_error"):
            active_tool_span.span.log(error=_tool_span_output(result_item))
        else:
            active_tool_span.span.log(output=_tool_span_output(result_item))

    active_tool_span.span.end(end_time=end_time)
    _cleanup_interaction_tool_span_state(active_tool_spans)
    return True


def _activate_interaction_tool_span(
    call_item: dict[str, Any], *, parent_export: str, start_time: float | None = None, set_current: bool = False
) -> None:
    # Keep the tool span open across local tool execution so any nested spans
    # started by user code naturally inherit from it until the corresponding
    # function_result is submitted on a follow-up interactions.create call.
    call_id = call_item.get("id")
    if not isinstance(call_id, str):
        _log_posthoc_interaction_tool_span(call_item, None)
        return

    active_tool_spans = _get_active_interaction_tool_spans()
    if call_id in active_tool_spans:
        return

    tool_span = start_span(
        name=_tool_span_name(call_item, None),
        type=SpanTypeAttribute.TOOL,
        input=_tool_span_input(call_item),
        metadata=_tool_span_metadata(call_item, None),
        parent=parent_export,
        start_time=start_time,
        set_current=True,
    )
    active_tool_spans[call_id] = _ActiveInteractionToolSpan(span=tool_span, is_current=False)

    if set_current:
        tool_span.set_current()
        active_tool_spans[call_id].is_current = True


def _serialize_interaction_items(value: Any) -> list[dict[str, Any]]:
    serialized = _serialize_interaction_value(value)
    if serialized is None:
        return []
    items = serialized if isinstance(serialized, list) else [serialized]
    return [item for item in items if isinstance(item, dict)]


def _close_interaction_tool_spans_from_input(input_value: Any) -> None:
    # Tool spans should end when the client hands the tool result back to the
    # interactions API, before the follow-up LLM/TASK span begins.
    end_time = time.time()
    for item in _serialize_interaction_items(input_value):
        if item.get("type") not in _TOOL_RESULT_TYPES:
            continue
        call_id = item.get("call_id")
        if isinstance(call_id, str):
            _close_active_interaction_tool_span(call_id, item, end_time=end_time)


def _finalize_interaction_tool_spans(
    output: Any, metrics: dict[str, Any], metadata: dict[str, Any] | None, parent_export: str
) -> None:
    del metadata

    if not isinstance(output, dict):
        return

    outputs = output.get("outputs")
    if not isinstance(outputs, list):
        return

    active_tool_spans = _get_active_interaction_tool_spans()
    calls_by_id: dict[str, dict[str, Any]] = {}
    pending_results: dict[str, list[dict[str, Any]]] = {}
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    emitted_call_ids: set[str] = set()

    for item in outputs:
        item_type = item.get("type")
        if item_type in _TOOL_CALL_TYPES:
            call_id = item.get("id")
            if isinstance(call_id, str):
                calls_by_id[call_id] = item
                for pending in pending_results.pop(call_id, []):
                    pairs.append((item, pending))
                    emitted_call_ids.add(call_id)
            else:
                pairs.append((item, None))
        elif item_type in _TOOL_RESULT_TYPES:
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id in active_tool_spans:
                _close_active_interaction_tool_span(call_id, item, end_time=time.time())
                emitted_call_ids.add(call_id)
            elif isinstance(call_id, str) and call_id in calls_by_id:
                pairs.append((calls_by_id[call_id], item))
                emitted_call_ids.add(call_id)
            elif isinstance(call_id, str):
                pending_results.setdefault(call_id, []).append(item)
            else:
                pairs.append((None, item))

    for call_id, result_items in pending_results.items():
        call_item = calls_by_id.get(call_id)
        for result_item in result_items:
            pairs.append((call_item, result_item))
            if call_item is not None:
                emitted_call_ids.add(call_id)

    unpaired_call_items: list[dict[str, Any]] = []
    for call_id, call_item in calls_by_id.items():
        if call_id not in emitted_call_ids:
            unpaired_call_items.append(call_item)

    for call_item, result_item in pairs:
        _log_posthoc_interaction_tool_span(call_item, result_item)

    activatable_call_items = [
        call_item
        for call_item in unpaired_call_items
        if isinstance(call_item.get("id"), str) and call_item.get("id") not in active_tool_spans
    ]
    claim_current = len(activatable_call_items) == 1 and not any(
        active_tool_span.is_current for active_tool_span in active_tool_spans.values()
    )

    for call_item in unpaired_call_items:
        call_id = call_item.get("id")
        if not isinstance(call_id, str):
            _log_posthoc_interaction_tool_span(call_item, None)
            continue
        if call_id in active_tool_spans:
            continue
        _activate_interaction_tool_span(
            call_item,
            parent_export=parent_export,
            start_time=metrics.get("end"),
            set_current=claim_current and call_item is activatable_call_items[0],
        )


def _aggregate_interaction_events(
    events: list[Any], start: float, first_token_time: float | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = _extract_generic_timing_metrics(start)
    if first_token_time is not None:
        metrics["time_to_first_token"] = first_token_time - start

    metadata = clean_nones(
        {"stream_event_types": [et for event in events if (et := getattr(event, "event_type", None))]}
    )
    reconstructed_outputs = _reconstruct_interaction_outputs_from_events(events)

    final_interaction = next(
        (
            event.interaction
            for event in reversed(events)
            if hasattr(event, "interaction") and getattr(event, "interaction", None) is not None
        ),
        None,
    )
    if final_interaction is None:
        if reconstructed_outputs:
            return (
                {"outputs": reconstructed_outputs, "text": _extract_interaction_text(reconstructed_outputs)},
                clean_nones(metrics),
                metadata,
            )
        error_event = next(
            (
                event
                for event in reversed(events)
                if getattr(event, "event_type", None) == "error" and getattr(event, "error", None) is not None
            ),
            None,
        )
        if error_event is not None:
            metadata["stream_error"] = _serialize_interaction_value(error_event.error)
        return {"events": _serialize_interaction_value(events)}, clean_nones(metrics), metadata

    final_outputs_list = _serialize_interaction_outputs(final_interaction)

    _extract_interaction_usage_metrics(getattr(final_interaction, "usage", None), metrics)
    metadata.update(_extract_interaction_metadata(final_interaction))

    output = _extract_interaction_output(final_interaction, final_outputs_list)
    if reconstructed_outputs and not output.get("outputs"):
        output["outputs"] = reconstructed_outputs
        output["text"] = _extract_interaction_text(reconstructed_outputs)

    return output, clean_nones(metrics), clean_nones(metadata)


# ---------------------------------------------------------------------------
# Traced call orchestration
# ---------------------------------------------------------------------------


def _normalize_logged_result(result: Any) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    if isinstance(result, tuple) and len(result) == 3:
        output, metrics, metadata = result
        return output, metrics, metadata
    if isinstance(result, tuple) and len(result) == 2:
        output, metrics = result
        return output, metrics, None
    raise ValueError("Expected process_result/aggregate to return a 2-tuple or 3-tuple")


def _run_traced_call(
    api_client: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    name: str,
    invoke: Callable[[], Any],
    process_result: Callable[[Any, float], tuple[Any, dict[str, Any]] | tuple[Any, dict[str, Any], dict[str, Any]]],
    prepare_call: Callable[
        [Any, list[Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]
    ] = _prepare_traced_call,
    span_type: SpanTypeAttribute = SpanTypeAttribute.LLM,
    before_invoke: Callable[[], None] | None = None,
    finalize_logged_output: Callable[[Any, dict[str, Any], dict[str, Any] | None, str], None] | None = None,
) -> Any:
    input, clean_kwargs = prepare_call(api_client, args, kwargs)

    if before_invoke is not None:
        before_invoke()

    start = time.time()
    parent_export = None
    output = None
    metrics = None
    metadata = None
    with start_span(name=name, type=span_type, input=input, metadata=clean_kwargs or None) as span:
        result = invoke()
        output, metrics, metadata = _normalize_logged_result(process_result(result, start))
        span.log(output=output, metrics=metrics, metadata=metadata)
        parent_export = span.export()

    if finalize_logged_output is not None and parent_export is not None and metrics is not None:
        finalize_logged_output(output, metrics, metadata, parent_export)

    return result


async def _run_async_traced_call(
    api_client: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    name: str,
    invoke: Callable[[], Awaitable[Any]],
    process_result: Callable[[Any, float], tuple[Any, dict[str, Any]] | tuple[Any, dict[str, Any], dict[str, Any]]],
    prepare_call: Callable[
        [Any, list[Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]
    ] = _prepare_traced_call,
    span_type: SpanTypeAttribute = SpanTypeAttribute.LLM,
    before_invoke: Callable[[], None] | None = None,
    finalize_logged_output: Callable[[Any, dict[str, Any], dict[str, Any] | None, str], None] | None = None,
) -> Any:
    input, clean_kwargs = prepare_call(api_client, args, kwargs)

    if before_invoke is not None:
        before_invoke()

    start = time.time()
    parent_export = None
    output = None
    metrics = None
    metadata = None
    with start_span(name=name, type=span_type, input=input, metadata=clean_kwargs or None) as span:
        result = await invoke()
        output, metrics, metadata = _normalize_logged_result(process_result(result, start))
        span.log(output=output, metrics=metrics, metadata=metadata)
        parent_export = span.export()

    if finalize_logged_output is not None and parent_export is not None and metrics is not None:
        finalize_logged_output(output, metrics, metadata, parent_export)

    return result


def _run_stream_traced_call(
    api_client: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    name: str,
    invoke: Callable[[], Any],
    aggregate: Callable[
        [list[Any], float, float | None], tuple[Any, dict[str, Any]] | tuple[Any, dict[str, Any], dict[str, Any]]
    ],
    span_type: SpanTypeAttribute = SpanTypeAttribute.LLM,
    first_token_predicate: Callable[[Any], bool] | None = None,
    prepare_call: Callable[
        [Any, list[Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]
    ] = _prepare_traced_call,
    before_invoke: Callable[[], None] | None = None,
    finalize_logged_output: Callable[[Any, dict[str, Any], dict[str, Any] | None, str], None] | None = None,
) -> Any:
    input, clean_kwargs = prepare_call(api_client, args, kwargs)

    if before_invoke is not None:
        before_invoke()

    start = time.time()
    first_token_time = None
    output = None
    metrics = None
    metadata = None
    parent_export = None
    with start_span(name=name, type=span_type, input=input, metadata=clean_kwargs or None) as span:
        chunks = []
        for chunk in invoke():
            if first_token_time is None and (
                first_token_predicate(chunk) if first_token_predicate is not None else True
            ):
                first_token_time = time.time()
            chunks.append(chunk)
            yield chunk

        output, metrics, metadata = _normalize_logged_result(aggregate(chunks, start, first_token_time))
        span.log(output=output, metrics=metrics, metadata=metadata)
        parent_export = span.export()

    if finalize_logged_output is not None and parent_export is not None and metrics is not None:
        finalize_logged_output(output, metrics, metadata, parent_export)

    return output


def _run_async_stream_traced_call(
    api_client: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    name: str,
    invoke: Callable[[], Awaitable[Any]],
    aggregate: Callable[
        [list[Any], float, float | None], tuple[Any, dict[str, Any]] | tuple[Any, dict[str, Any], dict[str, Any]]
    ],
    span_type: SpanTypeAttribute = SpanTypeAttribute.LLM,
    first_token_predicate: Callable[[Any], bool] | None = None,
    prepare_call: Callable[
        [Any, list[Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]
    ] = _prepare_traced_call,
    before_invoke: Callable[[], None] | None = None,
    finalize_logged_output: Callable[[Any, dict[str, Any], dict[str, Any] | None, str], None] | None = None,
) -> Any:
    input, clean_kwargs = prepare_call(api_client, args, kwargs)

    async def stream_generator():
        if before_invoke is not None:
            before_invoke()

        start = time.time()
        first_token_time = None
        output = None
        metrics = None
        metadata = None
        parent_export = None
        with start_span(name=name, type=span_type, input=input, metadata=clean_kwargs or None) as span:
            chunks = []
            async for chunk in await invoke():
                if first_token_time is None and (
                    first_token_predicate(chunk) if first_token_predicate is not None else True
                ):
                    first_token_time = time.time()
                chunks.append(chunk)
                yield chunk

            output, metrics, metadata = _normalize_logged_result(aggregate(chunks, start, first_token_time))
            span.log(output=output, metrics=metrics, metadata=metadata)
            parent_export = span.export()

        if finalize_logged_output is not None and parent_export is not None and metrics is not None:
            finalize_logged_output(output, metrics, metadata, parent_export)

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


def _generate_images_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_traced_call(
        instance._api_client,
        args,
        kwargs,
        name="generate_images",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_generate_images_process_result,
        prepare_call=_prepare_generate_images_traced_call,
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


async def _async_generate_images_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return await _run_async_traced_call(
        instance._api_client,
        args,
        kwargs,
        name="generate_images",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_generate_images_process_result,
        prepare_call=_prepare_generate_images_traced_call,
    )


def _interactions_create_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    before_invoke = lambda: _close_interaction_tool_spans_from_input(kwargs.get("input"))

    if kwargs.get("stream"):
        return _run_stream_traced_call(
            getattr(instance, "_client", None),
            args,
            kwargs,
            name="interactions.create",
            invoke=lambda: wrapped(*args, **kwargs),
            aggregate=_aggregate_interaction_events,
            first_token_predicate=_is_interaction_content_event,
            prepare_call=_prepare_interaction_create_traced_call,
            span_type=SpanTypeAttribute.LLM,
            before_invoke=before_invoke,
            finalize_logged_output=_finalize_interaction_tool_spans,
        )

    return _run_traced_call(
        getattr(instance, "_client", None),
        args,
        kwargs,
        name="interactions.create",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_interaction_process_result,
        prepare_call=_prepare_interaction_create_traced_call,
        span_type=SpanTypeAttribute.LLM,
        before_invoke=before_invoke,
        finalize_logged_output=_finalize_interaction_tool_spans,
    )


async def _async_interactions_create_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    before_invoke = lambda: _close_interaction_tool_spans_from_input(kwargs.get("input"))

    if kwargs.get("stream"):
        return _run_async_stream_traced_call(
            getattr(instance, "_client", None),
            args,
            kwargs,
            name="interactions.create",
            invoke=lambda: wrapped(*args, **kwargs),
            aggregate=_aggregate_interaction_events,
            first_token_predicate=_is_interaction_content_event,
            prepare_call=_prepare_interaction_create_traced_call,
            span_type=SpanTypeAttribute.LLM,
            before_invoke=before_invoke,
            finalize_logged_output=_finalize_interaction_tool_spans,
        )

    return await _run_async_traced_call(
        getattr(instance, "_client", None),
        args,
        kwargs,
        name="interactions.create",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_interaction_process_result,
        prepare_call=_prepare_interaction_create_traced_call,
        span_type=SpanTypeAttribute.LLM,
        before_invoke=before_invoke,
        finalize_logged_output=_finalize_interaction_tool_spans,
    )


def _interactions_get_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    if kwargs.get("stream"):
        return _run_stream_traced_call(
            getattr(instance, "_client", None),
            args,
            kwargs,
            name="interactions.get",
            invoke=lambda: wrapped(*args, **kwargs),
            aggregate=_aggregate_interaction_events,
            first_token_predicate=_is_interaction_content_event,
            prepare_call=_prepare_interaction_get_traced_call,
            span_type=SpanTypeAttribute.TASK,
            finalize_logged_output=_finalize_interaction_tool_spans,
        )

    return _run_traced_call(
        getattr(instance, "_client", None),
        args,
        kwargs,
        name="interactions.get",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_interaction_process_result,
        prepare_call=_prepare_interaction_get_traced_call,
        span_type=SpanTypeAttribute.TASK,
        finalize_logged_output=_finalize_interaction_tool_spans,
    )


async def _async_interactions_get_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    if kwargs.get("stream"):
        return _run_async_stream_traced_call(
            getattr(instance, "_client", None),
            args,
            kwargs,
            name="interactions.get",
            invoke=lambda: wrapped(*args, **kwargs),
            aggregate=_aggregate_interaction_events,
            first_token_predicate=_is_interaction_content_event,
            prepare_call=_prepare_interaction_get_traced_call,
            span_type=SpanTypeAttribute.TASK,
            finalize_logged_output=_finalize_interaction_tool_spans,
        )

    return await _run_async_traced_call(
        getattr(instance, "_client", None),
        args,
        kwargs,
        name="interactions.get",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_interaction_process_result,
        prepare_call=_prepare_interaction_get_traced_call,
        span_type=SpanTypeAttribute.TASK,
        finalize_logged_output=_finalize_interaction_tool_spans,
    )


def _interactions_cancel_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_traced_call(
        getattr(instance, "_client", None),
        args,
        kwargs,
        name="interactions.cancel",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_interaction_process_result,
        prepare_call=_prepare_interaction_id_traced_call,
        span_type=SpanTypeAttribute.TASK,
    )


async def _async_interactions_cancel_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return await _run_async_traced_call(
        getattr(instance, "_client", None),
        args,
        kwargs,
        name="interactions.cancel",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_interaction_process_result,
        prepare_call=_prepare_interaction_id_traced_call,
        span_type=SpanTypeAttribute.TASK,
    )


def _interactions_delete_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_traced_call(
        getattr(instance, "_client", None),
        args,
        kwargs,
        name="interactions.delete",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_generic_process_result,
        prepare_call=_prepare_interaction_id_traced_call,
        span_type=SpanTypeAttribute.TASK,
    )


async def _async_interactions_delete_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return await _run_async_traced_call(
        getattr(instance, "_client", None),
        args,
        kwargs,
        name="interactions.delete",
        invoke=lambda: wrapped(*args, **kwargs),
        process_result=_generic_process_result,
        prepare_call=_prepare_interaction_id_traced_call,
        span_type=SpanTypeAttribute.TASK,
    )
