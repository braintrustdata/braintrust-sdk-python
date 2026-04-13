"""Mistral-specific tracing helpers."""

import logging
import re
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.integrations.utils import (
    _camel_to_snake,
    _is_supported_metric_value,
    _log_and_end_span,
    _log_error_and_end_span,
    _materialize_attachment,
    _merge_timing_and_usage_metrics,
)
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute


logger = logging.getLogger(__name__)

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_TOKEN_NAME_MAP = {
    "total_tokens": "tokens",
}
_CHAT_METADATA_KEYS = (
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "random_seed",
    "response_format",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "prediction",
    "parallel_tool_calls",
    "reasoning_effort",
    "prompt_mode",
    "guardrails",
    "safe_prompt",
)
_AGENTS_METADATA_KEYS = (
    "agent_id",
    "max_tokens",
    "stop",
    "random_seed",
    "response_format",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "prediction",
    "parallel_tool_calls",
    "prompt_mode",
)
_EMBEDDINGS_METADATA_KEYS = (
    "model",
    "output_dimension",
    "output_dtype",
    "encoding_format",
)
_FIM_METADATA_KEYS = (
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "random_seed",
    "min_tokens",
)
_OCR_METADATA_KEYS = (
    "model",
    "id",
    "pages",
    "include_image_base64",
    "image_limit",
    "image_min_size",
    "bbox_annotation_format",
    "document_annotation_format",
    "document_annotation_prompt",
    "table_format",
    "extract_header",
    "extract_footer",
)


def _is_unset(value: Any) -> bool:
    return value.__class__.__name__ == "Unset"


def _normalize_base64_payload(value: str) -> str | None:
    normalized = value.strip().replace("\n", "")
    if len(normalized) >= 64 and len(normalized) % 4 == 0 and _BASE64_RE.fullmatch(normalized) is not None:
        return normalized
    return None


def _convert_input_audio_to_attachment(value: str) -> Any:
    normalized = _normalize_base64_payload(value)
    if normalized is None:
        return value

    return (
        resolved.attachment
        if (
            resolved := _materialize_attachment(
                normalized,
                mime_type="application/octet-stream",
                filename="input_audio.bin",
            )
        )
        is not None
        else value
    )


def _normalize_special_payloads(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    item_type = value.get("type")
    if item_type == "image_url":
        image_url = value.get("image_url")
        if isinstance(image_url, str):
            resolved = _materialize_attachment(image_url)
            return {
                **value,
                "image_url": {
                    "url": resolved.attachment if resolved is not None else image_url,
                },
            }
        if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            resolved = _materialize_attachment(image_url["url"])
            return {
                **value,
                "image_url": {
                    **image_url,
                    "url": resolved.attachment if resolved is not None else image_url["url"],
                },
            }

    if item_type == "document_url" and isinstance(value.get("document_url"), str):
        resolved = _materialize_attachment(
            value["document_url"],
            filename=value.get("document_name"),
            prefix="document",
        )
        if resolved is not None:
            return {
                "type": "file",
                "file": {
                    "file_data": resolved.attachment,
                    "filename": resolved.filename,
                },
            }

    if item_type == "input_audio" and isinstance(value.get("input_audio"), str):
        return {
            **value,
            "input_audio": _convert_input_audio_to_attachment(value["input_audio"]),
        }

    return value


def sanitize_mistral_logged_value(value: Any) -> Any:
    if _is_unset(value):
        return None

    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json", by_alias=True)
        except TypeError:
            value = value.model_dump()

    safe = bt_safe_deep_copy(value)
    safe = _normalize_special_payloads(safe)

    if callable(safe):
        return "[Function]"
    if isinstance(safe, list):
        return [sanitize_mistral_logged_value(item) for item in safe]
    if isinstance(safe, tuple):
        return [sanitize_mistral_logged_value(item) for item in safe]
    if isinstance(safe, dict):
        sanitized = {}
        for key, entry in safe.items():
            if _is_unset(entry):
                continue
            sanitized[key] = sanitize_mistral_logged_value(entry)
        return sanitized
    return safe


def _build_request_metadata(
    kwargs: dict[str, Any], keys: tuple[str, ...], *, stream: bool | None = None
) -> dict[str, Any]:
    metadata = {"provider": "mistral"}

    for key in keys:
        value = kwargs.get(key)
        if value is None or _is_unset(value):
            continue
        metadata[key] = sanitize_mistral_logged_value(value)

    request_metadata = kwargs.get("metadata")
    if request_metadata is not None and not _is_unset(request_metadata):
        metadata["request_metadata"] = sanitize_mistral_logged_value(request_metadata)

    if stream is not None:
        metadata["stream"] = stream

    return metadata


def _build_chat_metadata(kwargs: dict[str, Any], *, stream: bool | None = None) -> dict[str, Any]:
    return _build_request_metadata(kwargs, _CHAT_METADATA_KEYS, stream=stream)


def _build_agents_metadata(kwargs: dict[str, Any], *, stream: bool | None = None) -> dict[str, Any]:
    return _build_request_metadata(kwargs, _AGENTS_METADATA_KEYS, stream=stream)


def _build_embeddings_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _build_request_metadata(kwargs, _EMBEDDINGS_METADATA_KEYS)


def _build_fim_metadata(kwargs: dict[str, Any], *, stream: bool | None = None) -> dict[str, Any]:
    return _build_request_metadata(kwargs, _FIM_METADATA_KEYS, stream=stream)


def _document_type(document: Any) -> str | None:
    if _is_unset(document):
        return None

    document_type = getattr(document, "type", None)
    if document_type is None and isinstance(document, dict):
        document_type = document.get("type")

    if isinstance(document_type, str) and document_type:
        return document_type

    return None


def _build_ocr_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    metadata = _build_request_metadata(kwargs, _OCR_METADATA_KEYS)
    document_type = _document_type(kwargs.get("document"))
    if document_type is not None:
        metadata["document_type"] = document_type
    return metadata


def _fim_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    span_input = {"prompt": kwargs.get("prompt")}
    suffix = kwargs.get("suffix")
    if suffix is not None and not _is_unset(suffix):
        span_input["suffix"] = suffix
    return span_input


def _ocr_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"document": kwargs.get("document")}


def _start_span(name: str, span_input: Any, metadata: dict[str, Any]):
    return start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=sanitize_mistral_logged_value(span_input),
        metadata=metadata,
    )


def _parse_usage_metrics(usage: Any) -> dict[str, float]:
    usage_data = sanitize_mistral_logged_value(usage)
    if not isinstance(usage_data, dict):
        return {}

    metrics = {}
    for key, value in usage_data.items():
        if not _is_supported_metric_value(value):
            continue
        metrics[_TOKEN_NAME_MAP.get(key, _camel_to_snake(key))] = float(value)

    if "tokens" not in metrics and "prompt_tokens" in metrics and "completion_tokens" in metrics:
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]

    return metrics


def _merge_metrics(start_time: float, usage: Any, first_token_time: float | None = None) -> dict[str, Any]:
    return _merge_timing_and_usage_metrics(
        start_time,
        usage,
        _parse_usage_metrics,
        first_token_time,
    )


def _parse_ocr_usage_metrics(usage: Any) -> dict[str, float]:
    usage_data = sanitize_mistral_logged_value(usage)
    if not isinstance(usage_data, dict):
        return {}

    return {
        _camel_to_snake(key): float(value) for key, value in usage_data.items() if _is_supported_metric_value(value)
    }


def _merge_ocr_metrics(start_time: float, usage: Any) -> dict[str, Any]:
    return _merge_timing_and_usage_metrics(start_time, usage, _parse_ocr_usage_metrics)


def _response_to_metadata(response: Any) -> dict[str, Any]:
    data = sanitize_mistral_logged_value(response)
    if not isinstance(data, dict):
        return {}

    metadata = {}
    for key in ("id", "model", "object", "created"):
        value = data.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _completion_response_to_output(response: Any) -> Any:
    data = sanitize_mistral_logged_value(response)
    if isinstance(data, dict):
        return data.get("choices")
    return None


def _embeddings_output(response: Any) -> dict[str, Any]:
    items = getattr(response, "data", None) or []
    first = items[0] if items else None
    embedding = getattr(first, "embedding", None) if first is not None else None

    output = {
        "embeddings_count": len(items),
        "embedding_length": len(embedding) if isinstance(embedding, list) else None,
    }
    if first is not None and getattr(first, "index", None) is not None:
        output["first_index"] = first.index
    return output


def _ocr_output(response: Any) -> dict[str, Any]:
    data = sanitize_mistral_logged_value(response)
    if not isinstance(data, dict):
        return {"pages": []}

    output = {
        "pages": data.get("pages") or [],
    }
    document_annotation = data.get("document_annotation")
    if document_annotation is not None:
        output["document_annotation"] = document_annotation
    return output


def _ocr_response_to_metadata(response: Any) -> dict[str, Any]:
    data = sanitize_mistral_logged_value(response)
    if not isinstance(data, dict):
        return {}

    pages = data.get("pages")
    return {
        "page_count": len(pages) if isinstance(pages, list) else 0,
    }


def _call_with_error_logging(span: Any, wrapped: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        return wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise


async def _call_async_with_error_logging(
    span: Any, wrapped: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    try:
        return await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise


def _append_delta_content(message: dict[str, Any], delta_content: Any) -> None:
    if delta_content is None:
        return

    content = sanitize_mistral_logged_value(delta_content)
    existing = message.get("content")

    if isinstance(content, str):
        if isinstance(existing, str):
            message["content"] = existing + content
        elif isinstance(existing, list):
            existing.append({"type": "text", "text": content})
        elif existing is None:
            message["content"] = content
        else:
            message["content"] = sanitize_mistral_logged_value(existing)
        return

    if isinstance(content, list):
        if isinstance(existing, list):
            existing.extend(content)
        elif isinstance(existing, str) and existing:
            message["content"] = [{"type": "text", "text": existing}, *content]
        else:
            message["content"] = content


def _merge_tool_calls(message: dict[str, Any], tool_calls: Any) -> None:
    if not isinstance(tool_calls, list):
        return

    accumulated = message.setdefault("tool_calls", [])
    for tool_call in tool_calls:
        call = sanitize_mistral_logged_value(tool_call)
        if not isinstance(call, dict):
            continue

        index = call.get("index")
        if not isinstance(index, int) or index < 0:
            index = len(accumulated)

        while len(accumulated) <= index:
            accumulated.append({"id": None, "type": None, "function": {"name": "", "arguments": ""}})

        target = accumulated[index]
        if call.get("id") not in (None, "null"):
            target["id"] = call["id"]
        if call.get("type") is not None:
            target["type"] = call["type"]

        function = call.get("function")
        if not isinstance(function, dict):
            continue

        target_function = target.setdefault("function", {"name": "", "arguments": ""})
        name = function.get("name")
        if isinstance(name, str) and name:
            target_function["name"] = f"{target_function.get('name', '')}{name}"

        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            existing_arguments = target_function.get("arguments", "")
            if isinstance(existing_arguments, str):
                target_function["arguments"] = f"{existing_arguments}{arguments}"
            else:
                target_function["arguments"] = arguments
        elif isinstance(arguments, dict):
            target_function["arguments"] = {
                **(target_function.get("arguments") if isinstance(target_function.get("arguments"), dict) else {}),
                **arguments,
            }


def _chunk_has_output(item: Any) -> bool:
    data = getattr(item, "data", item)
    choices = getattr(data, "choices", None) or []
    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        tool_calls = getattr(delta, "tool_calls", None)
        if isinstance(content, str) and content:
            return True
        if isinstance(content, list) and content:
            return True
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def _aggregate_completion_events(items: list[Any]) -> dict[str, Any]:
    response_id = None
    model = None
    object_type = None
    created = None
    usage = None
    choices: dict[int, dict[str, Any]] = {}

    for item in items:
        data = getattr(item, "data", item)
        response_id = response_id or getattr(data, "id", None)
        model = model or getattr(data, "model", None)
        object_type = object_type or getattr(data, "object", None)
        created = created or getattr(data, "created", None)
        usage = getattr(data, "usage", None) or usage

        for choice in getattr(data, "choices", None) or []:
            index = getattr(choice, "index", 0)
            if not isinstance(index, int):
                index = 0
            accumulated = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )
            message = accumulated["message"]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            role = getattr(delta, "role", None)
            if isinstance(role, str) and role:
                message["role"] = role

            _append_delta_content(message, getattr(delta, "content", None))
            _merge_tool_calls(message, getattr(delta, "tool_calls", None))

            finish_reason = getattr(choice, "finish_reason", None)
            if isinstance(finish_reason, str) and finish_reason:
                accumulated["finish_reason"] = finish_reason

    result: dict[str, Any] = {
        "choices": [choices[idx] for idx in sorted(choices)],
    }
    if response_id is not None:
        result["id"] = response_id
    if model is not None:
        result["model"] = model
    if object_type is not None:
        result["object"] = object_type
    if created is not None:
        result["created"] = created
    if usage is not None:
        result["usage"] = sanitize_mistral_logged_value(usage)
    return result


def _finalize_completion_response(span: Any, request_metadata: dict[str, Any], response: Any, start_time: float):
    response_metadata = _response_to_metadata(response)
    _log_and_end_span(
        span,
        output=_completion_response_to_output(response),
        metrics=_merge_metrics(start_time, getattr(response, "usage", None)),
        metadata={**request_metadata, **response_metadata},
    )


def _finalize_embeddings_response(span: Any, request_metadata: dict[str, Any], response: Any, start_time: float):
    response_metadata = _response_to_metadata(response)
    _log_and_end_span(
        span,
        output=_embeddings_output(response),
        metrics=_merge_metrics(start_time, getattr(response, "usage", None)),
        metadata={**request_metadata, **response_metadata},
    )


def _finalize_ocr_response(span: Any, request_metadata: dict[str, Any], response: Any, start_time: float):
    response_metadata = _ocr_response_to_metadata(response)
    _log_and_end_span(
        span,
        output=_ocr_output(response),
        metrics=_merge_ocr_metrics(start_time, getattr(response, "usage_info", None)),
        metadata={**request_metadata, **response_metadata},
    )


class _TracedMistralSyncStream:
    def __init__(self, stream: Any, span: Any, metadata: dict[str, Any], start_time: float):
        self._stream = stream
        self._span = span
        self._metadata = metadata
        self._start_time = start_time
        self._first_token_time = None
        self._items = []
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            item = next(self._stream)
        except StopIteration:
            self._finalize()
            raise
        except Exception as error:
            self._finalize(error=error)
            raise

        if self._first_token_time is None and _chunk_has_output(item):
            self._first_token_time = time.time()
        self._items.append(item)
        return item

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if hasattr(self._stream, "__exit__"):
                return self._stream.__exit__(exc_type, exc_value, traceback)
            return False
        finally:
            self._finalize(error=exc_value)

    def _finalize(self, *, error: Exception | None = None):
        if self._closed:
            return
        self._closed = True

        if error is not None:
            _log_error_and_end_span(self._span, error)
            return

        response = _aggregate_completion_events(self._items)
        _log_and_end_span(
            self._span,
            output=response.get("choices"),
            metrics=_merge_metrics(self._start_time, response.get("usage"), self._first_token_time),
            metadata={**self._metadata, **_response_to_metadata(response)},
        )


class _TracedMistralAsyncStream:
    def __init__(self, stream: Any, span: Any, metadata: dict[str, Any], start_time: float):
        self._stream = stream
        self._span = span
        self._metadata = metadata
        self._start_time = start_time
        self._first_token_time = None
        self._items = []
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            item = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as error:
            self._finalize(error=error)
            raise

        if self._first_token_time is None and _chunk_has_output(item):
            self._first_token_time = time.time()
        self._items.append(item)
        return item

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        try:
            if hasattr(self._stream, "__aexit__"):
                return await self._stream.__aexit__(exc_type, exc_value, traceback)
            return False
        finally:
            self._finalize(error=exc_value)

    def _finalize(self, *, error: Exception | None = None):
        if self._closed:
            return
        self._closed = True

        if error is not None:
            _log_error_and_end_span(self._span, error)
            return

        response = _aggregate_completion_events(self._items)
        _log_and_end_span(
            self._span,
            output=response.get("choices"),
            metrics=_merge_metrics(self._start_time, response.get("usage"), self._first_token_time),
            metadata={**self._metadata, **_response_to_metadata(response)},
        )


def _chat_complete_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_chat_metadata(kwargs, stream=bool(kwargs.get("stream")))
    span = _start_span("mistral.chat.complete", kwargs.get("messages"), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)

    if kwargs.get("stream"):
        return _TracedMistralSyncStream(result, span, request_metadata, start_time)

    _finalize_completion_response(span, request_metadata, result, start_time)
    return result


async def _chat_complete_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_chat_metadata(kwargs, stream=bool(kwargs.get("stream")))
    span = _start_span("mistral.chat.complete", kwargs.get("messages"), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)

    if kwargs.get("stream"):
        return _TracedMistralAsyncStream(result, span, request_metadata, start_time)

    _finalize_completion_response(span, request_metadata, result, start_time)
    return result


def _chat_stream_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_chat_metadata(kwargs, stream=True)
    span = _start_span("mistral.chat.stream", kwargs.get("messages"), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)

    return _TracedMistralSyncStream(result, span, request_metadata, start_time)


async def _chat_stream_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_chat_metadata(kwargs, stream=True)
    span = _start_span("mistral.chat.stream", kwargs.get("messages"), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)

    return _TracedMistralAsyncStream(result, span, request_metadata, start_time)


def _agents_complete_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_agents_metadata(kwargs, stream=bool(kwargs.get("stream")))
    span = _start_span("mistral.agents.complete", kwargs.get("messages"), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)

    if kwargs.get("stream"):
        return _TracedMistralSyncStream(result, span, request_metadata, start_time)

    _finalize_completion_response(span, request_metadata, result, start_time)
    return result


async def _agents_complete_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_agents_metadata(kwargs, stream=bool(kwargs.get("stream")))
    span = _start_span("mistral.agents.complete", kwargs.get("messages"), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)

    if kwargs.get("stream"):
        return _TracedMistralAsyncStream(result, span, request_metadata, start_time)

    _finalize_completion_response(span, request_metadata, result, start_time)
    return result


def _agents_stream_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_agents_metadata(kwargs, stream=True)
    span = _start_span("mistral.agents.stream", kwargs.get("messages"), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)

    return _TracedMistralSyncStream(result, span, request_metadata, start_time)


async def _agents_stream_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_agents_metadata(kwargs, stream=True)
    span = _start_span("mistral.agents.stream", kwargs.get("messages"), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)

    return _TracedMistralAsyncStream(result, span, request_metadata, start_time)


def _embeddings_create_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_embeddings_metadata(kwargs)
    span = _start_span("mistral.embeddings.create", kwargs.get("inputs"), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)

    _finalize_embeddings_response(span, request_metadata, result, start_time)
    return result


async def _embeddings_create_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_embeddings_metadata(kwargs)
    span = _start_span("mistral.embeddings.create", kwargs.get("inputs"), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)

    _finalize_embeddings_response(span, request_metadata, result, start_time)
    return result


def _fim_complete_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_fim_metadata(kwargs, stream=bool(kwargs.get("stream")))
    span = _start_span("mistral.fim.complete", _fim_input(kwargs), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)

    if kwargs.get("stream"):
        return _TracedMistralSyncStream(result, span, request_metadata, start_time)

    _finalize_completion_response(span, request_metadata, result, start_time)
    return result


async def _fim_complete_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_fim_metadata(kwargs, stream=bool(kwargs.get("stream")))
    span = _start_span("mistral.fim.complete", _fim_input(kwargs), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)

    if kwargs.get("stream"):
        return _TracedMistralAsyncStream(result, span, request_metadata, start_time)

    _finalize_completion_response(span, request_metadata, result, start_time)
    return result


def _fim_stream_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_fim_metadata(kwargs, stream=True)
    span = _start_span("mistral.fim.stream", _fim_input(kwargs), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)

    return _TracedMistralSyncStream(result, span, request_metadata, start_time)


async def _fim_stream_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_fim_metadata(kwargs, stream=True)
    span = _start_span("mistral.fim.stream", _fim_input(kwargs), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)

    return _TracedMistralAsyncStream(result, span, request_metadata, start_time)


def _ocr_process_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_ocr_metadata(kwargs)
    span = _start_span("mistral.ocr.process", _ocr_input(kwargs), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)

    _finalize_ocr_response(span, request_metadata, result, start_time)
    return result


async def _ocr_process_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_ocr_metadata(kwargs)
    span = _start_span("mistral.ocr.process", _ocr_input(kwargs), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)

    _finalize_ocr_response(span, request_metadata, result, start_time)
    return result


def wrap_mistral(client: Any) -> Any:
    """Wrap a single Mistral client instance for tracing."""
    from .patchers import AgentsPatcher, ChatPatcher, EmbeddingsPatcher, FimPatcher, OcrPatcher

    chat = getattr(client, "chat", None)
    if chat is not None:
        ChatPatcher.wrap_target(chat)

    embeddings = getattr(client, "embeddings", None)
    if embeddings is not None:
        EmbeddingsPatcher.wrap_target(embeddings)

    fim = getattr(client, "fim", None)
    if fim is not None:
        FimPatcher.wrap_target(fim)

    agents = getattr(client, "agents", None)
    if agents is not None:
        AgentsPatcher.wrap_target(agents)

    ocr = getattr(client, "ocr", None)
    if ocr is not None:
        OcrPatcher.wrap_target(ocr)

    return client
