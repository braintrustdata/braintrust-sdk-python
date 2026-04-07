"""Cohere-specific tracing helpers."""

import time
from collections.abc import AsyncIterator, Iterator
from numbers import Real
from typing import Any

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute


_V2_CHAT_METADATA_KEYS = (
    "model",
    "documents",
    "tools",
    "citation_options",
    "response_format",
    "safety_mode",
    "max_tokens",
    "stop_sequences",
    "temperature",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "k",
    "p",
)
_V2_EMBED_METADATA_KEYS = (
    "model",
    "input_type",
    "embedding_types",
    "truncate",
    "images",
)
_V2_RERANK_METADATA_KEYS = (
    "model",
    "top_n",
    "max_tokens_per_doc",
)


def sanitize_cohere_logged_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json", by_alias=True)
        except TypeError:
            value = value.model_dump()

    safe = bt_safe_deep_copy(value)

    if callable(safe):
        return "[Function]"
    if isinstance(safe, list):
        return [sanitize_cohere_logged_value(item) for item in safe]
    if isinstance(safe, tuple):
        return [sanitize_cohere_logged_value(item) for item in safe]
    if isinstance(safe, dict):
        sanitized = {}
        for key, entry in safe.items():
            if entry is None:
                continue
            sanitized[key] = sanitize_cohere_logged_value(entry)
        return sanitized
    return safe


def _is_supported_metric_value(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _timing_metrics(start_time: float, first_token_time: float | None = None) -> dict[str, float]:
    end_time = time.time()
    metrics = {
        "start": start_time,
        "end": end_time,
        "duration": end_time - start_time,
    }
    if first_token_time is not None:
        metrics["time_to_first_token"] = first_token_time - start_time
    return metrics


def _normalize_usage_tokens(data: dict[str, Any], metrics: dict[str, float]) -> None:
    input_tokens = data.get("input_tokens")
    output_tokens = data.get("output_tokens")

    if _is_supported_metric_value(input_tokens):
        metrics["prompt_tokens"] = float(input_tokens)
    if _is_supported_metric_value(output_tokens):
        metrics["completion_tokens"] = float(output_tokens)


def _metrics_from_usage_like(usage_or_meta: Any) -> dict[str, float]:
    data = sanitize_cohere_logged_value(usage_or_meta)
    if not isinstance(data, dict):
        return {}

    metrics: dict[str, float] = {}

    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        _normalize_usage_tokens(tokens, metrics)

    billed_units = data.get("billed_units")
    if isinstance(billed_units, dict):
        if "prompt_tokens" not in metrics or "completion_tokens" not in metrics:
            _normalize_usage_tokens(billed_units, metrics)
        for key, value in billed_units.items():
            if _is_supported_metric_value(value):
                metrics[f"billed_{key}"] = float(value)

    cached_tokens = data.get("cached_tokens")
    if _is_supported_metric_value(cached_tokens):
        metrics["cached_tokens"] = float(cached_tokens)

    if "tokens" not in metrics:
        if "prompt_tokens" in metrics and "completion_tokens" in metrics:
            metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
        elif "prompt_tokens" in metrics:
            metrics["tokens"] = metrics["prompt_tokens"]
        elif "completion_tokens" in metrics:
            metrics["tokens"] = metrics["completion_tokens"]

    return metrics


def _merge_metrics(start_time: float, usage_or_meta: Any, first_token_time: float | None = None) -> dict[str, float]:
    return {
        **_timing_metrics(start_time, first_token_time),
        **_metrics_from_usage_like(usage_or_meta),
    }


def _build_metadata(
    kwargs: dict[str, Any],
    keys: tuple[str, ...],
    *,
    stream: bool | None = None,
) -> dict[str, Any]:
    metadata = {
        "provider": "cohere",
        "api_version": "2",
    }

    for key in keys:
        value = kwargs.get(key)
        if value is None:
            continue
        metadata[key] = sanitize_cohere_logged_value(value)

    if stream is not None:
        metadata["stream"] = stream

    return metadata


def _chat_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    span_input = {
        "messages": sanitize_cohere_logged_value(kwargs.get("messages")),
    }
    for key in ("documents", "tools"):
        value = kwargs.get(key)
        if value is not None:
            span_input[key] = sanitize_cohere_logged_value(value)
    return span_input


def _embed_input(kwargs: dict[str, Any]) -> Any:
    span_input = {}
    for key in ("texts", "images", "inputs"):
        value = kwargs.get(key)
        if value is not None:
            span_input[key] = sanitize_cohere_logged_value(value)
    if len(span_input) == 1:
        return next(iter(span_input.values()))
    return span_input


def _rerank_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": kwargs.get("query"),
        "documents": sanitize_cohere_logged_value(kwargs.get("documents")),
    }


def _start_span(name: str, span_input: Any, metadata: dict[str, Any]):
    return start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=sanitize_cohere_logged_value(span_input),
        metadata=metadata,
    )


def _response_metadata(response: Any, *, finish_reason: Any | None = None) -> dict[str, Any]:
    data = sanitize_cohere_logged_value(response)
    if not isinstance(data, dict):
        return {}

    metadata = {}
    if data.get("id") is not None:
        metadata["id"] = data["id"]
    if data.get("finish_reason") is not None:
        metadata["finish_reason"] = data["finish_reason"]
    elif finish_reason is not None:
        metadata["finish_reason"] = finish_reason

    meta = data.get("meta")
    if isinstance(meta, dict):
        warnings = meta.get("warnings")
        if warnings:
            metadata["warnings"] = warnings

    return metadata


def _chat_output(response: Any) -> Any:
    data = sanitize_cohere_logged_value(response)
    if not isinstance(data, dict):
        return data
    return data.get("message") or data


def _embed_output(response: Any) -> dict[str, Any]:
    data = sanitize_cohere_logged_value(response)
    if not isinstance(data, dict):
        return {"embeddings_count": 0, "embedding_length": None}

    output = {
        "embeddings_count": 0,
        "embedding_length": None,
    }
    embeddings = data.get("embeddings")
    if isinstance(embeddings, dict):
        embedding_types = []
        for key, value in embeddings.items():
            if not value:
                continue
            embedding_types.append(key.rstrip("_"))
            if output["embeddings_count"] == 0 and isinstance(value, list):
                output["embeddings_count"] = len(value)
                first = value[0] if value else None
                if isinstance(first, list):
                    output["embedding_length"] = len(first)
        if embedding_types:
            output["embedding_types"] = embedding_types
    return output


def _rerank_output(response: Any) -> Any:
    data = sanitize_cohere_logged_value(response)
    if not isinstance(data, dict):
        return data
    return data.get("results") or data


def _log_and_end(
    span: Any,
    *,
    output: Any = None,
    metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    event = {}
    if output is not None:
        event["output"] = output
    if metrics:
        event["metrics"] = metrics
    if metadata:
        event["metadata"] = metadata
    if event:
        span.log(**event)
    span.end()


def _log_error_and_end(span: Any, error: Exception):
    span.log(error=error)
    span.end()


def _call_with_error_logging(span: Any, wrapped: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        return wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise


async def _call_async_with_error_logging(
    span: Any,
    wrapped: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    try:
        return await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise


def _finalize_response(
    span: Any,
    *,
    output: Any,
    response: Any,
    request_metadata: dict[str, Any],
    start_time: float,
    usage_or_meta: Any,
):
    _log_and_end(
        span,
        output=output,
        metrics=_merge_metrics(start_time, usage_or_meta),
        metadata={
            **request_metadata,
            **_response_metadata(response),
        },
    )


def _append_text_content(content_by_index: dict[int, dict[str, Any]], index: int, delta: dict[str, Any]) -> None:
    item = content_by_index.setdefault(index, {"type": "text", "text": ""})
    if delta.get("type") is not None:
        item["type"] = delta["type"]
    text = delta.get("text")
    if isinstance(text, str):
        item["text"] = f"{item.get('text', '')}{text}"


def _merge_tool_call(target: dict[str, Any], delta: dict[str, Any]) -> None:
    for key in ("id", "type"):
        value = delta.get(key)
        if value is not None:
            target[key] = value

    function = delta.get("function")
    if not isinstance(function, dict):
        return

    target_function = target.setdefault("function", {"name": "", "arguments": ""})
    name = function.get("name")
    if isinstance(name, str):
        target_function["name"] = f"{target_function.get('name', '')}{name}"
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        target_function["arguments"] = f"{target_function.get('arguments', '')}{arguments}"


def _aggregate_chat_stream(chunks: list[Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    message = {"role": "assistant", "content": []}
    content_by_index: dict[int, dict[str, Any]] = {}
    tool_calls: dict[int, dict[str, Any]] = {}
    response_id = None
    finish_reason = None
    usage = None

    for chunk in chunks:
        data = sanitize_cohere_logged_value(chunk)
        if not isinstance(data, dict):
            continue

        chunk_type = data.get("type")
        if chunk_type == "message-start":
            response_id = data.get("id") or response_id
            delta_message = (data.get("delta") or {}).get("message") or {}
            role = delta_message.get("role")
            if role is not None:
                message["role"] = role
        elif chunk_type == "content-start":
            index = int(data.get("index", 0) or 0)
            content = ((data.get("delta") or {}).get("message") or {}).get("content") or {}
            if isinstance(content, dict):
                _append_text_content(content_by_index, index, content)
        elif chunk_type == "content-delta":
            index = int(data.get("index", 0) or 0)
            content = ((data.get("delta") or {}).get("message") or {}).get("content") or {}
            if isinstance(content, dict):
                _append_text_content(content_by_index, index, content)
        elif chunk_type == "tool-plan-delta":
            delta_message = (data.get("delta") or {}).get("message") or {}
            tool_plan = delta_message.get("tool_plan")
            if isinstance(tool_plan, str):
                message["tool_plan"] = f"{message.get('tool_plan', '')}{tool_plan}"
        elif chunk_type in ("tool-call-start", "tool-call-delta"):
            index = int(data.get("index", 0) or 0)
            tool_call = tool_calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
            delta_tool_call = ((data.get("delta") or {}).get("message") or {}).get("tool_calls") or {}
            if isinstance(delta_tool_call, dict):
                _merge_tool_call(tool_call, delta_tool_call)
        elif chunk_type == "message-end":
            delta = data.get("delta") or {}
            finish_reason = delta.get("finish_reason") or finish_reason
            usage = delta.get("usage") or usage

    if content_by_index:
        message["content"] = [content_by_index[index] for index in sorted(content_by_index)]
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]

    metadata = {}
    if response_id is not None:
        metadata["id"] = response_id
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason

    return message, usage, metadata


class _TracedSyncChatStream:
    def __init__(self, stream: Any, span: Any, metadata: dict[str, Any], start_time: float):
        self._stream = stream
        self._span = span
        self._metadata = metadata
        self._start_time = start_time
        self._first_token_time = None
        self._items: list[Any] = []
        self._closed = False

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

        if self._first_token_time is None and getattr(item, "type", None) in (
            "content-delta",
            "tool-plan-delta",
            "tool-call-start",
            "tool-call-delta",
        ):
            self._first_token_time = time.time()
        self._items.append(item)
        return item

    def _finalize(self, *, error: Exception | None = None):
        if self._closed:
            return
        self._closed = True

        if error is not None:
            _log_error_and_end(self._span, error)
            return

        output, usage, response_metadata = _aggregate_chat_stream(self._items)
        _log_and_end(
            self._span,
            output=output,
            metrics=_merge_metrics(self._start_time, usage, self._first_token_time),
            metadata={**self._metadata, **response_metadata},
        )


class _TracedAsyncChatStream:
    def __init__(self, stream: Any, span: Any, metadata: dict[str, Any], start_time: float):
        self._stream = stream
        self._span = span
        self._metadata = metadata
        self._start_time = start_time
        self._first_token_time = None
        self._items: list[Any] = []
        self._closed = False

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

        if self._first_token_time is None and getattr(item, "type", None) in (
            "content-delta",
            "tool-plan-delta",
            "tool-call-start",
            "tool-call-delta",
        ):
            self._first_token_time = time.time()
        self._items.append(item)
        return item

    def _finalize(self, *, error: Exception | None = None):
        if self._closed:
            return
        self._closed = True

        if error is not None:
            _log_error_and_end(self._span, error)
            return

        output, usage, response_metadata = _aggregate_chat_stream(self._items)
        _log_and_end(
            self._span,
            output=output,
            metrics=_merge_metrics(self._start_time, usage, self._first_token_time),
            metadata={**self._metadata, **response_metadata},
        )


def _v2_chat_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_metadata(kwargs, _V2_CHAT_METADATA_KEYS)
    span = _start_span("cohere.chat", _chat_input(kwargs), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)
    _finalize_response(
        span,
        output=_chat_output(result),
        response=result,
        request_metadata=request_metadata,
        start_time=start_time,
        usage_or_meta=getattr(result, "usage", None),
    )
    return result


async def _v2_chat_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_metadata(kwargs, _V2_CHAT_METADATA_KEYS)
    span = _start_span("cohere.chat", _chat_input(kwargs), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)
    _finalize_response(
        span,
        output=_chat_output(result),
        response=result,
        request_metadata=request_metadata,
        start_time=start_time,
        usage_or_meta=getattr(result, "usage", None),
    )
    return result


def _v2_chat_stream_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_metadata(kwargs, _V2_CHAT_METADATA_KEYS, stream=True)
    span = _start_span("cohere.chat", _chat_input(kwargs), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)
    return _TracedSyncChatStream(result, span, request_metadata, start_time)


def _v2_chat_stream_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_metadata(kwargs, _V2_CHAT_METADATA_KEYS, stream=True)
    span = _start_span("cohere.chat", _chat_input(kwargs), request_metadata)
    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise
    return _TracedAsyncChatStream(result, span, request_metadata, start_time)


def _v2_embed_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_metadata(kwargs, _V2_EMBED_METADATA_KEYS)
    span = _start_span("cohere.embed", _embed_input(kwargs), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)
    _finalize_response(
        span,
        output=_embed_output(result),
        response=result,
        request_metadata=request_metadata,
        start_time=start_time,
        usage_or_meta=getattr(result, "meta", None),
    )
    return result


async def _v2_embed_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_metadata(kwargs, _V2_EMBED_METADATA_KEYS)
    span = _start_span("cohere.embed", _embed_input(kwargs), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)
    _finalize_response(
        span,
        output=_embed_output(result),
        response=result,
        request_metadata=request_metadata,
        start_time=start_time,
        usage_or_meta=getattr(result, "meta", None),
    )
    return result


def _v2_rerank_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_metadata(kwargs, _V2_RERANK_METADATA_KEYS)
    span = _start_span("cohere.rerank", _rerank_input(kwargs), request_metadata)
    start_time = time.time()
    result = _call_with_error_logging(span, wrapped, args, kwargs)
    _finalize_response(
        span,
        output=_rerank_output(result),
        response=result,
        request_metadata=request_metadata,
        start_time=start_time,
        usage_or_meta=getattr(result, "meta", None),
    )
    return result


async def _v2_rerank_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_metadata(kwargs, _V2_RERANK_METADATA_KEYS)
    span = _start_span("cohere.rerank", _rerank_input(kwargs), request_metadata)
    start_time = time.time()
    result = await _call_async_with_error_logging(span, wrapped, args, kwargs)
    _finalize_response(
        span,
        output=_rerank_output(result),
        response=result,
        request_metadata=request_metadata,
        start_time=start_time,
        usage_or_meta=getattr(result, "meta", None),
    )
    return result


def _is_v2_client_instance(client: Any) -> bool:
    return any(
        base.__name__ in {"ClientV2", "AsyncClientV2", "V2Client", "AsyncV2Client"} for base in type(client).__mro__
    )


def _is_async_v2_client_instance(client: Any) -> bool:
    return any(base.__name__ in {"AsyncClientV2", "AsyncV2Client"} for base in type(client).__mro__)


def _wrap_v2_target(target: Any) -> None:
    from .patchers import (
        _V2ChatAsyncPatcher,
        _V2ChatPatcher,
        _V2ChatStreamAsyncPatcher,
        _V2ChatStreamPatcher,
        _V2EmbedAsyncPatcher,
        _V2EmbedPatcher,
        _V2RerankAsyncPatcher,
        _V2RerankPatcher,
    )

    if _is_async_v2_client_instance(target):
        patchers = (
            _V2ChatAsyncPatcher,
            _V2ChatStreamAsyncPatcher,
            _V2EmbedAsyncPatcher,
            _V2RerankAsyncPatcher,
        )
    else:
        patchers = (
            _V2ChatPatcher,
            _V2ChatStreamPatcher,
            _V2EmbedPatcher,
            _V2RerankPatcher,
        )

    for patcher in patchers:
        patcher.wrap_target(target)


def wrap_cohere(client: Any) -> Any:
    """Wrap a single Cohere client or Cohere V2 client instance for tracing."""
    if _is_v2_client_instance(client):
        _wrap_v2_target(client)
        return client

    nested_v2 = getattr(client, "v2", None)
    if nested_v2 is not None:
        _wrap_v2_target(nested_v2)

    return client
