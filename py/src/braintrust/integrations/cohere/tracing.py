"""Cohere-specific tracing helpers.

Span shape follows the JS plugin (`js/src/instrumentation/plugins/cohere-plugin.ts`):

- ``input``   — the meaningful user request (``message``/``messages`` for chat,
                ``texts``/``images`` for embed, ``{query, documents}`` for rerank).
- ``output``  — the normalized provider result.
- ``metadata`` — a small allowlist of request params plus provider/response ids.
- ``metrics`` — timing plus ``tokens``/``prompt_tokens``/``completion_tokens``
                derived from ``usage`` / ``meta.billed_units`` / ``meta.tokens``.
"""

import logging
import time
from typing import Any

from braintrust.integrations.utils import (
    _log_and_end_span,
    _log_error_and_end_span,
    _materialize_attachment,
    _timing_metrics,
    _try_to_dict,
)
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import is_numeric


logger = logging.getLogger(__name__)


_PROVIDER = "cohere"

_CHAT_METADATA_KEYS = (
    "model",
    "temperature",
    "max_tokens",
    "max_input_tokens",
    "top_p",
    "k",
    "p",
    "seed",
    "stop_sequences",
    "preamble",
    "conversation_id",
    "prompt_truncation",
    "response_format",
    "safety_mode",
    "frequency_penalty",
    "presence_penalty",
    "raw_prompting",
    "search_queries_only",
    "strict_tools",
    "tool_choice",
    "tools",
    "priority",
)
_EMBED_METADATA_KEYS = (
    "model",
    "input_type",
    "embedding_types",
    "output_dimension",
    "max_tokens",
    "truncate",
    "priority",
)
_RERANK_METADATA_KEYS = (
    "model",
    "top_n",
    "return_documents",
    "max_chunks_per_doc",
    "max_tokens_per_doc",
    "rank_fields",
    "priority",
)
_AUDIO_TRANSCRIPTION_METADATA_KEYS = (
    "model",
    "language",
    "temperature",
)
_RESPONSE_METADATA_KEYS = (
    "id",
    "generation_id",
    "response_id",
    "response_type",
    "finish_reason",
)


# ---------------------------------------------------------------------------
# Metadata / metrics helpers
# ---------------------------------------------------------------------------


def _get_field(obj: Any, key: str) -> Any:
    """Return a field from either a mapping or a Cohere SDK model object.

    Cohere responses and stream events are not represented by one uniform type
    across SDK versions and transports: some are plain dictionaries, while
    others are Pydantic-style objects with attributes.  Tracing code only needs
    read-only access to named fields, so this helper centralizes that small
    compatibility layer instead of scattering ``isinstance(..., dict)`` checks
    throughout the normalization logic.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _pick_allowed_metadata(kwargs: dict[str, Any], allowlist: tuple[str, ...]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": _PROVIDER}
    for key in allowlist:
        value = kwargs.get(key)
        if value is None:
            continue
        metadata[key] = value
    return metadata


def _extract_response_metadata(result: Any) -> dict[str, Any]:
    """Pull response-id / finish-reason / api-version metadata from *result*."""
    if result is None:
        return {}

    metadata: dict[str, Any] = {}
    for key in _RESPONSE_METADATA_KEYS:
        value = _get_field(result, key)
        if value is not None:
            metadata[key] = value

    api_version = _get_field(_get_field(result, "meta"), "api_version")
    version_value = _get_field(api_version, "version")
    if version_value is not None:
        metadata["api_version"] = version_value
    return metadata


def _merge_usage_metrics(metrics: dict[str, float], usage: Any) -> None:
    """Pull numeric token counts out of a Cohere ``usage`` / ``meta`` payload."""
    if usage is None:
        return

    # top-level fields (rare, but present on some v2 shapes)
    for key, metric in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "tokens"),
        ("cached_tokens", "prompt_cached_tokens"),
    ):
        value = _get_field(usage, key)
        if is_numeric(value):
            metrics[metric] = float(value)

    tokens_block = _get_field(usage, "tokens")
    for src_key, metric in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "tokens"),
    ):
        value = _get_field(tokens_block, src_key)
        if is_numeric(value):
            metrics[metric] = float(value)

    # billed_units is Cohere's authoritative counter for billing — it intentionally
    # overrides values from the top-level or ``tokens`` block when both are present.
    billed = _get_field(usage, "billed_units")
    for src_key, metric in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("search_units", "search_units"),
        ("classifications", "classifications"),
        ("images", "images"),
        ("image_tokens", "image_tokens"),
    ):
        value = _get_field(billed, src_key)
        if is_numeric(value):
            metrics[metric] = float(value)


def _parse_usage_metrics(result: Any) -> dict[str, float]:
    """Extract token / billed-unit metrics from a Cohere response."""
    if result is None:
        return {}

    metrics: dict[str, float] = {}
    _merge_usage_metrics(metrics, result)
    _merge_usage_metrics(metrics, _get_field(result, "usage"))
    _merge_usage_metrics(metrics, _get_field(result, "meta"))

    if "tokens" not in metrics and "prompt_tokens" in metrics and "completion_tokens" in metrics:
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    return metrics


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _chat_input(kwargs: dict[str, Any]) -> Any:
    messages = kwargs.get("messages")
    if messages is not None:
        return messages
    return kwargs.get("message")


def _embed_input(kwargs: dict[str, Any]) -> Any:
    for key in ("inputs", "texts", "images"):
        value = kwargs.get(key)
        if value is not None:
            return value
    return None


def _rerank_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": kwargs.get("query"),
        "documents": kwargs.get("documents"),
    }


def _rerank_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    metadata = _pick_allowed_metadata(kwargs, _RERANK_METADATA_KEYS)
    documents = kwargs.get("documents")
    if isinstance(documents, (list, tuple)):
        metadata["document_count"] = len(documents)
    return metadata


def _audio_transcription_input(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize the ``file=`` parameter into a traced ``Attachment`` input.

    Cohere's ``TranscriptionsClient.create`` accepts a ``core.File`` input,
    which can be bytes, a file-like object, or a ``(filename, content,
    content_type[, headers])`` tuple.  ``_materialize_attachment`` already
    unpacks those tuple shapes, so we hand it the raw value and fall back
    to a placeholder only when materialization fails.
    """
    file_value = kwargs.get("file")
    if file_value is None:
        return None

    resolved = _materialize_attachment(file_value, prefix="input_audio")
    attachment = resolved.attachment if resolved is not None else None
    return {"file": attachment if attachment is not None else "[audio]"}


def _audio_transcription_output(result: Any) -> str | None:
    """Return the transcribed text string for a transcription response."""
    if result is None:
        return None
    text = _get_field(result, "text")
    return text if isinstance(text, str) else None


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------


def _chat_output(result: Any) -> Any:
    """Return the user-visible chat output from a Cohere response.

    For v2 responses this is the ``message`` sub-object; for v1 it's the
    ``text`` string (or a ``{role, content, tool_calls}`` dict when tool
    calls are present).
    """
    if result is None:
        return None

    message = _get_field(result, "message")
    if message is not None:
        return message

    text = _get_field(result, "text")
    tool_calls = _get_field(result, "tool_calls")
    if tool_calls:
        return {
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        }
    if isinstance(text, str):
        return text
    return None


def _iter_embedding_lists(embeddings: Any):
    """Yield candidate ``list[list[float]]`` entries from an embeddings payload.

    Handles both the v1 plain list-of-lists shape and the v2 typed
    (``float`` / ``int8`` / ...) container — the latter may be a dict or a
    Pydantic model.
    """
    if isinstance(embeddings, list):
        yield embeddings
        return
    if embeddings is None:
        return
    if isinstance(embeddings, dict):
        yield from embeddings.values()
        return
    # Pydantic v2 exposes model_fields on the class, not the instance.
    fields = getattr(type(embeddings), "model_fields", None) or ()
    for name in fields:
        yield getattr(embeddings, name, None)


def _embed_output(result: Any) -> dict[str, Any] | None:
    """Return ``{embedding_count, embedding_length}`` summary for an embed response."""
    embeddings = _get_field(result, "embeddings")
    for entry in _iter_embedding_lists(embeddings):
        if isinstance(entry, list) and entry and isinstance(entry[0], list):
            return {
                "embedding_count": len(entry),
                "embedding_length": len(entry[0]),
            }
    return None


def _rerank_output(result: Any) -> list[dict[str, Any]] | None:
    """Return the top-N rerank results (capped at 100) from a rerank response.

    Each result is summarized to ``{index, relevance_score}`` — the document
    payload (if present via ``return_documents=True``) is dropped on purpose.
    """
    results = _get_field(result, "results")
    if not isinstance(results, list):
        return None

    out: list[dict[str, Any]] = []
    for item in results[:100]:
        if item is None:
            continue
        out.append(
            {
                "index": _get_field(item, "index"),
                "relevance_score": _get_field(item, "relevance_score"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Stream aggregation
# ---------------------------------------------------------------------------


def _merge_tool_call(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge a streaming tool-call delta into its accumulator."""
    base = dict(existing or {})
    base.update({k: v for k, v in incoming.items() if k != "function"})

    incoming_fn = incoming.get("function")
    if not isinstance(incoming_fn, dict):
        return base

    existing_fn = base.get("function")
    merged_fn = dict(existing_fn) if isinstance(existing_fn, dict) else {}
    incoming_args = incoming_fn.get("arguments")
    if isinstance(incoming_args, str):
        merged_fn["arguments"] = f"{merged_fn.get('arguments', '') or ''}{incoming_args}"
    for key, value in incoming_fn.items():
        if key == "arguments":
            continue
        merged_fn[key] = value
    base["function"] = merged_fn
    return base


def _v2_delta_text(chunk: Any) -> str | None:
    content = _get_field(_get_field(_get_field(chunk, "delta"), "message"), "content")
    if isinstance(content, str):
        return content
    text = _get_field(content, "text")
    return text if isinstance(text, str) else None


def _delta_message_tool_calls(chunk: Any) -> list[Any]:
    """Return tool-call deltas carried in a v2 ``delta.message.tool_calls`` field.

    The SDK may emit a single tool-call object or a list; this normalizes to
    a list so callers can iterate uniformly.
    """
    tool_calls = _get_field(_get_field(_get_field(chunk, "delta"), "message"), "tool_calls")
    if tool_calls is None:
        return []
    if isinstance(tool_calls, list):
        return tool_calls
    return [tool_calls]


def _as_merge_dict(value: Any) -> dict[str, Any] | None:
    """Coerce a tool-call delta to a plain dict for merging, or return ``None``."""
    if isinstance(value, dict):
        return value
    converted = _try_to_dict(value)
    return converted if isinstance(converted, dict) else None


def _upsert_tool_calls(
    tool_calls_by_index: dict[int, dict[str, Any]],
    tool_call_order: list[int],
    tool_calls: list[Any],
    *,
    fallback_index: int | None = None,
    merge: bool = False,
) -> None:
    """Merge tool-call deltas into the per-index accumulator.

    Used by both the v1 ``tool-calls-generation`` handler and the v2
    ``tool-call-start`` / ``tool-call-delta`` handlers.  When *merge* is
    true, arguments strings are concatenated via :func:`_merge_tool_call`;
    otherwise incoming fields shallow-overwrite existing ones.
    """
    for positional, tool_call in enumerate(tool_calls):
        incoming = _as_merge_dict(tool_call)
        if incoming is None:
            continue
        incoming_idx = incoming.get("index")
        idx = (
            fallback_index
            if isinstance(fallback_index, int)
            else incoming_idx
            if isinstance(incoming_idx, int)
            else positional
        )
        if idx not in tool_call_order:
            tool_call_order.append(idx)
        if merge:
            tool_calls_by_index[idx] = _merge_tool_call(tool_calls_by_index.get(idx), incoming)
        else:
            tool_calls_by_index[idx] = {
                **(tool_calls_by_index.get(idx) or {}),
                **incoming,
            }


def _aggregate_chat_stream(chunks: list[Any]) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """Aggregate a list of chat-stream events into (output, metrics, metadata).

    Handles both the v1 event shape (``event_type``: ``text-generation`` /
    ``stream-end``) and the v2 event shape (``type``: ``message-start`` /
    ``content-delta`` / ``message-end`` / ``tool-call-*``).

    Chunks may be dicts or provider objects; fields are read via :func:`_get`.
    """
    text_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    tool_call_order: list[int] = []
    terminal_response: Any = None
    role: str | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = {}
    metrics: dict[str, float] = {}

    for chunk in chunks:
        if chunk is None:
            continue
        event_type = _get_field(chunk, "event_type") or _get_field(chunk, "type")

        # -- v1 shape ---------------------------------------------------------
        if event_type == "text-generation":
            text = _get_field(chunk, "text")
            if isinstance(text, str):
                text_parts.append(text)
            continue
        if event_type == "stream-end":
            response = _get_field(chunk, "response")
            if response is not None:
                terminal_response = response
                metrics.update(_parse_usage_metrics(response))
                metadata.update(_extract_response_metadata(response))
                fr = _get_field(response, "finish_reason")
                if isinstance(fr, str):
                    finish_reason = fr
            continue
        if event_type == "tool-calls-generation":
            tool_calls = _get_field(chunk, "tool_calls")
            if isinstance(tool_calls, list):
                _upsert_tool_calls(tool_calls_by_index, tool_call_order, tool_calls)
            continue

        # -- v2 shape ---------------------------------------------------------
        if event_type == "message-start":
            msg_id = _get_field(chunk, "id")
            if isinstance(msg_id, str):
                metadata["id"] = msg_id
            role_value = _get_field(_get_field(_get_field(chunk, "delta"), "message"), "role")
            if isinstance(role_value, str):
                role = role_value
            continue
        if event_type == "content-delta":
            text = _v2_delta_text(chunk)
            if text:
                text_parts.append(text)
            continue
        if event_type == "tool-call-start":
            chunk_index = _get_field(chunk, "index")
            _upsert_tool_calls(
                tool_calls_by_index,
                tool_call_order,
                _delta_message_tool_calls(chunk),
                fallback_index=chunk_index if isinstance(chunk_index, int) else None,
            )
            continue
        if event_type == "tool-call-delta":
            chunk_index = _get_field(chunk, "index")
            _upsert_tool_calls(
                tool_calls_by_index,
                tool_call_order,
                _delta_message_tool_calls(chunk)[:1],
                fallback_index=chunk_index if isinstance(chunk_index, int) else None,
                merge=True,
            )
            continue
        if event_type == "message-end":
            delta = _get_field(chunk, "delta")
            fr = _get_field(delta, "finish_reason")
            if isinstance(fr, str):
                finish_reason = fr
            usage = _get_field(delta, "usage")
            if usage is not None:
                _merge_usage_metrics(metrics, usage)
                if "tokens" not in metrics and "prompt_tokens" in metrics and "completion_tokens" in metrics:
                    metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
            continue

    merged_tool_calls = [
        tool_calls_by_index[i] for i in sorted(tool_call_order) if isinstance(tool_calls_by_index.get(i), dict)
    ]

    output: Any = _chat_output(terminal_response) if terminal_response is not None else None
    if output is None:
        merged_text = "".join(text_parts)
        if merged_tool_calls or role or merged_text:
            output = {}
            if role:
                output["role"] = role
            if merged_text:
                output["content"] = merged_text
            if merged_tool_calls:
                output["tool_calls"] = merged_tool_calls

    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason

    return output, metrics, metadata


# ---------------------------------------------------------------------------
# Span creation / sync-async wrappers
# ---------------------------------------------------------------------------


def _start_span(name: str, span_input: Any, metadata: dict[str, Any]):
    return start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=span_input,
        metadata=metadata,
    )


def _tool_call_function(tool_call: Any) -> Any:
    return _get_field(tool_call, "function")


def _tool_call_name(tool_call: Any) -> str | None:
    function = _tool_call_function(tool_call)
    name = _get_field(function, "name") or _get_field(tool_call, "name")
    return name if isinstance(name, str) and name else None


def _tool_call_input(tool_call: Any) -> Any:
    function = _tool_call_function(tool_call)
    arguments = _get_field(function, "arguments")
    if arguments is not None:
        return arguments
    parameters = _get_field(function, "parameters")
    if parameters is not None:
        return parameters
    return _get_field(tool_call, "parameters")


def _tool_call_metadata(tool_call: Any) -> dict[str, Any] | None:
    metadata = {
        "tool_call_id": _get_field(tool_call, "id") or _get_field(tool_call, "call_id"),
        "tool_type": _get_field(tool_call, "type"),
    }
    return {k: v for k, v in metadata.items() if v is not None} or None


def _iter_tool_calls(output: Any):
    if output is None:
        return
    output_dict = output if isinstance(output, dict) else _try_to_dict(output)
    if not isinstance(output_dict, dict):
        return
    tool_calls = output_dict.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for tool_call in tool_calls:
        if tool_call is not None:
            yield tool_call


def _log_tool_call_spans(output: Any, *, parent_export: str | None) -> None:
    for tool_call in _iter_tool_calls(output):
        name = _tool_call_name(tool_call)
        if name is None:
            continue
        span_args = {
            "name": f"tool: {name}",
            "type": SpanTypeAttribute.TOOL,
            "input": _tool_call_input(tool_call),
            "metadata": _tool_call_metadata(tool_call),
        }
        if parent_export is not None:
            span_args["parent"] = parent_export
        with start_span(**span_args):
            pass


def _log_call_result(span, output_fn, start_time: float, result: Any) -> None:
    """Log output/metrics/metadata for *result* and end *span*."""
    metrics = {
        **_timing_metrics(start_time, time.time()),
        **_parse_usage_metrics(result),
    }
    output = output_fn(result)
    _log_tool_call_spans(output, parent_export=span.export())
    _log_and_end_span(
        span,
        output=output,
        metrics=metrics,
        metadata=_extract_response_metadata(result) or None,
    )


def _trace_sync_call(name: str, span_input: Any, metadata: dict[str, Any], output_fn, wrapped, args, kwargs):
    span = _start_span(name, span_input, metadata)
    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    _log_call_result(span, output_fn, start_time, result)
    return result


async def _trace_async_call(name: str, span_input: Any, metadata: dict[str, Any], output_fn, wrapped, args, kwargs):
    span = _start_span(name, span_input, metadata)
    start_time = time.time()
    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    _log_call_result(span, output_fn, start_time, result)
    return result


# ---- sync wrappers ---------------------------------------------------------


def _chat_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _chat_input(kwargs)
    metadata = _pick_allowed_metadata(kwargs, _CHAT_METADATA_KEYS)
    return _trace_sync_call("cohere.chat", span_input, metadata, _chat_output, wrapped, args, kwargs)


def _embed_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _embed_input(kwargs)
    metadata = _pick_allowed_metadata(kwargs, _EMBED_METADATA_KEYS)
    return _trace_sync_call("cohere.embed", span_input, metadata, _embed_output, wrapped, args, kwargs)


def _rerank_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _rerank_input(kwargs)
    metadata = _rerank_metadata(kwargs)
    return _trace_sync_call("cohere.rerank", span_input, metadata, _rerank_output, wrapped, args, kwargs)


def _audio_transcription_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _audio_transcription_input(kwargs)
    metadata = _pick_allowed_metadata(kwargs, _AUDIO_TRANSCRIPTION_METADATA_KEYS)
    return _trace_sync_call(
        "cohere.audio.transcriptions.create",
        span_input,
        metadata,
        _audio_transcription_output,
        wrapped,
        args,
        kwargs,
    )


def _chat_stream_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _chat_input(kwargs)
    metadata = _pick_allowed_metadata(kwargs, _CHAT_METADATA_KEYS)
    span = _start_span("cohere.chat_stream", span_input, metadata)
    start_time = time.time()
    try:
        iterator = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise
    return _TracedChatStream(iterator, span, start_time)


# ---- async wrappers --------------------------------------------------------


async def _async_chat_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _chat_input(kwargs)
    metadata = _pick_allowed_metadata(kwargs, _CHAT_METADATA_KEYS)
    return await _trace_async_call("cohere.chat", span_input, metadata, _chat_output, wrapped, args, kwargs)


async def _async_embed_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _embed_input(kwargs)
    metadata = _pick_allowed_metadata(kwargs, _EMBED_METADATA_KEYS)
    return await _trace_async_call("cohere.embed", span_input, metadata, _embed_output, wrapped, args, kwargs)


async def _async_rerank_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _rerank_input(kwargs)
    metadata = _rerank_metadata(kwargs)
    return await _trace_async_call("cohere.rerank", span_input, metadata, _rerank_output, wrapped, args, kwargs)


async def _async_audio_transcription_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _audio_transcription_input(kwargs)
    metadata = _pick_allowed_metadata(kwargs, _AUDIO_TRANSCRIPTION_METADATA_KEYS)
    return await _trace_async_call(
        "cohere.audio.transcriptions.create",
        span_input,
        metadata,
        _audio_transcription_output,
        wrapped,
        args,
        kwargs,
    )


async def _async_chat_stream_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span_input = _chat_input(kwargs)
    metadata = _pick_allowed_metadata(kwargs, _CHAT_METADATA_KEYS)
    span = _start_span("cohere.chat_stream", span_input, metadata)
    start_time = time.time()
    try:
        iterator = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    # Some async methods return a coroutine that yields an async iterator;
    # others directly return an async iterator.  Handle both.
    if hasattr(iterator, "__await__"):
        try:
            iterator = await iterator
        except Exception as error:
            _log_error_and_end_span(span, error)
            raise

    return _AsyncTracedChatStream(iterator, span, start_time)


# ---- Traced stream iterators ----------------------------------------------


class _ChatStreamTracker:
    """Shared bookkeeping for the sync/async traced chat-stream iterators."""

    def __init__(self, iterator: Any, span: Any, start_time: float):
        self._iterator = iterator
        self._span = span
        self._start_time = start_time
        self._first_token_time: float | None = None
        self._chunks: list[Any] = []
        self._finished = False

    def _record(self, event: Any) -> None:
        if self._first_token_time is None:
            self._first_token_time = time.time()
        self._chunks.append(event)

    def _finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        if error is not None:
            _log_error_and_end_span(self._span, error)
            return

        output, usage_metrics, extra_metadata = _aggregate_chat_stream(self._chunks)
        metrics = {
            **_timing_metrics(self._start_time, time.time(), self._first_token_time),
            **usage_metrics,
        }
        _log_tool_call_spans(output, parent_export=self._span.export())
        _log_and_end_span(
            self._span,
            output=output,
            metrics=metrics,
            metadata=extra_metadata or None,
        )


class _TracedChatStream(_ChatStreamTracker):
    """Wrap a sync chat-stream iterator so exhaustion logs the aggregated span."""

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._iterator)
        except StopIteration:
            self._finish()
            raise
        except Exception as error:
            self._finish(error=error)
            raise
        self._record(event)
        return event


class _AsyncTracedChatStream(_ChatStreamTracker):
    """Async counterpart of :class:`_TracedChatStream`."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except Exception as error:
            self._finish(error=error)
            raise
        self._record(event)
        return event


# ---------------------------------------------------------------------------
# Manual wrapping — `wrap_cohere(client)`
# ---------------------------------------------------------------------------

_BRAINTRUST_TRACED_COHERE = "__braintrust_cohere_traced__"


def _client_is_async(client: Any) -> bool:
    type_name = type(client).__name__
    return type_name.startswith("Async")


def _patch_v1_client(client: Any) -> None:
    from .patchers import (
        AsyncChatPatcher,
        AsyncChatStreamPatcher,
        AsyncEmbedPatcher,
        AsyncRerankPatcher,
        AsyncTranscriptionsCreatePatcher,
        ChatPatcher,
        ChatStreamPatcher,
        EmbedPatcher,
        RerankPatcher,
        TranscriptionsCreatePatcher,
    )

    if _client_is_async(client):
        for patcher in (AsyncChatPatcher, AsyncChatStreamPatcher, AsyncEmbedPatcher, AsyncRerankPatcher):
            patcher.wrap_target(client)
        _patch_audio_transcriptions(client, AsyncTranscriptionsCreatePatcher)
    else:
        for patcher in (ChatPatcher, ChatStreamPatcher, EmbedPatcher, RerankPatcher):
            patcher.wrap_target(client)
        _patch_audio_transcriptions(client, TranscriptionsCreatePatcher)


def _patch_audio_transcriptions(client: Any, patcher: Any) -> None:
    """Patch ``client.audio.transcriptions.create`` for manual ``wrap_cohere``.

    ``client.audio`` is a lazy Cohere property that constructs an
    ``AudioClient`` on first access, and ``.transcriptions`` in turn
    constructs a ``TranscriptionsClient``.  We instrument the leaf
    ``create`` method on the instance so per-client wrapping does not
    affect global class state.

    Skips silently on older Cohere SDKs that predate the audio surface.
    """
    audio = getattr(client, "audio", None)
    if audio is None:
        return
    transcriptions = getattr(audio, "transcriptions", None)
    if transcriptions is None:
        return
    patcher.wrap_target(transcriptions)


def _patch_v2_client(client: Any) -> None:
    from .patchers import (
        AsyncV2ChatPatcher,
        AsyncV2ChatStreamPatcher,
        AsyncV2EmbedPatcher,
        AsyncV2RerankPatcher,
        V2ChatPatcher,
        V2ChatStreamPatcher,
        V2EmbedPatcher,
        V2RerankPatcher,
    )

    if _client_is_async(client):
        for patcher in (AsyncV2ChatPatcher, AsyncV2ChatStreamPatcher, AsyncV2EmbedPatcher, AsyncV2RerankPatcher):
            patcher.wrap_target(client)
    else:
        for patcher in (V2ChatPatcher, V2ChatStreamPatcher, V2EmbedPatcher, V2RerankPatcher):
            patcher.wrap_target(client)


def _is_cohere_v1_client(client: Any) -> bool:
    type_name = type(client).__name__
    return type_name in {"Client", "AsyncClient"}


def _is_cohere_v2_client(client: Any) -> bool:
    type_name = type(client).__name__
    return type_name in {"ClientV2", "AsyncClientV2", "V2Client", "AsyncV2Client"}


def wrap_cohere(client: Any) -> Any:
    """Wrap a Cohere client instance for Braintrust tracing.

    Supports ``cohere.Client``, ``cohere.AsyncClient``, ``cohere.ClientV2`` and
    ``cohere.AsyncClientV2``.  Wrapping is idempotent; a client is returned
    unchanged if it's already traced or if it's not a recognized Cohere client.
    """
    if client is None:
        return client
    if getattr(client, _BRAINTRUST_TRACED_COHERE, False):
        return client

    if _is_cohere_v1_client(client):
        _patch_v1_client(client)
        # Also instrument the nested v2 client exposed on v1 clients.
        nested_v2 = getattr(client, "v2", None)
        if nested_v2 is not None and _is_cohere_v2_client(nested_v2):
            _patch_v2_client(nested_v2)
            setattr(nested_v2, _BRAINTRUST_TRACED_COHERE, True)
        setattr(client, _BRAINTRUST_TRACED_COHERE, True)
        return client

    if _is_cohere_v2_client(client):
        _patch_v2_client(client)
        setattr(client, _BRAINTRUST_TRACED_COHERE, True)
        return client

    logger.warning("Unsupported Cohere client %s; not wrapping.", type(client).__name__)
    return client
