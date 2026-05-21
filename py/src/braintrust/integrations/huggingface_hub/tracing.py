"""HuggingFace Hub-specific tracing helpers.

Span shape:

- span ``name``: ``huggingface.<task>`` for non-streaming, ``huggingface.<task>_stream``
  for streaming variants. Covered tasks: ``chat_completion``, ``text_generation``,
  ``feature_extraction``, and ``sentence_similarity``.
- ``input``: ``messages`` (chat), ``prompt`` (text generation), the raw input
  string/list (feature extraction), or ``{sentence, other_sentences}``
  (sentence similarity).
- ``output``: the SDK ``choices`` list verbatim for chat (or the aggregated
  ``{choices: [...]}`` shape for streaming), ``{generated_text, finish_reason?}``
  for text generation, and an ``{embedding_count, embedding_length, ...}``
  summary for feature extraction.
- ``metadata``: ``provider`` is always present (defaults to ``"huggingface"``;
  the user's ``provider=`` kwarg, when given, overrides the default so the
  span reflects the actual routing target). Other allowlisted request params
  (``model``, ``temperature``, ``max_tokens``, ``tools``, ...) and response-
  level fields (``id``, ``model``, ``object``, ``created``, ``finish_reason``)
  are merged in.
- ``metrics``: timing plus ``prompt_tokens`` / ``completion_tokens`` / ``tokens``
  from the response ``usage`` block, with text-generation falling back to
  ``details.prefill`` length and ``details.generated_tokens``.
"""

import logging
import time
from typing import Any, Protocol

from braintrust.integrations.utils import (
    _log_and_end_span,
    _log_error_and_end_span,
    _normalize_chat_messages,
    _timing_metrics,
    _try_to_dict,
)
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import is_numeric


logger = logging.getLogger(__name__)


# Default ``metadata.provider`` value; the user's ``provider=`` kwarg, when
# given, overrides this so the span reflects the actual HuggingFace routing
# target (e.g. ``"cerebras"``, ``"together"``).
_PROVIDER = "huggingface"


# Subset of chat_completion kwargs that should appear under span metadata.
_CHAT_METADATA_KEYS = (
    "model",
    "provider",
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
    "stream",
    "tools",
    "tool_choice",
    "tool_prompt",
    "response_format",
    "logprobs",
    "top_logprobs",
    "n",
    "logit_bias",
    "stream_options",
    "extra_body",
)

# Subset of text_generation kwargs that should appear under span metadata.
_TEXT_GENERATION_METADATA_KEYS = (
    "model",
    "provider",
    "details",
    "adapter_id",
    "best_of",
    "decoder_input_details",
    "do_sample",
    "frequency_penalty",
    "grammar",
    "max_new_tokens",
    "repetition_penalty",
    "return_full_text",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_k",
    "top_n_tokens",
    "top_p",
    "truncate",
    "typical_p",
    "watermark",
)

_FEATURE_EXTRACTION_METADATA_KEYS = (
    "model",
    "provider",
    "normalize",
    "prompt_name",
    "truncate",
    "truncation_direction",
    "dimensions",
    "encoding_format",
)

_SENTENCE_SIMILARITY_METADATA_KEYS = (
    "model",
    "provider",
)

_RESPONSE_METADATA_KEYS = (
    "id",
    "model",
    "object",
    "created",
)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _get_field(obj: Any, key: str) -> Any:
    """Return a field from either a mapping or a HuggingFace SDK model object.

    The inference SDK returns dataclass-like objects (``BaseInferenceType``)
    for most responses, but ``parse_obj_as_instance`` may surface them as
    plain dicts when fields don't match the schema, and streaming chunks may
    arrive as either.  Centralizing the access keeps the rest of the module
    indifferent to the runtime shape.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first_nonempty_str(*candidates: Any, default: str | None = None) -> str | None:
    """Return the first non-empty string in *candidates*, else *default*."""
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return default


def _resolve_provider_and_model(kwargs: dict[str, Any], instance: Any) -> tuple[str, str | None]:
    """Return (provider, model) using request kwargs first, instance second.

    The HuggingFace Python SDK lets callers pin a default ``provider`` and
    ``model`` on the ``InferenceClient`` instance and override them per-call.
    Spans should mirror what actually went on the wire: per-call values win
    over instance defaults, with ``"huggingface"`` as the final fallback for
    provider so the integration identity is always present. ``model`` has no
    such fallback — it stays ``None`` when neither side supplies one.
    """
    provider = _first_nonempty_str(
        kwargs.get("provider"),
        _get_field(instance, "provider"),
        default=_PROVIDER,
    )
    model = _first_nonempty_str(
        kwargs.get("model"),
        _get_field(instance, "model"),
    )
    return provider, model


def _pick_allowed_metadata(kwargs: dict[str, Any], allowlist: tuple[str, ...]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in allowlist:
        value = kwargs.get(key)
        if value is None:
            continue
        metadata[key] = value
    return metadata


def _build_request_metadata(
    kwargs: dict[str, Any],
    allowlist: tuple[str, ...],
    instance: Any,
) -> dict[str, Any]:
    """Build span metadata from request kwargs + instance defaults.

    Allowlisted request params are copied verbatim, then ``provider`` and
    ``model`` are resolved with kwarg-over-instance-over-default precedence.
    """
    metadata = _pick_allowed_metadata(kwargs, allowlist)
    provider, model = _resolve_provider_and_model(kwargs, instance)
    metadata["provider"] = provider
    if model is not None and "model" not in metadata:
        metadata["model"] = model
    return metadata


def _extract_response_metadata(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    metadata: dict[str, Any] = {}
    for key in _RESPONSE_METADATA_KEYS:
        value = _get_field(result, key)
        if value is not None:
            metadata[key] = value

    choices = _get_field(result, "choices")
    if isinstance(choices, list) and choices:
        finish_reason = _get_field(choices[0], "finish_reason")
        if isinstance(finish_reason, str):
            metadata["finish_reason"] = finish_reason
    return metadata


def _parse_usage_metrics(result: Any) -> dict[str, float]:
    """Extract token usage from a chat or text-generation response."""
    if result is None:
        return {}

    metrics: dict[str, float] = {}
    usage = _get_field(result, "usage")
    if usage is None:
        return metrics

    for key, metric in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "tokens"),
    ):
        value = _get_field(usage, key)
        if is_numeric(value):
            metrics[metric] = float(value)

    if "tokens" not in metrics and "prompt_tokens" in metrics and "completion_tokens" in metrics:
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    return metrics


def _text_generation_metrics(details: Any) -> dict[str, float]:
    """Extract metrics from a ``TextGenerationOutput.details`` payload.

    - ``prompt_tokens`` from ``details.prefill`` length when available.
    - ``completion_tokens`` from ``details.generated_tokens`` (or the length
      of ``details.tokens`` if ``generated_tokens`` is missing).
    - ``tokens`` = ``prompt_tokens + completion_tokens`` when either side is
      known (missing side counted as 0).
    """
    if details is None:
        return {}

    prefill = _get_field(details, "prefill")
    prompt_tokens: float | None = None
    if isinstance(prefill, list):
        prompt_tokens = float(len(prefill))

    completion_tokens: float | None = None
    generated_tokens = _get_field(details, "generated_tokens")
    if is_numeric(generated_tokens):
        completion_tokens = float(generated_tokens)
    else:
        tokens_list = _get_field(details, "tokens")
        if isinstance(tokens_list, list):
            completion_tokens = float(len(tokens_list))

    metrics: dict[str, float] = {}
    if prompt_tokens is not None:
        metrics["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        metrics["completion_tokens"] = completion_tokens
    if prompt_tokens is not None or completion_tokens is not None:
        metrics["tokens"] = (prompt_tokens or 0.0) + (completion_tokens or 0.0)
    return metrics


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------


def _chat_output(result: Any) -> Any:
    """Return the chat response's ``choices`` list verbatim.

    The full ``choices`` array, with each entry's ``message`` / ``finish_reason``
    intact, is what the span logs as ``output``. This keeps tool calls,
    logprobs, multiple choices, and any future fields available to consumers
    without extra normalization.
    """
    if result is None:
        return None
    choices = _get_field(result, "choices")
    if not isinstance(choices, list):
        return None
    return choices


def _text_generation_output(result: Any) -> Any:
    """Return ``{generated_text: ...}`` for any text-generation response shape.

    Always emit an object so consumers can rely on a stable key regardless of
    whether ``details=True`` was passed.  When ``details=False`` the SDK
    returns a plain ``str``; we wrap it.  When ``details=True`` we pull
    ``generated_text`` from the ``TextGenerationOutput``.
    """
    if result is None:
        return None
    if isinstance(result, str):
        return {"generated_text": result}
    generated_text = _get_field(result, "generated_text")
    if isinstance(generated_text, str):
        return {"generated_text": generated_text}
    return result


def _feature_extraction_output(result: Any) -> dict[str, Any] | None:
    """Summarize a feature-extraction numpy array or nested list.

    - 1D vector → ``{embedding_length}``
    - 2D matrix → ``{embedding_count, embedding_length}``
    - 3D batch  → ``{embedding_batch_count, embedding_count, embedding_length}``

    Returning a summary instead of the raw array keeps spans readable for
    payloads that can be many KB of floats.
    """
    if result is None:
        return None

    shape = getattr(result, "shape", None)
    if shape is not None:
        try:
            dims = len(shape)
            if dims == 1:
                return {"embedding_length": int(shape[0])}
            if dims == 2:
                return {"embedding_count": int(shape[0]), "embedding_length": int(shape[1])}
            if dims >= 3:
                return {
                    "embedding_batch_count": int(shape[0]),
                    "embedding_count": int(shape[1]),
                    "embedding_length": int(shape[-1]),
                }
        except (TypeError, ValueError):
            pass

    if isinstance(result, list):
        if not result:
            return None
        first = result[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            return {
                "embedding_batch_count": len(result),
                "embedding_count": len(first),
                "embedding_length": len(first[0]),
            }
        if isinstance(first, list):
            return {"embedding_count": len(result), "embedding_length": len(first)}
        return {"embedding_length": len(result)}

    return None


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _chat_input(args: tuple, kwargs: dict[str, Any]) -> Any:
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]
    return _normalize_chat_messages(messages)


def _text_generation_input(args: tuple, kwargs: dict[str, Any]) -> Any:
    prompt = kwargs.get("prompt")
    if prompt is None and args:
        prompt = args[0]
    return prompt


def _feature_extraction_input(args: tuple, kwargs: dict[str, Any]) -> Any:
    text = kwargs.get("text")
    if text is None and args:
        text = args[0]
    return text


def _sentence_similarity_input(args: tuple, kwargs: dict[str, Any]) -> Any:
    """Normalize the (sentence, other_sentences) inputs from args/kwargs.

    ``sentence_similarity`` accepts ``sentence`` positionally; callers may
    pass it positionally or by name. Use ``args`` to capture the positional
    form, then fall back to ``kwargs``.
    """
    sentence = kwargs.get("sentence")
    other_sentences = kwargs.get("other_sentences")
    if sentence is None and args:
        sentence = args[0]
    if other_sentences is None and len(args) > 1:
        other_sentences = args[1]
    return {"sentence": sentence, "other_sentences": other_sentences}


# ---------------------------------------------------------------------------
# Stream aggregation — chat_completion
# ---------------------------------------------------------------------------


def _merge_tool_call_delta(
    tool_calls_by_index: dict[int, dict[str, Any]],
    delta_tool_calls: Any,
) -> None:
    """Merge a chat-completion streaming tool-call delta into the accumulator.

    Each delta contains ``index``, ``id``, ``type`` and a ``function`` block
    whose ``arguments`` string is delivered piecewise across chunks.  The
    accumulator keeps function ``name`` once seen and concatenates the
    incremental ``arguments`` strings. Tool-call entries are emitted in
    ``sorted(index)`` order downstream, so no explicit order list is needed
    here.
    """
    if not isinstance(delta_tool_calls, list):
        return
    for entry in delta_tool_calls:
        entry_dict = entry if isinstance(entry, dict) else _try_to_dict(entry)
        if not isinstance(entry_dict, dict):
            continue
        index = entry_dict.get("index")
        if not isinstance(index, int):
            index = len(tool_calls_by_index)
        existing = tool_calls_by_index.get(index, {})
        # Carry forward static fields (id, type) when first seen.
        for key in ("id", "type"):
            value = entry_dict.get(key)
            if value is not None:
                existing[key] = value
        incoming_fn = entry_dict.get("function")
        if isinstance(incoming_fn, dict):
            existing_fn = existing.get("function") or {}
            name = incoming_fn.get("name")
            if name is not None:
                existing_fn["name"] = name
            args_delta = incoming_fn.get("arguments")
            if isinstance(args_delta, str):
                existing_fn["arguments"] = f"{existing_fn.get('arguments', '') or ''}{args_delta}"
            existing["function"] = existing_fn
        tool_calls_by_index[index] = existing


class _ChatStreamAggregator:
    """Incrementally aggregate streamed chat-completion chunks."""

    def __init__(self) -> None:
        # ``dict`` preserves insertion order (Python 3.7+), so iterating these
        # accumulators in their natural order gives us first-seen choice order.
        self.choices_by_index: dict[int, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}

    def record(self, chunk: Any) -> None:
        if chunk is None:
            return
        for key in _RESPONSE_METADATA_KEYS:
            value = _get_field(chunk, key)
            if value is not None and key not in self.metadata:
                self.metadata[key] = value

        choices = _get_field(chunk, "choices")
        if isinstance(choices, list):
            for choice in choices:
                idx_raw = _get_field(choice, "index")
                index = idx_raw if isinstance(idx_raw, int) else 0
                acc = self.choices_by_index.setdefault(
                    index,
                    {
                        "content_parts": [],
                        "role": None,
                        "finish_reason": None,
                        "tool_calls_by_index": {},
                    },
                )

                delta = _get_field(choice, "delta")
                content = _get_field(delta, "content")
                if isinstance(content, str) and content:
                    acc["content_parts"].append(content)

                role_value = _get_field(delta, "role")
                if isinstance(role_value, str):
                    acc["role"] = role_value

                _merge_tool_call_delta(acc["tool_calls_by_index"], _get_field(delta, "tool_calls"))

                fr = _get_field(choice, "finish_reason")
                if isinstance(fr, str):
                    acc["finish_reason"] = fr

        self.metrics.update(_parse_usage_metrics(chunk))

    def finish(self) -> tuple[Any, dict[str, float], dict[str, Any]]:
        aggregated_choices: list[dict[str, Any]] = []
        for index, acc in self.choices_by_index.items():
            tool_calls_by_index = acc["tool_calls_by_index"]
            tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
            message: dict[str, Any] = {
                "content": "".join(acc["content_parts"]),
                "role": acc["role"] or "assistant",
            }
            if tool_calls:
                message["tool_calls"] = tool_calls
            entry: dict[str, Any] = {"index": index, "message": message}
            if acc["finish_reason"] is not None:
                entry["finish_reason"] = acc["finish_reason"]
            aggregated_choices.append(entry)

        for entry in aggregated_choices:
            if "finish_reason" in entry:
                self.metadata["finish_reason"] = entry["finish_reason"]
                break

        return {"choices": aggregated_choices}, self.metrics, self.metadata


def _aggregate_chat_stream(chunks: list[Any]) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """Aggregate streamed chat-completion chunks into (output, metrics, metadata)."""
    aggregator = _ChatStreamAggregator()
    for chunk in chunks:
        aggregator.record(chunk)
    return aggregator.finish()


# ---------------------------------------------------------------------------
# Stream aggregation — text_generation
# ---------------------------------------------------------------------------


class _TextGenerationStreamAggregator:
    """Incrementally aggregate streamed text-generation chunks."""

    def __init__(self) -> None:
        self.pieces: list[str] = []
        self.final_text: str | None = None
        self.last_details: Any = None
        self.metadata: dict[str, Any] = {}

    def record(self, chunk: Any) -> None:
        if chunk is None:
            return
        if isinstance(chunk, str):
            self.pieces.append(chunk)
            return

        token_text = _get_field(_get_field(chunk, "token"), "text")
        if isinstance(token_text, str):
            self.pieces.append(token_text)

        generated_text = _get_field(chunk, "generated_text")
        if isinstance(generated_text, str):
            self.final_text = generated_text

        details = _get_field(chunk, "details")
        if details is not None:
            self.last_details = details
            self.metadata.update(_text_generation_extra_metadata(details))

    def finish(self) -> tuple[Any, dict[str, float], dict[str, Any]]:
        text = self.final_text if self.final_text is not None else "".join(self.pieces)
        output: dict[str, Any] = {"generated_text": text}
        if "finish_reason" in self.metadata:
            output["finish_reason"] = self.metadata["finish_reason"]

        return output, _text_generation_metrics(self.last_details), self.metadata


def _aggregate_text_generation_stream(chunks: list[Any]) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """Aggregate streamed text-generation chunks into (output, metrics, metadata)."""
    aggregator = _TextGenerationStreamAggregator()
    for chunk in chunks:
        aggregator.record(chunk)
    return aggregator.finish()


# ---------------------------------------------------------------------------
# Span creation / wrappers
# ---------------------------------------------------------------------------


def _start_span(name: str, span_input: Any, metadata: dict[str, Any]):
    return start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=span_input,
        metadata=metadata,
    )


def _chat_metadata(kwargs: dict[str, Any], instance: Any) -> dict[str, Any]:
    return _build_request_metadata(kwargs, _CHAT_METADATA_KEYS, instance)


def _text_generation_metadata(kwargs: dict[str, Any], instance: Any) -> dict[str, Any]:
    return _build_request_metadata(kwargs, _TEXT_GENERATION_METADATA_KEYS, instance)


def _feature_extraction_metadata(kwargs: dict[str, Any], instance: Any) -> dict[str, Any]:
    return _build_request_metadata(kwargs, _FEATURE_EXTRACTION_METADATA_KEYS, instance)


def _sentence_similarity_metadata(kwargs: dict[str, Any], instance: Any) -> dict[str, Any]:
    return _build_request_metadata(kwargs, _SENTENCE_SIMILARITY_METADATA_KEYS, instance)


def _is_streaming(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("stream"))


# ---- chat_completion -------------------------------------------------------


def _chat_span_name(streaming: bool) -> str:
    return "huggingface.chat_completion_stream" if streaming else "huggingface.chat_completion"


def _chat_completion_wrapper(wrapped, instance, args, kwargs):
    span_input = _chat_input(args, kwargs)
    metadata = _chat_metadata(kwargs, instance)
    streaming = _is_streaming(kwargs)
    span = _start_span(_chat_span_name(streaming), span_input, metadata)
    start_time = time.time()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if streaming:
        return _TracedStream(result, span, start_time, _ChatStreamAggregator())

    _log_chat_result(span, start_time, result)
    return result


async def _async_chat_completion_wrapper(wrapped, instance, args, kwargs):
    span_input = _chat_input(args, kwargs)
    metadata = _chat_metadata(kwargs, instance)
    streaming = _is_streaming(kwargs)
    span = _start_span(_chat_span_name(streaming), span_input, metadata)
    start_time = time.time()

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if streaming:
        return _AsyncTracedStream(result, span, start_time, _ChatStreamAggregator())

    _log_chat_result(span, start_time, result)
    return result


def _log_chat_result(span, start_time: float, result: Any) -> None:
    metrics = {
        **_timing_metrics(start_time, time.time()),
        **_parse_usage_metrics(result),
    }
    _log_and_end_span(
        span,
        output=_chat_output(result),
        metrics=metrics,
        metadata=_extract_response_metadata(result) or None,
    )


# ---- text_generation -------------------------------------------------------


def _text_generation_span_name(streaming: bool) -> str:
    return "huggingface.text_generation_stream" if streaming else "huggingface.text_generation"


def _text_generation_wrapper(wrapped, instance, args, kwargs):
    span_input = _text_generation_input(args, kwargs)
    metadata = _text_generation_metadata(kwargs, instance)
    streaming = _is_streaming(kwargs)
    span = _start_span(_text_generation_span_name(streaming), span_input, metadata)
    start_time = time.time()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if streaming:
        return _TracedStream(result, span, start_time, _TextGenerationStreamAggregator())

    _log_text_generation_result(span, start_time, result)
    return result


async def _async_text_generation_wrapper(wrapped, instance, args, kwargs):
    span_input = _text_generation_input(args, kwargs)
    metadata = _text_generation_metadata(kwargs, instance)
    streaming = _is_streaming(kwargs)
    span = _start_span(_text_generation_span_name(streaming), span_input, metadata)
    start_time = time.time()

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if streaming:
        return _AsyncTracedStream(result, span, start_time, _TextGenerationStreamAggregator())

    _log_text_generation_result(span, start_time, result)
    return result


def _text_generation_extra_metadata(details: Any) -> dict[str, Any]:
    """Pull ``finish_reason`` / ``seed`` from a text-generation ``details`` payload.

    Shared by the non-streaming and streaming code paths so the two stay in
    sync when new ``details`` fields are added.
    """
    metadata: dict[str, Any] = {}
    finish_reason = _get_field(details, "finish_reason")
    if isinstance(finish_reason, str):
        metadata["finish_reason"] = finish_reason
    seed = _get_field(details, "seed")
    if seed is not None:
        metadata["seed"] = seed
    return metadata


def _log_text_generation_result(span, start_time: float, result: Any) -> None:
    details = _get_field(result, "details")
    metrics = {
        **_timing_metrics(start_time, time.time()),
        **_text_generation_metrics(details),
    }
    metadata = _text_generation_extra_metadata(details)
    output = _text_generation_output(result)
    # Include ``finish_reason`` inside the output dict when known, so
    # consumers get the same field whether they look in ``span.output`` or
    # ``span.metadata``.
    if isinstance(output, dict) and "finish_reason" in metadata:
        output = {**output, "finish_reason": metadata["finish_reason"]}
    _log_and_end_span(
        span,
        output=output,
        metrics=metrics,
        metadata=metadata or None,
    )


# ---- feature_extraction ----------------------------------------------------


def _feature_extraction_wrapper(wrapped, instance, args, kwargs):
    span_input = _feature_extraction_input(args, kwargs)
    metadata = _feature_extraction_metadata(kwargs, instance)
    span = _start_span("huggingface.feature_extraction", span_input, metadata)
    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    metrics = _timing_metrics(start_time, time.time())
    _log_and_end_span(span, output=_feature_extraction_output(result), metrics=metrics)
    return result


async def _async_feature_extraction_wrapper(wrapped, instance, args, kwargs):
    span_input = _feature_extraction_input(args, kwargs)
    metadata = _feature_extraction_metadata(kwargs, instance)
    span = _start_span("huggingface.feature_extraction", span_input, metadata)
    start_time = time.time()
    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    metrics = _timing_metrics(start_time, time.time())
    _log_and_end_span(span, output=_feature_extraction_output(result), metrics=metrics)
    return result


# ---- sentence_similarity ---------------------------------------------------


def _sentence_similarity_wrapper(wrapped, instance, args, kwargs):
    span_input = _sentence_similarity_input(args, kwargs)
    metadata = _sentence_similarity_metadata(kwargs, instance)
    span = _start_span("huggingface.sentence_similarity", span_input, metadata)
    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    metrics = _timing_metrics(start_time, time.time())
    _log_and_end_span(span, output=result, metrics=metrics)
    return result


async def _async_sentence_similarity_wrapper(wrapped, instance, args, kwargs):
    span_input = _sentence_similarity_input(args, kwargs)
    metadata = _sentence_similarity_metadata(kwargs, instance)
    span = _start_span("huggingface.sentence_similarity", span_input, metadata)
    start_time = time.time()
    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    metrics = _timing_metrics(start_time, time.time())
    _log_and_end_span(span, output=result, metrics=metrics)
    return result


# ---------------------------------------------------------------------------
# Traced stream iterators
# ---------------------------------------------------------------------------


class _Aggregator(Protocol):
    def record(self, event: Any) -> None: ...

    def finish(self) -> tuple[Any, dict[str, float], dict[str, Any]]: ...


class _BaseTracedStream:
    """Shared bookkeeping for the sync/async traced stream wrappers.

    Accepts an *aggregator* callable so the same wrapper class can carry
    chat-completion and text-generation streams without per-task subclasses.
    """

    def __init__(self, iterator: Any, span: Any, start_time: float, aggregator: _Aggregator):
        self._iterator = iterator
        self._span = span
        self._start_time = start_time
        self._aggregator = aggregator
        self._first_token_time: float | None = None
        self._finished = False

    def _record(self, event: Any) -> None:
        if self._first_token_time is None:
            self._first_token_time = time.time()
        self._aggregator.record(event)

    def _finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        if error is not None:
            _log_error_and_end_span(self._span, error)
            return

        output, usage_metrics, extra_metadata = self._aggregator.finish()
        metrics = {
            **_timing_metrics(self._start_time, time.time(), self._first_token_time),
            **usage_metrics,
        }
        _log_and_end_span(
            self._span,
            output=output,
            metrics=metrics,
            metadata=extra_metadata or None,
        )


class _TracedStream(_BaseTracedStream):
    """Wrap a sync provider iterator and log the aggregated span on exhaustion.

    Supports three completion paths so spans always finalize:

    1. Iterator exhaustion — ``StopIteration`` in :meth:`__next__` calls
       :meth:`_finish` with the aggregated payload.
    2. Early exit — ``close()`` (or ``with`` block exit) finalizes the span
       with whatever chunks have been collected so callers can ``break`` out
       of the loop without leaking the span.
    3. GC backstop — ``__del__`` calls ``close()`` for callers who never
       finalize the iterator explicitly. ``__del__`` runs only when CPython
       reclaims the iterator, which is best-effort but covers most leaks.
    """

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

    def close(self) -> None:
        """Finalize the span without consuming more chunks."""
        self._finish()
        inner_close = getattr(self._iterator, "close", None)
        if callable(inner_close):
            try:
                inner_close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self._finish(error=exc)
        else:
            self.close()
        return False

    def __del__(self):
        # Best-effort backstop; never raise out of ``__del__``.
        try:
            self.close()
        except Exception:
            pass


class _AsyncTracedStream(_BaseTracedStream):
    """Async counterpart of :class:`_TracedStream`.

    See :class:`_TracedStream` for the finalization contract; ``aclose`` /
    ``async with`` is the async-safe way to finalize on early exit.
    """

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

    async def aclose(self) -> None:
        """Finalize the span without consuming more chunks."""
        self._finish()
        inner_aclose = getattr(self._iterator, "aclose", None)
        if callable(inner_aclose):
            try:
                await inner_aclose()
            except Exception:
                pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc is not None:
            self._finish(error=exc)
        else:
            await self.aclose()
        return False

    def __del__(self):
        # GC backstop for callers that never iterate to completion. Avoids
        # touching the event loop — we only finalize the in-memory span and
        # leave the underlying async iterator for the runtime to clean up.
        try:
            self._finish()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Manual wrapping — `wrap_huggingface_hub(client)`
# ---------------------------------------------------------------------------

_BRAINTRUST_TRACED_HF_HUB = "__braintrust_huggingface_hub_traced__"


def _client_is_async(client: Any) -> bool:
    return type(client).__name__.startswith("Async")


def _is_inference_client(client: Any) -> bool:
    return type(client).__name__ in {"InferenceClient", "AsyncInferenceClient"}


def wrap_huggingface_hub(client: Any) -> Any:
    """Wrap an ``InferenceClient`` or ``AsyncInferenceClient`` for Braintrust tracing.

    The wrapped client traces ``chat_completion``, ``text_generation``,
    ``feature_extraction``, and ``sentence_similarity``.  ``client.chat.completions.create``
    is covered automatically because the HuggingFace SDK implements it as a
    thin proxy to ``chat_completion``.

    Wrapping is idempotent; a client is returned unchanged if it's already
    traced or if it's not a recognized HuggingFace inference client.
    """
    from .patchers import (
        AsyncChatCompletionPatcher,
        AsyncFeatureExtractionPatcher,
        AsyncSentenceSimilarityPatcher,
        AsyncTextGenerationPatcher,
        ChatCompletionPatcher,
        FeatureExtractionPatcher,
        SentenceSimilarityPatcher,
        TextGenerationPatcher,
    )

    if client is None:
        return client
    if getattr(client, _BRAINTRUST_TRACED_HF_HUB, False):
        return client
    if not _is_inference_client(client):
        logger.warning("Unsupported HuggingFace inference client %s; not wrapping.", type(client).__name__)
        return client

    if _client_is_async(client):
        patchers = (
            AsyncChatCompletionPatcher,
            AsyncTextGenerationPatcher,
            AsyncFeatureExtractionPatcher,
            AsyncSentenceSimilarityPatcher,
        )
    else:
        patchers = (
            ChatCompletionPatcher,
            TextGenerationPatcher,
            FeatureExtractionPatcher,
            SentenceSimilarityPatcher,
        )

    for patcher in patchers:
        patcher.wrap_target(client)

    setattr(client, _BRAINTRUST_TRACED_HF_HUB, True)
    return client
