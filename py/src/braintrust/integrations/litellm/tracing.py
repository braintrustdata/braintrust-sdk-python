"""LiteLLM tracing helpers — spans, metadata extraction, stream handling."""

import logging
import time
from collections.abc import AsyncGenerator, Generator
from functools import lru_cache
from types import TracebackType
from typing import Any

from braintrust.integrations.utils import (
    _extract_audio_output,
    _materialize_attachment,
    _parse_openai_usage_metrics,
    _prettify_response_params,
    _ResolvedAttachment,
    _timing_metrics,
    _try_to_dict,
)
from braintrust.logger import Span
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import clean_nones, is_numeric, merge_dicts


_INSTRUMENTATION = "litellm-auto"

_log = logging.getLogger(__name__)


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


# LiteLLM's representation to Braintrust's representation
TOKEN_NAME_MAP: dict[str, str] = {
    # chat API
    "total_tokens": "tokens",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    # responses API
    "tokens": "tokens",
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
}

TOKEN_PREFIX_MAP: dict[str, str] = {
    "input": "prompt",
    "output": "completion",
}


_RERANK_MAX_RESULTS = 100

# litellm reports the legacy /v1/completions endpoint under its own routing
# key, but pricing/billing is against the underlying provider — normalize so
# metadata.provider stays consistent across chat and text-completion calls.
_PROVIDER_ALIASES: dict[str, str] = {
    "text-completion-openai": "openai",
    "azure_text": "azure",
    "text-completion-codestral": "codestral",
    "text-completion-inception": "inception",
}


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* off *obj* whether it's a dict or an SDK/pydantic object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@lru_cache(maxsize=256)
def _resolve_provider(model: str) -> str:
    # Lazy import: tracing.py loads eagerly via auto.py even when litellm isn't installed.
    try:
        import litellm

        _, provider, _, _ = litellm.get_llm_provider(model)
        if isinstance(provider, str) and provider:
            return _PROVIDER_ALIASES.get(provider, provider)
    except Exception:
        _log.debug("litellm.get_llm_provider failed for model=%r", model, exc_info=True)
    return "litellm"


# ---------------------------------------------------------------------------
# Async response wrapper (preserves async context manager / iterator behavior)
# ---------------------------------------------------------------------------


class AsyncResponseWrapper:
    """Wrapper that properly preserves async context manager behavior for LiteLLM responses."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def __aenter__(self) -> Any:
        if hasattr(self._response, "__aenter__"):
            return await self._response.__aenter__()
        return self._response

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> bool | None:
        if hasattr(self._response, "__aexit__"):
            return await self._response.__aexit__(exc_type, exc_val, exc_tb)
        return None

    def __aiter__(self) -> AsyncGenerator[Any, None]:
        if hasattr(self._response, "__aiter__"):
            return self._response.__aiter__()
        raise TypeError("Response object is not an async iterator")

    async def __anext__(self) -> Any:
        if hasattr(self._response, "__anext__"):
            return await self._response.__anext__()
        raise StopAsyncIteration

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _handle_completion_streaming(
    raw_response: Any, span: Span, start_time: float, is_async: bool = False
) -> AsyncResponseWrapper | Generator[Any, None, None]:
    """Handle streaming response for completion (sync and async)."""
    if is_async:

        async def async_gen() -> AsyncGenerator[Any, None]:
            try:
                first = True
                all_results: list[dict[str, Any]] = []
                async for item in raw_response:
                    if first:
                        span.log(metrics={"time_to_first_token": time.time() - start_time})
                        first = False
                    all_results.append(_try_to_dict(item))
                    yield item

                span.log(**_postprocess_completion_streaming_results(all_results))
            finally:
                span.end()

        return AsyncResponseWrapper(async_gen())
    else:

        def sync_gen() -> Generator[Any, None, None]:
            try:
                first = True
                all_results: list[dict[str, Any]] = []
                for item in raw_response:
                    if first:
                        span.log(metrics={"time_to_first_token": time.time() - start_time})
                        first = False
                    all_results.append(_try_to_dict(item))
                    yield item

                span.log(**_postprocess_completion_streaming_results(all_results))
            finally:
                span.end()

        return sync_gen()


def _handle_responses_streaming(
    raw_response: Any, span: Span, start_time: float, is_async: bool = False
) -> AsyncResponseWrapper | Generator[Any, None, None]:
    """Handle streaming response for responses API (sync and async)."""
    if is_async:

        async def async_gen() -> AsyncGenerator[Any, None]:
            try:
                first = True
                all_results: list[Any] = []
                async for item in raw_response:
                    if first:
                        span.log(metrics={"time_to_first_token": time.time() - start_time})
                        first = False
                    all_results.append(item)
                    yield item

                span.log(**_postprocess_responses_streaming_results(all_results))
            finally:
                span.end()

        return AsyncResponseWrapper(async_gen())
    else:

        def sync_gen() -> Generator[Any, None, None]:
            try:
                first = True
                all_results: list[Any] = []
                for item in raw_response:
                    if first:
                        span.log(metrics={"time_to_first_token": time.time() - start_time})
                        first = False
                    all_results.append(item)
                    yield item

                span.log(**_postprocess_responses_streaming_results(all_results))
            finally:
                span.end()

        return sync_gen()


# ---------------------------------------------------------------------------
# wrapt-style wrapper functions (used by FunctionWrapperPatcher)
# ---------------------------------------------------------------------------


def _completion_wrapper_impl(wrapped, args, kwargs, *, input_key: str):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key=input_key)
    is_streaming = kwargs.get("stream", False)

    span = start_span(
        **merge_dicts(dict(name="Completion", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    )
    should_end = True

    try:
        start = time.time()
        completion_response = wrapped(*args, **kwargs)

        if is_streaming:
            should_end = False
            return _handle_completion_streaming(completion_response, span, start, is_async=False)

        metrics = _parse_metrics_from_usage(_get_field(completion_response, "usage"))
        metrics["time_to_first_token"] = time.time() - start
        span.log(metrics=metrics, output=_get_field(completion_response, "choices"))
        return completion_response
    finally:
        if should_end:
            span.end()


async def _acompletion_wrapper_impl(wrapped, args, kwargs, *, input_key: str):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key=input_key)
    is_streaming = kwargs.get("stream", False)

    span = start_span(
        **merge_dicts(dict(name="Completion", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    )
    should_end = True

    try:
        start = time.time()
        completion_response = await wrapped(*args, **kwargs)

        if is_streaming:
            should_end = False
            return _handle_completion_streaming(completion_response, span, start, is_async=True)

        metrics = _parse_metrics_from_usage(_get_field(completion_response, "usage"))
        metrics["time_to_first_token"] = time.time() - start
        span.log(metrics=metrics, output=_get_field(completion_response, "choices"))
        return completion_response
    finally:
        if should_end:
            span.end()


def _completion_wrapper(wrapped, instance, args, kwargs):
    return _completion_wrapper_impl(wrapped, args, kwargs, input_key="messages")


def _text_completion_wrapper(wrapped, instance, args, kwargs):
    return _completion_wrapper_impl(wrapped, args, kwargs, input_key="prompt")


async def _acompletion_wrapper_async(wrapped, instance, args, kwargs):
    return await _acompletion_wrapper_impl(wrapped, args, kwargs, input_key="messages")


async def _atext_completion_wrapper_async(wrapped, instance, args, kwargs):
    return await _acompletion_wrapper_impl(wrapped, args, kwargs, input_key="prompt")


def _responses_wrapper(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")
    is_streaming = kwargs.get("stream", False)

    span = start_span(
        **merge_dicts(dict(name="Response", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    )
    should_end = True

    try:
        start = time.time()
        response = wrapped(*args, **kwargs)

        if is_streaming:
            should_end = False
            return _handle_responses_streaming(response, span, start, is_async=False)

        metrics = _parse_metrics_from_usage(_get_field(response, "usage"))
        metrics["time_to_first_token"] = time.time() - start
        span.log(metrics=metrics, output=_get_field(response, "output"))
        return response
    finally:
        if should_end:
            span.end()


async def _aresponses_wrapper_async(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")
    is_streaming = kwargs.get("stream", False)

    span = start_span(
        **merge_dicts(dict(name="Response", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    )
    should_end = True

    try:
        start = time.time()
        response = await wrapped(*args, **kwargs)

        if is_streaming:
            should_end = False
            return _handle_responses_streaming(response, span, start, is_async=True)

        metrics = _parse_metrics_from_usage(_get_field(response, "usage"))
        metrics["time_to_first_token"] = time.time() - start
        span.log(metrics=metrics, output=_get_field(response, "output"))
        return response
    finally:
        if should_end:
            span.end()


def _image_attachment_from_base64(
    data: Any, *, output_format: Any, index: int
) -> tuple[_ResolvedAttachment | None, int | None]:
    if not isinstance(data, str):
        return None, None

    extension = output_format if isinstance(output_format, str) and output_format else "png"
    mime_type = extension if "/" in extension else f"image/{extension}"
    resolved_attachment = _materialize_attachment(
        data,
        mime_type=mime_type,
        prefix=f"generated_image_{index}",
    )
    if resolved_attachment is None:
        return None, None
    return resolved_attachment, len(resolved_attachment.attachment.data)


def _extract_image_generation_output(response: Any) -> Any:
    images: list[dict[str, Any]] = []
    output_format = _get_field(response, "output_format")

    for index, image in enumerate(_get_field(response, "data") or []):
        image_entry = clean_nones({"revised_prompt": _get_field(image, "revised_prompt")})

        url = _get_field(image, "url")
        if isinstance(url, str):
            image_entry["image_url"] = {"url": url}

        b64_json = _get_field(image, "b64_json")
        resolved_attachment, image_size_bytes = _image_attachment_from_base64(
            b64_json,
            output_format=output_format,
            index=index,
        )
        if resolved_attachment is not None:
            image_entry.update(resolved_attachment.multimodal_part_payload)
            image_entry["image_size_bytes"] = image_size_bytes
            image_entry["mime_type"] = resolved_attachment.mime_type
        elif isinstance(b64_json, str):
            image_entry["b64_json_present"] = True

        images.append(image_entry)

    return clean_nones(
        {
            "created": _get_field(response, "created"),
            "background": _get_field(response, "background"),
            "output_format": output_format,
            "quality": _get_field(response, "quality"),
            "size": _get_field(response, "size"),
            "images_count": len(images),
            "images": images,
        }
    )


def _log_image_generation(span: Span, image_response: Any, start: float) -> None:
    metrics = _timing_metrics(start, time.time())
    metrics.update(_parse_metrics_from_usage(_get_field(image_response, "usage")))
    span.log(metrics=metrics, output=_extract_image_generation_output(image_response))


def _image_generation_wrapper(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="prompt")
    with start_span(
        **merge_dicts(
            dict(name="Image Generation", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload
        )
    ) as span:
        start = time.time()
        image_response = wrapped(*args, **kwargs)
        _log_image_generation(span, image_response, start)
        return image_response


async def _aimage_generation_wrapper_async(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="prompt")
    with start_span(
        **merge_dicts(
            dict(name="Image Generation", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload
        )
    ) as span:
        start = time.time()
        image_response = await wrapped(*args, **kwargs)
        _log_image_generation(span, image_response, start)
        return image_response


def _log_embedding_response(span: Span, embedding_response: Any) -> None:
    data = _get_field(embedding_response, "data") or []
    first = data[0] if data else None
    length = len(_get_field(first, "embedding") or [])
    span.log(
        metrics=_parse_metrics_from_usage(_get_field(embedding_response, "usage")),
        output={"embedding_length": length},
    )


def _embedding_wrapper(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")
    with start_span(
        **merge_dicts(dict(name="Embedding", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        embedding_response = wrapped(*args, **kwargs)
        _log_embedding_response(span, embedding_response)
        return embedding_response


async def _aembedding_wrapper_async(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")
    with start_span(
        **merge_dicts(dict(name="Embedding", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        embedding_response = await wrapped(*args, **kwargs)
        _log_embedding_response(span, embedding_response)
        return embedding_response


def _log_moderation_response(span: Span, moderation_response: Any) -> None:
    span.log(
        metrics=_parse_metrics_from_usage(_get_field(moderation_response, "usage")),
        output=_get_field(moderation_response, "results"),
    )


def _moderation_wrapper(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")
    with start_span(
        **merge_dicts(dict(name="Moderation", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        moderation_response = wrapped(*args, **kwargs)
        _log_moderation_response(span, moderation_response)
        return moderation_response


async def _amoderation_wrapper_async(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")
    with start_span(
        **merge_dicts(dict(name="Moderation", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        moderation_response = await wrapped(*args, **kwargs)
        _log_moderation_response(span, moderation_response)
        return moderation_response


def _speech_wrapper(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")
    with start_span(
        **merge_dicts(dict(name="Speech", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        start = time.time()
        speech_response = wrapped(*args, **kwargs)
        span.log(
            metrics=_timing_metrics(start, time.time()),
            output=_extract_audio_output(
                speech_response,
                response_format=kwargs.get("response_format"),
                prefix="generated_speech",
            ),
        )
        return speech_response


async def _aspeech_wrapper_async(wrapped, instance, args, kwargs):
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")
    with start_span(
        **merge_dicts(dict(name="Speech", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        start = time.time()
        speech_response = await wrapped(*args, **kwargs)
        span.log(
            metrics=_timing_metrics(start, time.time()),
            output=_extract_audio_output(
                speech_response,
                response_format=kwargs.get("response_format"),
                prefix="generated_speech",
            ),
        )
        return speech_response


def _rerank_wrapper(wrapped, instance, args, kwargs):
    updated_span_payload = _update_rerank_span_payload_from_params(kwargs)
    with start_span(
        **merge_dicts(dict(name="Rerank", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        start = time.time()
        rerank_response = wrapped(*args, **kwargs)
        metrics = _timing_metrics(start, time.time())
        metrics.update(_parse_rerank_metrics(rerank_response))
        span.log(metrics=metrics, output=_extract_rerank_output(rerank_response))
        return rerank_response


async def _arerank_wrapper_async(wrapped, instance, args, kwargs):
    updated_span_payload = _update_rerank_span_payload_from_params(kwargs)
    with start_span(
        **merge_dicts(dict(name="Rerank", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        start = time.time()
        rerank_response = await wrapped(*args, **kwargs)
        metrics = _timing_metrics(start, time.time())
        metrics.update(_parse_rerank_metrics(rerank_response))
        span.log(metrics=metrics, output=_extract_rerank_output(rerank_response))
        return rerank_response


def _transcription_wrapper(wrapped, instance, args, kwargs):
    updated_span_payload = _update_audio_span_payload_from_params(kwargs)
    with start_span(
        **merge_dicts(
            dict(name="Transcription", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload
        )
    ) as span:
        transcription_response = wrapped(*args, **kwargs)
        span.log(
            metrics=_parse_metrics_from_usage(_get_field(transcription_response, "usage")),
            output=_extract_transcription_text(transcription_response),
        )
        return transcription_response


async def _atranscription_wrapper_async(wrapped, instance, args, kwargs):
    updated_span_payload = _update_audio_span_payload_from_params(kwargs)
    with start_span(
        **merge_dicts(
            dict(name="Transcription", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload
        )
    ) as span:
        transcription_response = await wrapped(*args, **kwargs)
        span.log(
            metrics=_parse_metrics_from_usage(_get_field(transcription_response, "usage")),
            output=_extract_transcription_text(transcription_response),
        )
        return transcription_response


# ---------------------------------------------------------------------------
# Streaming post-processing
# ---------------------------------------------------------------------------


def _postprocess_completion_streaming_results(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Process streaming results to extract final response."""
    role = None
    content = None
    text = None
    tool_calls: list[Any] | None = None
    finish_reason = None
    metrics: dict[str, float] = {}

    for result in all_results:
        usage = result.get("usage")
        if usage:
            metrics.update(_parse_metrics_from_usage(usage))

        choices = result["choices"]
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
        if choice.get("text") is not None:
            text = (text or "") + choice.get("text")
            continue

        delta = choice.get("delta")
        if not delta:
            continue

        if role is None and delta.get("role") is not None:
            role = delta.get("role")

        if delta.get("finish_reason") is not None:
            finish_reason = delta.get("finish_reason")

        if delta.get("content") is not None:
            content = (content or "") + delta.get("content")

        if delta.get("tool_calls") is not None:
            delta_tool_calls = delta.get("tool_calls")
            if not delta_tool_calls:
                continue
            tool_delta = delta_tool_calls[0]

            # pylint: disable=unsubscriptable-object
            if not tool_calls or (tool_delta.get("id") and tool_calls[-1]["id"] != tool_delta.get("id")):
                tool_calls = (tool_calls or []) + [
                    {
                        "id": tool_delta.get("id"),
                        "type": tool_delta.get("type"),
                        "function": tool_delta.get("function"),
                    }
                ]
            else:
                # pylint: disable=unsubscriptable-object
                tool_calls[-1]["function"]["arguments"] += delta["tool_calls"][0]["function"]["arguments"]

    if text is not None:
        output = [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}]
    else:
        output = [
            {
                "index": 0,
                "message": {"role": role, "content": content, "tool_calls": tool_calls},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ]

    return {"metrics": metrics, "output": output}


def _postprocess_responses_streaming_results(all_results: list[Any]) -> dict[str, Any]:
    """Process responses API streaming results."""
    metrics: dict[str, Any] = {}
    output: list[dict[str, Any]] = []

    for result in all_results:
        usage = None
        if hasattr(result, "usage"):
            usage = getattr(result, "usage")
        elif result.type == "response.completed" and hasattr(result, "response"):
            usage = getattr(result.response, "usage")

        if usage:
            parsed_metrics = _parse_metrics_from_usage(usage)
            metrics.update(parsed_metrics)

        if result.type == "response.output_item.added":
            output.append({"id": result.item.get("id"), "type": result.item.get("type")})
            continue

        if not hasattr(result, "output_index"):
            continue

        output_index = result.output_index
        current_output = output[output_index]
        if result.type == "response.output_item.done":
            current_output["status"] = result.item.get("status")
            continue

        if result.type == "response.output_item.delta":
            current_output["delta"] = result.delta
            continue

        if hasattr(result, "content_index"):
            if "content" not in current_output:
                current_output["content"] = []
            content_index = result.content_index
            if content_index == len(current_output["content"]):
                current_output["content"].append({})
            current_content = current_output["content"][content_index]
            if hasattr(result, "delta") and result.delta:
                current_content["text"] = (current_content.get("text") or "") + result.delta

            if result.type == "response.output_text.annotation.added":
                annotation_index = result.annotation_index
                if "annotations" not in current_content:
                    current_content["annotations"] = []
                if annotation_index == len(current_content["annotations"]):
                    current_content["annotations"].append({})
                current_content["annotations"][annotation_index] = _try_to_dict(result.annotation)

    return {
        "metrics": metrics,
        "output": output,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


_SAFE_METADATA_KEYS = frozenset(
    {
        "allowed_openai_params",
        "audio",
        "background",
        "best_of",
        "caching",
        "custom_llm_provider",
        "deployment_id",
        "dimensions",
        "echo",
        "enable_json_schema_validation",
        "encoding_format",
        "frequency_penalty",
        "function_call",
        "functions",
        "include",
        "include_server_side_tool_invocations",
        "instructions",
        "language",
        "litellm_call_id",
        "logit_bias",
        "logprobs",
        "max_chunks_per_doc",
        "max_completion_tokens",
        "max_output_tokens",
        "max_retries",
        "max_tokens",
        "max_tokens_per_doc",
        "metadata",
        "modalities",
        "n",
        "organization",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "previous_response_id",
        "project",
        "quality",
        "rank_fields",
        "reasoning",
        "reasoning_effort",
        "response_format",
        "return_documents",
        "safety_identifier",
        "seed",
        "service_tier",
        "size",
        "speed",
        "stop",
        "store",
        "stream",
        "stream_options",
        "style",
        "suffix",
        "temperature",
        "text",
        "text_format",
        "thinking",
        "timeout",
        "timestamp_granularities",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_n",
        "top_p",
        "truncation",
        "user",
        "verbosity",
        "voice",
        "web_search_options",
    }
)


def _build_span_metadata(params: dict[str, Any], model: Any) -> dict[str, Any]:
    """Build span metadata from known-safe LiteLLM request parameters plus provider/model."""
    metadata = {key: value for key, value in params.items() if key in _SAFE_METADATA_KEYS}
    provider = _resolve_provider(model) if isinstance(model, str) and model else "litellm"
    metadata.update(provider=provider, model=model)
    return metadata


def _update_span_payload_from_params(params: dict[str, Any], input_key: str = "input") -> dict[str, Any]:
    """Updates the span payload with the parameters into LiteLLM's completion/acompletion methods.

    Works on a shallow copy so the caller's kwargs dict is never mutated.
    """
    params = _prettify_response_params(params)
    span_info_d = params.pop("span_info", {})
    input_data = params.pop(input_key, None)
    model = params.pop("model", None)
    metadata = _build_span_metadata(params, model)

    return merge_dicts(
        span_info_d,
        {"input": input_data, "metadata": metadata},
    )


def _update_rerank_span_payload_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """Build the span payload for a LiteLLM rerank/arerank call.

    The request shape is modeled after Cohere's rerank API: ``query`` plus a
    list of ``documents``.  The span input captures both, metadata records the
    model and reranker-specific request parameters, and ``document_count`` is
    derived for convenience.
    """
    params = _prettify_response_params(params)
    span_info_d = params.pop("span_info", {})
    query = params.pop("query", None)
    documents = params.pop("documents", None)
    model = params.pop("model", None)

    metadata = _build_span_metadata(params, model)
    if isinstance(documents, (list, tuple)):
        metadata["document_count"] = len(documents)

    return merge_dicts(
        span_info_d,
        {"input": {"query": query, "documents": documents}, "metadata": metadata},
    )


def _update_audio_span_payload_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """Update the span payload for audio transcription calls."""
    params = _prettify_response_params(params)
    span_info_d = params.pop("span_info", {})
    audio_file = _materialize_attachment(params.pop("file", None))
    model = params.pop("model", None)

    input_data = {"file": audio_file.attachment} if audio_file is not None else None
    metadata = _build_span_metadata(params, model)

    return merge_dicts(
        span_info_d,
        {"input": input_data, "metadata": metadata},
    )


def _extract_transcription_text(response: Any) -> str | None:
    if isinstance(response, str):
        return response.strip()
    return _get_field(response, "text")


def _parse_metrics_from_usage(usage: Any) -> dict[str, Any]:
    """Parse usage metrics from API response."""
    return _parse_openai_usage_metrics(
        usage,
        token_name_map=TOKEN_NAME_MAP,
        token_prefix_map=TOKEN_PREFIX_MAP,
    )


_RERANK_BILLED_UNITS_MAP: dict[str, str] = {
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "search_units": "search_units",
    "classifications": "classifications",
    "total_tokens": "tokens",
}

_RERANK_TOKENS_MAP: dict[str, str] = {
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "total_tokens": "tokens",
}


def _parse_rerank_metrics(response: Any) -> dict[str, Any]:
    """Parse token / billed-unit metrics from a LiteLLM rerank response.

    Why: LiteLLM follows Cohere's shape — usage lives under ``meta.billed_units``
    (authoritative billing counter) and ``meta.tokens``. ``billed_units`` wins
    when both are present.
    """
    metrics: dict[str, Any] = {}
    meta = _get_field(response, "meta")
    if meta is None:
        return metrics

    def _read(block: Any, mapping: dict[str, str]) -> None:
        if block is None:
            return
        for src_key, dst_key in mapping.items():
            value = _get_field(block, src_key)
            if is_numeric(value):
                metrics[dst_key] = value

    _read(_get_field(meta, "tokens"), _RERANK_TOKENS_MAP)
    _read(_get_field(meta, "billed_units"), _RERANK_BILLED_UNITS_MAP)

    if "tokens" not in metrics and "prompt_tokens" in metrics and "completion_tokens" in metrics:
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    return metrics


def _extract_rerank_output(response: Any) -> list[dict[str, Any]] | None:
    """Return the ranked ``{index, relevance_score}`` list from a rerank response.

    The document payload (from ``return_documents=True``) is dropped and the
    list is capped at ``_RERANK_MAX_RESULTS`` to keep the span compact.
    """
    results = _get_field(response, "results")
    if not isinstance(results, list):
        return None

    out: list[dict[str, Any]] = []
    for item in results[:_RERANK_MAX_RESULTS]:
        if item is None:
            continue
        out.append(
            {
                "index": _get_field(item, "index"),
                "relevance_score": _get_field(item, "relevance_score"),
            }
        )
    return out
