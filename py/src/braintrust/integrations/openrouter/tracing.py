"""OpenRouter-specific tracing helpers."""

import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from braintrust.integrations.utils import (
    _camel_to_snake,
    _is_supported_metric_value,
    _log_and_end_span,
    _log_error_and_end_span,
    _merge_timing_and_usage_metrics,
    _try_to_dict,
)
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute


_INSTRUMENTATION = "openrouter-auto"


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


logger = logging.getLogger(__name__)

_TOKEN_NAME_MAP = {
    "promptTokens": "prompt_tokens",
    "inputTokens": "prompt_tokens",
    "completionTokens": "completion_tokens",
    "outputTokens": "completion_tokens",
    "totalTokens": "tokens",
    "prompt_tokens": "prompt_tokens",
    "input_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "output_tokens": "completion_tokens",
    "total_tokens": "tokens",
}
_TOKEN_DETAIL_PREFIX_MAP = {
    "promptTokensDetails": "prompt",
    "inputTokensDetails": "prompt",
    "completionTokensDetails": "completion",
    "outputTokensDetails": "completion",
    "costDetails": "cost",
    "prompt_tokens_details": "prompt",
    "input_tokens_details": "prompt",
    "completion_tokens_details": "completion",
    "output_tokens_details": "completion",
    "cost_details": "cost",
}

_CHAT_REQUEST_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "top_a",
    "min_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "n",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "tool_choice",
    "tools",
    "parallel_tool_calls",
    "reasoning",
    "prediction",
    "modalities",
    "web_search_options",
    "user",
    "verbosity",
    "stream",
    "stream_options",
    "service_tier",
    "transforms",
    "models",
    "route",
    "plugins",
)
_EMBEDDINGS_REQUEST_KEYS = (
    "encoding_format",
    "dimensions",
    "input_type",
    "user",
)
_RESPONSES_REQUEST_KEYS = (
    "temperature",
    "top_p",
    "max_output_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "top_logprobs",
    "logprobs",
    "response_format",
    "tool_choice",
    "tools",
    "parallel_tool_calls",
    "max_tool_calls",
    "reasoning",
    "background",
    "previous_response_id",
    "service_tier",
    "truncation",
    "store",
    "instructions",
    "text",
    "safety_identifier",
    "prompt_cache_key",
    "include",
    "user",
    "stream",
    "metadata",
)
_CHAT_RESPONSE_KEYS = (
    "id",
    "object",
    "created",
    "system_fingerprint",
    "service_tier",
)
_RESPONSES_RESPONSE_KEYS = (
    "id",
    "object",
    "created_at",
    "completed_at",
    "status",
    "service_tier",
    "background",
    "previous_response_id",
    "safety_identifier",
    "prompt_cache_key",
    "truncation",
    "incomplete_details",
)
_EMBEDDINGS_RESPONSE_KEYS = (
    "id",
    "object",
)


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_openrouter_model_string(model: Any) -> tuple[Any, str | None]:
    if not isinstance(model, str):
        return model, None
    slash = model.find("/")
    if 0 < slash < len(model) - 1:
        return model[slash + 1 :], model[:slash]
    return model, None


def _stamp_model_and_provider(metadata: dict[str, Any], raw_model: Any, routing: Any) -> Any:
    """Set model / provider / provider_routing on ``metadata``. Returns the parsed model."""
    model, provider = _parse_openrouter_model_string(raw_model)
    if model is not None:
        metadata["model"] = model
    if provider is not None:
        metadata["provider"] = provider
    if routing is not None:
        metadata["provider_routing"] = routing
    return model


def _build_request_metadata(
    kwargs: dict[str, Any], keys: tuple[str, ...], *, embedding: bool = False
) -> dict[str, Any]:
    metadata = {key: kwargs[key] for key in keys if key in kwargs and kwargs[key] is not None}
    model = _stamp_model_and_provider(metadata, kwargs.get("model"), kwargs.get("provider"))
    metadata.setdefault("provider", "openrouter")
    if embedding and isinstance(model, str):
        metadata["embedding_model"] = model
    return metadata


def _pick_response_metadata(response: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if response is None:
        return {}

    picked: dict[str, Any] = {}
    for key in keys:
        value = _get_field(response, key)
        if value is not None:
            picked[key] = value

    _stamp_model_and_provider(picked, _get_field(response, "model"), _get_field(response, "provider"))
    return picked


def _usage_metadata(usage: Any) -> dict[str, Any]:
    is_byok = _get_field(usage, "is_byok")
    if is_byok is None:
        is_byok = _get_field(usage, "isByok")
    return {"is_byok": is_byok} if isinstance(is_byok, bool) else {}


def _parse_openrouter_metrics_from_usage(usage: Any) -> dict[str, float]:
    usage_dict = _try_to_dict(usage)
    if not isinstance(usage_dict, dict):
        return {}

    metrics: dict[str, float] = {}
    for name, value in usage_dict.items():
        if _is_supported_metric_value(value):
            metrics[_TOKEN_NAME_MAP.get(name, _camel_to_snake(name))] = float(value)
            continue
        prefix = _TOKEN_DETAIL_PREFIX_MAP.get(name)
        if prefix is None or not isinstance(value, dict):
            continue
        for nested_name, nested_value in value.items():
            if _is_supported_metric_value(nested_value):
                metrics[f"{prefix}_{_camel_to_snake(nested_name)}"] = float(nested_value)

    return metrics


def _merge_metrics(start_time: float, usage: Any, first_token_time: float | None = None) -> dict[str, Any]:
    return _merge_timing_and_usage_metrics(
        start_time,
        usage,
        _parse_openrouter_metrics_from_usage,
        first_token_time,
    )


def _embeddings_output(response: Any) -> dict[str, Any]:
    items = _get_field(response, "data") or []
    first = items[0] if items else None
    embedding = _get_field(first, "embedding") if first is not None else None
    return {
        "embedding_length": len(embedding) if embedding is not None else None,
        "embeddings_count": len(items),
    }


def _start_span(name: str, span_input: Any, metadata: dict[str, Any]):
    return start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=span_input,
        metadata=metadata,
    )


class _TracedOpenRouterSyncStream:
    def __init__(self, stream: Any, span: Any, metadata: dict[str, Any], kind: str, start_time: float):
        self._stream = stream
        self._span = span
        self._metadata = metadata
        self._kind = kind
        self._start_time = start_time
        self._first_token_time = None
        self._items = []
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

        _finalize_stream(self._span, self._metadata, self._items, self._kind, self._start_time, self._first_token_time)


class _TracedOpenRouterAsyncStream:
    def __init__(self, stream: Any, span: Any, metadata: dict[str, Any], kind: str, start_time: float):
        self._stream = stream
        self._span = span
        self._metadata = metadata
        self._kind = kind
        self._start_time = start_time
        self._first_token_time = None
        self._items = []
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

        _finalize_stream(self._span, self._metadata, self._items, self._kind, self._start_time, self._first_token_time)


def _finalize_stream(
    span: Any,
    request_metadata: dict[str, Any],
    items: list[Any],
    kind: str,
    start_time: float,
    first_token_time: float | None,
) -> None:
    if kind == "chat":
        output, usage = _aggregate_chat_stream(items)
        metadata = {**request_metadata, **_usage_metadata(usage)}
    else:
        output, usage, response_metadata = _aggregate_responses_stream(items)
        metadata = {**request_metadata, **response_metadata, **_usage_metadata(usage)}

    _log_and_end_span(
        span,
        output=output,
        metrics=_merge_metrics(start_time, usage, first_token_time),
        metadata=metadata,
    )


def _chunk_has_output(item: Any) -> bool:
    item_type = getattr(item, "type", None)
    if isinstance(item_type, str) and ".delta" in item_type:
        return True

    if hasattr(item, "choices"):
        for choice in getattr(item, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if (
                getattr(delta, "content", None)
                or getattr(delta, "reasoning", None)
                or getattr(delta, "tool_calls", None)
            ):
                return True

    return False


def _aggregate_chat_stream(chunks: list[Any]) -> tuple[list[dict[str, Any]], Any]:
    choices: dict[int, dict[str, Any]] = {}
    usage = None

    for chunk in chunks:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage

        for choice in getattr(chunk, "choices", []) or []:
            index = int(getattr(choice, "index", 0) or 0)
            state = choices.setdefault(
                index,
                {
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )
            delta = getattr(choice, "delta", None)
            if delta is not None:
                role = getattr(delta, "role", None)
                if role is not None:
                    state["message"]["role"] = role

                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    state["message"]["content"] += content

                reasoning = getattr(delta, "reasoning", None)
                if isinstance(reasoning, str):
                    state["message"]["reasoning"] = state["message"].get("reasoning", "") + reasoning

                refusal = getattr(delta, "refusal", None)
                if isinstance(refusal, str):
                    state["message"]["refusal"] = state["message"].get("refusal", "") + refusal

                tool_calls = getattr(delta, "tool_calls", None) or []
                if tool_calls:
                    tools = state["message"].setdefault("tool_calls", [])
                    for tool_call in tool_calls:
                        tool_index = int(getattr(tool_call, "index", len(tools)) or 0)
                        while len(tools) <= tool_index:
                            tools.append({"function": {"arguments": ""}})
                        current = tools[tool_index]
                        tool_id = getattr(tool_call, "id", None)
                        if tool_id is not None:
                            current["id"] = tool_id
                        tool_type = getattr(tool_call, "type", None)
                        if tool_type is not None:
                            current["type"] = tool_type
                        function = getattr(tool_call, "function", None)
                        if function is not None:
                            if getattr(function, "name", None) is not None:
                                current.setdefault("function", {})["name"] = function.name
                            arguments = getattr(function, "arguments", None)
                            if isinstance(arguments, str):
                                current.setdefault("function", {}).setdefault("arguments", "")
                                current["function"]["arguments"] += arguments

            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None:
                state["finish_reason"] = finish_reason

    output = []
    for index in sorted(choices):
        choice = choices[index]
        if not choice["message"].get("tool_calls"):
            choice["message"].pop("tool_calls", None)
        output.append(choice)
    return output, usage


def _aggregate_responses_stream(chunks: list[Any]) -> tuple[Any, Any, dict[str, Any]]:
    completed_response = None
    usage = None
    output_items: dict[int, Any] = {}

    for chunk in chunks:
        chunk_type = getattr(chunk, "type", None)
        if chunk_type == "response.completed":
            completed_response = getattr(chunk, "response", None)
            usage = getattr(completed_response, "usage", None)
        elif chunk_type == "response.output_item.done":
            output_index = int(getattr(chunk, "output_index", 0) or 0)
            output_items[output_index] = getattr(chunk, "item", None)

    if completed_response is not None:
        return (
            _get_field(completed_response, "output"),
            getattr(completed_response, "usage", None),
            _pick_response_metadata(completed_response, _RESPONSES_RESPONSE_KEYS),
        )

    output = [output_items[index] for index in sorted(output_items)]
    return output, usage, {}


def _finalize_response(
    span: Any,
    request_metadata: dict[str, Any],
    result: Any,
    start_time: float,
    *,
    output: Any,
    response_keys: tuple[str, ...],
) -> None:
    usage = _get_field(result, "usage")
    metadata = {
        **request_metadata,
        **_pick_response_metadata(result, response_keys),
        **_usage_metadata(usage),
    }
    _log_and_end_span(
        span,
        output=output,
        metrics=_merge_metrics(start_time, usage),
        metadata=metadata,
    )


def _chat_send_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_request_metadata(kwargs, _CHAT_REQUEST_KEYS)
    span = _start_span("openrouter.chat.send", kwargs.get("messages"), request_metadata)
    start_time = time.time()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if kwargs.get("stream"):
        return _TracedOpenRouterSyncStream(result, span, request_metadata, "chat", start_time)

    _finalize_response(
        span,
        request_metadata,
        result,
        start_time,
        output=_get_field(result, "choices"),
        response_keys=_CHAT_RESPONSE_KEYS,
    )
    return result


async def _chat_send_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_request_metadata(kwargs, _CHAT_REQUEST_KEYS)
    span = _start_span("openrouter.chat.send", kwargs.get("messages"), request_metadata)
    start_time = time.time()

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if kwargs.get("stream"):
        return _TracedOpenRouterAsyncStream(result, span, request_metadata, "chat", start_time)

    _finalize_response(
        span,
        request_metadata,
        result,
        start_time,
        output=_get_field(result, "choices"),
        response_keys=_CHAT_RESPONSE_KEYS,
    )
    return result


def _embeddings_generate_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_request_metadata(kwargs, _EMBEDDINGS_REQUEST_KEYS, embedding=True)
    span = _start_span("openrouter.embeddings.generate", kwargs.get("input"), request_metadata)
    start_time = time.time()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    _finalize_response(
        span,
        request_metadata,
        result,
        start_time,
        output=_embeddings_output(result),
        response_keys=_EMBEDDINGS_RESPONSE_KEYS,
    )
    return result


async def _embeddings_generate_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_request_metadata(kwargs, _EMBEDDINGS_REQUEST_KEYS, embedding=True)
    span = _start_span("openrouter.embeddings.generate", kwargs.get("input"), request_metadata)
    start_time = time.time()

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    _finalize_response(
        span,
        request_metadata,
        result,
        start_time,
        output=_embeddings_output(result),
        response_keys=_EMBEDDINGS_RESPONSE_KEYS,
    )
    return result


def _responses_send_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_request_metadata(kwargs, _RESPONSES_REQUEST_KEYS)
    span = _start_span("openrouter.beta.responses.send", kwargs.get("input"), request_metadata)
    start_time = time.time()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if kwargs.get("stream"):
        return _TracedOpenRouterSyncStream(result, span, request_metadata, "responses", start_time)

    _finalize_response(
        span,
        request_metadata,
        result,
        start_time,
        output=_get_field(result, "output") or _get_field(result, "output_text"),
        response_keys=_RESPONSES_RESPONSE_KEYS,
    )
    return result


async def _responses_send_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_request_metadata(kwargs, _RESPONSES_REQUEST_KEYS)
    span = _start_span("openrouter.beta.responses.send", kwargs.get("input"), request_metadata)
    start_time = time.time()

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if kwargs.get("stream"):
        return _TracedOpenRouterAsyncStream(result, span, request_metadata, "responses", start_time)

    _finalize_response(
        span,
        request_metadata,
        result,
        start_time,
        output=_get_field(result, "output") or _get_field(result, "output_text"),
        response_keys=_RESPONSES_RESPONSE_KEYS,
    )
    return result


def wrap_openrouter(client: Any) -> Any:
    """Wrap a single OpenRouter client instance for tracing."""
    from .patchers import ChatPatcher, EmbeddingsPatcher, ResponsesPatcher

    chat = getattr(client, "chat", None)
    if chat is not None:
        ChatPatcher.wrap_target(chat)

    embeddings = getattr(client, "embeddings", None)
    if embeddings is not None:
        EmbeddingsPatcher.wrap_target(embeddings)

    beta = getattr(client, "beta", None)
    responses = getattr(beta, "responses", None) if beta is not None else None
    if responses is not None:
        ResponsesPatcher.wrap_target(responses)

    return client
