"""OpenRouter-specific tracing helpers."""

import logging
import time
from collections.abc import AsyncIterator, Iterator
from numbers import Real
from typing import TYPE_CHECKING, Any

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_OMITTED_KEYS = {
    "execute",
    "render",
    "nextTurnParams",
    "next_turn_params",
    "requireApproval",
    "require_approval",
}
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


def _camel_to_snake(value: str) -> str:
    out = []
    for char in value:
        if char.isupper():
            out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    return "".join(out).lstrip("_")


def _is_supported_metric_value(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def sanitize_openrouter_logged_value(value: Any) -> Any:
    safe = bt_safe_deep_copy(value)

    if callable(safe):
        return "[Function]"
    if isinstance(safe, list):
        return [sanitize_openrouter_logged_value(item) for item in safe]
    if isinstance(safe, tuple):
        return [sanitize_openrouter_logged_value(item) for item in safe]
    if isinstance(safe, dict):
        sanitized = {}
        for key, entry in safe.items():
            if key in _OMITTED_KEYS:
                continue
            sanitized[key] = sanitize_openrouter_logged_value(entry)
        return sanitized
    return safe


def _parse_openrouter_model_string(model: Any) -> dict[str, Any]:
    if not isinstance(model, str):
        return {"model": model}

    slash_index = model.find("/")
    if 0 < slash_index < len(model) - 1:
        return {
            "provider": model[:slash_index],
            "model": model[slash_index + 1 :],
        }

    return {"model": model}


def _build_openrouter_metadata(metadata: dict[str, Any], *, embedding: bool = False) -> dict[str, Any]:
    sanitized = sanitize_openrouter_logged_value(metadata)
    record = sanitized if isinstance(sanitized, dict) else {}
    model = record.pop("model", None)
    provider_routing = record.pop("provider", None)
    normalized_model = _parse_openrouter_model_string(model)

    result = dict(record)
    if normalized_model.get("model") is not None:
        result["model"] = normalized_model["model"]
    if provider_routing is not None:
        result["provider_routing"] = provider_routing
    result["provider"] = normalized_model.get("provider") or "openrouter"
    if embedding and isinstance(result.get("model"), str):
        result["embedding_model"] = result["model"]
    return result


def _extract_openrouter_usage_metadata(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        usage = sanitize_openrouter_logged_value(usage)
    if not isinstance(usage, dict):
        return {}

    if isinstance(usage.get("is_byok"), bool):
        return {"is_byok": usage["is_byok"]}
    if isinstance(usage.get("isByok"), bool):
        return {"is_byok": usage["isByok"]}
    return {}


def _parse_openrouter_metrics_from_usage(usage: Any) -> dict[str, float]:
    if not isinstance(usage, dict):
        usage = sanitize_openrouter_logged_value(usage)
    if not isinstance(usage, dict):
        return {}

    metrics = {}
    for name, value in usage.items():
        if _is_supported_metric_value(value):
            metrics[_TOKEN_NAME_MAP.get(name, _camel_to_snake(name))] = float(value)
            continue

        if not isinstance(value, dict):
            continue

        prefix = _TOKEN_DETAIL_PREFIX_MAP.get(name)
        if prefix is None:
            continue

        for nested_name, nested_value in value.items():
            if _is_supported_metric_value(nested_value):
                metrics[f"{prefix}_{_camel_to_snake(nested_name)}"] = float(nested_value)

    return metrics


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


def _merge_metrics(start_time: float, usage: Any, first_token_time: float | None = None) -> dict[str, float]:
    return {
        **_timing_metrics(start_time, first_token_time),
        **_parse_openrouter_metrics_from_usage(usage),
    }


def _response_to_output(response: Any, *, fallback_output: Any | None = None) -> Any:
    if hasattr(response, "output") and getattr(response, "output") is not None:
        return sanitize_openrouter_logged_value(getattr(response, "output"))
    if hasattr(response, "choices") and getattr(response, "choices") is not None:
        return sanitize_openrouter_logged_value(getattr(response, "choices"))
    if fallback_output is not None:
        return sanitize_openrouter_logged_value(fallback_output)
    return None


def _response_to_metadata(response: Any, *, embedding: bool = False) -> dict[str, Any]:
    if response is None:
        return {}

    data = sanitize_openrouter_logged_value(response)
    if not isinstance(data, dict):
        return {}

    data.pop("output", None)
    data.pop("choices", None)
    data.pop("data", None)
    usage = data.pop("usage", None)
    metadata = _build_openrouter_metadata(data, embedding=embedding)
    metadata.update(_extract_openrouter_usage_metadata(usage))
    return metadata


def _embeddings_output(response: Any) -> dict[str, Any]:
    items = getattr(response, "data", None) or []
    first = items[0] if items else None
    embedding = getattr(first, "embedding", None) if first is not None else None
    return {
        "embedding_length": len(embedding) if embedding is not None else None,
        "embeddings_count": len(items),
    }


def _start_span(name: str, span_input: Any, metadata: dict[str, Any]):
    return start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=sanitize_openrouter_logged_value(span_input),
        metadata=metadata,
    )


def _log_and_end(
    span: Any, *, output: Any = None, metrics: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None
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
            _log_error_and_end(self._span, error)
            return

        if self._kind == "chat":
            output, usage = _aggregate_chat_stream(self._items)
            metadata = dict(self._metadata)
        else:
            output, usage, response_metadata = _aggregate_responses_stream(self._items)
            metadata = {**self._metadata, **response_metadata}

        _log_and_end(
            self._span,
            output=output,
            metrics=_merge_metrics(self._start_time, usage, self._first_token_time),
            metadata=metadata,
        )


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
            _log_error_and_end(self._span, error)
            return

        if self._kind == "chat":
            output, usage = _aggregate_chat_stream(self._items)
            metadata = dict(self._metadata)
        else:
            output, usage, response_metadata = _aggregate_responses_stream(self._items)
            metadata = {**self._metadata, **response_metadata}

        _log_and_end(
            self._span,
            output=output,
            metrics=_merge_metrics(self._start_time, usage, self._first_token_time),
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
    choices = {}
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
    output_items = {}

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
            _response_to_output(completed_response),
            getattr(completed_response, "usage", None),
            _response_to_metadata(completed_response),
        )

    output = [output_items[index] for index in sorted(output_items)]
    return sanitize_openrouter_logged_value(output), usage, {}


def _finalize_chat_response(span: Any, request_metadata: dict[str, Any], result: Any, start_time: float):
    _log_and_end(
        span,
        output=_response_to_output(result),
        metrics=_merge_metrics(start_time, getattr(result, "usage", None)),
        metadata={**request_metadata, **_response_to_metadata(result)},
    )


def _finalize_embeddings_response(span: Any, request_metadata: dict[str, Any], result: Any, start_time: float):
    _log_and_end(
        span,
        output=_embeddings_output(result),
        metrics=_merge_metrics(start_time, getattr(result, "usage", None)),
        metadata={**request_metadata, **_response_to_metadata(result, embedding=True)},
    )


def _finalize_responses_response(span: Any, request_metadata: dict[str, Any], result: Any, start_time: float):
    _log_and_end(
        span,
        output=_response_to_output(result, fallback_output=getattr(result, "output_text", None)),
        metrics=_merge_metrics(start_time, getattr(result, "usage", None)),
        metadata={**request_metadata, **_response_to_metadata(result)},
    )


def _chat_send_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_openrouter_metadata(dict(kwargs))
    span = _start_span("openrouter.chat.send", kwargs.get("messages"), request_metadata)
    start_time = time.time()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise

    if kwargs.get("stream"):
        return _TracedOpenRouterSyncStream(result, span, request_metadata, "chat", start_time)

    _finalize_chat_response(span, request_metadata, result, start_time)
    return result


async def _chat_send_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_openrouter_metadata(dict(kwargs))
    span = _start_span("openrouter.chat.send", kwargs.get("messages"), request_metadata)
    start_time = time.time()

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise

    if kwargs.get("stream"):
        return _TracedOpenRouterAsyncStream(result, span, request_metadata, "chat", start_time)

    _finalize_chat_response(span, request_metadata, result, start_time)
    return result


def _embeddings_generate_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_openrouter_metadata(dict(kwargs), embedding=True)
    span = _start_span("openrouter.embeddings.generate", kwargs.get("input"), request_metadata)
    start_time = time.time()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise

    _finalize_embeddings_response(span, request_metadata, result, start_time)
    return result


async def _embeddings_generate_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_openrouter_metadata(dict(kwargs), embedding=True)
    span = _start_span("openrouter.embeddings.generate", kwargs.get("input"), request_metadata)
    start_time = time.time()

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise

    _finalize_embeddings_response(span, request_metadata, result, start_time)
    return result


def _responses_send_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_openrouter_metadata(dict(kwargs))
    span = _start_span("openrouter.beta.responses.send", kwargs.get("input"), request_metadata)
    start_time = time.time()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise

    if kwargs.get("stream"):
        return _TracedOpenRouterSyncStream(result, span, request_metadata, "responses", start_time)

    _finalize_responses_response(span, request_metadata, result, start_time)
    return result


async def _responses_send_async_wrapper(wrapped, instance, args, kwargs):
    request_metadata = _build_openrouter_metadata(dict(kwargs))
    span = _start_span("openrouter.beta.responses.send", kwargs.get("input"), request_metadata)
    start_time = time.time()

    try:
        result = await wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end(span, error)
        raise

    if kwargs.get("stream"):
        return _TracedOpenRouterAsyncStream(result, span, request_metadata, "responses", start_time)

    _finalize_responses_response(span, request_metadata, result, start_time)
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
