"""LiteLLM tracing helpers — spans, metadata extraction, stream handling."""

import time
from collections.abc import AsyncGenerator, Generator
from types import TracebackType
from typing import Any

from braintrust.logger import Span, start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import is_numeric, merge_dicts


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


def _completion_wrapper(wrapped, instance, args, kwargs):
    """wrapt wrapper for litellm.completion."""
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="messages")
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
        else:
            log_response = _try_to_dict(completion_response)
            metrics = _parse_metrics_from_usage(log_response.get("usage", {}))
            metrics["time_to_first_token"] = time.time() - start
            span.log(metrics=metrics, output=log_response["choices"])
            return completion_response
    finally:
        if should_end:
            span.end()


async def _acompletion_wrapper_async(wrapped, instance, args, kwargs):
    """wrapt wrapper for litellm.acompletion."""
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="messages")
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
        else:
            log_response = _try_to_dict(completion_response)
            metrics = _parse_metrics_from_usage(log_response.get("usage", {}))
            metrics["time_to_first_token"] = time.time() - start
            span.log(metrics=metrics, output=log_response["choices"])
            return completion_response
    finally:
        if should_end:
            span.end()


def _responses_wrapper(wrapped, instance, args, kwargs):
    """wrapt wrapper for litellm.responses."""
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
        else:
            log_response = _try_to_dict(response)
            metrics = _parse_metrics_from_usage(log_response.get("usage", {}))
            metrics["time_to_first_token"] = time.time() - start
            span.log(metrics=metrics, output=log_response["output"])
            return response
    finally:
        if should_end:
            span.end()


async def _aresponses_wrapper_async(wrapped, instance, args, kwargs):
    """wrapt wrapper for litellm.aresponses."""
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
        else:
            log_response = _try_to_dict(response)
            metrics = _parse_metrics_from_usage(log_response.get("usage", {}))
            metrics["time_to_first_token"] = time.time() - start
            span.log(metrics=metrics, output=log_response["output"])
            return response
    finally:
        if should_end:
            span.end()


def _embedding_wrapper(wrapped, instance, args, kwargs):
    """wrapt wrapper for litellm.embedding."""
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")

    with start_span(
        **merge_dicts(dict(name="Embedding", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        embedding_response = wrapped(*args, **kwargs)
        log_response = _try_to_dict(embedding_response)
        usage = log_response.get("usage")
        metrics = _parse_metrics_from_usage(usage)
        span.log(
            metrics=metrics,
            output={"embedding_length": len(log_response["data"][0]["embedding"])},
        )
        return embedding_response


async def _aembedding_wrapper_async(wrapped, instance, args, kwargs):
    """wrapt wrapper for litellm.aembedding."""
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")

    with start_span(
        **merge_dicts(dict(name="Embedding", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        embedding_response = await wrapped(*args, **kwargs)
        log_response = _try_to_dict(embedding_response)
        usage = log_response.get("usage")
        metrics = _parse_metrics_from_usage(usage)
        span.log(
            metrics=metrics,
            output={"embedding_length": len(log_response["data"][0]["embedding"])},
        )
        return embedding_response


def _moderation_wrapper(wrapped, instance, args, kwargs):
    """wrapt wrapper for litellm.moderation."""
    updated_span_payload = _update_span_payload_from_params(kwargs, input_key="input")

    with start_span(
        **merge_dicts(dict(name="Moderation", span_attributes={"type": SpanTypeAttribute.LLM}), updated_span_payload)
    ) as span:
        moderation_response = wrapped(*args, **kwargs)
        log_response = _try_to_dict(moderation_response)
        usage = log_response.get("usage")
        metrics = _parse_metrics_from_usage(usage)
        span.log(
            metrics=metrics,
            output=log_response["results"],
        )
        return moderation_response


# ---------------------------------------------------------------------------
# Streaming post-processing
# ---------------------------------------------------------------------------


def _postprocess_completion_streaming_results(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Process streaming results to extract final response."""
    role = None
    content = None
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
        delta = choices[0]["delta"]
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

    return {
        "metrics": metrics,
        "output": [
            {
                "index": 0,
                "message": {"role": role, "content": content, "tool_calls": tool_calls},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


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


def _update_span_payload_from_params(params: dict[str, Any], input_key: str = "input") -> dict[str, Any]:
    """Updates the span payload with the parameters into LiteLLM's completion/acompletion methods.

    Works on a shallow copy so the caller's kwargs dict is never mutated.
    """
    params = params.copy()
    span_info_d = params.pop("span_info", {})

    params = prettify_params(params)
    input_data = params.pop(input_key, None)
    model = params.pop("model", None)

    return merge_dicts(
        span_info_d,
        {"input": input_data, "metadata": {**params, "provider": "litellm", "model": model}},
    )


def _parse_metrics_from_usage(usage: Any) -> dict[str, Any]:
    """Parse usage metrics from API response."""
    metrics: dict[str, Any] = {}

    if not usage:
        return metrics

    usage = _try_to_dict(usage)
    if not isinstance(usage, dict):
        return metrics

    for oai_name, value in usage.items():
        if oai_name.endswith("_tokens_details"):
            if not isinstance(value, dict):
                continue
            raw_prefix = oai_name[: -len("_tokens_details")]
            prefix = TOKEN_PREFIX_MAP.get(raw_prefix, raw_prefix)
            for k, v in value.items():
                if is_numeric(v):
                    metrics[f"{prefix}_{k}"] = v
        elif is_numeric(value):
            name = TOKEN_NAME_MAP.get(oai_name, oai_name)
            metrics[name] = value

    return metrics


def prettify_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *params* with response_format serialized for logging."""

    if "response_format" in params:
        ret = params.copy()
        ret["response_format"] = serialize_response_format(ret["response_format"])
        return ret

    return params


def _try_to_dict(obj: Any) -> dict[str, Any] | Any:
    """Try to convert an object to a dictionary."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            result = obj.model_dump()
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            result = obj.dict()
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return obj


def serialize_response_format(response_format: Any) -> Any:
    """Serialize response format for logging."""
    try:
        from pydantic import BaseModel
    except ImportError:
        return response_format

    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return dict(
            type="json_schema",
            json_schema=dict(
                name=response_format.__name__,
                schema=response_format.model_json_schema(),
            ),
        )
    else:
        return response_format
