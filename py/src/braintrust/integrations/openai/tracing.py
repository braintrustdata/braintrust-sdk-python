"""OpenAI-specific tracing wrappers, stream proxies, and serialization helpers."""

import abc
import inspect
import json
import time
from collections.abc import Callable
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
from braintrust.logger import Span, start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import clean_nones, merge_dicts
from wrapt import FunctionWrapper


X_LEGACY_CACHED_HEADER = "x-cached"
X_CACHED_HEADER = "x-bt-cached"
RAW_RESPONSE_HEADER = "x-stainless-raw-response"


class NamedWrapper:
    def __init__(self, wrapped: Any):
        self.__wrapped = wrapped

    @property
    def _wrapped(self) -> Any:
        return self.__wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__wrapped, name)


class AsyncResponseWrapper:
    """Wrapper that properly preserves async context manager behavior for OpenAI responses."""

    def __init__(self, response: Any):
        self._response = response

    async def __aenter__(self):
        if hasattr(self._response, "__aenter__"):
            await self._response.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._response, "__aexit__"):
            return await self._response.__aexit__(exc_type, exc_val, exc_tb)

    def __aiter__(self):
        if hasattr(self._response, "__aiter__"):
            return self._response.__aiter__()
        raise TypeError("Response object is not an async iterator")

    async def __anext__(self):
        if hasattr(self._response, "__anext__"):
            return await self._response.__anext__()
        raise StopAsyncIteration

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def __class__(self):  # type: ignore
        return self._response.__class__

    def __str__(self) -> str:
        return str(self._response)

    def __repr__(self) -> str:
        return repr(self._response)


def log_headers(response: Any, span: Span):
    cached_value = response.headers.get(X_CACHED_HEADER) or response.headers.get(X_LEGACY_CACHED_HEADER)

    if cached_value:
        span.log(
            metrics={
                "cached": 1 if cached_value.lower() in ["true", "hit"] else 0,
            }
        )


def _raw_response_requested(kwargs: dict[str, Any]) -> bool:
    extra_headers = kwargs.get("extra_headers")
    if not isinstance(extra_headers, dict):
        return False

    for key, value in extra_headers.items():
        if isinstance(key, str) and key.lower() == RAW_RESPONSE_HEADER:
            if isinstance(value, str):
                return value.lower() == "true"
            return bool(value)

    return False


def _materialize_logged_file_input(value: Any) -> Any:
    if isinstance(value, list):
        return [_materialize_logged_file_input(item) for item in value]

    resolved = _materialize_attachment(value)
    return resolved.attachment if resolved is not None else value


def _process_attachments_in_input(input_data: Any) -> Any:
    """Process input to convert data URL images and base64 documents to Attachment objects."""
    if isinstance(input_data, list):
        return [_process_attachments_in_input(item) for item in input_data]

    if isinstance(input_data, dict):
        if (
            input_data.get("type") == "image_url"
            and isinstance(input_data.get("image_url"), dict)
            and isinstance(input_data["image_url"].get("url"), str)
        ):
            url = input_data["image_url"]["url"]
            resolved = _materialize_attachment(url)
            return {
                **input_data,
                "image_url": {
                    **input_data["image_url"],
                    "url": resolved.attachment if resolved is not None else url,
                },
            }

        if (
            input_data.get("type") == "file"
            and isinstance(input_data.get("file"), dict)
            and isinstance(input_data["file"].get("file_data"), str)
        ):
            file_data = input_data["file"]["file_data"]
            file_filename = input_data["file"].get("filename")
            resolved = _materialize_attachment(
                file_data,
                filename=file_filename if isinstance(file_filename, str) else None,
            )
            return {
                **input_data,
                "file": {
                    **input_data["file"],
                    "file_data": resolved.attachment if resolved is not None else file_data,
                },
            }

        return {key: _process_attachments_in_input(value) for key, value in input_data.items()}

    return input_data


def _is_async_callable(fn: Any) -> bool:
    fn = getattr(fn, "__func__", fn)
    # Walk the __wrapped__ chain to see through decorators (e.g. OpenAI's
    # @required_args) that hide the underlying coroutine function.
    while fn is not None:
        if inspect.iscoroutinefunction(fn):
            return True
        next_fn = getattr(fn, "__wrapped__", None)
        if next_fn is fn:
            break
        fn = next_fn
    return False


def _get_raw_callable(instance: Any, method_name: str) -> Any | None:
    """Return the ``with_raw_response`` variant of *method_name* on *instance*.

    This allows wrappers to route through the raw-response path so that
    ``log_headers`` can capture Braintrust proxy headers (e.g. cache status)
    even when the user called the regular (non-raw) method.

    Returns ``None`` when:
    * the resource does not expose ``with_raw_response``
    * the requested method does not exist on it
    * the class-level method is already a wrapt ``FunctionWrapper`` — in that
      case the raw callable would internally call the patched class method,
      causing infinite recursion.
    """
    raw_resource = getattr(instance, "with_raw_response", None)
    if raw_resource is None:
        return None
    raw_callable = getattr(raw_resource, method_name, None)
    if raw_callable is None:
        return None
    # When setup() patches the class method or wrap_openai() patches the
    # instance method, the with_raw_response object captures the already-
    # wrapped method.  Calling it would re-enter our wrapper.  Detect this
    # by checking whether the descriptor (class-level from setup() or
    # instance-level from wrap_openai()) is a FunctionWrapper.
    cls_attr = inspect.getattr_static(type(instance), method_name, None)
    if isinstance(cls_attr, FunctionWrapper):
        return None
    inst_attr = inspect.getattr_static(instance, method_name, None)
    if isinstance(inst_attr, FunctionWrapper):
        return None
    return raw_callable


# ---------------------------------------------------------------------------
# wrapt wrapper callbacks — used by FunctionWrapperPatcher classes
# ---------------------------------------------------------------------------


def _chat_completion_create_wrapper(wrapped, instance, args, kwargs):
    # Route through with_raw_response to capture response headers.
    create_fn = _get_raw_callable(instance, "create") or wrapped
    if _is_async_callable(wrapped):

        async def call():
            response = await ChatCompletionWrapper(None, create_fn).acreate(*args, **kwargs)
            return AsyncResponseWrapper(response)

        return call()
    return ChatCompletionWrapper(create_fn, None).create(*args, **kwargs)


def _chat_completion_parse_wrapper(wrapped, instance, args, kwargs):
    if _is_async_callable(wrapped):

        async def call():
            response = await ChatCompletionWrapper(None, wrapped).acreate(*args, **kwargs)
            return AsyncResponseWrapper(response)

        return call()
    return ChatCompletionWrapper(wrapped, None).create(*args, **kwargs)


def _embedding_create_wrapper(wrapped, instance, args, kwargs):
    create_fn = _get_raw_callable(instance, "create") or wrapped
    if _is_async_callable(wrapped):

        async def call():
            response = await EmbeddingWrapper(None, create_fn).acreate(*args, **kwargs)
            return AsyncResponseWrapper(response)

        return call()
    return EmbeddingWrapper(create_fn, None).create(*args, **kwargs)


def _moderation_create_wrapper(wrapped, instance, args, kwargs):
    create_fn = _get_raw_callable(instance, "create") or wrapped
    if _is_async_callable(wrapped):

        async def call():
            response = await ModerationWrapper(None, create_fn).acreate(*args, **kwargs)
            return AsyncResponseWrapper(response)

        return call()
    return ModerationWrapper(create_fn, None).create(*args, **kwargs)


def _make_base_wrapper_callback(
    wrapper_cls: type["BaseWrapper"],
    *,
    method_name: str = "create",
):
    """Create a wrapt callback that routes through with_raw_response for header capture."""

    def wrapper(wrapped, instance, args, kwargs):
        stream = bool(kwargs.get("stream", False))
        create_fn = wrapped if stream else (_get_raw_callable(instance, method_name) or wrapped)
        if _is_async_callable(wrapped):

            async def call():
                response = await wrapper_cls(None, create_fn).acreate(*args, **kwargs)
                return AsyncResponseWrapper(response)

            return call()
        return wrapper_cls(create_fn, None).create(*args, **kwargs)

    return wrapper


def _responses_create_wrapper(wrapped, instance, args, kwargs):
    if _is_async_callable(wrapped):

        async def call():
            response = await ResponseWrapper(None, wrapped).acreate(*args, **kwargs)
            return AsyncResponseWrapper(response)

        return call()
    return ResponseWrapper(wrapped, None).create(*args, **kwargs)


def _responses_parse_wrapper(wrapped, instance, args, kwargs):
    if _is_async_callable(wrapped):

        async def call():
            response = await ResponseWrapper(None, wrapped, "openai.responses.parse").acreate(*args, **kwargs)
            return AsyncResponseWrapper(response)

        return call()
    return ResponseWrapper(wrapped, None, "openai.responses.parse").create(*args, **kwargs)


def _responses_raw_create_wrapper(wrapped, instance, args, kwargs):
    if _is_async_callable(wrapped):

        async def call():
            return await ResponseWrapper(None, wrapped, return_raw=True).acreate(*args, **kwargs)

        return call()
    return ResponseWrapper(wrapped, None, return_raw=True).create(*args, **kwargs)


def _responses_raw_parse_wrapper(wrapped, instance, args, kwargs):
    if _is_async_callable(wrapped):

        async def call():
            return await ResponseWrapper(
                None,
                wrapped,
                "openai.responses.parse",
                return_raw=True,
            ).acreate(*args, **kwargs)

        return call()
    return ResponseWrapper(wrapped, None, "openai.responses.parse", return_raw=True).create(*args, **kwargs)


# ---------------------------------------------------------------------------
# Core tracing wrappers
# ---------------------------------------------------------------------------


class ChatCompletionWrapper:
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        self.create_fn = create_fn
        self.acreate_fn = acreate_fn

    def create(self, *args: Any, **kwargs: Any) -> Any:
        raw_requested = _raw_response_requested(kwargs)
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        span = start_span(
            **merge_dicts(dict(name="Chat Completion", span_attributes={"type": SpanTypeAttribute.LLM}), params)
        )
        should_end = True

        try:
            start = time.time()
            create_response = self.create_fn(*args, **kwargs)
            if hasattr(create_response, "parse"):
                raw_response = create_response.parse()
                log_headers(create_response, span)
            else:
                raw_response = create_response
            if stream:

                def gen():
                    try:
                        first = True
                        all_results = []
                        for item in raw_response:
                            if first:
                                span.log(
                                    metrics={
                                        "time_to_first_token": time.time() - start,
                                    }
                                )
                                first = False
                            all_results.append(_try_to_dict(item))
                            yield item

                        span.log(**self._postprocess_streaming_results(all_results))
                    finally:
                        span.end()

                should_end = False
                if raw_requested and hasattr(create_response, "parse"):
                    return _RawResponseWithTracedStream(create_response, _TracedStream(raw_response, gen()))
                return _TracedStream(raw_response, gen())
            else:
                log_response = _try_to_dict(raw_response)
                metrics = _parse_metrics_from_usage(log_response.get("usage", {}))
                metrics["time_to_first_token"] = time.time() - start
                span.log(
                    metrics=metrics,
                    output=log_response["choices"],
                )
                return create_response if (raw_requested and hasattr(create_response, "parse")) else raw_response
        finally:
            if should_end:
                span.end()

    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        raw_requested = _raw_response_requested(kwargs)
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        span = start_span(
            **merge_dicts(dict(name="Chat Completion", span_attributes={"type": SpanTypeAttribute.LLM}), params)
        )
        should_end = True

        try:
            start = time.time()
            create_response = await self.acreate_fn(*args, **kwargs)

            if hasattr(create_response, "parse"):
                raw_response = create_response.parse()
                log_headers(create_response, span)
            else:
                raw_response = create_response

            if stream:

                async def gen():
                    try:
                        first = True
                        all_results = []
                        async for item in raw_response:
                            if first:
                                span.log(
                                    metrics={
                                        "time_to_first_token": time.time() - start,
                                    }
                                )
                                first = False
                            all_results.append(_try_to_dict(item))
                            yield item

                        span.log(**self._postprocess_streaming_results(all_results))
                    finally:
                        span.end()

                should_end = False
                streamer = gen()
                if raw_requested and hasattr(create_response, "parse"):
                    return _RawResponseWithTracedStream(create_response, _AsyncTracedStream(raw_response, streamer))
                return _AsyncTracedStream(raw_response, streamer)
            else:
                log_response = _try_to_dict(raw_response)
                metrics = _parse_metrics_from_usage(log_response.get("usage"))
                metrics["time_to_first_token"] = time.time() - start
                span.log(
                    metrics=metrics,
                    output=log_response["choices"],
                )
                return create_response if (raw_requested and hasattr(create_response, "parse")) else raw_response
        finally:
            if should_end:
                span.end()

    @classmethod
    def _parse_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        # First, destructively remove span_info
        ret = params.pop("span_info", {})

        # Then, copy the rest of the params
        params = prettify_params(params)
        messages = params.pop("messages", None)

        # Process attachments in input (convert data URLs to Attachment objects)
        processed_input = _process_attachments_in_input(messages)

        return merge_dicts(
            ret,
            {
                "input": processed_input,
                "metadata": {**params, "provider": "openai"},
            },
        )

    @classmethod
    def _postprocess_streaming_results(cls, all_results: list[dict[str, Any]]) -> dict[str, Any]:
        role = None
        content = None
        refusal = None
        tool_calls: list[Any] | None = None
        finish_reason = None
        logprobs_content: list[Any] | None = None
        logprobs_refusal: list[Any] | None = None
        saw_logprobs = False
        metrics: dict[str, float] = {}
        for result in all_results:
            usage = result.get("usage")
            if usage:
                metrics.update(_parse_metrics_from_usage(usage))

            choices = result["choices"]
            if not choices:
                continue

            choice = choices[0]
            fr = choice.get("finish_reason")
            if fr is not None:
                finish_reason = fr

            choice_logprobs = choice.get("logprobs")
            if choice_logprobs is not None:
                saw_logprobs = True

                chunk_content_logprobs = choice_logprobs.get("content")
                if chunk_content_logprobs is not None:
                    if logprobs_content is None:
                        logprobs_content = []
                    logprobs_content.extend(chunk_content_logprobs)

                chunk_refusal_logprobs = choice_logprobs.get("refusal")
                if chunk_refusal_logprobs is not None:
                    if logprobs_refusal is None:
                        logprobs_refusal = []
                    logprobs_refusal.extend(chunk_refusal_logprobs)

            delta = choice.get("delta")
            if not delta:
                continue

            if role is None and delta.get("role") is not None:
                role = delta.get("role")

            if delta.get("content") is not None:
                content = (content or "") + delta.get("content")

            if delta.get("refusal") is not None:
                refusal = (refusal or "") + delta.get("refusal")

            if delta.get("tool_calls") is not None:
                delta_tool_calls = delta.get("tool_calls")
                if not delta_tool_calls:
                    continue
                tool_delta = delta_tool_calls[0]

                # pylint: disable=unsubscriptable-object
                if not tool_calls or (tool_delta.get("id") and tool_calls[-1]["id"] != tool_delta.get("id")):
                    function_arg = tool_delta.get("function", {})
                    tool_calls = (tool_calls or []) + [
                        {
                            "id": tool_delta.get("id"),
                            "type": tool_delta.get("type"),
                            "function": {
                                "name": function_arg.get("name"),
                                "arguments": function_arg.get("arguments") or "",
                            },
                        }
                    ]
                else:
                    # pylint: disable=unsubscriptable-object
                    # append to existing tool call
                    function_arg = tool_delta.get("function", {})
                    args = function_arg.get("arguments") or ""
                    if isinstance(args, str):
                        # pylint: disable=unsubscriptable-object
                        tool_calls[-1]["function"]["arguments"] += args

        return {
            "metrics": metrics,
            "output": [
                {
                    "index": 0,
                    "message": {
                        "role": role,
                        "content": content,
                        "tool_calls": tool_calls,
                        **({"refusal": refusal} if refusal is not None else {}),
                    },
                    "logprobs": (
                        {
                            "content": logprobs_content,
                            "refusal": logprobs_refusal,
                        }
                        if saw_logprobs
                        else None
                    ),
                    "finish_reason": finish_reason,
                }
            ],
        }


class _TracedStream(NamedWrapper):
    """Traced sync stream. Iterates via the traced generator while delegating
    SDK-specific attributes (e.g. .close(), .response) to the original stream."""

    def __init__(self, original_stream: Any, traced_generator: Any) -> None:
        self._traced_generator = traced_generator
        super().__init__(original_stream)

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        return next(self._traced_generator)

    def __enter__(self) -> Any:
        if hasattr(self._wrapped, "__enter__"):
            self._wrapped.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        if hasattr(self._wrapped, "__exit__"):
            return self._wrapped.__exit__(exc_type, exc_val, exc_tb)
        return None


class _AsyncTracedStream(NamedWrapper):
    """Traced async stream. Iterates via the traced generator while delegating
    SDK-specific attributes (e.g. .close(), .response) to the original stream."""

    def __init__(self, original_stream: Any, traced_generator: Any) -> None:
        self._traced_generator = traced_generator
        super().__init__(original_stream)

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        return await self._traced_generator.__anext__()

    async def __aenter__(self) -> Any:
        if hasattr(self._wrapped, "__aenter__"):
            await self._wrapped.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        if hasattr(self._wrapped, "__aexit__"):
            return await self._wrapped.__aexit__(exc_type, exc_val, exc_tb)
        return None


class _RawResponseWithTracedStream(NamedWrapper):
    """Proxy for LegacyAPIResponse that replaces parse() with a traced stream,
    so that with_raw_response + stream=True preserves both headers and tracing."""

    def __init__(self, raw_response: Any, traced_stream: Any) -> None:
        self._traced_stream = traced_stream
        super().__init__(raw_response)

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        return self._traced_stream


_RESPONSE_TOOL_ITEM_INPUT_KEYS = {
    "function_call": ("arguments",),
    "web_search_call": ("action",),
    "file_search_call": ("queries",),
    "code_interpreter_call": ("code", "container_id"),
    "computer_call": ("action",),
    "image_generation_call": (),
    "mcp_call": ("arguments",),
}


def _maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _serialize_response_output_items(value: Any) -> list[dict[str, Any]]:
    serialized = _try_to_dict(value)
    if serialized is None:
        return []

    items = serialized if isinstance(serialized, list) else [serialized]
    serialized_items = []
    for item in items:
        item_dict = _try_to_dict(item)
        if isinstance(item_dict, dict):
            serialized_items.append(item_dict)
    return serialized_items


def _response_tool_span_name(item: dict[str, Any]) -> str:
    if item.get("server_label") and item.get("name"):
        return f"{item['server_label']}.{item['name']}"
    if item.get("name"):
        return str(item["name"])
    return str(item.get("type") or "response_tool")


def _response_tool_span_input(item: dict[str, Any]) -> Any:
    input_keys = _RESPONSE_TOOL_ITEM_INPUT_KEYS.get(item.get("type"), ())
    if not input_keys:
        return None

    input_data = clean_nones({key: _maybe_parse_json_string(item.get(key)) for key in input_keys})
    if not input_data:
        return None

    if input_keys == ("arguments",):
        return input_data["arguments"]
    return input_data


def _response_tool_span_output(item: dict[str, Any]) -> Any:
    if item.get("type") == "function_call":
        return None

    excluded_keys = {
        "id",
        "type",
        "name",
        "call_id",
        "server_label",
        "error",
        *(_RESPONSE_TOOL_ITEM_INPUT_KEYS.get(item.get("type"), ())),
    }
    output = clean_nones(
        {
            key: _maybe_parse_json_string(value) if key in {"output", "error"} else value
            for key, value in item.items()
            if key not in excluded_keys
        }
    )
    return output or None


def _response_tool_span_error(item: dict[str, Any]) -> Any:
    error = item.get("error")
    if error is None:
        return None
    parsed_error = _maybe_parse_json_string(error)
    if isinstance(parsed_error, (dict, list)):
        return parsed_error
    return str(parsed_error)


def _response_tool_span_metadata(item: dict[str, Any]) -> dict[str, Any] | None:
    return (
        clean_nones(
            {
                "tool_type": item.get("type"),
                "tool_id": item.get("id"),
                "call_id": item.get("call_id"),
                "status": item.get("status"),
                "server_label": item.get("server_label"),
            }
        )
        or None
    )


def _log_response_tool_spans(output: Any, *, parent_export: str | None) -> None:
    for item in _serialize_response_output_items(output):
        if item.get("type") not in _RESPONSE_TOOL_ITEM_INPUT_KEYS:
            continue

        span_args = {
            "name": _response_tool_span_name(item),
            "type": SpanTypeAttribute.TOOL,
            "input": _response_tool_span_input(item),
            "metadata": _response_tool_span_metadata(item),
        }
        if parent_export is not None:
            span_args["parent"] = parent_export

        with start_span(**span_args) as tool_span:
            error = _response_tool_span_error(item)
            if error is not None:
                tool_span.log(error=error)
                continue

            output_data = _response_tool_span_output(item)
            if output_data is not None:
                tool_span.log(output=output_data)


class ResponseWrapper:
    def __init__(
        self,
        create_fn: Callable[..., Any] | None,
        acreate_fn: Callable[..., Any] | None,
        name: str = "openai.responses.create",
        return_raw: bool = False,
    ):
        self.create_fn = create_fn
        self.acreate_fn = acreate_fn
        self.name = name
        self.return_raw = return_raw

    def create(self, *args: Any, **kwargs: Any) -> Any:
        raw_requested = self.return_raw or _raw_response_requested(kwargs)
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        span = start_span(**merge_dicts(dict(name=self.name, span_attributes={"type": SpanTypeAttribute.LLM}), params))
        should_end = True

        try:
            start = time.time()
            create_response = self.create_fn(*args, **kwargs)
            if hasattr(create_response, "parse"):
                raw_response = create_response.parse()
                log_headers(create_response, span)
            else:
                raw_response = create_response
            if stream:

                def gen():
                    try:
                        first = True
                        all_results = []
                        for item in raw_response:
                            if first:
                                span.log(
                                    metrics={
                                        "time_to_first_token": time.time() - start,
                                    }
                                )
                                first = False
                            all_results.append(item)
                            yield item

                        event_data = self._postprocess_streaming_results(all_results)
                        span.log(**event_data)
                        _log_response_tool_spans(event_data.get("output"), parent_export=span.export())
                    finally:
                        span.end()

                should_end = False
                if raw_requested and hasattr(create_response, "parse"):
                    return _RawResponseWithTracedStream(create_response, _TracedStream(raw_response, gen()))
                return _TracedStream(raw_response, gen())
            else:
                log_response = _try_to_dict(raw_response)
                event_data = self._parse_event_from_result(log_response)
                if "metrics" not in event_data:
                    event_data["metrics"] = {}
                event_data["metrics"]["time_to_first_token"] = time.time() - start
                span.log(**event_data)
                _log_response_tool_spans(event_data.get("output"), parent_export=span.export())
                return create_response if (raw_requested and hasattr(create_response, "parse")) else raw_response
        finally:
            if should_end:
                span.end()

    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        raw_requested = self.return_raw or _raw_response_requested(kwargs)
        params = self._parse_params(kwargs)
        stream = kwargs.get("stream", False)

        span = start_span(**merge_dicts(dict(name=self.name, span_attributes={"type": SpanTypeAttribute.LLM}), params))
        should_end = True

        try:
            start = time.time()
            create_response = await self.acreate_fn(*args, **kwargs)
            if hasattr(create_response, "parse"):
                raw_response = create_response.parse()
                log_headers(create_response, span)
            else:
                raw_response = create_response
            if stream:

                async def gen():
                    try:
                        first = True
                        all_results = []
                        async for item in raw_response:
                            if first:
                                span.log(
                                    metrics={
                                        "time_to_first_token": time.time() - start,
                                    }
                                )
                                first = False
                            all_results.append(item)
                            yield item

                        event_data = self._postprocess_streaming_results(all_results)
                        span.log(**event_data)
                        _log_response_tool_spans(event_data.get("output"), parent_export=span.export())
                    finally:
                        span.end()

                should_end = False
                streamer = gen()
                if raw_requested and hasattr(create_response, "parse"):
                    return _RawResponseWithTracedStream(create_response, _AsyncTracedStream(raw_response, streamer))
                return _AsyncTracedStream(raw_response, streamer)
            else:
                log_response = _try_to_dict(raw_response)
                event_data = self._parse_event_from_result(log_response)
                if "metrics" not in event_data:
                    event_data["metrics"] = {}
                event_data["metrics"]["time_to_first_token"] = time.time() - start
                span.log(**event_data)
                _log_response_tool_spans(event_data.get("output"), parent_export=span.export())
                return create_response if (raw_requested and hasattr(create_response, "parse")) else raw_response
        finally:
            if should_end:
                span.end()

    @classmethod
    def _parse_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        # First, destructively remove span_info
        ret = params.pop("span_info", {})

        # Then, copy the rest of the params
        params = prettify_params(params)
        input_data = params.pop("input", None)

        # Process attachments in input (convert data URLs to Attachment objects)
        processed_input = _process_attachments_in_input(input_data)

        return merge_dicts(
            ret,
            {
                "input": processed_input,
                "metadata": {**params, "provider": "openai"},
            },
        )

    @classmethod
    def _parse_event_from_result(cls, result: dict[str, Any]) -> dict[str, Any]:
        """Parse event from response result"""
        data = {"metrics": {}}

        if not result:
            return data

        if "output" in result:
            data["output"] = result["output"]

        metadata = {k: v for k, v in result.items() if k not in ["output", "usage"]}
        if metadata:
            data["metadata"] = metadata

        if "usage" in result:
            data["metrics"] = _parse_metrics_from_usage(result["usage"])

        return data

    @classmethod
    def _postprocess_streaming_results(cls, all_results: list[Any]) -> dict[str, Any]:
        """Process streaming results - minimal version focused on metrics extraction."""
        metrics = {}
        output = []

        for result in all_results:
            usage = getattr(result, "usage", None)
            if (
                not usage
                and hasattr(result, "type")
                and result.type == "response.completed"
                and hasattr(result, "response")
            ):
                # Handle summaries from completed response if present
                if hasattr(result.response, "output") and result.response.output:
                    output_by_id = {item.get("id"): item for item in output if item.get("id")}
                    for output_item in result.response.output:
                        if hasattr(output_item, "summary") and output_item.summary:
                            matched = output_by_id.get(output_item.id)
                            if matched:
                                matched["summary"] = output_item.summary
                usage = getattr(result.response, "usage", None)

            if usage:
                parsed_metrics = _parse_metrics_from_usage(usage)
                metrics.update(parsed_metrics)

            # Skip processing if result doesn't have a type attribute
            if not hasattr(result, "type"):
                continue

            if result.type == "response.output_item.added":
                item_data = {"id": result.item.id, "type": result.item.type}
                if hasattr(result.item, "role"):
                    item_data["role"] = result.item.role
                output.append(item_data)
                continue

            if result.type == "response.completed":
                if hasattr(result, "response") and hasattr(result.response, "output"):
                    return {
                        "metrics": metrics,
                        "output": result.response.output,
                    }
                continue

            # Handle output_index based updates
            if hasattr(result, "output_index"):
                output_index = result.output_index
                if output_index < len(output):
                    current_output = output[output_index]

                    if result.type == "response.output_item.done":
                        current_output["status"] = result.item.status
                        continue

                    if result.type == "response.output_item.delta":
                        current_output["delta"] = result.delta
                        continue

                    # Handle content_index based updates
                    if hasattr(result, "content_index"):
                        if "content" not in current_output:
                            current_output["content"] = []
                        content_index = result.content_index
                        # Fill any gaps in the content array
                        while len(current_output["content"]) <= content_index:
                            current_output["content"].append({})
                        current_content = current_output["content"][content_index]
                        current_content["type"] = "output_text"
                        if hasattr(result, "delta") and result.delta:
                            current_content["text"] = (current_content.get("text") or "") + result.delta

                        if result.type == "response.output_text.annotation.added":
                            annotation_index = result.annotation_index
                            if "annotations" not in current_content:
                                current_content["annotations"] = []
                            # Fill any gaps in the annotations array
                            while len(current_content["annotations"]) <= annotation_index:
                                current_content["annotations"].append({})
                            current_content["annotations"][annotation_index] = _try_to_dict(result.annotation)

        return {
            "metrics": metrics,
            "output": output,
        }


def _image_attachment_from_base64(
    data: Any,
    *,
    output_format: Any,
    index: int,
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


def _extract_images_output(response: dict[str, Any]) -> dict[str, Any]:
    images = []
    output_format = response.get("output_format")

    for index, image in enumerate(response.get("data") or []):
        image_dict = _try_to_dict(image)
        if not isinstance(image_dict, dict):
            continue

        image_entry = clean_nones(
            {
                "revised_prompt": image_dict.get("revised_prompt"),
            }
        )

        if isinstance(image_dict.get("url"), str):
            image_entry["image_url"] = {"url": image_dict["url"]}

        b64_json = image_dict.get("b64_json")
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
            "created": response.get("created"),
            "background": response.get("background"),
            "output_format": output_format,
            "quality": response.get("quality"),
            "size": response.get("size"),
            "images_count": len(images),
            "images": images,
        }
    )


class BaseWrapper(abc.ABC):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None, name: str):
        self._create_fn = create_fn
        self._acreate_fn = acreate_fn
        self._name = name

    @abc.abstractmethod
    def process_output(self, response: dict[str, Any], span: Span):
        """Process the API response and log relevant information to the span."""
        pass

    def create(self, *args: Any, **kwargs: Any) -> Any:
        params = self._parse_params(kwargs)

        with start_span(
            **merge_dicts(dict(name=self._name, span_attributes={"type": SpanTypeAttribute.LLM}), params)
        ) as span:
            create_response = self._create_fn(*args, **kwargs)
            if hasattr(create_response, "parse"):
                raw_response = create_response.parse()
                log_headers(create_response, span)
            else:
                raw_response = create_response

            log_response = _try_to_dict(raw_response)
            self.process_output(log_response, span)
            return raw_response

    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        params = self._parse_params(kwargs)

        with start_span(
            **merge_dicts(dict(name=self._name, span_attributes={"type": SpanTypeAttribute.LLM}), params)
        ) as span:
            create_response = await self._acreate_fn(*args, **kwargs)
            if hasattr(create_response, "parse"):
                raw_response = create_response.parse()
                log_headers(create_response, span)
            else:
                raw_response = create_response
            log_response = _try_to_dict(raw_response)
            self.process_output(log_response, span)
            return raw_response

    @classmethod
    def _parse_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        # First, destructively remove span_info
        ret = params.pop("span_info", {})

        params = prettify_params(params)
        input_data = params.pop("input", None)

        # Process attachments in input (convert data URLs to Attachment objects)
        processed_input = _process_attachments_in_input(input_data)

        return merge_dicts(
            ret,
            {
                "input": processed_input,
                "metadata": {**params, "provider": "openai"},
            },
        )


class _ImageBaseWrapper(BaseWrapper):
    def _log_result(self, log_response: Any, start: float, span: Span) -> None:
        end = time.time()
        metrics = _timing_metrics(start, end)
        if isinstance(log_response, dict):
            metrics.update(_parse_metrics_from_usage(log_response.get("usage")))
        output = _extract_images_output(log_response) if isinstance(log_response, dict) else log_response
        span.log(metrics=metrics, output=output)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        params = self._parse_params(kwargs)

        with start_span(
            **merge_dicts(dict(name=self._name, span_attributes={"type": SpanTypeAttribute.LLM}), params)
        ) as span:
            start = time.time()
            create_response = self._create_fn(*args, **kwargs)
            if hasattr(create_response, "parse"):
                raw_response = create_response.parse()
                log_headers(create_response, span)
            else:
                raw_response = create_response

            self._log_result(_try_to_dict(raw_response), start, span)
            return raw_response

    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        params = self._parse_params(kwargs)

        with start_span(
            **merge_dicts(dict(name=self._name, span_attributes={"type": SpanTypeAttribute.LLM}), params)
        ) as span:
            start = time.time()
            create_response = await self._acreate_fn(*args, **kwargs)
            if hasattr(create_response, "parse"):
                raw_response = create_response.parse()
                log_headers(create_response, span)
            else:
                raw_response = create_response

            self._log_result(_try_to_dict(raw_response), start, span)
            return raw_response

    def process_output(self, response: Any, span: Span):
        output = _extract_images_output(response) if isinstance(response, dict) else response
        span.log(output=output)

    @classmethod
    def _parse_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        ret = params.pop("span_info", {})
        params = prettify_params(params)
        prompt = params.pop("prompt", None)
        image = params.pop("image", None)
        mask = params.pop("mask", None)

        input_data = clean_nones(
            {
                "prompt": prompt,
                "image": _materialize_logged_file_input(image),
                "mask": _materialize_logged_file_input(mask),
            }
        )

        return merge_dicts(
            ret,
            {
                "input": prompt if (prompt is not None and len(input_data) == 1) else (input_data or None),
                "metadata": {**params, "provider": "openai"},
            },
        )


class ImageGenerateWrapper(_ImageBaseWrapper):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        super().__init__(create_fn, acreate_fn, "Image Generation")


class ImageEditWrapper(_ImageBaseWrapper):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        super().__init__(create_fn, acreate_fn, "Image Edit")


class ImageVariationWrapper(_ImageBaseWrapper):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        super().__init__(create_fn, acreate_fn, "Image Variation")


class EmbeddingWrapper(BaseWrapper):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        super().__init__(create_fn, acreate_fn, "Embedding")

    def process_output(self, response: dict[str, Any], span: Span):
        usage = response.get("usage")
        metrics = _parse_metrics_from_usage(usage)
        span.log(
            metrics=metrics,
            # TODO: Add a flag to control whether to log the full embedding vector,
            # possibly w/ JSON compression.
            output={"embedding_length": len(response["data"][0]["embedding"])},
        )


class ModerationWrapper(BaseWrapper):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        super().__init__(create_fn, acreate_fn, "Moderation")

    def process_output(self, response: Any, span: Span):
        span.log(
            output=response["results"],
        )


class SpeechWrapper(BaseWrapper):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        super().__init__(create_fn, acreate_fn, "Speech")

    def process_output(self, response: Any, span: Span):
        span.log(output=_extract_audio_output(response, prefix="generated_speech"))


class _AudioFileWrapper(BaseWrapper):
    """Base for transcription/translation wrappers that accept audio file input."""

    @classmethod
    def _parse_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        ret = params.pop("span_info", {})
        params = prettify_params(params)
        # Remove the file object after prettifying — prettify_params already
        # made a copy so the original kwargs (used by the API call) are preserved.
        file_input = _materialize_logged_file_input(params.pop("file", None))
        input_data = {"file": file_input} if file_input is not None else None
        return merge_dicts(
            ret,
            {
                "input": input_data,
                "metadata": {**params, "provider": "openai"},
            },
        )

    @staticmethod
    def _extract_text(response: Any) -> str | None:
        if isinstance(response, dict):
            return response.get("text")
        if isinstance(response, str):
            return response.strip()
        return getattr(response, "text", None)


class TranscriptionWrapper(_AudioFileWrapper):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        super().__init__(create_fn, acreate_fn, "Transcription")

    def process_output(self, response: Any, span: Span):
        metrics = {}
        if isinstance(response, dict):
            usage = response.get("usage")
            if usage:
                metrics = _parse_metrics_from_usage(usage)
        span.log(metrics=metrics, output=self._extract_text(response))


class TranslationWrapper(_AudioFileWrapper):
    def __init__(self, create_fn: Callable[..., Any] | None, acreate_fn: Callable[..., Any] | None):
        super().__init__(create_fn, acreate_fn, "Translation")

    def process_output(self, response: Any, span: Span):
        span.log(output=self._extract_text(response))


_audio_speech_create_wrapper = _make_base_wrapper_callback(SpeechWrapper)
_audio_transcription_create_wrapper = _make_base_wrapper_callback(TranscriptionWrapper)
_audio_translation_create_wrapper = _make_base_wrapper_callback(TranslationWrapper)


_image_generate_wrapper = _make_base_wrapper_callback(ImageGenerateWrapper, method_name="generate")
_image_edit_wrapper = _make_base_wrapper_callback(ImageEditWrapper, method_name="edit")
_image_create_variation_wrapper = _make_base_wrapper_callback(ImageVariationWrapper, method_name="create_variation")


# OpenAI's representation to Braintrust's representation
TOKEN_NAME_MAP = {
    # chat API
    "total_tokens": "tokens",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    # responses API
    "tokens": "tokens",
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
}

TOKEN_PREFIX_MAP = {
    "input": "prompt",
    "output": "completion",
}


def _parse_metrics_from_usage(usage: Any) -> dict[str, Any]:
    return _parse_openai_usage_metrics(
        usage,
        token_name_map=TOKEN_NAME_MAP,
        token_prefix_map=TOKEN_PREFIX_MAP,
    )


def prettify_params(params: dict[str, Any]) -> dict[str, Any]:
    # Filter out NOT_GIVEN parameters
    # https://linear.app/braintrustdata/issue/BRA-2467
    return _prettify_response_params(params, drop_not_given=True)
