import logging
import time
import warnings
from typing import Any

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.integrations.anthropic._utils import Wrapper, _try_to_dict, extract_anthropic_usage
from braintrust.integrations.utils import _materialize_attachment
from braintrust.logger import log_exc_info_to_span, start_span
from braintrust.span_types import SpanTypeAttribute


log = logging.getLogger(__name__)


# This tracer depends on an internal anthropic method used to merge
# streamed messages together. It's a bit tricky so I'm opting to use it
# here. If it goes away, this polyfill will make it a no-op and the only
# result will be missing `output` and metrics in our spans. Our tests always
# run against the latest version of anthropic's SDK, so we'll know.
# anthropic-sdk-python/blob/main/src/anthropic/lib/streaming/_messages.py#L392
try:
    from anthropic.lib.streaming._messages import accumulate_event
except ImportError:

    def accumulate_event(event=None, current_snapshot=None, **kwargs):
        warnings.warn("braintrust: missing method: anthropic.lib.streaming._messages.accumulate_event")
        return current_snapshot


# Anthropic model parameters that we want to track as span metadata.
METADATA_PARAMS = (
    "model",
    "max_tokens",
    "temperature",
    "top_k",
    "top_p",
    "stop_sequences",
    "tool_choice",
    "tools",
    "stream",
    "thinking",
    "output_config",
    "output_format",
)


class TracedAsyncAnthropic(Wrapper):
    def __init__(self, client):
        super().__init__(client)
        self.__client = client

    @property
    def messages(self):
        return AsyncMessages(self.__client.messages)

    @property
    def beta(self):
        return AsyncBeta(self.__client.beta)


class AsyncMessages(Wrapper):
    def __init__(self, messages):
        super().__init__(messages)
        self.__messages = messages

    @property
    def batches(self):
        return AsyncBatches(self.__messages.batches)

    async def create(self, *args, **kwargs):
        if kwargs.get("stream", False):
            return await self.__create_with_stream_true(*args, **kwargs)
        else:
            return await self.__create_with_stream_false(*args, **kwargs)

    async def __create_with_stream_false(self, *args, **kwargs):
        span = _start_span("anthropic.messages.create", kwargs)
        request_start_time = time.time()
        try:
            result = await self.__messages.create(*args, **kwargs)
            ttft = time.time() - request_start_time
            _log_message_to_span(result, span, time_to_first_token=ttft)
            return result
        except Exception as e:
            span.log(error=e)
            raise
        finally:
            span.end()

    async def __create_with_stream_true(self, *args, **kwargs):
        span = _start_span("anthropic.messages.stream", kwargs)
        request_start_time = time.time()
        try:
            stream = await self.__messages.create(*args, **kwargs)
        except Exception as e:
            span.log(error=e)
            span.end()
            raise

        traced_stream = TracedMessageStream(stream, span, request_start_time)

        async def async_stream():
            try:
                async for msg in traced_stream:
                    yield msg
            except Exception as e:
                span.log(error=e)
                raise
            finally:
                msg = traced_stream._get_final_traced_message()
                if msg:
                    ttft = traced_stream._get_time_to_first_token()
                    _log_message_to_span(msg, span, time_to_first_token=ttft)
                span.end()

        return async_stream()

    def stream(self, *args, **kwargs):
        span = _start_span("anthropic.messages.stream", kwargs)
        request_start_time = time.time()
        stream = self.__messages.stream(*args, **kwargs)
        return TracedMessageStreamManager(stream, span, request_start_time)


class AsyncBeta(Wrapper):
    def __init__(self, beta):
        super().__init__(beta)
        self.__beta = beta

    @property
    def messages(self):
        return AsyncMessages(self.__beta.messages)


class TracedAnthropic(Wrapper):
    def __init__(self, client):
        super().__init__(client)
        self.__client = client

    @property
    def messages(self):
        return Messages(self.__client.messages)

    @property
    def beta(self):
        return Beta(self.__client.beta)


class Messages(Wrapper):
    def __init__(self, messages):
        super().__init__(messages)
        self.__messages = messages

    @property
    def batches(self):
        return Batches(self.__messages.batches)

    def stream(self, *args, **kwargs):
        return self.__trace_stream(self.__messages.stream, *args, **kwargs)

    def create(self, *args, **kwargs):
        if kwargs.get("stream"):
            return self.__trace_stream(self.__messages.create, *args, **kwargs)

        span = _start_span("anthropic.messages.create", kwargs)
        request_start_time = time.time()
        try:
            msg = self.__messages.create(*args, **kwargs)
            ttft = time.time() - request_start_time
            _log_message_to_span(msg, span, time_to_first_token=ttft)
            return msg
        except Exception as e:
            span.log(error=e)
            raise
        finally:
            span.end()

    def __trace_stream(self, stream_func, *args, **kwargs):
        span = _start_span("anthropic.messages.stream", kwargs)
        request_start_time = time.time()
        s = stream_func(*args, **kwargs)
        return TracedMessageStreamManager(s, span, request_start_time)


class Beta(Wrapper):
    def __init__(self, beta):
        super().__init__(beta)
        self.__beta = beta

    @property
    def messages(self):
        return Messages(self.__beta.messages)


class Batches(Wrapper):
    """Wrapper for sync Anthropic Messages Batches resource."""

    def __init__(self, batches):
        super().__init__(batches)
        self.__batches = batches

    def create(self, *args, **kwargs):
        span = _start_batch_create_span(kwargs)
        try:
            result = self.__batches.create(*args, **kwargs)
            _log_batch_create_to_span(result, span)
            return result
        except Exception as e:
            span.log(error=e)
            raise
        finally:
            span.end()

    def results(self, *args, **kwargs):
        span = _start_batch_results_span(args, kwargs)
        try:
            result = self.__batches.results(*args, **kwargs)
            span.log(output={"type": "jsonl_stream"})
            return result
        except Exception as e:
            span.log(error=e)
            raise
        finally:
            span.end()


class AsyncBatches(Wrapper):
    """Wrapper for async Anthropic Messages Batches resource."""

    def __init__(self, batches):
        super().__init__(batches)
        self.__batches = batches

    async def create(self, *args, **kwargs):
        span = _start_batch_create_span(kwargs)
        try:
            result = await self.__batches.create(*args, **kwargs)
            _log_batch_create_to_span(result, span)
            return result
        except Exception as e:
            span.log(error=e)
            raise
        finally:
            span.end()

    async def results(self, *args, **kwargs):
        span = _start_batch_results_span(args, kwargs)
        try:
            result = await self.__batches.results(*args, **kwargs)
            span.log(output={"type": "jsonl_stream"})
            return result
        except Exception as e:
            span.log(error=e)
            raise
        finally:
            span.end()


class TracedMessageStreamManager(Wrapper):
    def __init__(self, msg_stream_mgr, span, request_start_time: float):
        super().__init__(msg_stream_mgr)
        self.__msg_stream_mgr = msg_stream_mgr
        self.__traced_message_stream = None
        self.__span = span
        self.__request_start_time = request_start_time

    async def __aenter__(self):
        ms = await self.__msg_stream_mgr.__aenter__()
        self.__traced_message_stream = TracedMessageStream(ms, self.__span, self.__request_start_time)
        return self.__traced_message_stream

    def __enter__(self):
        ms = self.__msg_stream_mgr.__enter__()
        self.__traced_message_stream = TracedMessageStream(ms, self.__span, self.__request_start_time)
        return self.__traced_message_stream

    def __aexit__(self, exc_type, exc_value, traceback):
        try:
            return self.__msg_stream_mgr.__aexit__(exc_type, exc_value, traceback)
        finally:
            self.__close(exc_type, exc_value, traceback)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.__msg_stream_mgr.__exit__(exc_type, exc_value, traceback)
        finally:
            self.__close(exc_type, exc_value, traceback)

    def __close(self, exc_type, exc_value, traceback):
        tms = self.__traced_message_stream
        msg = tms._get_final_traced_message()
        if msg:
            ttft = tms._get_time_to_first_token()
            _log_message_to_span(msg, self.__span, time_to_first_token=ttft)
        if exc_type:
            log_exc_info_to_span(self.__span, exc_type, exc_value, traceback)
        self.__span.end()


class TracedMessageStream(Wrapper):
    """TracedMessageStream wraps both sync and async message streams."""

    def __init__(self, msg_stream, span, request_start_time: float):
        super().__init__(msg_stream)
        self.__msg_stream = msg_stream
        self.__span = span
        self.__metrics = {}
        self.__snapshot = None
        self.__request_start_time = request_start_time
        self.__time_to_first_token: float | None = None

    def _get_final_traced_message(self):
        return self.__snapshot

    def _get_time_to_first_token(self):
        return self.__time_to_first_token

    def __await__(self):
        return self.__msg_stream.__await__()

    def __aiter__(self):
        return self

    def __iter__(self):
        return self

    async def __anext__(self):
        m = await self.__msg_stream.__anext__()
        self.__process_message(m)
        return m

    def __next__(self):
        m = next(self.__msg_stream)
        self.__process_message(m)
        return m

    @property
    def text_stream(self):
        if hasattr(self.__msg_stream, "__aiter__"):
            return self.__async_text_stream()
        return self.__sync_text_stream()

    def __sync_text_stream(self):
        for event in self:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "text_delta":
                    yield getattr(delta, "text", "")

    async def __async_text_stream(self):
        async for event in self:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "text_delta":
                    yield getattr(delta, "text", "")

    def __process_message(self, m):
        if self.__time_to_first_token is None:
            self.__time_to_first_token = time.time() - self.__request_start_time

        self.__snapshot = accumulate_event(event=m, current_snapshot=self.__snapshot)


def _start_batch_create_span(kwargs):
    requests = list(kwargs.get("requests", []))
    # Extract models from the batch requests for metadata
    models = set()
    for req in requests:
        params = req.get("params", {}) if isinstance(req, dict) else getattr(req, "params", {})
        model = params.get("model") if isinstance(params, dict) else getattr(params, "model", None)
        if model:
            models.add(model)

    metadata = {"provider": "anthropic", "num_requests": len(requests)}
    if len(models) == 1:
        metadata["model"] = next(iter(models))
    elif models:
        metadata["models"] = sorted(models)

    _input = [
        {"custom_id": req.get("custom_id") if isinstance(req, dict) else getattr(req, "custom_id", None)}
        for req in requests
    ]

    return start_span(name="anthropic.messages.batches.create", type="task", metadata=metadata, input=_input)


def _log_batch_create_to_span(result, span):
    output = {}
    if hasattr(result, "id"):
        output["id"] = result.id
    if hasattr(result, "processing_status"):
        output["processing_status"] = result.processing_status
    if hasattr(result, "request_counts"):
        rc = result.request_counts
        output["request_counts"] = {
            "processing": getattr(rc, "processing", 0),
            "succeeded": getattr(rc, "succeeded", 0),
            "errored": getattr(rc, "errored", 0),
            "canceled": getattr(rc, "canceled", 0),
            "expired": getattr(rc, "expired", 0),
        }

    span.log(output=output)


def _start_batch_results_span(args, kwargs):
    # message_batch_id can be passed as first positional arg or as kwarg
    batch_id = args[0] if args else kwargs.get("message_batch_id")
    metadata = {"provider": "anthropic"}
    _input = {"message_batch_id": batch_id}
    return start_span(name="anthropic.messages.batches.results", type="task", metadata=metadata, input=_input)


def _convert_base64_source_to_attachment(block_type, source):
    if not isinstance(source, dict):
        return None
    if source.get("type") != "base64":
        return None

    media_type = source.get("media_type")
    data = source.get("data")
    if not isinstance(media_type, str) or not isinstance(data, str):
        return None

    return _materialize_attachment(
        data,
        mime_type=media_type,
        prefix="image" if block_type == "image" else "document",
    )


def _process_input_attachments(value):
    if isinstance(value, list):
        return [_process_input_attachments(item) for item in value]

    if isinstance(value, dict):
        block_type = value.get("type")
        source = value.get("source")

        if block_type in {"image", "document"} and isinstance(source, dict):
            resolved_attachment = _convert_base64_source_to_attachment(block_type, source)
            if resolved_attachment is not None:
                processed = {k: _process_input_attachments(v) for k, v in value.items() if k != "source"}
                processed["source"] = {k: _process_input_attachments(v) for k, v in source.items() if k != "data"}
                processed.update(resolved_attachment.multimodal_part_payload)
                return processed

        return {k: _process_input_attachments(v) for k, v in value.items()}

    return value


def _get_input_from_kwargs(kwargs):
    msgs = bt_safe_deep_copy(list(kwargs.get("messages", [])))
    msgs = _process_input_attachments(msgs)

    system = bt_safe_deep_copy(kwargs.get("system", None))
    if system:
        msgs.append({"role": "system", "content": _process_input_attachments(system)})
    return msgs


def _get_metadata_from_kwargs(kwargs):
    metadata = {"provider": "anthropic"}
    for k in METADATA_PARAMS:
        v = kwargs.get(k, None)
        if v is not None:
            metadata[k] = v
    return metadata


def _start_span(name, kwargs):
    _input = _get_input_from_kwargs(kwargs)
    metadata = _get_metadata_from_kwargs(kwargs)
    return start_span(name=name, type="llm", metadata=metadata, input=_input)


_SERVER_TOOL_USE_TYPE = "server_tool_use"


def _is_server_tool_result_type(item_type: Any) -> bool:
    return isinstance(item_type, str) and item_type.endswith("_tool_result") and item_type != "tool_result"


def _tool_span_name(call_item: dict[str, Any] | None, result_item: dict[str, Any] | None) -> str:
    if isinstance((call_item or {}).get("name"), str):
        return call_item["name"]

    result_type = (result_item or {}).get("type")
    if isinstance(result_type, str) and result_type.endswith("_tool_result"):
        return result_type.removesuffix("_tool_result")

    return "server_tool"


def _tool_span_input(call_item: dict[str, Any] | None) -> Any:
    if not call_item:
        return None

    if call_item.get("input") is not None:
        return call_item["input"]

    return {key: value for key, value in call_item.items() if key not in {"id", "type", "name", "caller"}} or None


_SERVER_TOOL_SPAN_REDACTED_KEYS = frozenset({"encrypted_content"})


def _redact_server_tool_output(value: Any) -> Any:
    value = bt_safe_deep_copy(value)

    def _redact(inner: Any) -> Any:
        if isinstance(inner, list):
            return [_redact(item) for item in inner]
        if isinstance(inner, dict):
            return {
                key: ("<redacted>" if key in _SERVER_TOOL_SPAN_REDACTED_KEYS else _redact(item))
                for key, item in inner.items()
            }
        return inner

    return _redact(value)


def _tool_span_output(result_item: dict[str, Any] | None) -> Any:
    if not result_item:
        return None

    if "content" in result_item:
        return _redact_server_tool_output(result_item.get("content"))

    return (
        _redact_server_tool_output(
            {key: value for key, value in result_item.items() if key not in {"tool_use_id", "type", "caller"}}
        )
        or None
    )


def _tool_span_error(result_item: dict[str, Any] | None) -> str | None:
    if not result_item:
        return None

    content = result_item.get("content")
    if not isinstance(content, dict):
        content = _try_to_dict(content)
    if not isinstance(content, dict):
        return None

    content_type = content.get("type")
    if not (isinstance(content_type, str) and content_type.endswith("_error")):
        return None

    error_message = content.get("error_message")
    if isinstance(error_message, str) and error_message:
        return error_message

    error_code = content.get("error_code")
    if isinstance(error_code, str) and error_code:
        return error_code

    return content_type


def _tool_span_metadata(call_item: dict[str, Any] | None, result_item: dict[str, Any] | None) -> dict[str, Any] | None:
    metadata = {
        key: value
        for key, value in {
            "tool_use_id": (call_item or {}).get("id") or (result_item or {}).get("tool_use_id"),
            "tool_call_type": (call_item or {}).get("type"),
            "tool_result_type": (result_item or {}).get("type"),
            "caller": (call_item or {}).get("caller") or (result_item or {}).get("caller"),
        }.items()
        if value is not None
    }
    return metadata or None


def _log_server_tool_span(parent_span, call_item: dict[str, Any] | None, result_item: dict[str, Any] | None) -> None:
    tool_span = start_span(
        name=_tool_span_name(call_item, result_item),
        type=SpanTypeAttribute.TOOL,
        parent=parent_span.export(),
        input=_tool_span_input(call_item),
        metadata=_tool_span_metadata(call_item, result_item),
    )
    try:
        output = _tool_span_output(result_item)
        if output is None:
            return

        error = _tool_span_error(result_item)
        if error is not None:
            tool_span.log(output=output, error=error)
        else:
            tool_span.log(output=output)
    finally:
        tool_span.end()


def _log_server_tool_spans(content: Any, parent_span) -> None:
    if not isinstance(content, list):
        return

    calls_by_id: dict[str, dict[str, Any]] = {}
    pending_results_by_id: dict[str, list[dict[str, Any]]] = {}
    matched_call_ids: set[str] = set()
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []

    for item in content:
        item = _try_to_dict(item)
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == _SERVER_TOOL_USE_TYPE:
            call_id = item.get("id")
            if isinstance(call_id, str):
                calls_by_id[call_id] = item
                for pending_result in pending_results_by_id.pop(call_id, []):
                    pairs.append((item, pending_result))
                    matched_call_ids.add(call_id)
            else:
                pairs.append((item, None))
            continue

        if not _is_server_tool_result_type(item_type):
            continue

        tool_use_id = item.get("tool_use_id")
        if isinstance(tool_use_id, str) and tool_use_id in calls_by_id:
            pairs.append((calls_by_id[tool_use_id], item))
            matched_call_ids.add(tool_use_id)
        elif isinstance(tool_use_id, str):
            pending_results_by_id.setdefault(tool_use_id, []).append(item)
        else:
            pairs.append((None, item))

    for call_item, result_item in pairs:
        _log_server_tool_span(parent_span, call_item, result_item)

    for call_id, call_item in calls_by_id.items():
        if call_id not in matched_call_ids:
            _log_server_tool_span(parent_span, call_item, None)

    for pending_results in pending_results_by_id.values():
        for result_item in pending_results:
            _log_server_tool_span(parent_span, None, result_item)


def _log_message_to_span(message, span, time_to_first_token: float | None = None):
    usage = getattr(message, "usage", {})
    metrics, metadata = extract_anthropic_usage(usage)

    if time_to_first_token is not None:
        metrics["time_to_first_token"] = time_to_first_token

    content = getattr(message, "content", None)
    output = {
        k: v
        for k, v in {
            "role": getattr(message, "role", None),
            "content": content,
            "model": getattr(message, "model", None),
            "stop_reason": getattr(message, "stop_reason", None),
            "stop_sequence": getattr(message, "stop_sequence", None),
        }.items()
        if v is not None
    } or None

    span.log(output=output, metrics=metrics, metadata=metadata)
    _log_server_tool_spans(content, span)


_BRAINTRUST_TRACED = "__braintrust_traced__"


def _wrap_anthropic(client):
    """Wrap an Anthropic object (or AsyncAnthropic) to add tracing.

    If the client is already traced (e.g. via ``AnthropicIntegration.setup()``),
    it is returned unchanged to avoid double-wrapping.
    """
    if getattr(client, _BRAINTRUST_TRACED, False):
        return client

    type_name = getattr(type(client), "__name__")
    if "AsyncAnthropic" in type_name:
        return TracedAsyncAnthropic(client)
    elif "Anthropic" in type_name:
        return TracedAnthropic(client)
    else:
        return client


wrap_anthropic = _wrap_anthropic


def _apply_anthropic_wrapper(client):
    if getattr(client, _BRAINTRUST_TRACED, False):
        return
    wrapped = _wrap_anthropic(client)
    client.messages = wrapped.messages
    if hasattr(wrapped, "beta"):
        client.beta = wrapped.beta
    setattr(client, _BRAINTRUST_TRACED, True)


def _apply_async_anthropic_wrapper(client):
    if getattr(client, _BRAINTRUST_TRACED, False):
        return
    wrapped = _wrap_anthropic(client)
    client.messages = wrapped.messages
    if hasattr(wrapped, "beta"):
        client.beta = wrapped.beta
    setattr(client, _BRAINTRUST_TRACED, True)


def _anthropic_init_wrapper(wrapped, instance, args, kwargs):
    wrapped(*args, **kwargs)
    _apply_anthropic_wrapper(instance)


def _async_anthropic_init_wrapper(wrapped, instance, args, kwargs):
    wrapped(*args, **kwargs)
    _apply_async_anthropic_wrapper(instance)
