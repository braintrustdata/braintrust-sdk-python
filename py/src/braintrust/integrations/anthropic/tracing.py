import base64
import logging
import time
import warnings
from contextlib import contextmanager

from braintrust.bt_json import bt_safe_deep_copy
from braintrust.integrations.anthropic._utils import Wrapper, extract_anthropic_usage
from braintrust.logger import NOOP_SPAN, Attachment, log_exc_info_to_span, start_span


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
            with _catch_exceptions():
                _log_message_to_span(result, span, time_to_first_token=ttft)
            return result
        except Exception as e:
            with _catch_exceptions():
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
            with _catch_exceptions():
                span.log(error=e)
                span.end()
            raise

        traced_stream = TracedMessageStream(stream, span, request_start_time)

        async def async_stream():
            try:
                async for msg in traced_stream:
                    yield msg
            except Exception as e:
                with _catch_exceptions():
                    span.log(error=e)
                raise
            finally:
                with _catch_exceptions():
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
            with _catch_exceptions():
                _log_batch_create_to_span(result, span)
            return result
        except Exception as e:
            with _catch_exceptions():
                span.log(error=e)
            raise
        finally:
            span.end()

    def results(self, *args, **kwargs):
        span = _start_batch_results_span(args, kwargs)
        try:
            result = self.__batches.results(*args, **kwargs)
            with _catch_exceptions():
                span.log(output={"type": "jsonl_stream"})
            return result
        except Exception as e:
            with _catch_exceptions():
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
            with _catch_exceptions():
                _log_batch_create_to_span(result, span)
            return result
        except Exception as e:
            with _catch_exceptions():
                span.log(error=e)
            raise
        finally:
            span.end()

    async def results(self, *args, **kwargs):
        span = _start_batch_results_span(args, kwargs)
        try:
            result = await self.__batches.results(*args, **kwargs)
            with _catch_exceptions():
                span.log(output={"type": "jsonl_stream"})
            return result
        except Exception as e:
            with _catch_exceptions():
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
            with _catch_exceptions():
                self.__close(exc_type, exc_value, traceback)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.__msg_stream_mgr.__exit__(exc_type, exc_value, traceback)
        finally:
            with _catch_exceptions():
                self.__close(exc_type, exc_value, traceback)

    def __close(self, exc_type, exc_value, traceback):
        with _catch_exceptions():
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
        with _catch_exceptions():
            self.__process_message(m)
        return m

    def __next__(self):
        m = next(self.__msg_stream)
        with _catch_exceptions():
            self.__process_message(m)
        return m

    def __process_message(self, m):
        if self.__time_to_first_token is None:
            self.__time_to_first_token = time.time() - self.__request_start_time

        with _catch_exceptions():
            self.__snapshot = accumulate_event(event=m, current_snapshot=self.__snapshot)


def _start_batch_create_span(kwargs):
    with _catch_exceptions():
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

    return NOOP_SPAN


def _log_batch_create_to_span(result, span):
    with _catch_exceptions():
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
    with _catch_exceptions():
        # message_batch_id can be passed as first positional arg or as kwarg
        batch_id = args[0] if args else kwargs.get("message_batch_id")
        metadata = {"provider": "anthropic"}
        _input = {"message_batch_id": batch_id}
        return start_span(name="anthropic.messages.batches.results", type="task", metadata=metadata, input=_input)

    return NOOP_SPAN


def _attachment_filename_for_media_type(media_type: str, block_type: str) -> str:
    extension = media_type.split("/", 1)[1] if "/" in media_type else "bin"
    extension = extension.split("+", 1)[0]
    prefix = "image" if block_type == "image" else "document"
    return f"{prefix}.{extension}"


def _convert_base64_source_to_attachment(block_type, source):
    if not isinstance(source, dict):
        return None
    if source.get("type") != "base64":
        return None

    media_type = source.get("media_type")
    data = source.get("data")
    if not isinstance(media_type, str) or not isinstance(data, str):
        return None

    try:
        binary_data = base64.b64decode(data, validate=True)
    except Exception:
        return None

    return Attachment(
        data=binary_data,
        filename=_attachment_filename_for_media_type(media_type, block_type),
        content_type=media_type,
    )


def _process_input_attachments(value):
    if isinstance(value, list):
        return [_process_input_attachments(item) for item in value]

    if isinstance(value, dict):
        block_type = value.get("type")
        source = value.get("source")

        if block_type in {"image", "document"} and isinstance(source, dict):
            attachment = _convert_base64_source_to_attachment(block_type, source)
            if attachment is not None:
                processed = {k: _process_input_attachments(v) for k, v in value.items() if k != "source"}
                processed["source"] = {k: _process_input_attachments(v) for k, v in source.items() if k != "data"}
                processed["image_url"] = {"url": attachment}
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
    with _catch_exceptions():
        _input = _get_input_from_kwargs(kwargs)
        metadata = _get_metadata_from_kwargs(kwargs)
        return start_span(name=name, type="llm", metadata=metadata, input=_input)

    return NOOP_SPAN


def _log_message_to_span(message, span, time_to_first_token: float | None = None):
    with _catch_exceptions():
        usage = getattr(message, "usage", {})
        metrics, metadata = extract_anthropic_usage(usage)

        if time_to_first_token is not None:
            metrics["time_to_first_token"] = time_to_first_token

        output = {
            k: v
            for k, v in {
                "role": getattr(message, "role", None),
                "content": getattr(message, "content", None),
                "model": getattr(message, "model", None),
                "stop_reason": getattr(message, "stop_reason", None),
                "stop_sequence": getattr(message, "stop_sequence", None),
            }.items()
            if v is not None
        } or None

        span.log(output=output, metrics=metrics, metadata=metadata)


@contextmanager
def _catch_exceptions():
    try:
        yield
    except Exception as e:
        log.warning("swallowing exception in tracing code", exc_info=e)


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
