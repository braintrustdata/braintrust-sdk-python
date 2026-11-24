"""Observe binary speech responses without consuming application audio."""

import time
import weakref
from email.message import Message
from typing import Any

from braintrust.env import BraintrustEnv
from braintrust.integrations.utils import _infer_audio_mime_type, _is_not_given, _materialize_attachment
from braintrust.logger import NOOP_SPAN
from wrapt import ObjectProxy


def speech_input(params: dict[str, Any]) -> dict[str, Any]:
    parameters = {
        target: params[source]
        for source, target in (
            ("voice", "voice"),
            ("response_format", "format"),
            ("speed", "speed"),
            ("language", "language"),
            ("instructions", "instructions"),
        )
        if source in params and params[source] is not None and not _is_not_given(params[source])
    }
    return {
        "operation": "speech",
        "prompt": params.get("input"),
        **({"parameters": parameters} if parameters else {}),
    }


def speech_filename(response: Any) -> str | None:
    # Streaming APIResponse exposes headers directly; the non-streaming
    # HttpxBinaryResponseContent keeps them on its httpx response.
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(getattr(response, "response", None), "headers", None)
    disposition = headers.get("content-disposition") if headers is not None else None
    if not disposition:
        return None
    message = Message()
    message["Content-Disposition"] = disposition
    return message.get_filename()


def speech_output(data: bytes, mime_type: str, filename: str | None = None) -> dict[str, Any]:
    if data and BraintrustEnv.CAPTURE_AGENT_AUDIO_ATTACHMENTS.get(False):
        resolved = _materialize_attachment(
            data,
            mime_type=mime_type,
            filename=filename
            or "speech."
            + {
                "audio/wav": "wav",
                "audio/mpeg": "mp3",
                "audio/pcm": "pcm",
                "audio/opus": "opus",
                "audio/aac": "aac",
                "audio/flac": "flac",
            }.get(mime_type, "bin"),
        )
        if resolved is not None:
            return {
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": resolved.filename,
                            "file_data": resolved.attachment,
                            "byte_size": len(data),
                        },
                    }
                ]
            }
    return {"content": []}


class _SpeechTrace:
    def __init__(self, span, response, response_format, started):
        self.span = span
        content_type = response.headers.get("content-type")
        self.mime_type = (
            content_type.split(";", 1)[0].strip()
            if isinstance(content_type, str) and content_type
            else _infer_audio_mime_type(response, response_format)
        )
        self.filename = speech_filename(response)
        self.started = started
        self.first_content = False
        self.finished = False
        self.audio = bytearray()
        self.capture = span is not NOOP_SPAN and BraintrustEnv.CAPTURE_AGENT_AUDIO_ATTACHMENTS.get(False)

    def observe(self, chunk, *, audio=True):
        """Record a consumed chunk; ``audio=False`` marks decoded text chunks,
        which count toward timing but cannot be captured as audio bytes."""
        if self.finished or not chunk:
            return
        if not self.first_content:
            self.first_content = True
            self.span.log(metrics={"time_to_first_token": time.monotonic() - self.started})
        if not audio:
            self.capture = False
            self.audio.clear()
        elif self.capture:
            self.audio.extend(chunk)

    def finish(self, error=None, *, complete=False):
        if self.finished:
            return
        self.finished = True
        if error is not None and not isinstance(error, GeneratorExit):
            self.span.log(error=error)
        self.span.log(output=speech_output(bytes(self.audio) if complete else b"", self.mime_type, self.filename))
        self.audio.clear()
        self.span.end()


def _observe_iter(trace, iterator, *, audio=True):
    try:
        for chunk in iterator:
            trace.observe(chunk, audio=audio)
            yield chunk
    except BaseException as exc:
        trace.finish(exc)
        raise
    else:
        trace.finish(complete=True)


async def _aobserve_iter(trace, iterator, *, audio=True):
    try:
        async for chunk in iterator:
            trace.observe(chunk, audio=audio)
            yield chunk
    except BaseException as exc:
        trace.finish(exc)
        raise
    else:
        trace.finish(complete=True)


class SpeechResponse(ObjectProxy):
    def __init__(self, response, trace):
        super().__init__(response)
        self._self_trace = trace
        # The finalizer retains only trace state, never the response/proxy.
        weakref.finalize(self, trace.finish)

    def iter_bytes(self, *args, **kwargs):
        return _observe_iter(self._self_trace, self.__wrapped__.iter_bytes(*args, **kwargs))

    def iter_text(self, *args, **kwargs):
        return _observe_iter(self._self_trace, self.__wrapped__.iter_text(*args, **kwargs), audio=False)

    def iter_lines(self, *args, **kwargs):
        return _observe_iter(self._self_trace, self.__wrapped__.iter_lines(*args, **kwargs), audio=False)

    def read(self):
        try:
            data = self.__wrapped__.read()
        except BaseException as exc:
            self._self_trace.finish(exc)
            raise
        self._self_trace.observe(data)
        self._self_trace.finish(complete=True)
        return data

    # Run these SDK methods with this proxy as self, so their internal
    # read()/iter_bytes() calls go through the observation above while the
    # provider's type and cache behavior is preserved.
    def parse(self, *args, **kwargs):
        return type(self.__wrapped__).parse(self, *args, **kwargs)

    def text(self, *args, **kwargs):
        return type(self.__wrapped__).text(self, *args, **kwargs)

    def json(self, *args, **kwargs):
        return type(self.__wrapped__).json(self, *args, **kwargs)

    def stream_to_file(self, *args, **kwargs):
        return type(self.__wrapped__).stream_to_file(self, *args, **kwargs)

    def close(self):
        try:
            return self.__wrapped__.close()
        finally:
            self._self_trace.finish()


class AsyncSpeechResponse(ObjectProxy):
    def __init__(self, response, trace):
        super().__init__(response)
        self._self_trace = trace
        weakref.finalize(self, trace.finish)

    def iter_bytes(self, *args, **kwargs):
        return _aobserve_iter(self._self_trace, self.__wrapped__.iter_bytes(*args, **kwargs))

    def iter_text(self, *args, **kwargs):
        return _aobserve_iter(self._self_trace, self.__wrapped__.iter_text(*args, **kwargs), audio=False)

    def iter_lines(self, *args, **kwargs):
        return _aobserve_iter(self._self_trace, self.__wrapped__.iter_lines(*args, **kwargs), audio=False)

    async def read(self):
        try:
            data = await self.__wrapped__.read()
        except BaseException as exc:
            self._self_trace.finish(exc)
            raise
        self._self_trace.observe(data)
        self._self_trace.finish(complete=True)
        return data

    async def parse(self, *args, **kwargs):
        return await type(self.__wrapped__).parse(self, *args, **kwargs)

    async def text(self, *args, **kwargs):
        return await type(self.__wrapped__).text(self, *args, **kwargs)

    async def json(self, *args, **kwargs):
        return await type(self.__wrapped__).json(self, *args, **kwargs)

    async def stream_to_file(self, *args, **kwargs):
        return await type(self.__wrapped__).stream_to_file(self, *args, **kwargs)

    async def close(self):
        try:
            return await self.__wrapped__.close()
        finally:
            self._self_trace.finish()


def trace_stream(wrapped, args, kwargs, *, is_async, start_span, log_headers, params):
    """Trace a ``with_streaming_response`` speech request until its body is consumed.

    The sync SDK calls this only when the response context is entered. The async
    SDK calls it when ``create()`` is called, but the returned coroutine only runs
    on ``__aenter__``, so the async span is started inside that coroutine.
    """

    def start():
        return start_span(**{"name": "Speech", "type": "llm", "set_current": False, **params}), time.monotonic()

    def traced(span, response, started):
        log_headers(response, span)
        return _SpeechTrace(span, response, kwargs.get("response_format"), started)

    if is_async:

        async def call():
            span, started = start()
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exc:
                span.log(error=exc)
                span.end()
                raise
            return AsyncSpeechResponse(response, traced(span, response, started))

        return call()
    span, started = start()
    try:
        response = wrapped(*args, **kwargs)
    except BaseException as exc:
        span.log(error=exc)
        span.end()
        raise
    return SpeechResponse(response, traced(span, response, started))
