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
    headers = getattr(response, "headers", None)
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
        # APIResponse exposes headers directly; the shared helper also supports
        # the non-streaming HttpxBinaryResponseContent.response representation.
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

    def observe(self, chunk):
        if self.finished or not chunk:
            return
        if not self.first_content:
            self.first_content = True
            self.span.log(metrics={"time_to_first_token": time.monotonic() - self.started})
        if self.capture and BraintrustEnv.CAPTURE_AGENT_AUDIO_ATTACHMENTS.get(False):
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


class SpeechResponse(ObjectProxy):
    def __init__(self, response, trace):
        super().__init__(response)
        self._self_trace = trace
        # The finalizer retains only trace state, never the response/proxy.
        weakref.finalize(self, trace.finish)

    def iter_bytes(self, *args, **kwargs):
        try:
            for chunk in self.__wrapped__.iter_bytes(*args, **kwargs):
                self._self_trace.observe(chunk)
                yield chunk
        except BaseException as exc:
            self._self_trace.finish(exc)
            raise
        else:
            self._self_trace.finish(complete=True)

    def read(self):
        try:
            data = self.__wrapped__.read()
        except BaseException as exc:
            self._self_trace.finish(exc)
            raise
        self._self_trace.observe(data)
        self._self_trace.finish(complete=True)
        return data

    def parse(self, *args, **kwargs):
        # Execute SDK parsing with this proxy as self, so its read goes through
        # the observation above. Preserve the provider's type/cache behavior.
        return type(self.__wrapped__).parse(self, *args, **kwargs)

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

    async def iter_bytes(self, *args, **kwargs):
        try:
            async for chunk in self.__wrapped__.iter_bytes(*args, **kwargs):
                self._self_trace.observe(chunk)
                yield chunk
        except BaseException as exc:
            self._self_trace.finish(exc)
            raise
        else:
            self._self_trace.finish(complete=True)

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

    async def stream_to_file(self, *args, **kwargs):
        return await type(self.__wrapped__).stream_to_file(self, *args, **kwargs)

    async def close(self):
        try:
            return await self.__wrapped__.close()
        finally:
            self._self_trace.finish()


def trace_stream(wrapped, args, kwargs, *, is_async, start_span, params):
    """The SDK invokes this request only when its response context is entered."""
    span = start_span(**{"name": "Speech", "type": "llm", "set_current": False, **params})
    started = time.monotonic()
    if is_async:

        async def call():
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exc:
                span.log(error=exc)
                span.end()
                raise
            return AsyncSpeechResponse(response, _SpeechTrace(span, response, kwargs.get("response_format"), started))

        return call()
    try:
        response = wrapped(*args, **kwargs)
    except BaseException as exc:
        span.log(error=exc)
        span.end()
        raise
    return SpeechResponse(response, _SpeechTrace(span, response, kwargs.get("response_format"), started))
