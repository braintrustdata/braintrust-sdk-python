"""Tracing helpers for the ElevenLabs Python SDK."""

import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from braintrust.integrations.utils import _try_to_dict
from braintrust.logger import Attachment, Span, SpanTypeAttribute, start_span


_OMIT_REPR = "Ellipsis"
_METADATA_KEYS = (
    "model_id",
    "voice_id",
    "language_code",
    "output_format",
    "sample_rate",
    "seed",
    "similarity_boost",
    "stability",
    "style",
    "use_speaker_boost",
)


def traced_tts(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return _trace_sync_audio_iterator("elevenlabs_text_to_speech", wrapped, args, kwargs, ("voice_id", "text"))


def traced_tts_with_timestamps(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return _trace_sync_result("elevenlabs_text_to_speech_with_timestamps", wrapped, args, kwargs, ("voice_id", "text"))


def traced_speech_to_text(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return _trace_sync_result("elevenlabs_speech_to_text", wrapped, args, kwargs, ("file",))


def traced_speech_to_speech(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return _trace_sync_audio_iterator("elevenlabs_speech_to_speech", wrapped, args, kwargs, ("voice_id", "audio"))


def traced_sound_effects(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return _trace_sync_audio_iterator("elevenlabs_text_to_sound_effects", wrapped, args, kwargs, ("text",))


def traced_async_tts(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return _trace_async_audio_iterator("elevenlabs_text_to_speech", wrapped, args, kwargs, ("voice_id", "text"))


async def traced_async_tts_with_timestamps(
    wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    return await _trace_async_result(
        "elevenlabs_text_to_speech_with_timestamps", wrapped, args, kwargs, ("voice_id", "text")
    )


async def traced_async_speech_to_text(
    wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    return await _trace_async_result("elevenlabs_speech_to_text", wrapped, args, kwargs, ("file",))


def traced_async_speech_to_speech(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return _trace_async_audio_iterator("elevenlabs_speech_to_speech", wrapped, args, kwargs, ("voice_id", "audio"))


def traced_async_sound_effects(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return _trace_async_audio_iterator("elevenlabs_text_to_sound_effects", wrapped, args, kwargs, ("text",))


def _trace_sync_result(
    name: str, wrapped: Any, args: tuple[Any, ...], kwargs: dict[str, Any], positional_names: tuple[str, ...]
) -> Any:
    start = time.monotonic()
    with start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=_input(args, kwargs, positional_names),
        metadata=_metadata(args, kwargs, positional_names),
    ) as span:
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as exc:
            span.log(error=_error(exc), metrics=_metrics(start, args, kwargs, positional_names=positional_names))
            raise
        span.log(output=_output(result), metrics=_metrics(start, args, kwargs, positional_names=positional_names))
        return result


async def _trace_async_result(
    name: str, wrapped: Any, args: tuple[Any, ...], kwargs: dict[str, Any], positional_names: tuple[str, ...]
) -> Any:
    start = time.monotonic()
    with start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=_input(args, kwargs, positional_names),
        metadata=_metadata(args, kwargs, positional_names),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
            span.log(error=_error(exc), metrics=_metrics(start, args, kwargs, positional_names=positional_names))
            raise
        span.log(output=_output(result), metrics=_metrics(start, args, kwargs, positional_names=positional_names))
        return result


def _trace_sync_audio_iterator(
    name: str, wrapped: Any, args: tuple[Any, ...], kwargs: dict[str, Any], positional_names: tuple[str, ...]
) -> Iterator[bytes]:
    start = time.monotonic()
    span = start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=_input(args, kwargs, positional_names),
        metadata=_metadata(args, kwargs, positional_names),
    )
    try:
        iterator = wrapped(*args, **kwargs)
    except BaseException as exc:
        span.log(error=_error(exc), metrics=_metrics(start, args, kwargs, positional_names=positional_names))
        span.end()
        raise
    return _traced_bytes_iterator(iterator, span, start, args, kwargs, positional_names)


async def _trace_async_audio_iterator(
    name: str, wrapped: Any, args: tuple[Any, ...], kwargs: dict[str, Any], positional_names: tuple[str, ...]
) -> AsyncIterator[bytes]:
    start = time.monotonic()
    span = start_span(
        name=name,
        type=SpanTypeAttribute.LLM,
        input=_input(args, kwargs, positional_names),
        metadata=_metadata(args, kwargs, positional_names),
    )
    try:
        iterator = wrapped(*args, **kwargs)
    except BaseException as exc:
        span.log(error=_error(exc), metrics=_metrics(start, args, kwargs, positional_names=positional_names))
        span.end()
        raise
    async for chunk in _traced_async_bytes_iterator(iterator, span, start, args, kwargs, positional_names):
        yield chunk


def _traced_bytes_iterator(
    iterator: Iterator[bytes],
    span: Span,
    start: float,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    positional_names: tuple[str, ...],
) -> Iterator[bytes]:
    chunks = 0
    size = 0
    data = bytearray()
    first_token_time = None
    try:
        for chunk in iterator:
            if first_token_time is None:
                first_token_time = time.monotonic()
            chunks += 1
            if isinstance(chunk, bytes):
                size += len(chunk)
                data.extend(chunk)
            yield chunk
    except BaseException as exc:
        span.log(
            error=_error(exc),
            metrics={
                **_metrics(start, args, kwargs, first_token_time=first_token_time, positional_names=positional_names),
                "chunk_count": chunks,
                "audio_bytes": size,
            },
        )
        raise
    finally:
        span.log(
            output=_audio_output(bytes(data)),
            metrics={
                **_metrics(start, args, kwargs, first_token_time=first_token_time, positional_names=positional_names),
                "chunk_count": chunks,
                "audio_bytes": size,
            },
        )
        span.end()


async def _traced_async_bytes_iterator(
    iterator: AsyncIterator[bytes],
    span: Span,
    start: float,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    positional_names: tuple[str, ...],
) -> AsyncIterator[bytes]:
    chunks = 0
    size = 0
    data = bytearray()
    first_token_time = None
    try:
        async for chunk in iterator:
            if first_token_time is None:
                first_token_time = time.monotonic()
            chunks += 1
            if isinstance(chunk, bytes):
                size += len(chunk)
                data.extend(chunk)
            yield chunk
    except BaseException as exc:
        span.log(
            error=_error(exc),
            metrics={
                **_metrics(start, args, kwargs, first_token_time=first_token_time, positional_names=positional_names),
                "chunk_count": chunks,
                "audio_bytes": size,
            },
        )
        raise
    finally:
        span.log(
            output=_audio_output(bytes(data)),
            metrics={
                **_metrics(start, args, kwargs, first_token_time=first_token_time, positional_names=positional_names),
                "chunk_count": chunks,
                "audio_bytes": size,
            },
        )
        span.end()


def _input(args: tuple[Any, ...], kwargs: dict[str, Any], positional_names: tuple[str, ...]) -> dict[str, Any]:
    data = {}
    extra_args = []
    for index, arg in enumerate(args):
        key = positional_names[index] if index < len(positional_names) else None
        if key is None:
            extra_args.append(_safe_input_value(arg))
        elif key not in _METADATA_KEYS:
            data[key] = _safe_input_value(arg)
    if extra_args:
        data["args"] = extra_args
    for key, value in kwargs.items():
        if key == "request_options" or key in _METADATA_KEYS or _is_omitted(value):
            continue
        data[key] = _safe_input_value(value)
    return data


def _metadata(args: tuple[Any, ...], kwargs: dict[str, Any], positional_names: tuple[str, ...]) -> dict[str, Any]:
    metadata = {}
    for index, arg in enumerate(args):
        key = positional_names[index] if index < len(positional_names) else None
        if key in _METADATA_KEYS:
            metadata[key] = _safe_input_value(arg)
    for key in _METADATA_KEYS:
        value = kwargs.get(key)
        if value is not None and not _is_omitted(value):
            metadata[key] = _safe_input_value(value)
    if "model_id" in metadata and "model" not in metadata:
        metadata["model"] = metadata["model_id"]
    return metadata


def _safe_input_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"bytes": len(value)}
    if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[1], (bytes, bytearray)):
        return {"filename": value[0], "bytes": len(value[1]), "content_type": value[2] if len(value) > 2 else None}
    if hasattr(value, "read"):
        return {"file": getattr(value, "name", None)}
    return _try_to_dict(value)


def _output(result: Any) -> Any:
    return _try_to_dict(result)


def _audio_output(data: bytes) -> dict[str, Any]:
    output: dict[str, Any] = {"audio_bytes": len(data)}
    if data:
        output["audio"] = Attachment(data=data, filename="elevenlabs-audio.mp3", content_type="audio/mpeg")
    return output


def _metrics(
    start: float,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    first_token_time: float | None = None,
    positional_names: tuple[str, ...] = (),
) -> dict[str, float]:
    metrics: dict[str, float] = {"duration": time.monotonic() - start}
    if first_token_time is not None:
        metrics["time_to_first_token"] = first_token_time - start
    kwargs = kwargs or {}
    text = kwargs.get("text")
    if text is None and "text" in positional_names:
        text_index = positional_names.index("text")
        if text_index < len(args):
            text = args[text_index]
    if isinstance(text, str):
        metrics["input_characters"] = float(len(text))
    return metrics


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _is_omitted(value: Any) -> bool:
    return value is ... or repr(value) == _OMIT_REPR
