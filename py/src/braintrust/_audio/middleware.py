import importlib
import io
import logging
import wave
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Any, Protocol

from braintrust._audio.config import AudioEncoding

from .types import AudioResult, SegmentMeta
from .worker import run_native


AudioNext = Callable[[Any, SegmentMeta], Awaitable[AudioResult]]


class AudioMiddleware(Protocol):
    async def __call__(self, payload: Any, meta: SegmentMeta, next: AudioNext) -> AudioResult: ...


def _validate(meta: SegmentMeta, original: SegmentMeta):
    if not isinstance(meta, SegmentMeta) or (
        meta.index != original.index
        or Fraction(meta.start_sample, meta.sample_rate_hz) != Fraction(original.start_sample, original.sample_rate_hz)
        or Fraction(meta.sample_count, meta.sample_rate_hz) != Fraction(original.sample_count, original.sample_rate_hz)
        or meta.channels != original.channels
    ):
        raise ValueError("Audio middleware changed segment identity, placement, duration, or sources")
    if (
        meta.encoding == "pcm"
        and meta.mime_type == "audio/pcm"
        and (
            meta.attributes.get("pcm.sample_format") != "s16le"
            or meta.attributes.get("pcm.channel_layout") != "interleaved"
        )
    ):
        raise ValueError("Raw PCM requires sample format and channel layout")


async def run_middleware(stack: Sequence[AudioMiddleware], payload: bytes, meta: SegmentMeta) -> AudioResult:
    async def run(index, data, current):
        _validate(current, meta)
        if index == len(stack):
            return AudioResult(data, current)
        called = False

        def next(data, new_meta):
            nonlocal called
            if called:
                raise ValueError("Audio middleware called next more than once")
            called = True
            return run(index + 1, data, new_meta)

        return await stack[index](data, current, next)

    result = await run(0, payload, meta)
    if not isinstance(result, AudioResult) or not isinstance(result.data, bytes):
        raise ValueError("Audio middleware must return final bytes")
    _validate(result.meta, meta)
    if result.meta.encoding == "pcm" and result.meta.mime_type == "audio/pcm":
        if len(result.data) != result.meta.sample_count * len(result.meta.channels) * 2:
            raise ValueError("PCM byte length does not match playable duration")
    return result


def _wav(data, meta):
    buffer = io.BytesIO()
    with wave.Wave_write(buffer) as output:
        output.setnchannels(len(meta.channels))
        output.setsampwidth(2)
        output.setframerate(meta.sample_rate_hz)
        output.writeframes(data)
    return buffer.getvalue()


async def wav_middleware(data, meta, next):
    data = await run_native(_wav, data, meta)
    return await next(data, replace(meta, mime_type="audio/wav"))


_warned_missing_extension = False


def encoding_stack(config: AudioEncoding | None, *, channels: int | None = None) -> tuple[AudioMiddleware, ...]:
    global _warned_missing_extension
    selected = config or AudioEncoding()
    if selected.codec == "pcm":
        return (wav_middleware,) if selected.container == "wav" else ()
    if channels is not None:
        if channels not in (1, 2):
            raise ValueError("Opus/Ogg supports mono or stereo recordings")
        if selected.bitrate_bps > 256000 * channels:
            raise ValueError("Opus bitrate exceeds 256000 bits/s per output channel")
    try:
        extension = importlib.import_module("braintrust_audio")
        stack = tuple(extension.create_stack(selected, contract_version=1))
        if not all(callable(layer) for layer in stack):
            raise ValueError("Invalid audio extension middleware contract")
        return stack
    except Exception as error:
        if config is not None:
            raise ValueError("Opus/Ogg requires a compatible braintrust-audio installation") from error
        if not _warned_missing_extension:
            _warned_missing_extension = True
            logging.getLogger("braintrust").warning(
                "braintrust-audio is unavailable; recording raw PCM. Install braintrust-audio for Opus/Ogg."
            )
        return ()
