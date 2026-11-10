"""Optional Opus/Ogg encoding. PyAV is loaded only during codec setup."""

import io
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING

from .config import AudioEncoding
from .types import AudioResult, SegmentMeta
from .worker import run_native


if TYPE_CHECKING:
    from .middleware import AudioMiddleware, AudioNext


_SAMPLE_RATE = 48000
# Opus frames contain 20 ms of audio at 48 kHz.
_FRAME_SAMPLES = 960


def _encode(data: bytes, meta: SegmentMeta, config: AudioEncoding) -> bytes:
    import av

    layout = {1: "mono", 2: "stereo"}.get(len(meta.channels))
    if layout is None or meta.sample_rate_hz != _SAMPLE_RATE:
        raise ValueError("Opus recordings require mono/stereo 48 kHz PCM")
    stride = len(meta.channels) * 2
    if len(data) != meta.sample_count * stride:
        raise ValueError("PCM byte length does not match playable duration")
    destination = io.BytesIO()
    with av.open(destination, "w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=_SAMPLE_RATE)
        stream.layout = layout
        stream.bit_rate = config.bitrate_bps
        stream.codec_context.options = {
            "application": config.application,
            "compression_level": str(config.complexity),
        }
        for start in range(0, meta.sample_count, _FRAME_SAMPLES):
            count = min(_FRAME_SAMPLES, meta.sample_count - start)
            frame = av.AudioFrame(format="s16", layout=layout, samples=count)
            frame.planes[0].update(data[start * stride : (start + count) * stride])
            frame.sample_rate = _SAMPLE_RATE
            frame.time_base = Fraction(1, _SAMPLE_RATE)
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        # Flush delayed packets so the final partial frame is retained.
        for packet in stream.encode(None):
            container.mux(packet)
    return destination.getvalue()


def create_stack(config: AudioEncoding) -> tuple["AudioMiddleware", ...]:
    # Qualify backend availability during explicit setup, before sessions begin.
    import av

    av.codec.Codec("libopus", "w")

    async def opus_ogg(data: bytes, meta: SegmentMeta, next: "AudioNext") -> AudioResult:
        encoded = await run_native(_encode, data, meta, config)
        return await next(
            encoded,
            replace(
                meta,
                mime_type="audio/ogg",
                encoding="opus",
                attributes={
                    "opus.bitrate_bps": config.bitrate_bps,
                    "opus.complexity": config.complexity,
                    "opus.application": config.application,
                },
            ),
        )

    return (opus_ogg,)
