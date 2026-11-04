"""Versioned optional codec factory. Importing this module does not import PyAV."""

import io
from dataclasses import replace
from fractions import Fraction


def _encode(data, meta, config):
    import av

    destination = io.BytesIO()
    layout = {1: "mono", 2: "stereo"}.get(len(meta.channels))
    if layout is None or meta.sample_rate_hz != 48000:
        raise ValueError("Opus recordings require mono/stereo 48 kHz PCM")
    with av.open(destination, "w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=48000)
        stream.layout = layout
        stream.bit_rate = config.bitrate_bps
        stream.codec_context.options = {
            "application": config.application,
            "compression_level": str(config.complexity),
        }
        stride = len(meta.channels) * 2
        for start in range(0, meta.sample_count, 960):
            count = min(960, meta.sample_count - start)
            frame = av.AudioFrame(format="s16", layout=layout, samples=count)
            frame.planes[0].update(data[start * stride : (start + count) * stride])
            frame.sample_rate = 48000
            frame.time_base = Fraction(1, 48000)
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return destination.getvalue()


def create_stack(config, *, contract_version=1):
    if contract_version != 1 or (config.codec, config.container) != ("opus", "ogg"):
        raise ValueError("Unsupported Braintrust audio contract or encoding")
    # Qualify backend availability during explicit setup, before sessions begin.
    import av

    av.codec.Codec("libopus", "w")
    from braintrust._audio.worker import run_native

    async def opus_ogg(data, meta, next):
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
