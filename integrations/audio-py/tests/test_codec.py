import io
import math
from array import array

import av
import pytest
from braintrust._audio.middleware import encoding_stack, run_middleware
from braintrust._audio.types import AudioChannel, SegmentMeta
from braintrust.audio import AudioEncoding


@pytest.mark.asyncio
@pytest.mark.parametrize("samples", [1, 137, 960, 48137, 48000 * 60 + 137])
@pytest.mark.parametrize("channels", [1, 2])
async def test_independent_opus_segments_have_exact_playable_duration(samples, channels):
    meta = SegmentMeta(0, 0, samples, 48000, tuple(AudioChannel((str(i),)) for i in range(channels)))
    data = array("h", (1000 if i % channels == 0 else -1000 for i in range(samples * channels))).tobytes()
    for config in (None, AudioEncoding(codec="opus", container="ogg")):
        result = await run_middleware(encoding_stack(config), data, meta)
        assert result.meta.encoding == "opus"
        with av.open(io.BytesIO(result.data)) as container:
            decoded = list(container.decode(audio=0))
        assert sum(frame.samples for frame in decoded) == samples
        assert all(len(frame.layout.channels) == channels for frame in decoded)


@pytest.mark.asyncio
async def test_opus_stereo_channels_remain_separate():
    count = 48000
    tones = [500, 1700]
    data = array(
        "h",
        (round(12000 * math.sin(2 * math.pi * frequency * i / 48000)) for i in range(count) for frequency in tones),
    ).tobytes()
    meta = SegmentMeta(0, 0, count, 48000, (AudioChannel(("input",)), AudioChannel(("output",))))
    result = await run_middleware(encoding_stack(AudioEncoding()), data, meta)
    channels = [[], []]
    with av.open(io.BytesIO(result.data)) as container:
        for frame in container.decode(audio=0):
            assert frame.format.name == "fltp"
            for channel in range(2):
                samples = array("f")
                samples.frombytes(bytes(frame.planes[channel]))
                channels[channel].extend(samples[: frame.samples])
    for channel in range(2):

        def power(frequency):
            real = sum(
                value * math.cos(2 * math.pi * frequency * i / 48000) for i, value in enumerate(channels[channel])
            )
            imag = sum(
                value * math.sin(2 * math.pi * frequency * i / 48000) for i, value in enumerate(channels[channel])
            )
            return real * real + imag * imag

        assert power(tones[channel]) > 100 * power(tones[1 - channel])


@pytest.mark.asyncio
@pytest.mark.parametrize("bitrate,complexity,application", [(500, 0, "voip"), (512000, 10, "lowdelay")])
async def test_supported_opus_options(bitrate, complexity, application):
    config = AudioEncoding(bitrate_bps=bitrate, complexity=complexity, application=application)
    channels = 2 if bitrate > 256000 else 1
    meta = SegmentMeta(0, 0, 48000, 48000, tuple(AudioChannel((str(i),)) for i in range(channels)))
    result = await run_middleware(encoding_stack(config, channels=channels), bytes(96000 * channels), meta)
    assert result.meta.attributes["opus.application"] == application
    with av.open(io.BytesIO(result.data)) as container:
        assert sum(frame.samples for frame in container.decode(audio=0)) == 48000
