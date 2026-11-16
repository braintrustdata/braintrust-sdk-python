import io
import math
import random
import struct
from array import array

import pytest
from braintrust._audio.codec import _encode
from braintrust._audio.config import AudioEncoding
from braintrust._audio.middleware import encoding_stack, run_middleware
from braintrust._audio.types import AudioChannel, SegmentMeta
from braintrust.audio import _crc, _page_body, _pages, join_ogg_streams


av = pytest.importorskip("av")


@pytest.mark.asyncio
@pytest.mark.parametrize("samples", [1, 137, 960, 48137, 48000 * 60 + 137])
@pytest.mark.parametrize("channels", [1, 2])
async def test_opus_segments_have_exact_playable_duration(samples, channels):
    meta = SegmentMeta(0, 0, samples, 48000, tuple(AudioChannel((str(i),)) for i in range(channels)))
    data = array("h", (1000 if i % channels == 0 else -1000 for i in range(samples * channels))).tobytes()
    result = await run_middleware(encoding_stack(AudioEncoding()), data, meta)
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
    assert result.meta.attributes == {
        "opus.bitrate_bps": bitrate,
        "opus.complexity": complexity,
        "opus.application": application,
    }
    with av.open(io.BytesIO(result.data)) as container:
        assert sum(frame.samples for frame in container.decode(audio=0)) == 48000


@pytest.mark.asyncio
@pytest.mark.parametrize("byte_count", [1919, 1921, 3840])
async def test_opus_rejects_pcm_length_that_disagrees_with_metadata(byte_count):
    meta = SegmentMeta(0, 0, 960, 48000, (AudioChannel(("input",)),))
    with pytest.raises(ValueError, match="PCM byte length"):
        await run_middleware(encoding_stack(AudioEncoding()), bytes(byte_count), meta)


@pytest.mark.asyncio
async def test_recorder_defaults_to_independently_playable_opus_segments():
    from braintrust._audio.config import AudioRecordingOptions
    from braintrust._audio.recorder import AudioRecorder

    segments = []

    async def collect(segment):
        segments.append(segment)

    def unexpected_omission(meta, reason):
        pytest.fail(f"Unexpected omitted segment {meta.index}: {reason}")

    instance = AudioRecorder(
        sources=("input",),
        options=AudioRecordingOptions(segment_duration_seconds=0.02),
        middleware=encoding_stack(None),
        on_segment=collect,
        on_omitted=unexpected_omission,
    )
    count = 960 * 2 + 137
    data = array("h", (round(12000 * math.sin(2 * math.pi * 500 * i / 48000)) for i in range(count))).tobytes()
    assert instance.record("input", data, sample_rate_hz=48000, channels=1, offset_ns=0)
    await instance.finish()

    assert [segment.meta.index for segment in segments] == [0, 1, 2]
    assert [segment.meta.start_sample for segment in segments] == [0, 960, 1920]
    assert [segment.meta.sample_count for segment in segments] == [960, 960, 137]
    for segment in segments:
        assert segment.meta.mime_type == "audio/ogg"
        assert segment.meta.encoding == "opus"
        assert segment.meta.channels == (AudioChannel(("input",)),)
        with av.open(io.BytesIO(segment.data)) as container:
            assert container.streams.audio[0].codec_context.name == "opus"
            frames = list(container.decode(audio=0))
        assert sum(frame.samples for frame in frames) == segment.meta.sample_count
        assert frames[0].pts == 0
        assert all(frame.sample_rate == 48000 for frame in frames)


def _segment(samples, channels=1, bitrate_bps=64000):
    meta = SegmentMeta(0, 0, samples, 48000, tuple(AudioChannel((str(i),)) for i in range(channels)))
    return _encode(
        random.Random(samples).randbytes(samples * channels * 2), meta, AudioEncoding(bitrate_bps=bitrate_bps)
    )


def _split_first_packet(page):
    # Moves all but the first 255 bytes of the page's first packet to a continued page.
    n, body = page[26], 27 + page[26]
    partial = page[:6] + struct.pack("<q", -1) + page[14:26] + bytes([1, 255]) + page[body : body + 255]
    return partial + page[:5] + bytes([page[5] | 1]) + page[6:26] + bytes([n - 1]) + page[28:body] + page[body + 255 :]


def _with_split_page(segment, index):
    pages = _pages(segment)
    return b"".join(pages[:index] + [_split_first_packet(pages[index])] + pages[index + 1 :])


def _granule(page):
    return struct.unpack_from("<q", page, 6)[0]


@pytest.mark.parametrize(
    "make_segments",
    [
        lambda: [_segment(48000 * 60 + 137), _segment(48000 * 2 - 7)],
        lambda: [_segment(48000), b"".join(_pages(_segment(960))[:2]), _segment(48000)],
        lambda: [_segment(48000), _with_split_page(_segment(48000, bitrate_bps=256000), 2)],
    ],
    ids=["long-then-short", "headers-only-middle", "continued-packet"],
)
def test_join_ogg_streams_makes_one_logical_stream(make_segments):
    segments = make_segments()
    pages = _pages(join_ogg_streams(segments))
    n = len(pages)
    assert len({bytes(page[14:18]) for page in pages}) == 1
    assert [page[5] & 0x06 for page in pages] == [0x02] + [0] * (n - 2) + [0x04]
    assert [struct.unpack_from("<I", page, 18)[0] for page in pages] == list(range(n))
    for page in pages:
        assert _crc(page[:22] + bytes(4) + page[26:]) == struct.unpack_from("<I", page, 22)[0]
    granules = [_granule(page) for page in pages if _granule(page) != -1]
    assert granules == sorted(granules)
    assert sum(_page_body(page)[:8] in (b"OpusHead", b"OpusTags") for page in pages) == 2
    assert sum(page[5] & 1 for page in pages) == sum(page[5] & 1 for s in segments for page in _pages(s))
    packet_count = sum(v < 255 for s in segments[:-1] for page in _pages(s)[2:] for v in page[27 : 27 + page[26]])
    assert granules[-1] == 960 * packet_count + _granule(_pages(segments[-1])[-1])
    with av.open(io.BytesIO(b"".join(pages))) as container:
        assert len(container.streams) == 1
        decoded = sum(frame.samples for frame in container.decode(audio=0))
    assert decoded == granules[-1] - struct.unpack_from("<H", _page_body(pages[0]), 10)[0]


def test_join_ogg_streams_keeps_one_segment_unchanged():
    segment = _segment(48000)
    assert join_ogg_streams([segment]) == segment


def _with_multipage_tags(segment):
    pages = _pages(segment)
    tags = _page_body(pages[1]).ljust(300, b"\0")
    pages[1] = pages[1][:26] + bytes([2, 255, 45]) + tags
    return _with_split_page(b"".join(pages), 1)


@pytest.mark.parametrize(
    "make_segments,error",
    [
        (lambda: [_segment(960), _with_multipage_tags(_segment(960))], "OpusTags over more than one page"),
        (lambda: [_segment(960), _segment(960, channels=2)], "different channel layout"),
        (lambda: [b"not an ogg file"], "Malformed Ogg page"),
    ],
)
def test_join_ogg_streams_rejects_invalid_input(make_segments, error):
    with pytest.raises(ValueError, match=error):
        join_ogg_streams(make_segments())
