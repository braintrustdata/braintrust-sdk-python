import asyncio
import io
import subprocess
import sys
import threading
import wave
from array import array
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import pytest
from braintrust._audio.config import AudioEncoding, AudioRecordingOptions
from braintrust._audio.middleware import encoding_stack, run_middleware
from braintrust._audio.recorder import AudioRecorder
from braintrust._audio.types import AudioChannel, AudioResult, SegmentMeta
from braintrust._audio.worker import run_native


def pcm(values):
    return array("h", values).tobytes()


def recorder(*, duration=0.01, threshold=1024 * 1024, middleware=(), on_segment=None):
    segments, omitted = [], []

    async def collect(result):
        segments.append(result)
        if on_segment is not None:
            await on_segment(result)

    instance = AudioRecorder(
        sources=("input", "output"),
        options=AudioRecordingOptions(
            segment_duration_seconds=duration,
            flush_threshold_bytes=threshold,
        ),
        middleware=middleware,
        on_segment=collect,
        on_omitted=lambda *args: omitted.append(args),
    )
    return instance, segments, omitted


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [16000, 24000, 44100, 48000])
async def test_packet_splits_keep_exact_phase_and_length(rate):
    async def record(split):
        instance, segments, omitted = recorder()
        count = rate // 5
        values = [i % 1000 for i in range(count)]
        for start in range(0, count, split):
            instance.record(
                "input",
                pcm(values[start : start + split]),
                sample_rate_hz=rate,
                channels=1,
                offset_ns=Fraction(start * 1_000_000_000, rate),
            )
        instance.advance(200_000_000)
        await instance.finish()
        assert not omitted
        assert sum(s.meta.sample_count for s in segments) == 9600
        return b"".join(s.data for s in segments)

    assert await record(137) == await record(rate)


@pytest.mark.asyncio
async def test_sparse_timeline_downmix_and_channel_alignment():
    instance, segments, omitted = recorder(duration=0.1)
    instance.record("input", pcm([200, 400] * 480), sample_rate_hz=48000, channels=2, offset_ns=5_000_000)
    instance.record("output", pcm([800] * 240), sample_rate_hz=24000, channels=1, offset_ns=20_000_000)
    await instance.finish()
    samples = array("h")
    samples.frombytes(segments[0].data)
    assert len(samples) == 1440 * 2
    assert samples[240 * 2 : 720 * 2 : 2] == array("h", [300] * 480)
    assert samples[960 * 2 + 1 :: 2] == array("h", [800] * 480)
    assert samples[: 240 * 2] == array("h", [0] * 480)
    assert not omitted


@pytest.mark.asyncio
async def test_size_threshold_waits_for_watermark_and_congestion_preserves_samples():
    gate = asyncio.Event()

    async def slow_upload(_):
        await gate.wait()

    instance, segments, omitted = recorder(duration=60, threshold=32, on_segment=slow_upload)
    for index in range(10):
        instance.record("input", pcm([index] * 480), sample_rate_hz=48000, channels=1, offset_ns=index * 10_000_000)
    await asyncio.sleep(0)
    assert not segments
    instance.advance(50_000_000)
    for _ in range(50):
        if segments:
            break
        await asyncio.sleep(0.005)
    assert len(segments) == 1
    for index in range(10, 20):
        instance.record("input", pcm([index] * 480), sample_rate_hz=48000, channels=1, offset_ns=index * 10_000_000)
    instance.advance(200_000_000)
    gate.set()
    await instance.finish()
    samples = array("h")
    samples.frombytes(b"".join(s.data for s in segments))
    assert samples[::2] == array("h", (value for index in range(20) for value in [index] * 480))
    assert not omitted


@pytest.mark.parametrize("mapping", [dict, MappingProxyType])
def test_segment_attributes_are_reused_and_isolated_from_caller_mutations(mapping):
    attributes = {"nested": {"list": [1]}}
    meta = SegmentMeta(0, 0, 48, 48000, (AudioChannel(("input",)),), attributes=mapping(attributes))
    changed = replace(meta, mime_type="audio/wav")
    assert changed.attributes is meta.attributes

    attributes["nested"]["list"].append(2)
    attributes["extra"] = True
    assert meta.attributes == {"nested": {"list": (1,)}}
    with pytest.raises(TypeError):
        meta.attributes["extra"] = True
    with pytest.raises(TypeError):
        meta.attributes["nested"]["extra"] = True

    replacement = {"nested": {"list": [3]}}
    changed = replace(meta, attributes=mapping(replacement))
    replacement["nested"]["list"].append(4)
    assert changed.attributes is not meta.attributes
    assert changed.attributes == {"nested": {"list": (3,)}}


@pytest.mark.asyncio
async def test_middleware_validates_duration_sources_bytes_and_deep_freeze():
    meta = SegmentMeta(
        0,
        0,
        48,
        48000,
        (AudioChannel(("input",)),),
        attributes={
            "nested": {"list": [1]},
            "pcm.sample_format": "s16le",
            "pcm.channel_layout": "interleaved",
        },
    )
    with pytest.raises(TypeError):
        meta.attributes["nested"]["list"][0] = 2

    async def invalid(data, current, next):
        return AudioResult(data, replace(current, sample_count=49))

    with pytest.raises(ValueError, match="identity, placement"):
        await run_middleware((invalid,), bytes(96), meta)

    async def twice(data, current, next):
        await next(data, current)
        return await next(data, current)

    meta = replace(meta, attributes={"pcm.sample_format": "s16le", "pcm.channel_layout": "interleaved"})
    with pytest.raises(ValueError, match="more than once"):
        await run_middleware((twice,), bytes(96), meta)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "message"),
    [(None, "final bytes"), (bytearray(96), "final bytes"), (bytes(94), "PCM byte length")],
)
async def test_middleware_validates_final_result_after_nested_layers(data, message):
    meta = SegmentMeta(0, 0, 48, 48000, (AudioChannel(("input",)),))

    async def passthrough(data, current, next):
        return await next(data, current)

    async def invalid(payload, current, next):
        return None if data is None else AudioResult(data, current)

    with pytest.raises(ValueError, match=message):
        await run_middleware((passthrough, passthrough, invalid), bytes(96), meta)


@pytest.mark.asyncio
async def test_middleware_rejects_invalid_metadata_before_calling_downstream():
    meta = SegmentMeta(0, 0, 48, 48000, (AudioChannel(("input",)),))

    async def invalid(data, current, next):
        return await next(data, replace(current, sample_count=49))

    async def downstream(data, current, next):
        pytest.fail("Invalid segment metadata reached downstream middleware")

    with pytest.raises(ValueError, match="identity, placement"):
        await run_middleware((invalid, downstream), bytes(96), meta)


@pytest.mark.asyncio
async def test_wav_requires_no_optional_backend_and_preserves_tiny_tail():
    meta = SegmentMeta(0, 0, 137, 48000, (AudioChannel(("input",)),))
    result = await run_middleware(encoding_stack(AudioEncoding(codec="pcm", container="wav")), pcm(range(137)), meta)
    with wave.open(io.BytesIO(result.data)) as decoded:
        assert decoded.getnframes() == 137
        assert decoded.readframes(137) == pcm(range(137))


@pytest.mark.asyncio
async def test_native_worker_cancellation_keeps_buffer_owned_until_completion():
    started, release, completed = threading.Event(), threading.Event(), threading.Event()
    payload = bytearray(b"owned")

    def native(data):
        started.set()
        release.wait(5)
        assert data == b"owned"
        completed.set()

    task = asyncio.create_task(run_native(native, payload))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(completed.wait, 5)


@pytest.mark.asyncio
async def test_native_worker_admits_waiting_sessions_in_order_and_removes_cancelled_waiters():
    from braintrust._audio import worker

    started = [threading.Event() for _ in range(4)]
    release = [threading.Event() for _ in range(3)]

    def blocking(index):
        started[index].set()
        assert release[index].wait(5)

    tasks = [asyncio.create_task(run_native(blocking, index)) for index in range(2)]
    try:
        for event in started[:2]:
            assert await asyncio.to_thread(event.wait, 5)
        first = asyncio.create_task(run_native(blocking, 2))
        cancelled = asyncio.create_task(run_native(lambda: pytest.fail("Cancelled job was submitted")))
        last = asyncio.create_task(run_native(started[3].set))
        tasks.extend([first, cancelled, last])

        async def wait_for_queue():
            while worker._waiters is None or len(worker._waiters) != 3:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait_for_queue(), 5)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release[0].set()
        assert await asyncio.to_thread(started[2].wait, 5)
        assert not started[3].is_set()
        release[2].set()
        assert await asyncio.to_thread(started[3].wait, 5)
    finally:
        for event in release:
            event.set()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_options_reject_unsupported_configuration():
    with pytest.raises(ValueError):
        AudioEncoding(codec="opus/ogg")
    with pytest.raises(ValueError):
        AudioRecordingOptions(segment_duration_seconds=float("nan"))


@pytest.mark.asyncio
async def test_unavailable_audio_is_omitted_instead_of_encoded_as_silence():
    instance, segments, omitted = recorder(duration=0.01)
    instance.record("input", pcm([100] * 1440), sample_rate_hz=48000, channels=1, offset_ns=0)
    instance.unavailable(10_000_000, 20_000_000, "playback_timing_unavailable")
    instance.advance(30_000_000)
    await instance.finish()
    assert [segment.meta.index for segment in segments] == [0, 2]
    assert len(omitted) == 1
    assert omitted[0][0].placement() == {"start_offset_ms": 10, "duration_ms": 10}
    assert omitted[0][1] == "playback_timing_unavailable"


def test_missing_extension_warns_once_and_explicit_opus_fails(caplog, monkeypatch):
    from braintrust._audio import middleware

    monkeypatch.setattr(middleware, "_warned_missing_extension", False)
    assert encoding_stack(None) == ()
    assert encoding_stack(None) == ()
    assert len([record for record in caplog.records if "recording raw PCM" in record.message]) == 1
    with pytest.raises(ValueError, match="requires a compatible"):
        encoding_stack(AudioEncoding(codec="opus", container="ogg"))


def test_tracing_only_session_has_no_media_imports_or_audio_workers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from braintrust.logger import NOOP_SPAN
from braintrust._audio.session import AudioSessionRecorder
from braintrust._audio import worker
session = AudioSessionRecorder(NOOP_SPAN)
assert session.recorder is None
assert worker._executor is None
assert not {'av', 'braintrust_audio', 'livekit', 'numpy'} & sys.modules.keys()
""",
        ],
        check=True,
    )


@pytest.mark.asyncio
async def test_terminal_transform_failure_never_reencodes_omitted_tail_as_silence():
    async def fail(data, meta, next):
        raise RuntimeError("terminal codec failure")

    instance, segments, omitted = recorder(duration=0.01, middleware=(fail,))
    instance.record("input", pcm([100] * 1440), sample_rate_hz=48000, channels=1, offset_ns=0)
    instance.advance(30_000_000)
    await instance.finish()
    await instance.finish()
    assert not segments
    assert [meta.placement() for meta, _ in omitted] == [
        {"start_offset_ms": 0, "duration_ms": 10},
        {"start_offset_ms": 10, "duration_ms": 20},
    ]
