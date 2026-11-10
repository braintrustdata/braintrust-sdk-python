import asyncio
import io
import subprocess
import sys
import threading
import wave
from array import array
from dataclasses import replace
from fractions import Fraction
from itertools import permutations
from types import MappingProxyType

import pytest
from braintrust._audio.config import AudioEncoding, AudioRecordingOptions
from braintrust._audio.middleware import encoding_stack, run_middleware
from braintrust._audio.recorder import AudioRecorder
from braintrust._audio.types import AudioChannel, AudioResult, SegmentMeta
from braintrust._audio.worker import run_native


def pcm(values):
    return array("h", values).tobytes()


def recorder(*, duration=0.01, threshold=1024 * 1024, middleware=(), on_segment=None, max_duration=None):
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
            max_duration_seconds=max_duration,
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
@pytest.mark.parametrize("advance", [False, True])
@pytest.mark.parametrize("count", [480, 960])
async def test_duration_limit_waits_for_all_sources(advance, count):
    instance, segments, omitted = recorder(max_duration=0.01)
    assert instance.record("input", pcm([100] * count), sample_rate_hz=48000, channels=1, offset_ns=0)
    assert instance.record("output", pcm([200] * 480), sample_rate_hz=48000, channels=1, offset_ns=0)
    assert not instance.record("output", pcm([300] * 480), sample_rate_hz=48000, channels=1, offset_ns=10_000_000)
    if advance:
        instance.advance(10_000_000)
        assert instance.reason == "duration_limit"
        assert not instance.record("input", pcm([400]), sample_rate_hz=48000, channels=1, offset_ns=10_000_000)
    result = await instance.finish()
    samples = array("h")
    samples.frombytes(b"".join(s.data for s in segments))
    assert samples == array("h", [100, 200] * 480)
    assert result.reason == "duration_limit"
    assert not omitted


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [16000, 24000, 44100, 48000])
@pytest.mark.parametrize("threshold", [1, 1024 * 1024])
async def test_progressive_resampling_waits_for_packet_lookahead(rate, threshold):
    async def record(progressive):
        instance, segments, omitted = recorder(threshold=threshold)
        for index, value in enumerate((0, 10000, -10000)):
            assert instance.record(
                "input",
                pcm([value] * (rate // 100)),
                sample_rate_hz=rate,
                channels=1,
                offset_ns=index * 10_000_000,
            )
            if progressive:
                instance.advance((index + 1) * 10_000_000)
                if instance._task is not None:
                    await instance._task
                if index == 1:
                    assert segments  # Lookahead releases audio before finish.
        await instance.finish()
        assert not omitted
        return b"".join(s.data for s in segments)

    assert await record(True) == await record(False)


@pytest.mark.asyncio
@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("progressive", [False, True])
async def test_resampling_is_independent_of_packet_arrival_order(order, progressive):
    async def record(indices):
        instance, segments, omitted = recorder()
        for index in indices:
            assert instance.record(
                "input",
                pcm([index * 10000] * 240),
                sample_rate_hz=24000,
                channels=1,
                offset_ns=index * 10_000_000,
            )
        if progressive:
            instance.advance(20_000_000)
            await instance._task
            assert len(segments) == 2  # Both boundaries have contiguous lookahead.
        await instance.finish()
        assert not omitted
        return b"".join(s.data for s in segments)

    expected = await record(range(3))
    samples = array("h")
    samples.frombytes(expected)
    assert samples[959 * 2] == 15000
    assert await record(order) == expected


@pytest.mark.asyncio
async def test_resampling_tail_released_when_watermark_passes_packet_end():
    instance, segments, omitted = recorder()
    instance.record("input", pcm([100] * 240), sample_rate_hz=24000, channels=1, offset_ns=0)
    instance.advance(10_000_000)
    if instance._task is not None:
        await instance._task
    assert not segments
    # No contiguous packet can arrive once the watermark passes its start.
    instance.advance(10_100_000)
    await instance._task
    assert len(segments) == 1
    samples = array("h")
    samples.frombytes(segments[0].data)
    assert samples == array("h", [100, 0] * 480)
    await instance.finish()
    assert not omitted


@pytest.mark.asyncio
async def test_unavailable_audio_is_clamped_to_duration_limit():
    instance, segments, omitted = recorder(max_duration=0.01)
    instance.unavailable(0, 20_000_000, "unavailable")
    instance.unavailable(20_000_000, 30_000_000, "unavailable")
    result = await instance.finish()
    assert not segments
    assert len(omitted) == 1
    assert omitted[0][0].sample_count == 480
    assert result.reason == "duration_limit"


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
    with pytest.raises(TypeError):
        meta.attributes["nested"]["list"][0] = 2

    replacement = {"nested": {"list": [3]}}
    changed = replace(meta, attributes=mapping(replacement))
    replacement["nested"]["list"].append(4)
    assert changed.attributes is not meta.attributes
    assert changed.attributes == {"nested": {"list": (3,)}}


@pytest.mark.asyncio
async def test_middleware_rejects_calling_next_twice():
    meta = SegmentMeta(0, 0, 48, 48000, (AudioChannel(("input",)),))

    async def twice(data, current, next):
        await next(data, current)
        return await next(data, current)

    with pytest.raises(ValueError, match="more than once"):
        await run_middleware((twice,), bytes(96), meta)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "sample_count", "message"),
    [
        (None, 48, "final bytes"),
        (bytearray(96), 48, "final bytes"),
        (bytes(94), 48, "PCM byte length"),
        (bytes(96), 49, "identity, placement"),
    ],
)
async def test_middleware_validates_final_result_after_nested_layers(data, sample_count, message):
    meta = SegmentMeta(0, 0, 48, 48000, (AudioChannel(("input",)),))

    async def passthrough(data, current, next):
        return await next(data, current)

    async def invalid(payload, current, next):
        return None if data is None else AudioResult(data, replace(current, sample_count=sample_count))

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
async def test_native_work_completes_after_awaiter_cancellation():
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
@pytest.mark.parametrize("separate_loop", [False, True])
async def test_native_worker_admits_waiting_sessions_in_order_and_removes_cancelled_waiters(separate_loop):
    started = [threading.Event() for _ in range(4)]
    release = [threading.Event() for _ in range(3)]
    loop = asyncio.get_running_loop()
    queued = asyncio.Event()

    async def first_waiter():
        task = asyncio.create_task(run_native(blocking, 2))
        # Notify after the task has had a turn to request admission on its loop.
        asyncio.get_running_loop().call_soon(loop.call_soon_threadsafe, queued.set)
        return await task

    def blocking(index):
        started[index].set()
        assert release[index].wait(5)

    tasks = [asyncio.create_task(run_native(blocking, index)) for index in range(2)]
    try:
        for event in started[:2]:
            assert await asyncio.to_thread(event.wait, 5)
        first = asyncio.create_task(
            asyncio.to_thread(asyncio.run, first_waiter()) if separate_loop else first_waiter()
        )
        tasks.append(first)
        await asyncio.wait_for(queued.wait(), 5)
        cancelled = asyncio.create_task(run_native(lambda: pytest.fail("Cancelled job was submitted")))
        last = asyncio.create_task(run_native(started[3].set))
        tasks.extend([cancelled, last])

        queue_ready = asyncio.Event()
        loop.call_soon(queue_ready.set)
        await asyncio.wait_for(queue_ready.wait(), 5)
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


def test_missing_codec_warns_once_and_explicit_opus_fails(caplog, monkeypatch):
    from braintrust._audio import middleware

    monkeypatch.setattr(middleware, "_warned_missing_codec", False)
    monkeypatch.setitem(sys.modules, "av", None)
    assert encoding_stack(None) == ()
    assert encoding_stack(None) == ()
    assert len([record for record in caplog.records if "recording raw PCM" in record.message]) == 1
    with pytest.raises(ValueError, match=r"Install braintrust\[audio\]"):
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
assert not {'av', 'livekit', 'numpy'} & sys.modules.keys()
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
