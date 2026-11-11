import asyncio
import threading
from array import array

import pytest
from braintrust import SpanCustomizer, logger, set_span_customizers
from braintrust._audio.config import AudioEncoding, AudioRecordingOptions
from braintrust._audio.session import AudioSessionRecorder
from braintrust.test_helpers import init_test_logger


@pytest.fixture
def memory(monkeypatch):
    # init_test_logger replaces this global; restore it after each test.
    monkeypatch.setattr(logger, "_compute_logger_metadata", logger._compute_logger_metadata)
    init_test_logger("audio-tests")
    with logger._internal_with_memory_background_logger() as bgl:
        monkeypatch.setattr(logger._state, "_global_bg_logger", logger.LazyValue(lambda: bgl, use_mutex=False))
        yield bgl
    set_span_customizers(None)


def recording(**kwargs):
    span = logger.start_span(name="session")
    return AudioSessionRecorder(
        span,
        sources=("session-input",),
        options=AudioRecordingOptions(
            encoding=AudioEncoding(codec="pcm", container=None),
            **kwargs,
        ),
    )


def test_session_timeline_converts_nanoseconds_and_wall_seconds(memory):
    session = AudioSessionRecorder(logger.start_span(name="session"))
    origin = session.to_wall(0)
    later = session.to_wall(1_250_000_000)

    assert later - origin == pytest.approx(1.25, rel=0, abs=1e-6)
    # Unix seconds are floats: allow their sub-microsecond rounding error.
    assert session.from_wall(later) == pytest.approx(1_250_000_000, rel=0, abs=512)
    assert session.from_wall(origin - 1) == 0


@pytest.mark.asyncio
async def test_session_source_registry_is_framework_independent(memory):
    span = logger.start_span(name="standalone")
    session = AudioSessionRecorder(span)
    session.source("microphone-A", {"label": "Input"})
    session.source("speaker-B")
    assert session.session_metadata()["audio.sources"] == {
        "microphone-A": {"label": "Input"},
        "speaker-B": {},
    }
    await session.finish()
    metadata = memory.pop()[0]["metadata"]
    assert set(metadata["audio.sources"]) == {"microphone-A", "speaker-B"}


@pytest.mark.asyncio
async def test_ready_only_after_success_and_reference_is_not_reuploaded(memory, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    uploads = []

    def upload(attachment):
        uploads.append(attachment)
        entered.set()
        release.wait(5)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    session = recording()
    session.record("session-input", bytes(96), sample_rate_hz=48000, channels=1, offset_ns=0)
    finishing = asyncio.create_task(session.finish())
    assert await asyncio.to_thread(entered.wait, 5)
    records = memory.pop()
    assert records[0]["metadata"]["audio.segments"]["part-0"]["state"] == "pending"
    assert "audio" not in records[0].get("input", {})
    release.set()
    await finishing
    records = memory.pop()
    segment = records[0]["metadata"]["audio.segments"]["part-0"]
    assert segment["state"] == "ready"
    assert segment["channels"] == {"0": {"sources": ["session-input"]}}
    assert segment["attributes"] == {"pcm.sample_format": "s16le", "pcm.channel_layout": "interleaved"}
    assert records[0]["input"]["audio"]["part-0"] == uploads[0].reference
    assert not memory.upload_attempts
    assert len(uploads) == 1


@pytest.mark.asyncio
async def test_customizer_removal_does_not_veto_upload(memory, monkeypatch):
    uploads = []

    class RemoveAudio(SpanCustomizer):
        def on_span_export(self, data):
            data.pop("input", None)
            return data

    def upload(attachment):
        uploads.append(attachment)
        return {"upload_status": "done"}

    set_span_customizers([RemoveAudio()])
    monkeypatch.setattr(logger.Attachment, "upload", upload)
    session = recording()
    session.record("session-input", bytes(96), sample_rate_hz=48000, channels=1, offset_ns=0)
    await session.finish()
    assert len(uploads) == 1
    assert "input" not in memory.pop()[0]


@pytest.mark.asyncio
async def test_privacy_tightening_before_upload_discards_audio(memory, monkeypatch):
    def forbidden(_):
        pytest.fail("Restricted audio must never upload")

    monkeypatch.setattr(logger.Attachment, "upload", forbidden)
    allowed = True
    session = recording()
    session.allowed = lambda: allowed
    session.record("session-input", bytes(96), sample_rate_hz=48000, channels=1, offset_ns=0)
    allowed = False
    await session.finish()
    segment = memory.pop()[0]["metadata"]["audio.segments"]["part-0"]
    assert segment == {"state": "omitted", "reason": "privacy_restricted", "start_offset_ms": 0, "duration_ms": 1}


@pytest.mark.asyncio
async def test_terminal_upload_failure_preserves_known_placement(memory, monkeypatch):
    monkeypatch.setattr(logger.Attachment, "upload", lambda _: {"upload_status": "error"})
    session = recording()
    session.record("session-input", bytes(96), sample_rate_hz=48000, channels=1, offset_ns=0)
    result = await session.finish()
    assert result.reason == "encoding_or_upload_failed"
    segment = memory.pop()[0]["metadata"]["audio.segments"]["part-0"]
    assert segment["state"] == "omitted"
    assert segment["reason"] == "upload_failed"
    assert segment["duration_ms"] == 1
    assert "attachment" not in segment


@pytest.mark.asyncio
async def test_finish_propagates_cancellation_after_cleanup(memory):
    entered, cleaned_up = asyncio.Event(), asyncio.Event()

    async def blocking(data, meta, next):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    session = recording()
    session.recorder.middleware = (blocking,)
    session.record("session-input", bytes(96), sample_rate_hz=48000, channels=1, offset_ns=0)
    finishing = asyncio.create_task(session.finish())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        finishing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await finishing
        assert cleaned_up.is_set()
        assert finishing.cancelled()
        assert session.recorder.buffered_bytes == 0
        assert not session.record("session-input", bytes(96), sample_rate_hz=48000, channels=1, offset_ns=0)
        segment = memory.pop()[0]["metadata"]["audio.segments"]["part-0"]
        assert segment["state"] == "omitted"
        assert "attachment" not in segment
    finally:
        finishing.cancel()
        await asyncio.gather(finishing, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_deadline_fences_late_success(memory, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def upload(_):
        entered.set()
        release.wait(5)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    session = recording(segment_duration_seconds=0.001, shutdown_timeout_seconds=0.02)
    session.record("session-input", bytes(96), sample_rate_hz=48000, channels=1, offset_ns=0)
    # Start the upload before the shutdown deadline, even on a busy worker.
    session.advance(1_000_000)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        result = await session.finish()
        assert result.reason == "shutdown_timeout"
        records = memory.pop()
        assert records[0]["metadata"]["audio.segments"]["part-0"]["state"] == "omitted"
    finally:
        release.set()
    await asyncio.sleep(0.02)
    assert not memory.pop()


@pytest.mark.asyncio
async def test_stream_frames_arriving_in_a_burst_are_recorded_back_to_back(memory, monkeypatch):
    uploaded = []

    def upload(attachment):
        uploaded.append(attachment.data)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    session = recording()
    # Jitter: frames 1-3 (10 ms each at 48 kHz) arrive at once; frame 4 later.
    for value, now_ns in ((1, 0), (2, 0), (3, 0), (4, 100_000_000)):
        frame = array("h", [value] * 480).tobytes()
        assert session.record_stream("session-input", frame, sample_rate_hz=48000, channels=1, now_ns=now_ns)
    with pytest.raises(ValueError):
        session.record_stream("session-input", b"\x01", sample_rate_hz=48000, channels=1, now_ns=200_000_000)
    await session.finish()
    samples = array("h", uploaded[0])
    per_ms = [samples[index * 48] for index in range(len(samples) // 48)]
    # The burst plays out in order instead of overwriting itself at 0 ms; the
    # late frame keeps its arrival time; the invalid frame records nothing.
    assert per_ms == [1] * 10 + [2] * 10 + [3] * 10 + [0] * 70 + [4] * 10


@pytest.mark.asyncio
async def test_stream_bursts_queue_per_source_while_other_sources_overlap(memory, monkeypatch):
    uploaded = []

    def upload(attachment):
        uploaded.append(attachment.data)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    session = AudioSessionRecorder(
        logger.start_span(name="session"),
        sources=("user", "agent"),
        options=AudioRecordingOptions(encoding=AudioEncoding(codec="pcm", container=None)),
    )
    # Both sources burst at 0 ms: each queues behind only its own frames.
    for source, values in (("user", (1, 2)), ("agent", (7, 8, 9))):
        for value in values:
            frame = array("h", [value] * 480).tobytes()
            assert session.record_stream(source, frame, sample_rate_hz=48000, channels=1, now_ns=0)
    await session.finish()
    samples = array("h", uploaded[0])
    channels = [samples[channel::2] for channel in range(2)]
    user, agent = ([channel[index * 48] for index in range(len(channel) // 48)] for channel in channels)
    assert user == [1] * 10 + [2] * 10 + [0] * 10
    assert agent == [7] * 10 + [8] * 10 + [9] * 10
