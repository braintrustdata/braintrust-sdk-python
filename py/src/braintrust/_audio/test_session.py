import asyncio
import threading

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
    session = recording(shutdown_timeout_seconds=0.02)
    session.record("session-input", bytes(96), sample_rate_hz=48000, channels=1, offset_ns=0)
    result = await session.finish()
    assert result.reason == "shutdown_timeout"
    assert entered.is_set()
    records = memory.pop()
    assert records[0]["metadata"]["audio.segments"]["part-0"]["state"] == "omitted"
    release.set()
    await asyncio.sleep(0.02)
    assert not memory.pop()
