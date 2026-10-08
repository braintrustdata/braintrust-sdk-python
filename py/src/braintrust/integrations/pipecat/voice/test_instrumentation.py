import itertools
import unittest
from types import SimpleNamespace

from pipecat.frames.frames import (  # pylint: disable=import-error
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection  # pylint: disable=import-error

from .instrumentation import NativeObserver, native_value


class Span:
    ids = itertools.count()

    def __init__(self, **kwargs):
        self.span_id = f"span-{next(self.ids)}"
        self.rows = [kwargs]
        self.children = []

    def start_span(self, **kwargs):
        child = Span(**kwargs)
        self.children.append(child)
        return child

    def log(self, **kwargs):
        self.rows.append(kwargs)

    def end(self, end_time=None):
        pass

    def flush(self):
        pass


class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_tool_frames_keep_one_execution_span(self):
        observer = NativeObserver(Span(), retain_audio=True)

        async def push(frame):
            await observer.on_push_frame(
                SimpleNamespace(
                    frame=frame,
                    first_push=True,
                    source=SimpleNamespace(name="source"),
                    destination=SimpleNamespace(name="sink"),
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=1234,
                )
            )

        for _ in range(2):
            await push(
                FunctionCallInProgressFrame(
                    function_name="lookup_order",
                    tool_call_id="call-1",
                    arguments={"order_id": "1042"},
                    cancel_on_interruption=True,
                )
            )
        await push(
            FunctionCallResultFrame(
                function_name="lookup_order",
                tool_call_id="call-1",
                arguments={},
                result={"status": "in_transit"},
            )
        )
        self.assertEqual(observer.tool_names, ["lookup_order"])
        self.assertEqual(observer.tools, {})
        self.assertEqual(len(observer.root.children), 1)
        await observer.finish()

    async def test_concurrent_finish_waits_for_recording_before_flush(self):
        import asyncio
        import threading
        from unittest.mock import Mock

        observer = NativeObserver(Span(), retain_audio=True)
        observer.capture_transport = True
        observer.logger.flush = Mock()
        started, release = threading.Event(), threading.Event()

        def encode(audio_format):
            started.set()
            if not release.wait(2):
                raise TimeoutError("test encoding deadline")

        observer.call_recording.capture(0, b"\0\0" * 480, 24000, 1)
        from unittest.mock import patch

        from braintrust.audio.recording import CallRecording

        patcher = patch.object(CallRecording, "encode", lambda _self, audio_format: encode(audio_format))
        patcher.start()
        self.addCleanup(patcher.stop)
        first = asyncio.create_task(observer.finish())
        await asyncio.to_thread(started.wait, 1)
        second = asyncio.create_task(observer.finish())
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        observer.logger.flush.assert_not_called()
        release.set()
        await asyncio.gather(first, second)
        observer.logger.flush.assert_called_once()

    async def test_encoding_failure_still_flushes_trace(self):
        from unittest.mock import Mock

        observer = NativeObserver(Span(), retain_audio=True)
        observer.capture_transport = True
        from unittest.mock import patch

        from braintrust.audio.recording import CallRecording

        observer.call_recording.capture(0, b"\0\0" * 480, 24000, 1)
        patcher = patch.object(CallRecording, "encode", side_effect=RuntimeError("encoder failed"))
        patcher.start()
        self.addCleanup(patcher.stop)
        observer.logger.flush = Mock()
        await observer.finish()
        descriptors = next(
            row["metadata"]["audio.recordings"]
            for row in reversed(observer.root.rows)
            if "audio.recordings" in row.get("metadata", {})
        )
        self.assertEqual(descriptors[0]["reason"], "RuntimeError")
        self.assertEqual(descriptors[0]["state"], "omitted")
        self.assertNotIn("attachment", descriptors[0])
        self.assertEqual(descriptors[0]["gaps"][0]["end_offset_ms"], 20)
        self.assertEqual(observer.call_recording.retained_bytes, 0)
        observer.logger.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class FormatTests(unittest.IsolatedAsyncioTestCase):
    async def test_packet_pts_do_not_trigger_per_packet_export_and_invalid_clips_are_omitted(self):
        observer = NativeObserver(Span(), retain_audio=True, max_audio_bytes=100000)

        async def push(frame):
            await observer.on_push_frame(
                SimpleNamespace(
                    frame=frame,
                    first_push=True,
                    source=SimpleNamespace(name="tts"),
                    destination=SimpleNamespace(name="output"),
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        await push(TTSStartedFrame(context_id="format-test"))
        state = observer.tts["format-test"]
        for index in range(100):
            frame = TTSAudioRawFrame(b"\0\0" * 480, 24000, 1, context_id="format-test")
            frame.pts = index * 20_000_000
            self.assertNotIn("audio", native_value(frame))
            await push(frame)
        assert len(state.span.rows) <= 3
        await push(TTSAudioRawFrame(b"\0\0" * 320, 16000, 1, context_id="format-test"))
        await push(TTSStoppedFrame(context_id="format-test"))
        await push(TTSStartedFrame(context_id="oversized"))
        oversized = observer.tts["oversized"]
        await push(TTSAudioRawFrame(b"\0\0" * 50001, 24000, 1, context_id="oversized"))
        await push(TTSStoppedFrame(context_id="oversized"))
        await observer.finish()
        rejected = oversized.span.rows[-1]["metadata"]["audio.recordings"][0]
        self.assertEqual(rejected["state"], "omitted")
        self.assertEqual(rejected["reason"], "capture_byte_limit")
        descriptor = state.span.rows[-1]["metadata"]["audio.recordings"][0]
        assert descriptor["state"] == "omitted"
        assert descriptor["reason"] == "audio_format_changed"
        assert observer.synthesis.bytes == 0

    async def test_cancelled_finalizer_preserves_pending_segment_and_active_tail(self):
        import asyncio
        import threading
        from unittest.mock import patch

        from braintrust.audio.recording import CallRecording

        observer = NativeObserver(Span(), retain_audio=True)
        observer.capture_transport = True
        entered, release = threading.Event(), threading.Event()
        original = CallRecording.encode
        seen = []

        def encode(segment, audio_format):
            entered.set()
            release.wait(2)
            seen.append(segment.bytes)
            return original(segment, audio_format)

        with patch.object(CallRecording, "encode", encode):
            observer.call_recording.capture(0, b"\1\0" * 480, 24000, 1, observed_ns=0)
            observer.call_recording._rotate(20)
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            observer.call_recording.capture(0, b"\1\0" * 480, 24000, 1, observed_ns=20_000_000)
            waiter = asyncio.create_task(observer.finish())
            while observer.call_recording._finish_task is None:
                await asyncio.sleep(0)
            observer._finish_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertEqual(observer.call_recording.retained_bytes, 1920)
            release.set()
            await observer.call_recording.finish()
        self.assertEqual(seen, [960, 960])
        self.assertEqual(observer.call_recording.retained_bytes, 0)
        recordings = next(
            row["metadata"]["audio.recordings"]
            for row in reversed(observer.root.rows)
            if "audio.recordings" in row.get("metadata", {})
        )
        self.assertEqual(len(recordings), 2)
        self.assertTrue(all(recording["state"] == "ready" for recording in recordings))

    async def test_generated_clip_capacity_is_reused_after_publication(self):
        from unittest.mock import patch

        from braintrust.logger import Attachment

        uploaded = []

        def upload(attachment):
            uploaded.append(attachment.reference)
            if len(uploaded) == 2:
                return {"upload_status": "error", "error_message": "failed"}
            if len(uploaded) == 3:
                raise OSError("upload failed")
            return {"upload_status": "done"}

        uploader = patch.object(Attachment, "upload", upload)
        uploader.start()
        self.addCleanup(uploader.stop)
        from braintrust.audio.budget import source_budget

        observer = NativeObserver(Span(), retain_audio=True, audio_format="wav")
        before = source_budget.used
        for index in range(270):
            context = str(index)
            for frame in (
                TTSStartedFrame(context_id=context),
                TTSAudioRawFrame(b"\1\0" * 24, 24000, 1, context_id=context),
                TTSStoppedFrame(context_id=context),
            ):
                await observer.on_push_frame(
                    SimpleNamespace(
                        frame=frame,
                        first_push=True,
                        source=SimpleNamespace(name="tts"),
                        destination=SimpleNamespace(name="output"),
                        direction=FrameDirection.DOWNSTREAM,
                        timestamp=0,
                    )
                )
            span = observer.root.children[-1].children[-1]
            self.assertEqual(span.rows[-1]["metadata"]["audio.recordings"][0]["state"], "pending")
            await observer.synthesis.drain()
            descriptor = span.rows[-1]["metadata"]["audio.recordings"][0]
            self.assertEqual(descriptor["state"], "omitted" if index in (1, 2) else "ready")
            if index in (1, 2):
                self.assertEqual(descriptor["reason"], "upload_failed")
                self.assertFalse(
                    any(
                        d["state"] == "ready"
                        for row in span.rows
                        for d in row.get("metadata", {}).get("audio.recordings", [])
                    )
                )
            else:
                self.assertEqual(descriptor["duration_ms"], 1)
            self.assertEqual(source_budget.used, before)
        await observer.finish()

    async def test_failed_segment_publishes_omission_before_shutdown(self):

        from braintrust.audio import RecordingOptions, worker

        observer = NativeObserver(
            Span(), retain_audio=True, recording_options=RecordingOptions(segment_duration_seconds=0.1)
        )
        observer.capture_transport = True
        self.assertTrue(worker._slots.acquire(blocking=False))
        self.assertTrue(worker._slots.acquire(blocking=False))
        try:
            for index in range(60):
                observer.call_recording.capture(0, b"\0\0" * 480, 24000, 1, observed_ns=index * 20_000_000)
            self.assertEqual(observer.root.rows[-1]["metadata"]["audio.recordings"][0]["state"], "pending")
            await observer.call_recording.drain_exports()
            descriptor = observer.root.rows[-1]["metadata"]["audio.recordings"][0]
            self.assertEqual(descriptor["state"], "omitted")
            self.assertEqual(descriptor["reason"], "RecordingBusy")
        finally:
            worker._slots.release()
            worker._slots.release()
            await observer.finish()


# This tests local publication ordering, not the attachment HTTP protocol.
# Provider conversations are exercised separately through cassette replay.
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["done", "error", "exception", "flush_error"])
async def test_segment_readiness_waits_for_upload_and_preserves_earlier_audio(monkeypatch, outcome):
    import asyncio
    import threading

    from braintrust.audio import RecordingOptions
    from braintrust.logger import Attachment

    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def upload(attachment):
        assert threading.get_ident() != loop_thread
        if attachment.reference["filename"] == "call-0001.wav":
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test upload deadline")
            if outcome == "exception":
                raise OSError("upload failed")
            if outcome == "error":
                return {"upload_status": "error", "error_message": "upload failed"}
        return {"upload_status": "done"}

    monkeypatch.setattr(Attachment, "upload", upload)
    observer = NativeObserver(
        Span(),
        retain_audio=True,
        audio_format="wav",
        recording_options=RecordingOptions(segment_duration_seconds=0.02),
    )
    observer.capture_transport = True
    turn = Span()

    def recordings():
        return [
            row["metadata"]["audio.recordings"]
            for row in observer.root.rows
            if "audio.recordings" in row.get("metadata", {})
        ]

    def selections():
        return [s for row in turn.rows for s in row.get("metadata", {}).get("audio.selections", [])]

    try:
        for index in range(2):
            observer.call_recording.capture(0, b"\1\0" * 26400, 24000, 1, observed_ns=index * 1_100_000_000)
            observer.alignment.add(turn, [[index * 480, (index + 1) * 480]], 0)
            observer.call_recording.capture(1, b"\2\0" * 480, 24000, 1, observed_ns=(index + 1) * 1_100_000_000)
            if index == 0:
                await observer.call_recording.drain_exports()
                assert recordings()[-1][0]["state"] == "ready"
        assert await asyncio.to_thread(entered.wait, 1), "export must upload before marking ready"
        # An unrelated publication must not expose the encoded segment as ready.
        observer._publish_call_manifest()
        observer.alignment.publish()
        assert recordings()[-1][1]["state"] == "pending"
        assert all(s["recording_id"] != "call-0001" for s in selections())
        if outcome == "flush_error":

            def fail_flush():
                raise OSError("trace publication failed")

            monkeypatch.setattr(observer.logger, "flush", fail_flush)
        release.set()
        await observer.call_recording.drain_exports()
        observer._publish_call_manifest()
        expected = "omitted" if outcome in {"error", "exception"} else "ready"
        assert recordings()[-1][0]["state"] == "ready"
        assert recordings()[-1][1]["state"] == expected
        if expected == "omitted":
            assert recordings()[-1][1]["reason"]
            assert not any(d["id"] == "call-0001" and d["state"] == "ready" for row in recordings() for d in row)
            assert all(s["recording_id"] != "call-0001" for s in selections())
        else:
            selected = [s for s in selections() if s["recording_id"] == "call-0001"]
            assert selected and selected[-1]["start_offset_ms"] == 0
            assert selected[-1]["end_offset_ms"] == 20
        assert any(s["recording_id"] == "call-0000" for s in selections())
    finally:
        release.set()
        monkeypatch.setattr(observer.logger, "flush", lambda: None)
        await observer.finish()
