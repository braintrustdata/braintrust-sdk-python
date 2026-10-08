import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.frames.frames import (  # pylint: disable=import-error
    InputAudioRawFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData  # pylint: disable=import-error
from pipecat.processors.aggregators.llm_context import LLMContext  # pylint: disable=import-error
from pipecat.processors.aggregators.llm_response_universal import (  # pylint: disable=import-error
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection  # pylint: disable=import-error

from .instrumentation import NativeObserver, encode_wav, native_value
from .test_instrumentation import Span
from .user_capture import UserCapture


class STT:
    is_usable = True
    _audio_buffer = b""

    async def process_audio_frame(self, frame, direction):
        self._audio_buffer += frame.audio

    async def _handle_user_started_speaking(self, frame):
        pass

    async def _handle_user_stopped_speaking(self, frame):
        pass

    async def run_stt(self, audio):
        yield TranscriptionFrame("Where is my order?", "user", "2026-09-30T12:00:00Z")


class UserCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, enabled, max_bytes=8388608):
        observer = NativeObserver(Span(), retain_audio=enabled, max_audio_bytes=max_bytes)
        context = LLMContext()
        aggregator = LLMContextAggregatorPair(context).user()

        async def push_context():
            pass

        aggregator.push_context_frame = push_context
        stt = STT()
        stt.name = "test-stt"
        original_run = stt.run_stt

        async def measured_run(audio):
            await observer.on_push_frame(
                SimpleNamespace(
                    frame=MetricsFrame(data=[TTFBMetricsData(processor=stt.name, value=0.12)]),
                    first_push=True,
                    source=stt,
                    destination=aggregator,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=1234,
                )
            )
            async for result in original_run(audio):
                yield result

        stt.run_stt = measured_run
        original_audio = stt.process_audio_frame
        capture = UserCapture(observer, stt, aggregator, native_value)
        observer.user_capture = capture
        if not enabled:
            self.assertEqual(stt.process_audio_frame, original_audio)
        await stt.process_audio_frame(InputAudioRawFrame(b"\0\0" * 320, 16000, 1), FrameDirection.DOWNSTREAM)
        await stt._handle_user_started_speaking(VADUserStartedSpeakingFrame())
        await stt._handle_user_stopped_speaking(VADUserStoppedSpeakingFrame())
        audio = encode_wav([b"\x10\x10" * 1600], 16000, 1)
        async for frame in stt.run_stt(audio):
            # The frame is consumed before the aggregator creates its turn.
            await aggregator._handle_transcription(frame)
        turn = observer.turns.start("user", "UserStartedSpeakingFrame")
        observer.turns.confirm("user")
        await aggregator._push_aggregation()
        self.assertEqual(context.get_messages()[-1]["content"], "Where is my order?")
        self.assertEqual(capture.batches[turn["span"].span_id]["frames"][0]["id"], frame.id)
        self.assertEqual(len(capture.batches[turn["span"].span_id]["segments"][0]["boundaries"]), 2)
        stt_span = capture.batches[turn["span"].span_id]["segments"][0]["span"]
        metadata = {key: value for row in stt_span.rows for key, value in row.get("metadata", {}).items()}
        self.assertEqual(metadata["contrib.pipecat.ttfb"][0]["value"], 0.12)
        observer.turns.stop(
            "user",
            SimpleNamespace(content="Where is my order?", timestamp="t1", user_id="user"),
        )
        await asyncio.sleep(0)
        return observer, capture, turn

    async def test_opt_out_keeps_metadata_without_retaining_encoding_or_attaching_audio(
        self,
    ):
        with (
            patch(
                "braintrust.integrations.pipecat.voice.user_capture.encode_segments",
                side_effect=AssertionError("encoded"),
            ),
            patch("braintrust.audio.attachments.Attachment", side_effect=AssertionError("attached")),
        ):
            observer, capture, turn = await self.exercise(False)
            self.assertEqual(capture.bytes, 0)
            self.assertTrue(all(s["audio"] is None for b in capture.batches.values() for s in b["segments"]))
            await observer.finish()
        self.assertFalse(any("audio.selections" in row.get("metadata", {}) for row in turn["span"].rows))
        self.assertEqual(
            turn["span"].rows[-1]["metadata"]["audio.recordings"][0]["reason"],
            "disabled",
        )
        self.assertTrue(any("contrib.pipecat.transcriptions" in row.get("metadata", {}) for row in turn["span"].rows))

    async def test_observation_failure_does_not_drop_native_transcription(self):
        observer = NativeObserver(Span(), retain_audio=False)
        stt = STT()
        aggregator = LLMContextAggregatorPair(LLMContext()).user()
        capture = UserCapture(observer, stt, aggregator, lambda _: (_ for _ in ()).throw(ValueError("observation")))
        observer.user_capture = capture
        frames = [frame async for frame in stt.run_stt(b"unused")]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].text, "Where is my order?")
        self.assertEqual(capture.omitted, 1)
        await observer.finish()

    async def test_completed_turns_release_capacity_for_later_clips(self):
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
        observer = NativeObserver(Span(), retain_audio=True, audio_format="wav", max_audio_bytes=1024)
        aggregator = LLMContextAggregatorPair(LLMContext()).user()

        async def push_context():
            pass

        aggregator.push_context_frame = push_context
        stt = STT()
        capture = UserCapture(observer, stt, aggregator, native_value)
        observer.user_capture = capture
        audio = encode_wav([b"\x10\x10" * 160], 16000, 1)
        try:
            for index in range(270):
                # An oversized clip midway through the call must not prevent later clips.
                clip = encode_wav([b"\1\0" * 1024], 16000, 1) if index == 135 else audio
                async for frame in stt.run_stt(clip):
                    await aggregator._handle_transcription(frame)
                turn = observer.turns.start("user", "UserStartedSpeakingFrame")
                observer.turns.confirm("user")
                await aggregator._push_aggregation()
                observer.turns.stop("user", SimpleNamespace(content="order?", timestamp=str(index), user_id="user"))
                await capture.jobs.drain()
                descriptors = [
                    r["metadata"]["audio.recordings"]
                    for r in turn["span"].rows
                    if "audio.recordings" in r.get("metadata", {})
                ]
                self.assertEqual(descriptors[-1][0]["state"], "omitted" if index in (1, 2, 135) else "ready")
                if index in (1, 2, 135):
                    self.assertEqual(
                        descriptors[-1][0]["reason"], "capture_byte_limit" if index == 135 else "upload_failed"
                    )
                    self.assertFalse(any(d["state"] == "ready" for row in descriptors for d in row))
                self.assertFalse(capture.batches)
                self.assertFalse(capture.segments)
                self.assertEqual(capture.bytes, 0)
        finally:
            await observer.finish()

    def test_joined_clip_positions_include_padding_but_do_not_map_it(self):
        from braintrust.audio.alignment import InputRanges

        from .user_capture import encode_segments

        ranges = InputRanges()
        ranges.append(1440, {"start": 23520, "end": 24240})
        ranges.append(480, {"start": 24240, "end": 24480})
        ranges.trim(960)  # Retain only the final phrase, excluding earlier input.
        first = encode_wav([b"\1\0" * 480 + b"\0\0" * 240], 24000, 1)
        second = encode_wav([b"\2\0" * 480], 24000, 1)
        encoded = encode_segments([first, second], "wav", [ranges.drain_mapped(), [(0, 960, 240000, 240480)]])
        self.assertEqual(encoded["duration_ms"], 50)
        self.assertEqual(
            encoded["clip_timeline"].ranges,
            [
                dict(recording_start_ms=0, recording_end_ms=20, timeline_start_ms=1000, timeline_end_ms=1020),
                dict(recording_start_ms=30, recording_end_ms=50, timeline_start_ms=10000, timeline_end_ms=10020),
            ],
        )
