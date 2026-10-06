import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.frames.frames import (  # pylint: disable=import-error
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
        capture = UserCapture(observer, stt, aggregator, native_value)
        observer.user_capture = capture
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
        observer, capture, turn = await self.exercise(False)
        self.assertEqual(capture.bytes, 0)
        self.assertTrue(all(s["audio"] is None for b in capture.batches.values() for s in b["segments"]))
        with (
            patch(
                "braintrust.integrations.pipecat.voice.user_capture.encode_segments",
                side_effect=AssertionError("encoded"),
            ),
            patch("braintrust.audio.attachments.RecordingAttachment", side_effect=AssertionError("attached")),
        ):
            await observer.finish()
        self.assertEqual(
            turn["span"].rows[-1]["metadata"]["audio.recordings"][0]["reason"],
            "disabled",
        )
        self.assertTrue(any("contrib.pipecat.transcriptions" in row.get("metadata", {}) for row in turn["span"].rows))

    async def test_user_audio_limit_is_explicit(self):
        observer, capture, turn = await self.exercise(True, max_bytes=10)
        self.assertEqual(capture.bytes, 0)
        await observer.finish()
        self.assertEqual(
            turn["span"].rows[-1]["metadata"]["audio.recordings"][0]["reason"],
            "capture_byte_limit",
        )

    async def test_opt_out_does_not_install_audio_hook_or_dispatch_encoder(self):
        observer = NativeObserver(Span(), retain_audio=False)
        observer.capture_transport = True
        stt = STT()
        original = stt.process_audio_frame
        aggregator = LLMContextAggregatorPair(LLMContext()).user()
        observer.user_capture = UserCapture(observer, stt, aggregator, native_value)
        self.assertEqual(stt.process_audio_frame, original)
        with (
            patch.object(observer.call_recording, "capture", side_effect=AssertionError("capture")),
            patch.object(observer.call_recording, "encode", side_effect=AssertionError("encode")),
            patch(
                "braintrust.audio.alignment.InputRanges.append",
                side_effect=AssertionError("ranges"),
            ),
            patch(
                "braintrust.audio.attachments.RecordingAttachment",
                side_effect=AssertionError("attachment"),
            ),
        ):
            from pipecat.frames.frames import InputAudioRawFrame  # pylint: disable=import-error
            from pipecat.processors.frame_processor import FrameDirection  # pylint: disable=import-error

            await stt.process_audio_frame(InputAudioRawFrame(b"\0\0" * 320, 16000, 1), FrameDirection.DOWNSTREAM)
            await observer.finish()

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
        observer = NativeObserver(Span(), retain_audio=True, audio_format="wav")
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
                async for frame in stt.run_stt(audio):
                    await aggregator._handle_transcription(frame)
                turn = observer.turns.start("user", "UserStartedSpeakingFrame")
                observer.turns.confirm("user")
                await aggregator._push_aggregation()
                observer.turns.stop("user", SimpleNamespace(content="order?", timestamp=str(index), user_id="user"))
                await asyncio.gather(*capture.tasks)
                descriptors = [
                    r["metadata"]["audio.recordings"]
                    for r in turn["span"].rows
                    if "audio.recordings" in r.get("metadata", {})
                ]
                self.assertEqual(descriptors[-1][0]["state"], "ready")
                self.assertFalse(capture.batches)
                self.assertFalse(capture.segments)
                self.assertEqual(capture.bytes, 0)
        finally:
            await observer.finish()

    def test_joined_clip_positions_include_padding_but_do_not_map_it(self):
        from .user_capture import encode_segments

        first = encode_wav([b"\1\0" * 480 + b"\0\0" * 240], 24000, 1)
        second = encode_wav([b"\2\0" * 480], 24000, 1)
        encoded = encode_segments([first, second], "wav", [[(0, 960, 24000, 24480)], [(0, 960, 240000, 240480)]])
        self.assertEqual(encoded["duration_ms"], 50)
        self.assertEqual(
            encoded["clip_timeline"].ranges,
            [
                dict(recording_start_ms=0, recording_end_ms=20, timeline_start_ms=1000, timeline_end_ms=1020),
                dict(recording_start_ms=30, recording_end_ms=50, timeline_start_ms=10000, timeline_end_ms=10020),
            ],
        )
