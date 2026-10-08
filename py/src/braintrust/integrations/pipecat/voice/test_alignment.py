import asyncio
import io
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf  # pylint: disable=import-error
from braintrust.audio.export import Alignment
from braintrust.audio.recording import CallRecording
from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame  # pylint: disable=import-error
from pipecat.transports.base_output import BaseOutputTransport  # pylint: disable=import-error
from pipecat.transports.base_transport import TransportParams  # pylint: disable=import-error

from .alignment import instrument_output
from .test_instrumentation import Span


class AlignmentTests(unittest.IsolatedAsyncioTestCase):
    async def make_output(self, enabled=True):
        class Output:
            success = True

            def create_task(self, coroutine):
                coroutine.close()
                return object()

            async def start(self, frame):
                pass

            async def write_audio_frame(self, frame):
                return self.success

        output = Output()
        params = TransportParams(audio_out_enabled=True)
        sender = BaseOutputTransport.MediaSender(
            output,
            destination=None,
            sample_rate=24000,
            audio_chunk_size=8,
            params=params,
        )
        sender._audio_queue = asyncio.Queue()
        output._media_senders = {None: sender}
        recording = CallRecording(enabled=enabled)
        alignment = Alignment(Span(), recording)
        instrument_output(output, alignment)
        await output.start(None)
        self.addCleanup(sender._executor.shutdown)
        return output, sender, alignment

    async def test_native_chunk_crossing_contexts_and_flush_padding(self):
        output, sender, alignment = await self.make_output()
        first, second = Span(), Span()
        alignment.begin_output("a", [first])
        alignment.begin_output("b", [second])

        def audio(context, samples):
            return TTSAudioRawFrame(
                audio=np.array(samples, dtype="<i2").tobytes(),
                sample_rate=24000,
                num_channels=1,
                context_id=context,
            )

        await sender.handle_audio_frame(audio("a", [10, 11, 12]))
        await sender.handle_audio_frame(audio("b", [20, 21, 22]))
        await sender.handle_tts_stopped(TTSStoppedFrame(context_id="b"))
        with patch("time.monotonic_ns", return_value=0):
            while not sender._audio_queue.empty():
                frame = sender._audio_queue.get_nowait()
                if hasattr(frame, "audio"):
                    await output.write_audio_frame(frame)
        alignment.publish()
        self.assertEqual(first.rows[-1]["metadata"]["audio.selection"]["end_offset_ms"], 3 / 24)
        self.assertEqual(second.rows[-1]["metadata"]["audio.selection"]["start_offset_ms"], 3 / 24)
        decoded, _ = sf.read(
            io.BytesIO(alignment.recording.encode("wav")["data"]),
            dtype="int16",
            always_2d=True,
        )
        np.testing.assert_array_equal(decoded[:8, 1], [10, 11, 12, 20, 21, 22, 0, 0])
        alignment.publish()
        self.assertEqual(second.rows[-1]["metadata"]["audio.selection"]["end_offset_ms"], 6 / 24)

    async def test_discarded_output_and_failed_write_do_not_get_selections(self):
        output, sender, alignment = await self.make_output()

        owner = Span()
        alignment.begin_output("a", [owner])

        def audio(samples):
            return TTSAudioRawFrame(
                audio=np.array(samples, dtype="<i2").tobytes(),
                sample_rate=24000,
                num_channels=1,
                context_id="a",
            )

        await sender.handle_audio_frame(audio([1, 2]))
        sender._clear_audio_buffer()  # Native interruption discards the buffered tail.
        await sender.handle_audio_frame(audio([3, 4, 5, 6]))
        frame = sender._audio_queue.get_nowait()
        output.success = False
        await output.write_audio_frame(frame)
        alignment.publish()
        self.assertEqual(owner.rows, [{}])
        self.assertEqual(alignment.recording.bytes, 0)
        await sender.handle_audio_frame(audio([7, 8, 9, 10]))
        output.success = True
        await output.write_audio_frame(sender._audio_queue.get_nowait())
        alignment.publish_clip(owner, {"id": "tts-clip", "state": "ready"})
        mapping = owner.rows[-1]["metadata"]["audio.recordings"][0]["timeline"]["ranges"]
        self.assertEqual(len(mapping), 1)
        self.assertEqual(mapping[0]["recording_start_ms"], 6 / 24)
        self.assertEqual(mapping[0]["recording_end_ms"], 10 / 24)
        self.assertEqual(mapping[0]["timeline_start_ms"], 0)

    async def test_queue_recreation_keeps_context_mapping(self):
        output, sender, alignment = await self.make_output()
        owner = Span()
        alignment.begin_output("after-interruption", [owner])
        import gc
        import weakref

        old_queue = weakref.ref(sender._audio_queue)
        await sender._audio_queue.put(TTSAudioRawFrame(b"\1\0" * 480, 24000, 1))
        sender._create_audio_task()  # Pipecat replaces the queue after interruption.
        gc.collect()
        self.assertIsNone(old_queue(), "discarded queue and its audio must be released before shutdown")
        frame = TTSAudioRawFrame(
            audio=b"\x01\x00" * 4,
            sample_rate=24000,
            num_channels=1,
            context_id="after-interruption",
        )
        await sender.handle_audio_frame(frame)
        await output.write_audio_frame(sender._audio_queue.get_nowait())
        alignment.publish()
        selection = owner.rows[-1]["metadata"]["audio.selection"]
        self.assertEqual((selection["start_offset_ms"], selection["end_offset_ms"]), (0, 4 / 24))
