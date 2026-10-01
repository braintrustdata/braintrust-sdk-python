import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf  # pylint: disable=import-error
from braintrust.audio.alignment import Alignment, InputRanges
from braintrust.audio.recording import CallRecording
from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame  # pylint: disable=import-error
from pipecat.transports.base_output import BaseOutputTransport  # pylint: disable=import-error
from pipecat.transports.base_transport import TransportParams  # pylint: disable=import-error

from .alignment import instrument_output
from .test_instrumentation import Span


class AlignmentTests(unittest.IsolatedAsyncioTestCase):
    def test_input_trim_uses_sample_positions_not_span_time(self):
        recording = CallRecording()
        ledger = InputRanges()
        for index in range(3):
            frame = SimpleNamespace(
                audio=np.full(320, index + 1, dtype="<i2").tobytes(),
                sample_rate=16000,
                num_channels=1,
            )
            interval = recording.capture(
                0, frame.audio, frame.sample_rate, frame.num_channels, observed_ns=(100 + index * 22) * 1000000
            )
            ledger.append(len(frame.audio), interval)
        # Native STT retained the final 30 ms (half a packet plus a packet).
        ledger.trim(960)
        ranges = ledger.drain()
        self.assertEqual(ranges, [[720, 1440]])
        decoded, _ = sf.read(io.BytesIO(recording.encode("wav")["data"]), dtype="int16", always_2d=True)
        self.assertTrue(np.all(decoded[720:960, 0] == 2))
        self.assertTrue(np.all(decoded[960:1440, 0] == 3))

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
        alignment.contexts = {"a": [first], "b": [second]}

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
        self.assertEqual(alignment.owners[first.span_id][2], [[0, 3]])
        self.assertEqual(alignment.owners[second.span_id][2], [[3, 6]])
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
        alignment.contexts["a"] = [owner]

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
        self.assertEqual(alignment.owners, {})
        self.assertEqual(alignment.recording.bytes, 0)

    async def test_queue_recreation_keeps_context_mapping(self):
        output, sender, alignment = await self.make_output()
        owner = Span()
        alignment.contexts["after-interruption"] = [owner]
        sender._create_audio_task()  # Pipecat replaces the queue after interruption.
        frame = TTSAudioRawFrame(
            audio=b"\x01\x00" * 4,
            sample_rate=24000,
            num_channels=1,
            context_id="after-interruption",
        )
        await sender.handle_audio_frame(frame)
        await output.write_audio_frame(sender._audio_queue.get_nowait())
        self.assertEqual(alignment.owners[owner.span_id][2], [[0, 4]])

    def test_disabled_or_omitted_recording_never_publishes_selection(self):
        root, owner = Span(), Span()
        recording = CallRecording(enabled=False)
        alignment = Alignment(root, recording)
        alignment.add(owner, [[0, 100]], 0)
        alignment.publish()
        self.assertEqual(owner.rows, [{}])
