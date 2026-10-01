import asyncio
import unittest
from types import SimpleNamespace

from braintrust.audio.recording import CallRecording

from .recording import capture_transport_output


class TransportRecordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_successful_writes_are_recorded_and_failures_propagate(self):
        recording = CallRecording()

        class Output:
            result = False

            async def write_audio_frame(self, audio):
                if isinstance(self.result, Exception):
                    raise self.result
                await asyncio.sleep(0)
                return self.result

        output = Output()
        capture_transport_output(output, recording)
        audio = SimpleNamespace(audio=b"\x01\x00" * 160, sample_rate=16000, num_channels=1)
        self.assertFalse(await output.write_audio_frame(audio))
        self.assertEqual(recording.bytes, 0)
        output.result = True
        self.assertTrue(await output.write_audio_frame(audio))
        self.assertEqual(recording.bytes, 320)
        output.result = RuntimeError("transport failed")
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            await output.write_audio_frame(audio)
        self.assertEqual(recording.bytes, 320)
