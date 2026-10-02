import io
import unittest

import numpy as np
import soundfile as sf  # pylint: disable=import-error

from .recording import CallRecording, encode_audio


def frame(samples, rate=16000):
    return dict(
        pcm=np.asarray(samples, dtype="<i2").tobytes(),
        sample_rate=rate,
        channels=1,
    )


class RecordingTests(unittest.IsolatedAsyncioTestCase):
    def test_delayed_input_packets_do_not_insert_silence(self):
        recording = CallRecording()
        for index in range(100):
            recording.capture(0, **frame([1000] * 320), observed_ns=index * 22_000_000)
        result = recording.encode("wav")
        samples, _ = sf.read(io.BytesIO(result["data"]), dtype="int16", always_2d=True)
        self.assertEqual(result["duration_ms"], 2000)
        self.assertEqual(samples.shape, (48000, 2))
        self.assertTrue(np.all(samples[:, 0] == 1000))

    def test_input_anchor_and_real_silence_preserve_output_pauses(self):
        recording = CallRecording()
        recording.capture(1, **frame([-2000] * 320), observed_ns=0)
        recording.capture(0, **frame([1000] * 320), observed_ns=100_000_000)
        recording.capture(0, **frame([0] * 320), observed_ns=125_000_000)
        recording.capture(0, **frame([1000] * 320), observed_ns=126_000_000)
        recording.capture(1, **frame([-2000] * 320), observed_ns=200_000_000)
        result = recording.encode("wav")
        samples, _ = sf.read(io.BytesIO(result["data"]), dtype="int16", always_2d=True)
        self.assertEqual([p.start_ms for p in recording.packets], [0, 100, 120, 140, 200])
        self.assertTrue(np.all(samples[:2400, 0] == 0))
        self.assertTrue(np.all(samples[2880:3360, 0] == 0))
        self.assertTrue(np.all(samples[3360:3840, 0] == 1000))
        self.assertTrue(np.all(samples[480:4800, 1] == 0))
        self.assertEqual(result["duration_ms"], 220)

    def test_two_sources_keep_overlap_silence_and_duration(self):
        recording = CallRecording()
        recording.capture(0, **frame([1000] * 1600), observed_ns=0, observed_unix_ms=123456)
        recording.capture(1, **frame([-2000] * 2400, 24000), observed_ns=50_000_000)
        result = recording.encode("wav")
        samples, rate = sf.read(io.BytesIO(result["data"]), dtype="int16", always_2d=True)
        self.assertEqual(rate, 24000)
        self.assertEqual(samples.shape, (3600, 2))
        self.assertEqual(result["duration_ms"], 150)
        self.assertTrue(np.all(samples[:1200, 1] == 0))
        self.assertTrue(np.all(samples[1200:2400] == [1000, -2000]))
        self.assertTrue(np.all(samples[2400:, 0] == 0))

    def test_disabled_and_limits_release_all_audio(self):
        recording = CallRecording(max_bytes=10)
        recording.capture(0, **frame([1] * 6), observed_ns=0)
        self.assertEqual(recording.reason, "capture_byte_limit")
        self.assertEqual(recording.packets, [])
        self.assertEqual(recording.bytes, 0)
        self.assertIsNone(recording.encode())
        disabled = CallRecording(enabled=False)
        disabled.capture(0, **frame([1] * 100), observed_ns=0)
        self.assertEqual(disabled.bytes, 0)
        duration = CallRecording(max_duration_ms=1)
        duration.capture(0, **frame([1] * 160), observed_ns=0)
        self.assertEqual(duration.reason, "duration_limit")

    def test_ogg_decodes_to_original_length_and_is_smaller(self):
        samples = (np.sin(np.arange(24000) * 2 * np.pi * 440 / 24000) * 12000).astype("<i2")
        result = encode_audio([samples.tobytes()], 24000, 1)
        decoded, rate = sf.read(io.BytesIO(result["data"]), always_2d=True)
        self.assertEqual(rate, 24000)
        self.assertEqual(len(decoded), 24000)
        self.assertEqual(result["mime_type"], "audio/ogg")
        self.assertLess(len(result["data"]), samples.nbytes)
        self.assertGreater(np.max(np.abs(decoded)), 0.1)


if __name__ == "__main__":
    unittest.main()
