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
        recording.capture(1, **frame([-2000] * 960, 24000), observed_ns=140_000_000)
        recording.capture(1, **frame([-2000] * 320), observed_ns=200_000_000)
        recording.seal()
        self.assertIsNone(recording.capture(0, **frame([123] * 320)))
        result = recording.encode("wav")
        samples, rate = sf.read(io.BytesIO(result["data"]), dtype="int16", always_2d=True)
        self.assertEqual(rate, 24000)
        self.assertEqual(samples.shape, (5280, 2))
        self.assertTrue(np.all(samples[:2400, 0] == 0))
        self.assertTrue(np.all(samples[2880:3360, 0] == 0))
        self.assertTrue(np.all(samples[3360:3840, 0] == 1000))
        self.assertTrue(np.all(samples[480:3360, 1] == 0))
        self.assertTrue(np.all(samples[3360:3840] == [1000, -2000]))
        self.assertTrue(np.all(samples[3840:, 0] == 0))
        self.assertTrue(np.all(samples[4320:4800, 1] == 0))
        self.assertEqual(result["duration_ms"], 220)

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
