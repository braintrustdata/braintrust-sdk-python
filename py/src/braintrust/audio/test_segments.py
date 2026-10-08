"""Local PCM lifecycle tests: no provider or network behavior is simulated."""

import asyncio
import io
import unittest

import numpy as np
import soundfile as sf  # pylint: disable=import-error

from .segments import RecordingOptions, SegmentedRecording


class SegmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_rotates_during_capture_and_preserves_cross_boundary_samples(self):
        files = []
        pending = set()

        def publish_pending(segment):
            self.assertEqual(segment.state, "pending")
            self.assertIsNone(segment.encoded)
            pending.add(segment.segment_id)

        async def publish(segment):
            encoded = segment.encoded
            self.assertEqual(segment.state, "ready")
            self.assertIn(segment.segment_id, pending)
            pending.remove(segment.segment_id)
            files.append((segment.start_ms, encoded))

        recorder = SegmentedRecording(
            options=RecordingOptions(segment_duration_seconds=0.1),
            on_segment=publish,
            on_pending=publish_pending,
            audio_format="wav",
        )
        pcm = np.full(480, 1000, dtype="<i2").tobytes()
        for index in range(90):
            for channel in (0, 1):
                recorder.capture(channel, pcm, 24000, 1, observed_ns=index * 20_000_000)
            await asyncio.sleep(0.002)
        self.assertTrue(files, "segments must export before finish")
        await recorder.finish()
        self.assertEqual(recorder.retained_bytes, 0)
        position = 0
        for start, encoded in files:
            self.assertAlmostEqual(start, position)
            decoded, rate = sf.read(io.BytesIO(encoded["data"]), dtype="int16", always_2d=True)
            self.assertTrue(np.all(decoded == 1000))
            position += len(decoded) / rate * 1000
        self.assertAlmostEqual(position, 1800)
        self.assertFalse(pending)
        self.assertTrue(all(segment.encoded is None for segment in recorder.segments))
        self.assertTrue(all(segment.metadata["mime_type"] == "audio/wav" for segment in recorder.segments))

    async def test_large_frame_crosses_rotation_without_losing_samples(self):
        files = []

        async def publish(segment):
            encoded = segment.encoded
            files.append((segment.start_ms, encoded))

        recorder = SegmentedRecording(
            options=RecordingOptions(segment_duration_seconds=1), on_segment=publish, audio_format="wav"
        )
        # One native write contains three seconds. Subsequent writes advance
        # the export watermark across it; rotation must preserve every sample.
        audio = np.arange(24000 * 3, dtype=np.int16)
        recorder.capture(1, audio.tobytes(), 24000, 1, observed_ns=0)
        for index in range(4):
            recorder.capture(1, b"\0\0" * 24000, 24000, 1, observed_ns=(3 + index) * 1_000_000_000)
            await recorder.drain_exports()
        self.assertTrue(files, "rotation must export before shutdown")
        await recorder.finish()
        pieces = []
        position = 0
        for start, encoded in files:
            self.assertAlmostEqual(start, position)
            samples, rate = sf.read(io.BytesIO(encoded["data"]), dtype="int16", always_2d=True)
            pieces.append(samples[:, 1])
            position += len(samples) / rate * 1000
        np.testing.assert_array_equal(
            np.concatenate(pieces), np.concatenate([audio, np.zeros(24000 * 4, dtype=np.int16)])
        )
        self.assertIsNone(recorder.reason)
        self.assertEqual(recorder.retained_bytes, 0)

    async def test_duration_limit_preserves_partial_audio(self):
        files = []

        async def publish(segment):
            encoded = segment.encoded
            files.append(encoded)

        recorder = SegmentedRecording(
            options=RecordingOptions(max_duration_seconds=0.1), on_segment=publish, audio_format="wav"
        )
        for index in range(10):
            recorder.capture(0, b"\x01\x00" * 480, 24000, 1, observed_ns=index * 20_000_000)
        await recorder.finish()
        self.assertEqual(recorder.reason, "duration_limit")
        self.assertEqual(sum(f["duration_ms"] for f in files), 100)
        self.assertEqual(recorder.retained_bytes, 0)

    async def test_ten_minute_timeline_with_thirty_second_segments(self):
        durations = []
        peak = 0

        async def publish(segment):
            encoded = segment.encoded
            decoded, rate = sf.read(io.BytesIO(encoded["data"]), dtype="int16", always_2d=True)
            self.assertTrue(np.all(decoded == [1000, -2000]))
            self.assertAlmostEqual(segment.start_ms, sum(durations))
            durations.append(len(decoded) / rate * 1000)

        recorder = SegmentedRecording(
            options=RecordingOptions(segment_duration_seconds=30), on_segment=publish, audio_format="wav"
        )
        user = np.full(12000, 1000, dtype="<i2").tobytes()
        agent = np.full(12000, -2000, dtype="<i2").tobytes()
        for index in range(1200):
            recorder.capture(0, user, 24000, 1, observed_ns=index * 500_000_000)
            recorder.capture(1, agent, 24000, 1, observed_ns=index * 500_000_000)
            peak = max(peak, recorder.retained_bytes)
            # Accelerated replay: permit the real encoder/exporter to drain.
            await recorder.drain_exports()
        await recorder.finish()
        self.assertEqual(durations, [30000] * 20)
        self.assertIsNone(recorder.reason)
        self.assertLess(peak, 4 * 1024 * 1024)
        self.assertEqual(recorder.retained_bytes, 0)

    async def test_byte_rotation_and_slow_export_preserve_bounded_prefix(self):
        released = asyncio.Event()

        async def publish(segment):
            encoded = segment.encoded
            await released.wait()

        recorder = SegmentedRecording(
            options=RecordingOptions(segment_duration_seconds=300, max_buffer_bytes=200000),
            on_segment=publish,
            audio_format="wav",
        )
        for index in range(150):
            recorder.capture(0, b"\1\0" * 480, 24000, 1, observed_ns=index * 20_000_000)
            recorder.capture(1, b"\1\0" * 480, 24000, 1, observed_ns=index * 20_000_000)
            self.assertLessEqual(recorder.retained_bytes, 200000)
            await asyncio.sleep(0)
        self.assertGreater(recorder.sequence, 0)
        released.set()
        await recorder.finish()
        self.assertEqual(recorder.retained_bytes, 0)
        self.assertTrue(any(s["state"] == "ready" for s in recorder.completed))

    def test_selection_crossing_segment_boundary(self):
        from types import SimpleNamespace

        from .export import Alignment

        logs = []
        root = SimpleNamespace(span_id="root", log=lambda **_: None)
        owner = SimpleNamespace(span_id="turn", log=lambda **row: logs.append(row))
        recording = SimpleNamespace(
            reason=None,
            completed=[
                {"id": "call-0000", "start_ms": 0, "end_ms": 30000, "state": "ready"},
                {"id": "call-0001", "start_ms": 30000, "end_ms": 60000, "state": "ready"},
            ],
        )
        alignment = Alignment(root, recording)
        alignment.add(owner, [[29000 * 24, 32000 * 24]], 0)
        alignment.publish()
        selections = logs[-1]["metadata"]["audio.selections"]
        self.assertEqual(
            [(s["recording_id"], s["start_offset_ms"], s["end_offset_ms"]) for s in selections],
            [("call-0000", 29000, 30000), ("call-0001", 0, 2000)],
        )
        self.assertIsNone(logs[-1]["metadata"]["audio.selection"])

    async def test_cancelled_finish_waiter_does_not_release_encoder_input(self):
        import threading
        from unittest.mock import patch

        from .recording import CallRecording

        entered, release = threading.Event(), threading.Event()
        seen = []
        original = CallRecording.encode

        def encode(segment, audio_format):
            entered.set()
            release.wait(2)
            seen.append(segment.bytes)
            return original(segment, audio_format)

        recorder = SegmentedRecording(audio_format="wav")
        recorder.capture(0, b"\1\0" * 480, 24000, 1, observed_ns=0)
        with patch.object(CallRecording, "encode", encode):
            waiter = asyncio.create_task(recorder.finish())
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertEqual(recorder.retained_bytes, 960)
            release.set()
            await recorder.finish()
        self.assertEqual(seen, [960])
        self.assertEqual(recorder.retained_bytes, 0)

    def test_out_of_order_export_releases_only_resolved_ranges(self):
        from types import SimpleNamespace

        from .export import Alignment

        rows = []
        owner = SimpleNamespace(span_id="turn", log=lambda **row: rows.append(row))
        recording = SimpleNamespace(
            reason=None, completed=[{"id": "later", "start_ms": 1000, "end_ms": 2000, "state": "ready"}]
        )
        alignment = Alignment(SimpleNamespace(span_id="root", log=lambda **_: None), recording)
        alignment.add(owner, [[0, 48000]], 0)
        alignment.publish()
        self.assertEqual(rows[-1]["metadata"]["audio.selection"]["recording_id"], "later")
        recording.completed.append({"id": "earlier", "start_ms": 0, "end_ms": 1000, "state": "ready"})
        alignment.publish()
        self.assertEqual(alignment.count, 0)
        self.assertEqual({s["recording_id"] for s in rows[-1]["metadata"]["audio.selections"]}, {"earlier", "later"})
        # The admission budget is available again, even after many completed ranges.
        for _ in range(33000):
            alignment.add(owner, [[0, 1]], 0)
            alignment.publish()
        self.assertEqual(alignment.omitted, 0)
        self.assertEqual(len(rows), 3, "unchanged historical selections must not be exported again")
