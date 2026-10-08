"""Local scheduler tests; no provider responses are simulated."""

import asyncio
import threading
import unittest

from .worker import RecordingBusy, encode_in_worker


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_stays_reserved_after_waiter_cancel(self):
        entered, release = threading.Event(), threading.Event()

        def block():
            entered.set()
            release.wait(5)
            return 1

        first = asyncio.create_task(encode_in_worker(block))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.001)
            second = asyncio.create_task(encode_in_worker(lambda: 2))
            await asyncio.sleep(0.01)
            first.cancel()
            await asyncio.sleep(0)
            self.assertFalse(first.done())
            with self.assertRaises(RecordingBusy):
                await encode_in_worker(lambda: 3)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual(await second, 2)
        finally:
            release.set()
        self.assertEqual(await encode_in_worker(lambda: 4), 4)

    async def test_recording_jobs_release_once_on_success_failure_and_early_cancellation(self):
        from .jobs import RecordingJobs

        jobs = RecordingJobs()
        released = []
        executed = []
        flushed = []

        async def flush():
            self.assertEqual(released, ["success"])
            flushed.append("success")

        async def operation(outcome):
            executed.append(outcome)
            if outcome == "failure":
                raise ValueError("publication failed")

        for outcome in ("success", "failure", "cancelled"):
            task = jobs.submit(
                lambda: operation(outcome),
                lambda: released.append(outcome),
                after_release=flush if outcome == "success" else None,
            )
            if outcome == "cancelled":
                task.cancel()  # No operation body or finally block has run yet.
            if outcome == "failure":
                with self.assertLogs("braintrust._audio.jobs", level="WARNING"):
                    await jobs.drain()
            else:
                await jobs.drain()
        self.assertEqual(flushed, ["success"])
        self.assertEqual(executed, ["success", "failure"])
        self.assertEqual(released, ["success", "failure", "cancelled"])

    async def test_capture_limits_reject_and_release_retained_audio(self):
        from unittest.mock import patch

        from .budget import ByteBudget
        from .recording import CallRecording

        budget = ByteBudget(1000)
        with patch("braintrust._audio.recording.source_budget", budget):
            first, second = CallRecording(), CallRecording()
            pcm = b"\0\0" * 480
            self.assertIsNotNone(first.capture(0, pcm, 24000, 1))
            self.assertIsNone(second.capture(0, pcm, 24000, 1))
            self.assertEqual(second.reason, "process_capture_byte_limit")
            self.assertEqual(second.bytes, 0)
            first.clear()
            self.assertEqual(budget.used, 0)

        for options, reason in (
            ({"max_bytes": 1000}, "capture_byte_limit"),
            ({"max_duration_ms": 30}, "duration_limit"),
        ):
            recording = CallRecording(**options)
            self.assertIsNotNone(recording.capture(0, pcm, 24000, 1))
            self.assertIsNone(recording.capture(0, pcm, 24000, 1))
            self.assertEqual(recording.reason, reason)
            self.assertEqual(recording.bytes, 0)
            self.assertIsNone(recording.encode())
