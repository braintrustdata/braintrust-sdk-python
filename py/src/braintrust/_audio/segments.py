"""Progressively export independently decodable segments on one sample timeline."""

import asyncio
from collections.abc import Awaitable, Callable

from .budget import source_budget
from .constants import MAX_BUFFERED_PACKETS, MAX_SEGMENT_JOBS, SEGMENT_HEADROOM_MS
from .exporter import SegmentExporter
from .jobs import RecordingJobs
from .options import RecordingOptions
from .recording import CallRecording, CaptureState, Packet
from .segment import AudioSegment
from .timeline import ms_to_samples


class SegmentedRecording(CallRecording):
    """Capture remains synchronous; detached segments belong to encoder workers.

    Input samples and output observation time must both pass a cut before it is
    exported. One second of headroom permits in-flight output and packet skew.
    """

    def __init__(
        self,
        *,
        options: RecordingOptions | None = None,
        on_segment: Callable[[AudioSegment], Awaitable[None]] | None = None,
        on_pending: Callable[[AudioSegment], None] | None = None,
        audio_format: str = "ogg",
        enabled: bool = True,
    ):
        self.options = options or RecordingOptions()
        super().__init__(
            enabled=enabled,
            max_bytes=self.options.max_buffer_bytes,
            max_duration_ms=self.options.max_duration_seconds * 1000,
            max_packets=MAX_BUFFERED_PACKETS,
        )
        self.on_segment = on_segment
        self.on_pending = on_pending
        self.exporter = SegmentExporter(audio_format)
        self.start_ms = 0.0
        self.sequence = 0
        self.segments: list[AudioSegment] = []
        self.jobs = RecordingJobs()
        self.inflight = []
        self._finish_task = None

    @property
    def completed(self):
        return [segment.interval() for segment in self.segments if segment.state != "pending"]

    @property
    def retained_bytes(self):
        return self.bytes + sum(segment.bytes for segment in self.inflight)

    def omit(self, reason):
        # Keep the valid prefix. Neither an error nor a limit erases prior audio.
        self.stop(reason)

    def capture(self, channel, pcm, sample_rate, channels, *, observed_ns=None, observed_unix_ms=None):
        if self.state is not CaptureState.OPEN:
            return None
        if self.retained_bytes + len(pcm) > self.max_bytes:
            self.omit("capture_byte_limit")
            return None
        interval = super().capture(
            channel, pcm, sample_rate, channels, observed_ns=observed_ns, observed_unix_ms=observed_unix_ms
        )
        if interval is None:
            return None
        # Never rewrite audio that was already sealed/exported.
        if interval["start"] < ms_to_samples(self.start_ms):
            self.omit("capture_clock_discontinuity")
            packet = self.packets.pop()
            self.bytes -= len(packet.pcm)
            source_budget.release(len(packet.pcm))
            return None
        latest = self.packets[-1].start_ms
        watermark = min(self.ends[0], latest) if self.started[0] else latest
        watermark = max(self.start_ms, watermark - SEGMENT_HEADROOM_MS)
        cut = self.start_ms + self.options.segment_duration_seconds * 1000
        if watermark >= cut:
            self._rotate(cut)
        elif self.bytes >= self.max_bytes * self.options.flush_fraction and watermark > self.start_ms:
            self._rotate(watermark)
        return interval

    def _rotate(self, cut):
        if not self.packets:
            return
        if len(self.jobs) >= MAX_SEGMENT_JOBS:
            self.omit("recording_export_capacity")
            return
        snapshot = CallRecording()
        segment = AudioSegment(f"call-{self.sequence:04d}", self.start_ms, cut, self.origin_unix_ms, snapshot)
        later = []
        for packet in self.packets:
            size = len(packet.pcm) // (2 * packet.channels)
            before = min(size, max(0, round((cut - packet.start_ms) * packet.sample_rate / 1000)))
            if before:
                data = packet.pcm if before == size else packet.pcm[: before * 2 * packet.channels]
                snapshot.packets.append(
                    Packet(packet.channel, packet.start_ms - self.start_ms, data, packet.sample_rate, packet.channels)
                )
                snapshot.bytes += len(data)
            if before < size:
                data = packet.pcm if not before else packet.pcm[before * 2 * packet.channels :]
                later.append(
                    Packet(
                        packet.channel,
                        packet.start_ms + before / packet.sample_rate * 1000,
                        data,
                        packet.sample_rate,
                        packet.channels,
                    )
                )
        snapshot.ends = [cut - self.start_ms, cut - self.start_ms]
        snapshot.seal()
        self.packets = later
        self.bytes -= segment.bytes
        self.start_ms = cut
        self.sequence += 1
        self.inflight.append(segment)
        self.segments.append(segment)

        def release():
            segment.clear()
            self.inflight.remove(segment)

        self.jobs.submit(lambda: self._export(segment), release)
        if self.on_pending:
            self.on_pending(segment)

    async def _export(self, segment: AudioSegment) -> None:
        try:
            await self.exporter.export(segment)
            if segment.state == "omitted":
                self.omit("segment_export_failed")
            # Publishing is independent of upload success. A logging failure
            # cannot invalidate an already uploaded segment.
            if self.on_segment:
                await self.on_segment(segment)
        finally:
            segment.encoded = None

    def release_idle_buffer(self):
        """Release unsubmitted audio unless a final drain still owns it."""
        if self._finish_task is None or self._finish_task.done():
            self.clear()

    async def finish(self):
        if self._finish_task is None:
            self._finish_task = asyncio.create_task(self._finish())
        await asyncio.shield(self._finish_task)

    async def _finish(self):
        self.seal()
        await self.drain_exports()
        if self.packets:
            self._rotate(max(self.ends))
        await self.drain_exports()

    async def drain_exports(self):
        """Wait for detached segments without sealing the active recording."""
        await self.jobs.drain()
