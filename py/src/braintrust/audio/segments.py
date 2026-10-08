"""Progressively export independently decodable segments on one sample timeline."""

import asyncio
from dataclasses import dataclass, field

from .attachments import UploadFailed, prepare_recording, upload_recording
from .budget import source_budget
from .jobs import RecordingJobs
from .options import RecordingOptions
from .recording import CallRecording, Packet
from .timeline import ms_to_samples
from .worker import encode_in_worker


@dataclass
class AudioSegment:
    """One manifest throughout encoding and export; PCM is sealed at rotation.

    Only the worker releases submitted PCM. Encoded bytes live through export;
    afterwards only small metadata remains on the call's manifest.
    """

    segment_id: str
    start_ms: float
    end_ms: float
    origin_unix_ms: float
    source: CallRecording
    state: str = "pending"
    reason: str | None = None
    encoded: dict | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def bytes(self):
        return self.source.bytes

    def clear(self):
        self.source.clear()

    def ready(self, encoded):
        self.encoded = encoded
        self.metadata = {key: encoded[key] for key in ("mime_type", "duration_ms", "channel_count")}
        self.state = "ready"

    def omit(self, reason):
        self.state, self.reason = "omitted", reason
        self.encoded = None
        self.metadata.clear()

    def interval(self):
        return {"id": self.segment_id, "start_ms": self.start_ms, "end_ms": self.end_ms, "state": self.state}


class SegmentedRecording(CallRecording):
    """Capture remains synchronous; detached segments belong to encoder workers.

    Input samples and output observation time must both pass a cut before it is
    exported. One second of headroom permits in-flight output and packet skew.
    """

    def __init__(self, *, options=None, on_segment=None, on_pending=None, audio_format="ogg", enabled=True):
        self.options = options or RecordingOptions()
        super().__init__(
            enabled=enabled,
            max_bytes=self.options.max_buffer_bytes,
            max_duration_ms=self.options.max_duration_seconds * 1000,
            max_packets=100000,
        )
        self.on_segment = on_segment
        self.on_pending = on_pending
        self.audio_format = audio_format
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
        self.reason = reason

    def capture(self, channel, pcm, sample_rate, channels, *, observed_ns=None, observed_unix_ms=None):
        if self.reason or self.sealed:
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
        watermark = max(self.start_ms, watermark - 1000)
        cut = self.start_ms + self.options.segment_duration_seconds * 1000
        if watermark >= cut:
            self._rotate(cut)
        elif self.bytes >= self.max_bytes * self.options.flush_fraction and watermark > self.start_ms:
            self._rotate(watermark)
        return interval

    def _rotate(self, cut):
        if not self.packets:
            return
        if len(self.jobs) >= 2:
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

    async def _export(self, segment):
        def encode():
            try:
                return prepare_recording(segment.segment_id, segment.source.encode, self.audio_format)
            finally:
                # Only the encoder may release a segment handed to its thread.
                segment.clear()

        try:
            try:
                encoded = await encode_in_worker(encode)
                segment.encoded = encoded
                await upload_recording(encoded)
                segment.ready(encoded)
            except Exception as error:  # noqa: BLE001 - recording cannot stop speech
                segment.omit("upload_failed" if isinstance(error, UploadFailed) else type(error).__name__)
                self.omit("segment_export_failed")
            # Publication failures do not invalidate an already uploaded recording.
            # The authoritative manifest remains available for subsequent upserts.
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
