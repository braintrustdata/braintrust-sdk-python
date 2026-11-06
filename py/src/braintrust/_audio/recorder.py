import asyncio
import logging
import math
import sys
from array import array
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from fractions import Fraction

from braintrust._audio.config import AudioRecordingOptions

from .middleware import AudioMiddleware, run_middleware
from .types import AudioChannel, AudioResult, RecordingResult, SegmentMeta
from .worker import run_native


SAMPLE_RATE = 48000
NS = 1_000_000_000


@dataclass(frozen=True)
class Packet:
    source: str
    data: bytes
    rate: int
    channels: int
    start: Fraction  # Recording samples; retaining the fraction preserves run phase.

    @property
    def count(self):
        return len(self.data) // (2 * self.channels)

    @property
    def end(self):
        return self.start + Fraction(self.count * SAMPLE_RATE, self.rate)


def _samples(data):
    result = array("h")
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def render(packets: tuple[Packet, ...], sources: tuple[str, ...], start: int, end: int):
    """Sparse placement and phase-continuous linear resampling, in a worker.

    Phase is derived from absolute rational positions, never rounded per packet.
    A following packet supplies the interpolation sample at a packet boundary.
    Silence is materialized for only the segment being processed.
    """
    output = array("h", [0]) * ((end - start) * len(sources))
    represented = []
    for channel, source in enumerate(sources):
        stream = [packet for packet in packets if packet.source == source]
        present = False
        for index, packet in enumerate(stream):
            first = max(start, math.ceil(packet.start))
            last = min(end, math.ceil(packet.end))
            if first >= last:
                continue
            present = True
            samples = _samples(packet.data)
            # Downmix before interpolation. The next packet belongs to the same
            # run only when its position, rate and channel count agree exactly.
            mono = [
                sum(samples[i : i + packet.channels]) // packet.channels
                for i in range(0, len(samples), packet.channels)
            ]
            following = mono[-1]
            if index + 1 < len(stream):
                other = stream[index + 1]
                if other.start == packet.end and other.rate == packet.rate and other.channels == packet.channels:
                    frame = _samples(other.data[: other.channels * 2])
                    following = sum(frame) // other.channels
            phase = (Fraction(first) - packet.start) * packet.rate / SAMPLE_RATE
            step = Fraction(packet.rate, SAMPLE_RATE)
            denominator = math.lcm(phase.denominator, step.denominator)
            numerator = phase.numerator * (denominator // phase.denominator)
            increment = step.numerator * (denominator // step.denominator)
            for position in range(first, last):
                sample, remainder = divmod(numerator, denominator)
                value = mono[sample]
                next_value = mono[sample + 1] if sample + 1 < len(mono) else following
                output[(position - start) * len(sources) + channel] = (
                    value + (next_value - value) * remainder // denominator
                )
                numerator += increment
        represented.append(AudioChannel((source,) if present else ()))
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes(), tuple(represented)


class AudioRecorder:
    """Accept positioned PCM without waiting for normalization, encoding or IO.

    ``advance`` is a promise by the adapter: no source will add audio before its
    watermark. Thresholds request flushing; they never reject admitted samples.
    """

    def __init__(
        self,
        *,
        sources: tuple[str, ...],
        options: AudioRecordingOptions,
        middleware: tuple[AudioMiddleware, ...],
        on_segment: Callable[[AudioResult], Awaitable[None]],
        on_omitted: Callable[[SegmentMeta, str], None],
    ):
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("Recording sources must be nonempty and unique")
        self.sources = tuple(sources)
        self.options = options
        self.middleware = middleware
        self.on_segment = on_segment
        self.on_omitted = on_omitted
        self._packets: list[Packet] = []
        self._watermark = 0
        self._sealed = 0
        self._end = 0
        self._index = 0
        self._task: asyncio.Task | None = None
        self._finishing = False
        self._discarded = False
        self.reason: str | None = None
        self._warned_backlog = False
        self._processing: SegmentMeta | None = None
        self._gaps: list[tuple[int, int, str]] = []

    @property
    def buffered_bytes(self):
        return sum(len(packet.data) for packet in self._packets)

    def record(
        self, source_id: str, pcm: bytes, *, sample_rate_hz: int, channels: int, offset_ns: int | Fraction
    ) -> bool:
        if self._finishing or self.reason:
            return False
        if source_id not in self.sources or sample_rate_hz <= 0 or channels <= 0 or len(pcm) % (channels * 2):
            raise ValueError("Invalid positioned PCM")
        if not pcm:
            return True
        start = Fraction(offset_ns) * SAMPLE_RATE / NS
        if start < self._sealed or start < 0:
            raise ValueError("Audio arrived before the sealed recording watermark")
        packet = Packet(source_id, bytes(pcm), sample_rate_hz, channels, start)
        self._packets.append(packet)
        self._end = max(self._end, math.ceil(packet.end))
        if self.options.max_duration_seconds is not None:
            limit = round(self.options.max_duration_seconds * SAMPLE_RATE)
            if self._end >= limit:
                self._end = limit
                self.stop("duration_limit")
        buffered = self.buffered_bytes
        if buffered > self.options.flush_threshold_bytes * 4 and not self._warned_backlog:
            self._warned_backlog = True
            logging.getLogger("braintrust").warning(
                "Audio recording backlog exceeds its flush target; capture continues"
            )
        self._pump()
        return True

    def advance(self, through_ns: int | Fraction):
        self._watermark = max(self._watermark, math.floor(Fraction(through_ns) * SAMPLE_RATE / NS))
        self._pump()

    def unavailable(self, start_ns, end_ns, reason):
        """Known bounds of unavailable audio; never encode it as real silence."""
        start = max(0, math.floor(Fraction(start_ns) * SAMPLE_RATE / NS))
        end = math.ceil(Fraction(end_ns) * SAMPLE_RATE / NS)
        if end <= start or self._discarded:
            return
        if start < self._sealed:
            raise ValueError("Unavailable audio arrived before the sealed watermark")
        self._gaps.append((start, end, reason))
        self._end = max(self._end, end)
        self._pump()

    def stop(self, reason: str):
        self.reason = self.reason or reason

    def discard(self, reason: str):
        """Stop and erase queued PCM; in-flight workers retain their own copies."""
        if self._discarded:
            return
        self.stop(reason)
        self._discarded = True
        if self._processing is not None:
            self.on_omitted(self._processing, reason)
        if self._end > self._sealed:
            self.on_omitted(
                SegmentMeta(
                    self._index,
                    self._sealed,
                    self._end - self._sealed,
                    SAMPLE_RATE,
                    tuple(AudioChannel(()) for _ in self.sources),
                ),
                reason,
            )
            self._index += 1
        self._packets.clear()

    def _cut(self):
        eligible = min(self._end, self._watermark)
        target = self._sealed + max(1, round(self.options.segment_duration_seconds * SAMPLE_RATE))
        if eligible <= self._sealed:
            return None
        if eligible >= target:
            return target
        if self._finishing or self.buffered_bytes >= self.options.flush_threshold_bytes:
            return eligible
        return None

    def _pump(self):
        if self._task is None or self._task.done():
            if self._cut() is not None:
                self._task = asyncio.create_task(self._drain())

    async def _drain(self):
        while not self._discarded and (end := self._cut()) is not None:
            start = self._sealed
            meta = SegmentMeta(
                self._index, start, end - start, SAMPLE_RATE, tuple(AudioChannel(()) for _ in self.sources)
            )
            self._index += 1
            self._processing = meta
            # Snapshot references before detaching accepted buffers. Native work
            # owns them even if the awaiting coroutine is cancelled.
            packets = tuple(self._packets)
            self._sealed = end
            gap = next((gap for gap in self._gaps if gap[0] < end and gap[1] > start), None)
            if gap is not None:
                self.on_omitted(meta, gap[2])
                self._packets = [packet for packet in self._packets if packet.end > end]
                self._gaps = [gap for gap in self._gaps if gap[1] > end]
                self._processing = None
                continue
            try:
                data, channels = await run_native(render, packets, self.sources, start, end)
                meta = SegmentMeta(meta.index, start, end - start, SAMPLE_RATE, channels)
                result = await run_middleware(self.middleware, data, meta)
                if not self._discarded:
                    await self.on_segment(result)
                else:
                    self.on_omitted(meta, self.reason or "recording_stopped")
            except asyncio.CancelledError:
                self.on_omitted(meta, "shutdown_timeout")
                raise
            except Exception:
                self.stop("encoding_or_upload_failed")
                self.on_omitted(meta, self.reason)
                # Already accepted audio is unavailable; retain honest placement
                # rather than claim the remainder of the call was recorded.
                if self._end > end:
                    self.on_omitted(
                        SegmentMeta(self._index, end, self._end - end, SAMPLE_RATE, meta.channels), self.reason
                    )
                    self._index += 1
                self._packets.clear()
                self._sealed = self._end
                self._processing = None
                self._discarded = True
                return
            self._packets = [packet for packet in self._packets if packet.end > end]
            self._processing = None

    async def finish(self) -> RecordingResult:
        self._finishing = True
        # The adapter settles pending playback before calling finish.
        self._watermark = max(self._watermark, self._end)
        self._pump()
        if self._task is not None:
            await asyncio.shield(self._task)
        return RecordingResult(self._index, self.reason)
