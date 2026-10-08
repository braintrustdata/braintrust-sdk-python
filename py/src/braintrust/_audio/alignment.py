"""Bounded sample ranges and recording selections shared by voice integrations."""

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from .constants import MAX_PENDING_RANGES
from .timeline import ClipTimeline, ms_to_samples, samples_to_ms
from .types import SegmentInterval, Selection


def merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    result = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def resolve_selections(
    ranges: list[list[int]],
    segments: list[SegmentInterval],
    resolved: list[list[int]],
    recording_span_id: str,
    channel: int,
) -> tuple[list[Selection], list[list[int]]]:
    """Map call samples into ready files; retain ranges whose files are pending."""
    selections = []
    merged = merge_ranges(ranges)
    pending = []
    for segment in segments:
        if segment["state"] != "ready":
            continue
        lower, upper = ms_to_samples(segment["start_ms"]), ms_to_samples(segment["end_ms"])
        for start, end in merged:
            start, end = max(start, lower), min(end, upper)
            if end > start:
                selections.append(
                    {
                        "recording_span_id": recording_span_id,
                        "recording_id": segment["id"],
                        "start_offset_ms": samples_to_ms(start - lower),
                        "end_offset_ms": samples_to_ms(end - lower),
                        "channel_index": channel,
                    }
                )
    for start, end in ranges:
        for lower, upper in resolved:
            if lower > start:
                pending.append([start, min(lower, end)])
            start = max(start, upper) if lower < end else end
            if start >= end:
                break
        if start < end:
            pending.append([start, end])
    return selections, pending


class InputRanges:
    def __init__(self):
        self.runs = deque()
        self.size = 0

    def append(self, size, interval):
        self.runs.append([size, interval, size, 0])
        self.size += size

    def trim(self, retained):
        while self.size > retained and self.runs:
            run = self.runs[0]
            count = min(self.size - retained, run[0])
            run[0] -= count
            run[3] += count
            self.size -= count
            if not run[0]:
                self.runs.popleft()

    def drain_mapped(self):
        pieces = []
        position = 0
        for count, interval, original, offset in self.runs:
            if interval:
                length = interval["end"] - interval["start"]
                pieces.append(
                    (
                        position,
                        position + count,
                        interval["start"] + round(offset * length / original),
                        interval["start"] + round((offset + count) * length / original),
                    )
                )
            position += count
        self.runs.clear()
        self.size = 0
        return pieces

    def drain(self):
        return merge_ranges([[a, b] for _, _, a, b in self.drain_mapped()])


@dataclass
class _SelectionOwner:
    channel: int
    ranges: list[list[int]] = field(default_factory=list)
    published: list[Selection] = field(default_factory=list)


@dataclass
class _Clip:
    timeline: ClipTimeline = field(default_factory=ClipTimeline)
    descriptor: dict | None = None


class Alignment:
    """Resolve recording positions into metadata updates, without SDK objects or I/O."""

    def __init__(self, recording_span_id: str):
        self.recording_span_id = recording_span_id
        self._owners: dict[str, _SelectionOwner] = {}
        self._clips: dict[str, _Clip] = {}
        self._dirty_clips: set[str] = set()
        self.count = 0
        self.omitted = 0
        self.pending_owners = set()

    def add(self, span_id: str, ranges: list[list[int]], channel: int) -> None:
        item = self._owners.setdefault(span_id, _SelectionOwner(channel))
        for interval in ranges:
            if item.ranges and item.ranges[-1][0] <= interval[0] <= item.ranges[-1][1]:
                item.ranges[-1][1] = max(item.ranges[-1][1], interval[1])
                continue
            if self.count >= MAX_PENDING_RANGES:
                self.omitted += 1
                continue
            item.ranges.append(list(interval))
            self.count += 1

        if item.ranges:
            self.pending_owners.add(span_id)

    def begin_clip(self, span_id: str) -> None:
        self._clips[span_id] = _Clip()

    def add_clip_range(
        self, span_id: str, start_ms: float, end_ms: float, call_start_ms: float, call_end_ms: float
    ) -> None:
        clip = self._clips.get(span_id)
        if clip is not None:
            clip.timeline.add(start_ms, end_ms, call_start_ms, call_end_ms)
            self._dirty_clips.add(span_id)

    def set_clip(self, span_id: str, descriptor: dict) -> None:
        clip = self._clips.get(span_id)
        if clip is not None:
            clip.descriptor = descriptor
            self._dirty_clips.add(span_id)

    def clip_updates(self, origin: float | None) -> Iterator[tuple[str, dict]]:
        if origin is None:
            return
        for span_id in tuple(self._dirty_clips):
            clip = self._clips[span_id]
            if clip.descriptor is None:
                continue
            timeline = clip.timeline.descriptor(origin)
            if timeline != clip.descriptor.get("timeline"):
                yield span_id, {"audio.recordings": [{**clip.descriptor, "timeline": timeline}]}
                # Iteration resumes only after the publisher logs successfully.
                clip.descriptor["timeline"] = timeline
            self._dirty_clips.discard(span_id)

    def selection_updates(self, segments: list[SegmentInterval]) -> Iterator[tuple[str, dict]]:
        resolved = merge_ranges(
            [
                [ms_to_samples(s["start_ms"]), ms_to_samples(s["end_ms"])]
                for s in segments
                if s["state"] in {"ready", "omitted"}
            ]
        )
        for span_id in tuple(self.pending_owners):
            state = self._owners[span_id]
            channel, ranges = state.channel, state.ranges
            selections = list(state.published)
            additions, pending = resolve_selections(ranges, segments, resolved, self.recording_span_id, channel)
            selections.extend(additions)
            selections = [dict(items) for items in dict.fromkeys(tuple(s.items()) for s in selections)]
            if selections and selections != state.published:
                yield (
                    span_id,
                    {
                        "audio.selections": selections,
                        "audio.selection": selections[0] if len(selections) == 1 else None,
                    },
                )
                state.published = selections
            # Keep unresolved publication work intact if logging raises at yield.
            self.count -= len(ranges) - len(pending)
            ranges[:] = pending
            if not pending:
                self.pending_owners.discard(span_id)
