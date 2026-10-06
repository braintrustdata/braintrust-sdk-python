"""Bounded sample ranges and recording selections shared by voice integrations."""

from collections import deque
from dataclasses import dataclass, field

from braintrust.logger import Span

from .timeline import ClipTimeline


def merge_ranges(ranges):
    result = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


@dataclass
class _SelectionOwner:
    span: Span
    channel: int
    ranges: list[list[int]] = field(default_factory=list)
    published: list[dict] = field(default_factory=list)


@dataclass
class _Clip:
    timeline: ClipTimeline = field(default_factory=ClipTimeline)
    owner: Span | None = None
    descriptor: dict | None = None


class Alignment:
    def __init__(self, root, recording):
        self.root = root
        self.recording = recording
        self._owners: dict[str, _SelectionOwner] = {}
        self._contexts = {}
        self._clips: dict[str, _Clip] = {}
        self._dirty_clips: set[str] = set()
        self.count = 0
        self.omitted = 0
        self.pending_owners = set()

    def add(self, owner, ranges, channel):
        if self.recording.reason:
            return
        item = self._owners.setdefault(owner.span_id, _SelectionOwner(owner, channel))
        for interval in ranges:
            if item.ranges and item.ranges[-1][0] <= interval[0] <= item.ranges[-1][1]:
                item.ranges[-1][1] = max(item.ranges[-1][1], interval[1])
                continue
            if self.count >= 32000:
                self.omitted += 1
                continue
            item.ranges.append(list(interval))
            self.count += 1

        if item.ranges:
            self.pending_owners.add(owner.span_id)

    def begin_output(self, context, owners):
        if len(self._contexts) >= 16000:
            self.omitted += 1
            return
        self._contexts[context] = owners
        self._clips[owners[0].span_id] = _Clip()

    def output_owners(self, context):
        return self._contexts.get(context, [])

    def end_output(self, context):
        self._contexts.pop(context, None)

    def add_clip_range(self, owner, start_ms, end_ms, call_start_ms, call_end_ms):
        clip = self._clips.get(owner.span_id)
        if clip is not None:
            clip.timeline.add(start_ms, end_ms, call_start_ms, call_end_ms)
            self._dirty_clips.add(owner.span_id)

    def publish_clip(self, owner, descriptor):
        clip = self._clips.get(owner.span_id)
        if clip is not None:
            clip.owner, clip.descriptor = owner, descriptor
            self._dirty_clips.add(owner.span_id)
            self.publish_clips()

    def publish_clips(self):
        origin = getattr(self.recording, "origin_unix_ms", None)
        if origin is None:
            return
        for span_id in tuple(self._dirty_clips):
            clip = self._clips[span_id]
            if clip.descriptor is None:
                continue
            timeline = clip.timeline.descriptor(origin)
            if timeline != clip.descriptor.get("timeline"):
                clip.descriptor["timeline"] = timeline
                clip.owner.log(metadata={"audio.recordings": [dict(clip.descriptor)]})
            self._dirty_clips.discard(span_id)

    def publish(self):
        self.publish_clips()
        segments = getattr(self.recording, "completed", None)
        if segments is None:
            segments = (
                [{"id": "call", "start_ms": 0, "end_ms": max(self.recording.ends), "state": "ready"}]
                if not self.recording.reason and self.recording.packets
                else []
            )
        resolved = merge_ranges(
            [
                [round(s["start_ms"] * 24), round(s["end_ms"] * 24)]
                for s in segments
                if s["state"] in {"ready", "omitted"}
            ]
        )
        for span_id in tuple(self.pending_owners):
            state = self._owners[span_id]
            owner, channel, ranges = state.span, state.channel, state.ranges
            selections = list(state.published)
            merged = merge_ranges(ranges)
            pending = []
            for segment in segments:
                if segment["state"] != "ready":
                    continue
                lower, upper = round(segment["start_ms"] * 24), round(segment["end_ms"] * 24)
                for start, end in merged:
                    start, end = max(start, lower), min(end, upper)
                    if end > start:
                        selections.append(
                            {
                                "recording_span_id": self.root.span_id,
                                "recording_id": segment["id"],
                                "start_offset_ms": (start - lower) / 24,
                                "end_offset_ms": (end - lower) / 24,
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
            self.count -= len(ranges) - len(pending)
            ranges[:] = pending
            selections = [dict(items) for items in dict.fromkeys(tuple(s.items()) for s in selections)]
            if not pending:
                self.pending_owners.discard(span_id)
            if selections and selections != state.published:
                state.published = selections
                owner.log(
                    metadata={
                        "audio.selections": selections,
                        "audio.selection": selections[0] if len(selections) == 1 else None,
                    }
                )
        self.root.log(metadata={"braintrust.alignment.ranges_omitted": self.omitted})


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
