"""Serialize recording manifests without retaining their audio payloads."""

from dataclasses import dataclass, field

from braintrust.logger import Span

from .alignment import merge_ranges, resolve_selections
from .timeline import ClipTimeline, ms_to_samples


def segment_descriptor(segment, span_id, sources):
    descriptor = {
        "id": segment.segment_id,
        "recording_group_id": "call",
        "state": segment.state,
        "sources": sources,
        "timeline": {
            "origin_unix_ms": segment.origin_unix_ms,
            "recording_start_offset_ms": segment.start_ms,
            "basis": "input_sample_clock_and_output_write_observation",
        },
    }
    if segment.state == "ready":
        descriptor.update(segment.metadata)
        descriptor["attachment"] = {"span_id": span_id, "ref": f"/input/audio/{segment.segment_id}"}
    elif segment.state == "omitted":
        descriptor["reason"] = segment.reason
        descriptor["gaps"] = [
            {
                "start_offset_ms": segment.start_ms,
                "end_offset_ms": segment.end_ms,
                "reason": segment.reason,
            }
        ]
    return descriptor


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
    """Publish resolved selections and clip timelines to their owning spans."""

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
                [ms_to_samples(s["start_ms"]), ms_to_samples(s["end_ms"])]
                for s in segments
                if s["state"] in {"ready", "omitted"}
            ]
        )
        for span_id in tuple(self.pending_owners):
            state = self._owners[span_id]
            owner, channel, ranges = state.span, state.channel, state.ranges
            selections = list(state.published)
            additions, pending = resolve_selections(ranges, segments, resolved, self.root.span_id, channel)
            selections.extend(additions)
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
