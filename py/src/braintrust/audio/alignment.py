"""Bounded sample ranges and recording selections shared by voice integrations."""

from collections import deque


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


class Alignment:
    def __init__(self, root, recording):
        self.root = root
        self.recording = recording
        self.owners = {}
        self.contexts = {}
        self.clips = {}
        self.clip_descriptors = {}
        self.count = 0
        self.omitted = 0
        self.published = {}
        self.pending_owners = set()

    def add(self, owner, ranges, channel):
        if self.recording.reason:
            return
        item = self.owners.setdefault(owner.span_id, (owner, channel, []))
        for interval in ranges:
            if item[2] and item[2][-1][0] <= interval[0] <= item[2][-1][1]:
                item[2][-1][1] = max(item[2][-1][1], interval[1])
                continue
            if self.count >= 32000:
                self.omitted += 1
                continue
            item[2].append(list(interval))
            self.count += 1

        if item[2]:
            self.pending_owners.add(owner.span_id)

    def publish_clip(self, owner, descriptor):
        self.clip_descriptors[owner.span_id] = (owner, descriptor)
        self.publish_clips()

    def publish_clips(self):
        origin = getattr(self.recording, "origin_unix_ms", None)
        if origin is None:
            return
        for span_id, (owner, descriptor) in self.clip_descriptors.items():
            clip = self.clips.get(span_id)
            if clip is not None:
                timeline = clip.descriptor(origin)
                if timeline != descriptor.get("timeline"):
                    descriptor["timeline"] = timeline
                    owner.log(metadata={"audio.recordings": [dict(descriptor)]})

    def publish(self):
        self.publish_clips()
        segments = getattr(self.recording, "completed", None)
        if segments is None:
            segments = (
                [{"id": "call", "start_ms": 0, "end_ms": max(self.recording.ends), "state": "ready"}]
                if not self.recording.reason and self.recording.packets
                else []
            )
        for span_id in tuple(self.pending_owners):
            owner, channel, ranges = self.owners[span_id]
            selections = list(self.published.get(owner.span_id, []))
            pending = []
            for segment in segments:
                if segment["state"] != "ready":
                    continue
                lower, upper = round(segment["start_ms"] * 24), round(segment["end_ms"] * 24)
                for start, end in merge_ranges(ranges):
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
            resolved = merge_ranges(
                [
                    [round(s["start_ms"] * 24), round(s["end_ms"] * 24)]
                    for s in segments
                    if s["state"] in {"ready", "omitted"}
                ]
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
            if selections and selections != self.published.get(span_id):
                self.published[span_id] = selections
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
