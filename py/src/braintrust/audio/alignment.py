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
        self.count = 0
        self.omitted = 0

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

    def publish(self):
        segments = getattr(self.recording, "completed", None)
        if segments is None:
            segments = (
                [{"id": "call", "start_ms": 0, "end_ms": max(self.recording.ends), "state": "ready"}]
                if not self.recording.reason and self.recording.packets
                else []
            )
        for owner, channel, ranges in self.owners.values():
            selections = []
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
            if selections:
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

    def drain(self):
        ranges = []
        for count, interval, original, offset in self.runs:
            if interval:
                length = interval["end"] - interval["start"]
                ranges.append(
                    [
                        interval["start"] + round(offset * length / original),
                        interval["start"] + round((offset + count) * length / original),
                    ]
                )
        self.runs.clear()
        self.size = 0
        return merge_ranges(ranges)
