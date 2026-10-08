"""Bounded sample ranges and recording selections shared by voice integrations."""

from collections import deque

from .timeline import ms_to_samples, samples_to_ms


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


def resolve_selections(ranges, segments, resolved, recording_span_id, channel):
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
