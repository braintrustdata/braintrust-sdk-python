"""Compact source-file to call-clock mappings, without retaining audio."""

# Call recordings use a common 24 kHz sample clock regardless of input format.
CALL_SAMPLE_RATE = 24000


def ms_to_samples(milliseconds, sample_rate=CALL_SAMPLE_RATE):
    return round(milliseconds * sample_rate / 1000)


def samples_to_ms(samples, sample_rate=CALL_SAMPLE_RATE):
    return samples * 1000 / sample_rate


def pcm_bytes_to_ms(size, sample_rate, channels):
    """Duration of interleaved signed 16-bit PCM, in milliseconds."""
    return samples_to_ms(size / (2 * channels), sample_rate)


class ClipTimeline:
    def __init__(self):
        self.ranges = []

    def add(self, start_ms, end_ms, call_start_ms, call_end_ms):
        if end_ms <= start_ms:
            return
        if self.ranges:
            previous = self.ranges[-1]
            if (
                abs(previous["recording_end_ms"] - start_ms) < 0.000001
                and abs(previous["timeline_end_ms"] - call_start_ms) < 0.000001
            ):
                previous["recording_end_ms"] = end_ms
                previous["timeline_end_ms"] = call_end_ms
                return
        self.ranges.append(
            dict(
                recording_start_ms=start_ms,
                recording_end_ms=end_ms,
                timeline_start_ms=call_start_ms,
                timeline_end_ms=call_end_ms,
            )
        )

    def descriptor(self, origin_unix_ms):
        return {
            "origin_unix_ms": origin_unix_ms,
            "basis": "input_sample_clock_and_output_write_observation",
            "ranges": [dict(part) for part in self.ranges],
        }
