"""Public recording configuration; no codec or framework dependencies."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RecordingOptions:
    """Per-call limits. A rotation threshold is not a total recording limit."""

    segment_duration_seconds: float = 60
    max_duration_seconds: float = 1800
    max_buffer_bytes: int = 32 * 1024 * 1024
    flush_fraction: float = 0.5

    def __post_init__(self):
        for value in (self.segment_duration_seconds, self.max_duration_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("recording durations must be positive and finite")
        if self.max_buffer_bytes <= 0 or not 0 < self.flush_fraction < 1:
            raise ValueError("recording buffer must be positive and flush_fraction between zero and one")
