"""Lightweight, framework-independent audio recording configuration.

Configuration never enables capture. Integrations require their dedicated audio
opt-ins and enforce their framework's privacy policy independently.
"""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class AudioEncoding:
    codec: Literal["opus", "pcm"] = "opus"
    container: Literal["ogg", "wav"] | None = "ogg"
    bitrate_bps: int = 64000
    complexity: int = 10
    application: Literal["audio", "voip", "lowdelay"] = "audio"

    def __post_init__(self):
        if (self.codec, self.container) not in (("opus", "ogg"), ("pcm", None), ("pcm", "wav")):
            raise ValueError("Supported audio formats are opus/ogg, pcm, and pcm/wav")
        if (
            type(self.bitrate_bps) is not int
            or type(self.complexity) is not int
            or not 500 <= self.bitrate_bps <= 512000
            or not 0 <= self.complexity <= 10
        ):
            raise ValueError("Invalid Opus bitrate or complexity")
        if self.application not in ("audio", "voip", "lowdelay"):
            raise ValueError("Invalid Opus application")
        if self.codec == "pcm" and (self.bitrate_bps, self.complexity, self.application) != (64000, 10, "audio"):
            raise ValueError("Opus options cannot be used with PCM")


@dataclass(frozen=True, kw_only=True)
class AudioRecordingOptions:
    segment_duration_seconds: float = 60
    flush_threshold_bytes: int = 32 * 1024 * 1024
    encoding: AudioEncoding | None = None
    max_duration_seconds: float | None = None
    shutdown_timeout_seconds: float = 30

    def __post_init__(self):
        for value in (self.segment_duration_seconds, self.shutdown_timeout_seconds, self.max_duration_seconds):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Audio durations must be positive and finite")
        if self.segment_duration_seconds < 1 / 48000:
            raise ValueError("An audio segment must contain at least one sample")
        if type(self.flush_threshold_bytes) is not int or self.flush_threshold_bytes <= 0:
            raise ValueError("Audio flush threshold must be positive")
        if self.encoding is not None and not isinstance(self.encoding, AudioEncoding):
            raise TypeError("Audio encoding must be an AudioEncoding instance")
