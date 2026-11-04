from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Audio attributes must contain JSON values")


def thaw(value):
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class AudioChannel:
    sources: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True)
class SegmentMeta:
    index: int
    start_sample: int
    sample_count: int
    sample_rate_hz: int
    channels: tuple[AudioChannel, ...]
    mime_type: str = "audio/pcm"
    encoding: str = "pcm"
    attributes: Mapping[str, Any] = field(
        default_factory=lambda: {"pcm.sample_format": "s16le", "pcm.channel_layout": "interleaved"}
    )

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "attributes", freeze(self.attributes))
        if self.index < 0 or self.start_sample < 0 or self.sample_count <= 0 or self.sample_rate_hz <= 0:
            raise ValueError("Invalid audio segment placement")
        if not self.channels or not self.mime_type or not self.encoding:
            raise ValueError("Invalid audio segment format")

    def placement(self):
        start = round(self.start_sample * 1000 / self.sample_rate_hz)
        end = round((self.start_sample + self.sample_count) * 1000 / self.sample_rate_hz)
        return {"start_offset_ms": start, "duration_ms": end - start}


@dataclass(frozen=True)
class AudioResult:
    data: bytes
    meta: SegmentMeta


@dataclass(frozen=True)
class RecordingResult:
    segments: int
    reason: str | None = None
