"""Internal data contracts shared by capture, export, and alignment."""

from typing import Literal, TypedDict

from braintrust.logger import Attachment
from typing_extensions import NotRequired


RecordingState = Literal["pending", "ready", "omitted"]


class EncodedAudio(TypedDict):
    data: bytes
    duration_ms: float
    channel_count: int
    mime_type: str
    extension: str
    processing: NotRequired[dict[str, float | int]]
    attachment: NotRequired[Attachment]


class SegmentInterval(TypedDict):
    id: str
    start_ms: float
    end_ms: float
    state: RecordingState


class Selection(TypedDict):
    recording_span_id: str
    recording_id: str
    start_offset_ms: float
    end_offset_ms: float
    channel_index: int
