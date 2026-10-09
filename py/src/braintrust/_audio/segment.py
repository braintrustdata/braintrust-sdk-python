"""Authoritative lifecycle manifest for one sealed recording segment."""

from dataclasses import dataclass, field

from .recording import CallRecording
from .types import EncodedAudio, RecordingState, SegmentInterval


@dataclass
class AudioSegment:
    """One manifest throughout encoding and export; PCM is sealed at rotation.

    Only the worker releases submitted PCM. Encoded bytes live through export;
    afterwards only small metadata remains on the call's manifest.
    """

    segment_id: str
    start_ms: float
    end_ms: float
    origin_unix_ms: float
    source: CallRecording
    state: RecordingState = "pending"
    reason: str | None = None
    encoded: EncodedAudio | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def bytes(self) -> int:
        return self.source.bytes

    def clear(self) -> None:
        self.source.clear()

    def ready(self, encoded: EncodedAudio) -> None:
        self.encoded = encoded
        self.metadata = {key: encoded[key] for key in ("mime_type", "duration_ms", "channel_count")}
        self.state = "ready"

    def omit(self, reason: str) -> None:
        self.state, self.reason = "omitted", reason
        self.encoded = None
        self.metadata.clear()

    def interval(self) -> SegmentInterval:
        return {"id": self.segment_id, "start_ms": self.start_ms, "end_ms": self.end_ms, "state": self.state}
