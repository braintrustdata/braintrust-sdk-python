"""Serialize manifests and publish alignment updates through the SDK."""

from collections.abc import Iterable

from braintrust.logger import Span

from .alignment import Alignment
from .constants import MAX_OUTPUT_CONTEXTS
from .recording import CallRecording
from .segment import AudioSegment


def segment_descriptor(segment: AudioSegment, span_id: str, sources: list[dict]) -> dict:
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


class AlignmentPublisher:
    """Adapt observed span ownership to the pure alignment model."""

    def __init__(self, root: Span, recording: CallRecording):
        self.root = root
        self.recording = recording
        self.alignment = Alignment(root.span_id)
        self._spans: dict[str, Span] = {}
        self._contexts: dict[str, list[Span]] = {}

    def add(self, owner: Span, ranges: list[list[int]], channel: int) -> None:
        if not self.recording.reason:
            self._spans[owner.span_id] = owner
            self.alignment.add(owner.span_id, ranges, channel)

    def begin_output(self, context: str, owners: list[Span]) -> None:
        if len(self._contexts) >= MAX_OUTPUT_CONTEXTS:
            self.alignment.omitted += 1
            return
        self._contexts[context] = owners
        self.alignment.begin_clip(owners[0].span_id)

    def output_owners(self, context: str) -> list[Span]:
        return self._contexts.get(context, [])

    def end_output(self, context: str) -> None:
        self._contexts.pop(context, None)

    def add_clip_range(
        self, owner: Span, start_ms: float, end_ms: float, call_start_ms: float, call_end_ms: float
    ) -> None:
        self.alignment.add_clip_range(owner.span_id, start_ms, end_ms, call_start_ms, call_end_ms)

    def publish_clip(self, owner: Span, descriptor: dict) -> None:
        self._spans[owner.span_id] = owner
        self.alignment.set_clip(owner.span_id, descriptor)
        self.publish_clips()

    def _publish(self, updates: Iterable[tuple[str, dict]]) -> None:
        for span_id, metadata in updates:
            self._spans[span_id].log(metadata=metadata)

    def publish_clips(self) -> None:
        self._publish(self.alignment.clip_updates(getattr(self.recording, "origin_unix_ms", None)))

    def publish(self) -> None:
        self.publish_clips()
        segments = getattr(self.recording, "completed", None)
        if segments is None:
            segments = (
                [{"id": "call", "start_ms": 0, "end_ms": max(self.recording.ends), "state": "ready"}]
                if not self.recording.reason and self.recording.packets
                else []
            )
        self._publish(self.alignment.selection_updates(segments))
        self.root.log(metadata={"braintrust.alignment.ranges_omitted": self.alignment.omitted})
