"""Generated speech capture and publication, independent of frame dispatch."""

import asyncio
from dataclasses import dataclass, field

from braintrust._audio import (
    RecordingBusy,
    RecordingJobs,
    UploadFailed,
    encode_audio,
    encode_in_worker,
    prepare_recording,
    source_budget,
    upload_recording,
)
from braintrust.logger import Span


@dataclass
class Synthesis:
    span: Span
    chunks: list[bytes] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    omitted: bool = False
    reason: str | None = None
    rate: int | None = None
    channels: int | None = None
    observed_format: tuple[int, int] | None = None
    last_pts: int | None = None


class SynthesisRecordings:
    def __init__(self, logger, alignment, *, enabled, max_bytes, audio_format):
        self.logger = logger
        self.alignment = alignment
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.audio_format = audio_format
        self.bytes = 0
        self.omitted = 0
        self.jobs = RecordingJobs()

    def __del__(self):
        retained = getattr(self, "bytes", 0)
        if retained:
            source_budget.release(retained)
            self.bytes = 0

    async def drain(self):
        await self.jobs.drain()

    def capture(self, state, frame):
        audio_format = (frame.sample_rate, frame.num_channels)
        state.last_pts = frame.pts
        if state.observed_format != audio_format:
            state.observed_format = audio_format
            state.span.log(
                metadata={
                    "contrib.pipecat.sample_rate": frame.sample_rate,
                    "contrib.pipecat.num_channels": frame.num_channels,
                    "contrib.pipecat.frame.pts": frame.pts,
                }
            )
        if state.chunks and (state.rate, state.channels) != audio_format:
            state.omitted = True
            state.reason = "audio_format_changed"
            self.release(state)
        if self.enabled and not state.omitted:
            if self.bytes + len(frame.audio) <= self.max_bytes and source_budget.reserve(len(frame.audio)):
                state.chunks.append(frame.audio)
                self.bytes += len(frame.audio)
                state.rate, state.channels = (
                    frame.sample_rate,
                    frame.num_channels,
                )
            else:
                state.omitted = True
                state.reason = (
                    "capture_byte_limit"
                    if self.bytes + len(frame.audio) > self.max_bytes
                    else "process_capture_byte_limit"
                )
                self.release(state)

    def complete(self, state):
        if state.observed_format is not None:
            state.span.log(metadata={"contrib.pipecat.frame.pts": state.last_pts})
        if len(self.jobs) < 256:
            self.jobs.submit(
                lambda: self._publish(state),
                lambda: self.release(state),
                after_release=lambda: asyncio.to_thread(self.logger.flush),
            )
            if state.chunks and not state.omitted:
                state.span.log(
                    metadata={
                        "audio.recordings": [
                            {"id": "tts-clip", "state": "pending", "sources": [{"boundary": "tts_output"}]}
                        ]
                    }
                )
            return
        self.omitted += 1
        state.span.log(
            metadata={
                "audio.recordings": [
                    {
                        "id": "tts-clip",
                        "state": "omitted",
                        "reason": "recording_count_limit",
                        "sources": [{"boundary": "tts_output"}],
                    }
                ]
            }
        )
        self.release(state)

    def release(self, state):
        size = sum(map(len, state.chunks))
        state.chunks.clear()
        source_budget.release(size)
        self.bytes -= size

    async def _publish(self, state):
        span = state.span
        recording = {"id": "tts-clip", "sources": [{"boundary": "tts_output"}]}
        if state.chunks and not state.omitted:
            try:
                encoded = await encode_in_worker(
                    prepare_recording,
                    "tts-clip",
                    encode_audio,
                    state.chunks,
                    state.rate,
                    state.channels,
                    self.audio_format,
                )
                await upload_recording(encoded)
            except Exception as error:  # noqa: BLE001 - capture/export failures must not break the call
                span.log(
                    metadata={
                        "audio.recordings": [
                            {
                                **recording,
                                "state": "omitted",
                                "reason": str(error)
                                if isinstance(error, (RecordingBusy, UploadFailed))
                                else "encoding_failed",
                            }
                        ]
                    }
                )
                return
            recording.update(
                state="ready",
                attachment={
                    "span_id": span.span_id,
                    "ref": "/output/0/content/1/file/file_data",
                },
                mime_type=encoded["mime_type"],
                duration_ms=encoded["duration_ms"],
                channel_count=encoded["channel_count"],
            )
            span.log(
                output=[
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "".join(state.text)},
                            {
                                "type": "file",
                                "file": {
                                    "file_data": encoded["attachment"],
                                    "filename": f"tts-clip.{encoded['extension']}",
                                },
                            },
                        ],
                    }
                ],
                metadata={"braintrust.recording.processing": encoded["processing"]},
            )
        else:
            recording.update(
                state="omitted",
                reason=(state.reason or "no_audio_observed") if self.enabled else "disabled",
            )
        span.log(metadata={"audio.recordings": [recording]})
        if recording["state"] == "ready":
            self.alignment.publish_clip(span, recording)
