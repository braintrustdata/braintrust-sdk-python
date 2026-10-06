"""Pinned Pipecat 1.12 cascade hooks: STT segment -> accepted frames -> turn.

The native aggregator owns membership. No transcript or timestamp matching.
"""

import asyncio
import io
import time
import wave
from collections import deque

from braintrust.audio.alignment import InputRanges
from braintrust.audio.attachments import prepare_recording
from braintrust.audio.budget import source_budget
from braintrust.audio.recording import encode_audio
from braintrust.audio.worker import RecordingBusy, encode_in_worker
from pipecat.frames.frames import TranscriptionFrame  # pylint: disable=import-error

from ..llm_metrics import _metadata_from_processor


def encode_segments(segments, audio_format):
    chunks = []
    rate = channels = None
    for segment in segments:
        with wave.open(io.BytesIO(segment), "rb") as source:
            if source.getsampwidth() != 2:
                raise ValueError("Expected PCM16 STT input")
            current = source.getframerate(), source.getnchannels()
            if rate is not None and current != (rate, channels):
                raise ValueError("Mixed STT input formats")
            rate, channels = current
            chunks.append(source.readframes(source.getnframes()))
    return encode_audio(chunks, rate, channels, audio_format)


class UserCapture:
    def __init__(self, observer, stt, aggregator, native_value):
        self.observer = observer
        self.accepted = []
        self.by_frame = {}
        self.batches = {}
        self.bytes = 0
        self.boundaries = []
        self.segment_boundaries = deque(maxlen=256)
        self.omitted = 0
        self.input_ranges = InputRanges() if observer.capture_user_audio else None
        self.segments = []

        original_start = observer.hooks.original(stt, "_handle_user_started_speaking")
        original_stop = observer.hooks.original(stt, "_handle_user_stopped_speaking")
        original_run = observer.hooks.original(stt, "run_stt")
        original_accept = observer.hooks.original(aggregator, "_handle_transcription")
        original_commit = observer.hooks.original(aggregator, "_push_aggregation")
        original_reset = observer.hooks.original(aggregator, "reset")
        original_audio = observer.hooks.original(stt, "process_audio_frame")

        async def process_audio(frame, direction):
            try:
                interval = observer.call_recording.capture(0, frame.audio, frame.sample_rate, frame.num_channels)
            except Exception as error:  # noqa: BLE001 - recording must not stop input processing
                observer.call_recording.omit("capture_error")
                interval = None
            result = await original_audio(frame, direction)
            self.input_ranges.append(len(frame.audio), interval)
            # Mirror the actual retained native buffer length, not a guessed VAD time.
            self.input_ranges.trim(len(stt._audio_buffer))
            return result

        async def speech_start(frame):
            self.boundaries = [
                {
                    "contrib.pipecat.frame.type": type(frame).__name__,
                    "contrib.pipecat.frame": native_value(frame),
                }
            ]
            return await original_start(frame)

        async def speech_stop(frame):
            if stt.is_usable:
                self.segment_boundaries.append(
                    {
                        "ranges": self.input_ranges.drain() if self.input_ranges else [],
                        "events": [
                            *self.boundaries,
                            {
                                "contrib.pipecat.frame.type": type(frame).__name__,
                                "contrib.pipecat.frame": native_value(frame),
                            },
                        ],
                    }
                )
            elif self.input_ranges:
                self.input_ranges.drain()
            self.boundaries = []
            return await original_stop(frame)

        async def run(audio):
            queued = self.segment_boundaries.popleft() if self.segment_boundaries else {"events": [], "ranges": []}
            segment = {
                "service_metadata": _metadata_from_processor(stt),
                "boundaries": queued["events"],
                "ranges": queued["ranges"],
                "audio": None,
                "reason": "disabled",
                "start": time.time(),
                "end": None,
                "frames": [],
                "span": None,
            }
            tracked = len(self.segments) < 256
            if tracked:
                self.segments.append(segment)
            else:
                self.omitted += 1
            if observer.capture_user_audio and tracked:
                if self.bytes + len(audio) <= observer.max_audio_bytes and source_budget.reserve(len(audio)):
                    segment.update(audio=audio, reason=None)
                    self.bytes += len(audio)
                else:
                    segment["reason"] = (
                        "capture_byte_limit"
                        if self.bytes + len(audio) > observer.max_audio_bytes
                        else "process_capture_byte_limit"
                    )
            segment["ttfb_metadata"] = {}

            def log_ttfb(**event):
                segment["ttfb_metadata"].update(event["metadata"])
                if segment["span"] is not None:
                    segment["span"].log(**event)

            observer.ttfb.start(("stt", id(segment)), stt, log_ttfb)
            try:
                async for frame in original_run(audio):
                    try:
                        segment["end"] = time.time()
                        if isinstance(frame, TranscriptionFrame):
                            segment["frames"].append(native_value(frame))
                            if len(self.by_frame) < 256:
                                self.by_frame[frame.id] = segment
                            else:
                                self.omitted += 1
                        elif getattr(frame, "error", None):
                            segment["error"] = str(frame.error)
                    except Exception:  # noqa: BLE001 - always deliver the native STT frame
                        self.omitted += 1
                    yield frame
            except BaseException as error:
                segment["error"] = type(error).__name__
                raise
            finally:
                observer.ttfb.end(("stt", id(segment)))
                if segment["end"] is None:
                    segment["end"] = time.time()

        async def accept(frame):
            result = await original_accept(frame)
            if frame.text.strip():
                if len(self.accepted) < 256:
                    self.accepted.append((native_value(frame), self.by_frame.pop(frame.id, None)))
                else:
                    self.omitted += 1
            return result

        async def reset():
            self.accepted = []
            return await original_reset()

        async def commit(*args, **kwargs):
            # Snapshot before native reset/context push. The original method
            # can schedule turn-stop callbacks while it awaits downstream work.
            accepted = self.accepted[:]
            turn = observer.turns.user
            result = await original_commit(*args, **kwargs)
            if result and accepted:
                owner = turn["span"] if turn else observer.root
                segments = []
                for _, segment in accepted:
                    if segment is not None and not any(s is segment for s in segments):
                        segments.append(segment)
                        self.create_stt(segment, owner, turn)
                owner.log(
                    metadata={
                        "contrib.pipecat.transcriptions": [frame for frame, _ in accepted],
                        "contrib.pipecat.speech_events": [event for s in segments for event in s["boundaries"]],
                        "braintrust.user_capture.association": "aggregator_consumed_frames",
                    }
                )
                if turn and (turn["span"].span_id in self.batches or len(self.batches) < 256):
                    batch = self.batches.setdefault(
                        turn["span"].span_id,
                        {"turn": turn, "segments": [], "frames": []},
                    )
                    batch["frames"].extend(frame for frame, _ in accepted)
                    for segment in segments:
                        if not any(s is segment for s in batch["segments"]):
                            batch["segments"].append(segment)
                    owner.log(
                        metadata={
                            "contrib.pipecat.transcriptions": batch["frames"],
                            "contrib.pipecat.speech_events": [
                                event for s in batch["segments"] for event in s["boundaries"]
                            ],
                        }
                    )
                if turn:
                    self.queue_completed(turn)
            return result

        observer.hooks.set(stt, "_handle_user_started_speaking", speech_start)
        observer.hooks.set(stt, "_handle_user_stopped_speaking", speech_stop)
        observer.hooks.set(stt, "run_stt", run)
        if observer.capture_user_audio:
            observer.hooks.set(stt, "process_audio_frame", process_audio)
        observer.hooks.set(aggregator, "_handle_transcription", accept)
        observer.hooks.set(aggregator, "_push_aggregation", commit)
        observer.hooks.set(aggregator, "reset", reset)

    def __del__(self):
        retained = getattr(self, "bytes", 0)
        if retained:
            source_budget.release(retained)
            self.bytes = 0

    def create_stt(self, segment, owner, turn=None):
        if segment["span"] is not None:
            return
        metadata = {
            "contrib.pipecat.transcriptions": segment["frames"],
            "contrib.pipecat.function": "run_stt",
            **segment.get("service_metadata", {}),
            **segment.get("ttfb_metadata", {}),
        }
        if turn:
            metadata.update(self.observer.turns.metadata(turn))
        span = owner.start_span(
            name="stt",
            type="task",
            start_time=segment["start"],
            set_current=False,
            internal={"instrumentation": "pipecat-auto"},
            metadata=metadata,
        )
        text = " ".join(f["text"] for f in segment["frames"])
        span.log(output=[{"role": "user", "content": text}])
        if segment.get("error"):
            span.log(error=segment["error"])
        span.end(end_time=segment["end"] or time.time())
        segment["span"] = span
        self.observer.alignment.add(span, segment["ranges"], 0)
        if turn:
            self.observer.alignment.add(owner, segment["ranges"], 0)
            span.log(input={"recording_span_id": owner.span_id, "recording_id": "user-clip"})

    async def finish(self):
        for segment in self.segments:
            self.create_stt(segment, self.observer.root)
        for batch in list(self.batches.values()):
            self.queue_completed(batch["turn"], force=True)
        if getattr(self, "tasks", None):
            await asyncio.gather(*tuple(self.tasks))
        self.observer.root.log(metadata={"braintrust.user_capture.events_omitted": self.omitted})
        self.release()

    def queue_completed(self, turn, force=False):
        batch = self.batches.get(turn["span"].span_id)
        if not batch or batch.get("queued") or (not force and not turn.get("ended")):
            return
        batch["queued"] = True
        if not hasattr(self, "tasks"):
            self.tasks = set()
        task = asyncio.create_task(self._publish_batch(batch))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _publish_batch(self, batch):
        turn, segments = batch["turn"], batch["segments"]
        descriptor = {"id": "user-clip", "sources": [{"boundary": "stt_input"}]}
        reason = next((s["reason"] for s in segments if s["reason"]), None)
        audio = [s["audio"] for s in segments if s["audio"] is not None]
        if reason or not audio:
            descriptor.update(state="omitted", reason=reason or "no_audio_observed")
        else:
            try:
                encoded = await encode_in_worker(
                    prepare_recording, "user-clip", encode_segments, audio, self.observer.audio_format
                )
                descriptor.update(
                    state="ready",
                    attachment={
                        "span_id": turn["span"].span_id,
                        "ref": "/input/0/content/1/file/file_data",
                    },
                    mime_type=encoded["mime_type"],
                    duration_ms=encoded["duration_ms"],
                    channel_count=encoded["channel_count"],
                )
                attachment = encoded["attachment"]
                turn["span"].log(
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": turn.get("content") or ""},
                                {
                                    "type": "file",
                                    "file": {
                                        "file_data": attachment,
                                        "filename": f"user-clip.{encoded['extension']}",
                                    },
                                },
                            ],
                        }
                    ]
                )
            except Exception as error:  # noqa: BLE001 - attachment failure must not break flush
                descriptor.update(
                    state="omitted",
                    reason=str(error) if isinstance(error, RecordingBusy) else "encoding_failed",
                )
        turn["span"].log(metadata={"audio.recordings": [descriptor]})

        for segment in segments:
            if segment["audio"] is not None:
                size = len(segment["audio"])
                segment["audio"] = None
                self.bytes -= size
                source_budget.release(size)
        await asyncio.to_thread(self.observer.logger.flush)

    def release(self):
        self.batches.clear()
        self.by_frame.clear()
        self.accepted.clear()
        self.segments.clear()
        source_budget.release(self.bytes)
        self.bytes = 0
