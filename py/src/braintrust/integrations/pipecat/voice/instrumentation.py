"""Experimental native Pipecat observer and opt-in recording support."""

import asyncio
import dataclasses
import io
import json
import time
import wave
from collections import Counter, deque

from braintrust.audio.alignment import Alignment
from braintrust.audio.attachments import prepare_recording
from braintrust.audio.budget import source_budget
from braintrust.audio.recording import encode_audio
from braintrust.audio.segments import SegmentedRecording
from braintrust.audio.worker import RecordingBusy, encode_in_worker
from pipecat.observers.base_observer import BaseObserver  # pylint: disable=import-error

from ..ttfb import TTFBRouter
from ..turn_metrics import TURN_METRIC_TYPES, log_turn_metric
from .turns import Turns


# High-frequency payloads are handled by their owning operation or recorder.
# Only selected diagnostic/lifecycle facts become events.
EVENT_TYPES = {
    "StartFrame",
    "EndFrame",
    "CancelFrame",
    "ClientConnectedFrame",
    "OutputTransportReadyFrame",
    "BotStartedSpeakingFrame",
    "BotStoppedSpeakingFrame",
    "UserStartedSpeakingFrame",
    "UserStoppedSpeakingFrame",
    "VADUserStartedSpeakingFrame",
    "VADUserStoppedSpeakingFrame",
    "InterruptionFrame",
    "UserTurnInferenceCompletedFrame",
    "EagerEndOfTurnCancelFrame",
    "UserMuteStartedFrame",
    "UserMuteStoppedFrame",
    "STTMuteFrame",
    "ErrorFrame",
    "FatalErrorFrame",
    "MetricsFrame",
    "SpeechControlParamsFrame",
    "STTMetadataFrame",
    "LLMServiceMetadataFrame",
    "LLMUpdateSettingsFrame",
    "TTSUpdateSettingsFrame",
    "STTUpdateSettingsFrame",
    "VADParamsUpdateFrame",
}
OPERATION_TYPES = {
    "LLMContextFrame",
    "LLMFullResponseStartFrame",
    "LLMFullResponseEndFrame",
    "LLMTextFrame",
    "TranscriptionFrame",
    "FunctionCallsStartedFrame",
    "FunctionCallInProgressFrame",
    "FunctionCallResultFrame",
    "FunctionCallCancelFrame",
    "TTSStartedFrame",
    "TTSStoppedFrame",
    "TTSTextFrame",
}


def native_value(value):
    if isinstance(value, bytes):
        return None
    if dataclasses.is_dataclass(value):
        return {
            field.name: native_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name not in {"audio", "image"} and not field.name.startswith("_")
        }
    if hasattr(value, "model_dump"):
        return native_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): native_value(item) for key, item in value.items() if key not in {"audio", "image"}}
    if isinstance(value, (tuple, list)):
        return [native_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def encode_wav(chunks, sample_rate, channels):
    output = io.BytesIO()
    with wave.Wave_write(output) as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for chunk in chunks:
            wav.writeframesraw(chunk)
    return output.getvalue()


class NativeObserver(BaseObserver):
    def __init__(
        self,
        logger,
        *,
        retain_audio=False,
        max_audio_bytes=8 * 1024 * 1024,
        audio_format="ogg",
        capture_user_audio=None,
        capture_agent_audio=None,
        root=None,
        recording_options=None,
    ):
        if audio_format not in {"ogg", "wav"}:
            raise ValueError("AUDIO_FORMAT must be ogg or wav")
        super().__init__(observe_every_push=False)
        from .hooks import Hooks

        self.hooks = Hooks()
        self.capture_user_audio = retain_audio if capture_user_audio is None else capture_user_audio
        self.capture_agent_audio = retain_audio if capture_agent_audio is None else capture_agent_audio
        retain_audio = self.capture_user_audio or self.capture_agent_audio
        self.logger = logger
        self.audio_format = audio_format
        self.call_recording = SegmentedRecording(
            enabled=retain_audio,
            options=recording_options,
            audio_format=audio_format,
            on_segment=self._publish_call_segment,
        )
        self.call_descriptors = {}
        self.clip_tasks = set()
        self.frame_contexts = {}
        self.realtime = None
        self.input_processor = None
        self.capture_transport = False
        self.root = (
            root if root is not None else logger.start_span(name="pipecat.pipeline", type="task", set_current=False)
        )
        self.ttfb = TTFBRouter(self.root.log)
        self.turns = Turns(self.root, self.hooks)
        self.turns.on_completed = self._turn_completed
        self.alignment = Alignment(self.root, self.call_recording)
        self.user_capture = None
        self.llm_turn = None
        self.user_aggregator = None
        self.assistant_aggregator = None
        self.context_reply_to = None
        self.context_tool_results = []
        self.result_origins = {}
        self.event_groups = {}
        self.filtered_events = Counter()
        self.duplicate_events = 0
        self.clock_anchored = False
        self.llm = None
        self.llm_text = []
        self.user = None
        self.tools = {}
        self.requests = {}
        self.tts = {}
        self.events = []
        self.omitted_events = 0
        self.event_bytes = 0
        self.retain_audio = retain_audio
        self.max_audio_bytes = max_audio_bytes
        self.audio_bytes = 0
        self.recordings = []
        self.recordings_omitted = 0
        self.finished = False
        self._finish_task = None
        self.tool_names = []
        self.started_tools = set()
        self.last_context = None
        self.seen = set()
        self.seen_order = deque()

    def __del__(self):
        retained = getattr(self, "audio_bytes", 0)
        if retained:
            source_budget.release(retained)
            self.audio_bytes = 0

    def bind(self, *, transport, user_aggregator, assistant_aggregator, stt=None, realtime_service=None):
        """Bind one supported pipeline before it starts; does not alter routing."""
        import importlib.metadata

        if importlib.metadata.version("pipecat-ai") != "1.12.0":
            raise ValueError("Experimental voice hooks currently support Pipecat 1.12.0")
        if getattr(self, "_bound", False):
            raise ValueError("Voice observer is already bound to a pipeline")
        if (stt is None) == (realtime_service is None):
            raise ValueError("Bind either segmented STT or an OpenAI realtime service")
        self.turns.install(user_aggregator, assistant_aggregator)
        self.user_aggregator, self.assistant_aggregator = user_aggregator, assistant_aggregator
        self.capture_transport = True
        if stt is not None:
            from .user_capture import UserCapture

            self.input_processor = transport.input()
            self.user_capture = UserCapture(self, stt, user_aggregator, native_value)
        else:
            from .realtime import RealtimeCapture

            self.realtime = RealtimeCapture(self, realtime_service, user_aggregator)
        if self.capture_agent_audio:
            from .alignment import instrument_output

            instrument_output(transport.output(), self.alignment, self.frame_context, self.hooks)
        self._bound = True
        return self

    def frame_context(self, frame):
        return self.frame_contexts.get(frame.id, getattr(frame, "context_id", None))

    async def on_push_frame(self, data):
        if not data.first_push or self.finished:
            return
        frame = data.frame
        sibling = getattr(frame, "broadcast_sibling_id", None)
        if frame.id in self.seen or (sibling is not None and sibling in self.seen):
            self.duplicate_events += 1
            return
        if len(self.seen_order) >= 4096:
            self.seen.discard(self.seen_order.popleft())
        self.seen.add(frame.id)
        self.seen_order.append(frame.id)
        if (
            self.capture_user_audio
            and not self.user_capture
            and data.source is self.input_processor
            and type(frame).__name__
            in {
                "InputAudioRawFrame",
                "UserAudioRawFrame",
            }
        ):
            try:
                self.call_recording.capture(0, frame.audio, frame.sample_rate, frame.num_channels)
            except Exception:  # noqa: BLE001 - capture/export failures must not break the call
                self.call_recording.omit("capture_error")
        kind = type(frame).__name__
        # Audio data stays out of metadata; retain its format on the owning TTS span.
        if kind == "TTSAudioRawFrame":
            context = self.frame_context(frame)
            state = self.tts.get(context)
            if state:
                audio_format = (frame.sample_rate, frame.num_channels)
                state["last_pts"] = frame.pts
                if state.get("observed_format") != audio_format:
                    state["observed_format"] = audio_format
                    state["span"].log(
                        metadata={
                            "pipecat.sample_rate": frame.sample_rate,
                            "pipecat.num_channels": frame.num_channels,
                            "pipecat.frame.pts": frame.pts,
                        }
                    )
                if state["chunks"] and (state["rate"], state["channels"]) != audio_format:
                    state["omitted"] = True
                    state["reason"] = "audio_format_changed"
                    size = sum(map(len, state["chunks"]))
                    source_budget.release(size)
                    self.audio_bytes -= size
                    state["chunks"].clear()
                if self.capture_agent_audio and not state["omitted"]:
                    if self.audio_bytes + len(frame.audio) <= self.max_audio_bytes and source_budget.reserve(
                        len(frame.audio)
                    ):
                        state["chunks"].append(frame.audio)
                        self.audio_bytes += len(frame.audio)
                        state["rate"], state["channels"] = (
                            frame.sample_rate,
                            frame.num_channels,
                        )
                    else:
                        state["omitted"] = True
                        state["reason"] = (
                            "capture_byte_limit"
                            if self.audio_bytes + len(frame.audio) > self.max_audio_bytes
                            else "process_capture_byte_limit"
                        )
                        source_budget.release(sum(map(len, state["chunks"])))
                        self.audio_bytes -= sum(map(len, state["chunks"]))
                        state["chunks"].clear()
            return
        if "AudioRawFrame" in kind:
            return
        if (self.user_capture or self.realtime) and kind in {
            "VADUserStartedSpeakingFrame",
            "VADUserStoppedSpeakingFrame",
            "TranscriptionFrame",
        }:
            # These facts are attributed through the actual STT/aggregator path.
            return
        if kind not in EVENT_TYPES and kind not in OPERATION_TYPES:
            self.filtered_events[kind] += 1
            return
        fields = native_value(frame) if kind != "LLMContextFrame" else {}
        event_owner = self.root
        if kind == "MetricsFrame":
            remaining = []
            for metric in frame.data:
                if type(metric).__name__ == "TTFBMetricsData":
                    self.ttfb.capture(metric, data.source)
                else:
                    remaining.append(native_value(metric))
            if not remaining:
                return
            fields["data"] = remaining
        if kind == "MetricsFrame" and data.source is self.user_aggregator and self.turns.user:
            # The single discovered user aggregator emits its analyzer predictions
            # before stopping this turn. Other sources retain unassociated events.
            remaining = []
            for metric in frame.data:
                if type(metric).__name__ in TURN_METRIC_TYPES:
                    state = self.turns.user
                    # Turns assigns role dictionaries through setattr.
                    log_turn_metric(state["span"], state, metric)  # pylint: disable=unsubscriptable-object
                elif type(metric).__name__ != "TTFBMetricsData":
                    remaining.append(native_value(metric))
            if not remaining:
                return
            fields["data"] = remaining
        if kind == "StartFrame":
            self.root.log(
                metadata={
                    f"pipecat.{name}": getattr(frame, name)
                    for name in (
                        "audio_in_sample_rate",
                        "audio_out_sample_rate",
                        "enable_metrics",
                        "enable_usage_metrics",
                    )
                }
            )
        elif kind == "LLMContextFrame":
            self.last_context = native_value(frame.context.get_messages())
            self.context_tool_results = []
            self.context_reply_to = None
            if data.source is self.user_aggregator and self.turns.last_user:
                self.context_reply_to = self.turns.last_user["span"].span_id
            elif data.source is self.assistant_aggregator:
                # Only newly observed tool results establish a continuation.
                # Old tool messages remaining in context do not create links.
                ids = {
                    m.get("tool_call_id") for m in self.last_context if isinstance(m, dict) and m.get("role") == "tool"
                }
                self.context_tool_results = sorted(ids & self.result_origins.keys())
                origins = {self.result_origins.pop(i) for i in self.context_tool_results}
                if len(origins) == 1:
                    self.context_reply_to = origins.pop()
        elif kind in {"UserStartedSpeakingFrame", "VADUserStartedSpeakingFrame"}:
            if kind == "UserStartedSpeakingFrame":
                self.turns.start("user", kind)
            elif not self.user:
                self.user = self.root.start_span(
                    name="pipecat.user_speaking",
                    type="task",
                    set_current=False,
                    internal={"instrumentation": "pipecat-auto"},
                    metadata={"pipecat.start_frame": kind},
                )
                event_owner = self.user
        elif kind in {"UserStoppedSpeakingFrame", "VADUserStoppedSpeakingFrame"}:
            if kind == "VADUserStoppedSpeakingFrame" and self.user:
                event_owner = self.user
                self.user.log(metadata={"pipecat.end_frame": kind})
                self.user.end()
                self.user = None
        elif kind == "TranscriptionFrame":
            # STT can trigger the next user turn rather than belong to an open
            # one. The aggregator supplies authoritative turn text separately.
            span = self.root.start_span(
                name="pipecat.stt_transcription",
                type="task",
                set_current=False,
                internal={"instrumentation": "pipecat-auto"},
                metadata={f"pipecat.{key}": value for key, value in fields.items()},
            )
            span.log(input={"text": frame.text})
            span.end()
        elif kind == "LLMFullResponseStartFrame":
            if self.realtime and self.llm is not None:
                # Pipecat brackets both response creation and item arrival.
                # One still-open response must not create another turn.
                self.filtered_events["repeated_response_start"] += 1
                return
            if self.realtime:
                turn = self.realtime.user_turns.get(self.realtime.user_item)
                self.context_reply_to = turn["span"].span_id if turn else None
                self.context_tool_results = self.realtime.pending_tool_results[:]
                self.realtime.pending_tool_results.clear()
                self.result_origins.clear()
            self.llm_text = []
            # A new model operation may arrive before the previous aggregator
            # callback task finishes. Keep each native response lifecycle distinct.
            self.turns.assistant = None
            self.llm_turn = self.turns.start("assistant", kind, self.context_reply_to)
            self.llm = self.llm_turn["span"].start_span(
                name="pipecat.llm_response",
                type="llm",
                input=self.last_context,
                metadata={
                    **self.turns.metadata(self.llm_turn),
                    "braintrust.continuation.tool_call_ids": self.context_tool_results,
                },
                set_current=False,
                internal={"instrumentation": "pipecat-auto"},
            )
            self.ttfb.start("llm", data.source, self.llm.log)
        elif kind == "LLMTextFrame" and self.llm:
            self.llm_text.append(frame.text)
        elif kind == "FunctionCallsStartedFrame":
            if self.llm_turn:
                self.llm_turn["tool_calls"] = [
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.function_name,
                            "arguments": json.dumps(native_value(call.arguments)),
                        },
                    }
                    for call in frame.function_calls
                ]
                self.turns.log_message(self.llm_turn, "")
            for call in frame.function_calls:
                self.requests.setdefault(
                    call.tool_call_id,
                    (
                        self.llm or self.root,
                        self.turns.metadata(self.llm_turn)
                        or ({"braintrust.turn.reply_to": self.context_reply_to} if self.context_reply_to else {}),
                    ),
                )
            if self.llm:
                self.llm.log(metadata={"pipecat.function_calls": native_value(frame.function_calls)})
        elif kind == "FunctionCallInProgressFrame":
            if frame.tool_call_id in self.started_tools:
                return
            self.started_tools.add(frame.tool_call_id)
            owner, correlation = self.requests.get(frame.tool_call_id, (self.root, {}))
            self.tools[frame.tool_call_id] = owner.start_span(
                name=frame.function_name,
                type="tool",
                input=frame.arguments,
                metadata={
                    **correlation,
                    "pipecat.tool_call_id": frame.tool_call_id,
                    "pipecat.function_name": frame.function_name,
                    "pipecat.arguments": frame.arguments,
                    "pipecat.group_id": frame.group_id,
                    "pipecat.cancel_on_interruption": frame.cancel_on_interruption,
                },
                set_current=False,
                internal={"instrumentation": "pipecat-auto"},
            )
            if len(self.tool_names) < 512:
                self.tool_names.append(frame.function_name)
        elif kind in {"FunctionCallResultFrame", "FunctionCallCancelFrame"}:
            tool = self.tools.pop(frame.tool_call_id, None)
            if tool:
                if kind == "FunctionCallResultFrame":
                    request = self.requests.pop(frame.tool_call_id, None)
                    if request:
                        self.result_origins[frame.tool_call_id] = request[1].get("braintrust.turn.reply_to")
                    tool.log(
                        output=native_value(frame.result),
                        metadata={"pipecat.result": native_value(frame.result)},
                    )
                    if frame.error:
                        tool.log(error=frame.error)
                else:
                    tool.log(metadata={"pipecat.cancelled": True})
                tool.end()
        elif kind == "LLMFullResponseEndFrame" and self.llm:
            self.llm.log(
                output=[{"role": "assistant", "content": "".join(self.llm_text)}],
                metadata={"pipecat.text": "".join(self.llm_text)},
            )
            self.ttfb.end("llm")
            self.llm.end()
            self.llm = None
        elif kind == "TTSStartedFrame":
            context = self.frame_context(frame)
            if context in self.tts:
                return
            turn = self.turns.assistant or self.turns.start("assistant", kind)
            span = turn["span"].start_span(
                name="pipecat.tts_response",
                type="task",
                metadata={
                    **self.turns.metadata(turn),
                    "pipecat.context_id": getattr(frame, "context_id", None),
                    **({"openai.response.id": context} if self.realtime and context else {}),
                    "pipecat.append_to_context": frame.append_to_context,
                },
                set_current=False,
                internal={"instrumentation": "pipecat-auto"},
            )
            self.ttfb.start(("tts", context), data.source, span.log)
            self.tts[context] = {
                "span": span,
                "chunks": [],
                "omitted": False,
                "text": [],
            }
            if len(self.alignment.contexts) < 16000:
                self.alignment.contexts[context] = [span, turn["span"]]
            else:
                self.alignment.omitted += 1
        elif kind == "TTSTextFrame":
            state = self.tts.get(self.frame_context(frame))
            if state:
                state["text"].append(frame.text)
                state["span"].log(metadata={"pipecat.text": "".join(state["text"])})
        elif kind == "TTSStoppedFrame":
            self.ttfb.end(("tts", self.frame_context(frame)))
            state = self.tts.pop(self.frame_context(frame), None)
            if state:
                state["span"].end()
                self.complete_recording(state)
        elif kind == "InterruptionFrame":
            for state in self.tts.values():
                state["span"].log(metadata={"pipecat.end_frame": kind})
                state["span"].end()
                self.complete_recording(state)
            for context in self.tts:
                self.ttfb.end(("tts", context))
            self.tts.clear()
        elif kind in {"ErrorFrame", "FatalErrorFrame"}:
            self.root.log(error=str(frame.error))

        if kind in EVENT_TYPES:
            if kind.startswith("User"):
                turn = self.turns.user
                event_owner = turn["span"] if turn else self.root
            elif kind.startswith("Bot") or kind == "InterruptionFrame":
                turn = self.turns.assistant
                event_owner = turn["span"] if turn else self.root
            # The current LLM is not evidence that a concurrent service metric
            # belongs to it. Keep metrics at the root with native processor IDs.
            self.capture_event(data, fields, event_owner)

    def capture_event(self, data, fields, owner):
        if not self.clock_anchored:
            self.root.log(
                metadata={
                    "braintrust.clock": {
                        "observer_timestamp_ns": data.timestamp,
                        "observed_unix_ms": time.time() * 1000,
                        "basis": "observer_callback",
                    }
                }
            )
            self.clock_anchored = True
        event = {
            "pipecat.frame.type": type(data.frame).__name__,
            "pipecat.frame.id": data.frame.id,
            "pipecat.observer.timestamp": data.timestamp,
            "pipecat.observer.source": data.source.name,
            "pipecat.observer.destination": data.destination.name,
            "pipecat.observer.direction": data.direction.name,
            "pipecat.frame": fields,
        }
        size = len(json.dumps(event, ensure_ascii=False).encode("utf-8"))
        if len(self.events) < 256 and self.event_bytes + size < 65536:
            self.events.append(event)
            self.event_bytes += size
            group = self.event_groups.setdefault(owner.span_id, (owner, []))
            group[1].append(event)
        else:
            self.omitted_events += 1

    def complete_recording(self, state):
        if "last_pts" in state:
            state["span"].log(metadata={"pipecat.frame.pts": state["last_pts"]})
        if len(self.recordings) < 256:
            self.recordings.append(state)
            task = asyncio.create_task(self._publish_tts_clip(state))
            self.clip_tasks.add(task)
            task.add_done_callback(self.clip_tasks.discard)
            return
        self.recordings_omitted += 1
        state["span"].log(
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
        size = sum(map(len, state["chunks"]))
        state["chunks"].clear()
        source_budget.release(size)
        self.audio_bytes -= size

    async def cleanup(self):
        await self.finish()
        await super().cleanup()

    async def finish(self):
        if self._finish_task is None:
            self.finished = True
            self._finish_task = asyncio.create_task(self._finish())
        await asyncio.shield(self._finish_task)

    async def _finish(self):
        self.call_recording.seal()
        self.hooks.close()
        try:
            await self._finalize()
        finally:
            self.hooks.close()
            # A shielded segment drain can outlive cancellation of this finalizer.
            # Its active tail still belongs to that drain until it seals/exports it.
            drain = self.call_recording._finish_task
            if drain is None or drain.done():
                self.call_recording.clear()
            for state in self.recordings:
                state["chunks"].clear()
            source_budget.release(self.audio_bytes)
            self.audio_bytes = 0
            if self.user_capture:
                self.user_capture.release()

    async def _finalize(self):
        self.ttfb.clear()
        self.turns.finish()
        for span in [self.llm, self.user, *self.tools.values()]:
            if span:
                span.end()
        for state in self.tts.values():
            state["span"].end()
            state["omitted"] = True
            state["reason"] = "pipeline_closed_before_tts_stop"
            self.complete_recording(state)
        for owner, events in self.event_groups.values():
            owner.log(metadata={"pipecat.events": events})
        self.root.log(
            metadata={
                "braintrust.capture.events_omitted": self.omitted_events,
                "braintrust.capture.recordings_omitted": self.recordings_omitted,
                "braintrust.capture.filtered_event_counts": dict(self.filtered_events),
                "braintrust.capture.duplicate_events": self.duplicate_events,
                "braintrust.capture.retained_events": len(self.events),
            }
        )
        self.root.end()
        if self.user_capture:
            await self.user_capture.finish()
        if self.realtime:
            self.realtime.finish()
        # Trace lifetime ends before encoding; flush only after attachment updates.
        if self.clip_tasks:
            await asyncio.gather(*tuple(self.clip_tasks))
        if self.capture_transport:
            await self.call_recording.finish()
            self._publish_call_manifest()
            self.alignment.publish()
        await asyncio.to_thread(self.logger.flush)

    async def _publish_tts_clip(self, state):
        span = state["span"]
        recording = {"id": "tts-clip", "sources": [{"boundary": "tts_output"}]}
        if state["chunks"] and not state["omitted"]:
            try:
                encoded = await encode_in_worker(
                    prepare_recording,
                    "tts-clip",
                    encode_audio,
                    state["chunks"],
                    state["rate"],
                    state["channels"],
                    self.audio_format,
                )
            except Exception as error:  # noqa: BLE001 - capture/export failures must not break the call
                span.log(
                    metadata={
                        "audio.recordings": [
                            {
                                **recording,
                                "state": "omitted",
                                "reason": str(error) if isinstance(error, RecordingBusy) else "encoding_failed",
                            }
                        ]
                    }
                )
                source_budget.release(sum(map(len, state["chunks"])))
                self.audio_bytes -= sum(map(len, state["chunks"]))
                state["chunks"].clear()
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
                            {"type": "text", "text": "".join(state["text"])},
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
                reason=state.get("reason", "no_audio_observed") if self.capture_agent_audio else "disabled",
            )
        span.log(metadata={"audio.recordings": [recording]})
        source_budget.release(sum(map(len, state["chunks"])))
        self.audio_bytes -= sum(map(len, state["chunks"]))
        state["chunks"].clear()

        await asyncio.to_thread(self.logger.flush)

    def _call_sources(self):
        return [
            *(
                [{"boundary": "model_input" if self.realtime else "transport_input", "channel_index": 0}]
                if self.capture_user_audio
                else []
            ),
            *([{"boundary": "transport_output", "channel_index": 1}] if self.capture_agent_audio else []),
        ]

    async def _publish_call_segment(self, segment, encoded):
        if encoded is None:
            return
        recording_id = segment.segment_id
        self.call_descriptors[recording_id] = {
            "id": recording_id,
            "recording_group_id": "call",
            "state": "ready",
            "attachment": {"span_id": self.root.span_id, "ref": f"/input/audio/{recording_id}"},
            "mime_type": encoded["mime_type"],
            "duration_ms": encoded["duration_ms"],
            "channel_count": 2,
            "sources": self._call_sources(),
            "timeline": {
                "origin_unix_ms": self.call_recording.origin_unix_ms,
                "recording_start_offset_ms": segment.start_ms,
                "basis": "input_sample_clock_and_output_write_observation",
            },
        }
        self.root.log(input={"audio": {recording_id: encoded["attachment"]}})
        self._publish_call_manifest()
        self.alignment.publish()
        await asyncio.to_thread(self.logger.flush)

    def _publish_call_manifest(self):
        for segment in self.call_recording.completed:
            if segment["state"] == "omitted":
                self.call_descriptors[segment["id"]] = {
                    "id": segment["id"],
                    "recording_group_id": "call",
                    "state": "omitted",
                    "reason": segment["reason"],
                    "sources": self._call_sources(),
                    "timeline": {
                        "origin_unix_ms": self.call_recording.origin_unix_ms,
                        "recording_start_offset_ms": segment["start_ms"],
                        "basis": "input_sample_clock_and_output_write_observation",
                    },
                    "gaps": [
                        {
                            "start_offset_ms": segment["start_ms"],
                            "end_offset_ms": segment["end_ms"],
                            "reason": segment["reason"],
                        }
                    ],
                }
        descriptors = list(self.call_descriptors.values())
        reason = self.call_recording.reason
        if reason or not descriptors:
            descriptors.append(
                {
                    "id": "call",
                    "recording_group_id": "call",
                    "state": "omitted",
                    "reason": reason or "no_audio_observed",
                    "truncated": reason not in (None, "disabled"),
                    "sources": self._call_sources(),
                }
            )
        self.root.log(metadata={"audio.recordings": descriptors})

    def _turn_completed(self, turn):
        if turn["role"] == "user" and self.user_capture:
            self.user_capture.queue_completed(turn)
