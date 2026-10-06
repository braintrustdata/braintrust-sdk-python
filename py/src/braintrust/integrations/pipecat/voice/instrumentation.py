"""Experimental native Pipecat observer and opt-in recording support."""

import asyncio
import dataclasses
import io
import json
import time
import wave
from collections import Counter, deque

from braintrust.audio.alignment import Alignment
from braintrust.audio.segments import SegmentedRecording
from pipecat.observers.base_observer import BaseObserver  # pylint: disable=import-error

from ..llm_metrics import _llm_usage_metrics, _metadata_from_metric, _metadata_from_processor
from ..ttfb import TTFBRouter
from ..turn_metrics import TURN_METRIC_TYPES, log_turn_metric
from .synthesis import Synthesis, SynthesisRecordings
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


CONFIG_TYPES = {
    "SpeechControlParamsFrame",
    "STTMetadataFrame",
    "LLMServiceMetadataFrame",
    "LLMUpdateSettingsFrame",
    "TTSUpdateSettingsFrame",
    "STTUpdateSettingsFrame",
    "VADParamsUpdateFrame",
}


def compact_frame(fields):
    """Keep event content; omit transport bookkeeping and empty fields."""
    return {
        key: value
        for key, value in fields.items()
        if key
        not in {"id", "name", "broadcast_sibling_id", "interruptible", "transport_destination", "transport_source"}
        and value is not None
        and value != {}
        and value != []
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
            on_pending=self._pending_call_segment,
        )
        self.call_descriptors = {}
        self.realtime = None
        self.input_processor = None
        self.capture_transport = False
        self.root = (
            root if root is not None else logger.start_span(name="pipecat.pipeline", type="task", set_current=False)
        )
        self.ttfb = TTFBRouter(self.root.log)
        self.tts_requests = None
        self.turns = Turns(self.root, self.hooks)
        self.turns.on_completed = self._turn_completed
        self.alignment = Alignment(self.root, self.call_recording)
        self.synthesis = SynthesisRecordings(
            logger,
            self.alignment,
            enabled=self.capture_agent_audio,
            max_bytes=max_audio_bytes,
            audio_format=audio_format,
        )
        self.user_capture = None
        self.llm_turn = None
        self.user_aggregator = None
        self.assistant_aggregator = None
        self.context_reply_to = None
        self.context_tool_results = []
        self.result_origins = {}
        self.event_groups = {}
        self.configuration = {}
        self.filtered_events = Counter()
        self.duplicate_events = 0
        self.clock_anchored = False
        self.llm = None
        self.llm_tool_calls = []
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
        self.finished = False
        self._finish_task = None
        self.tool_names = []
        self.started_tools = set()
        self.last_context = None
        self.seen = set()
        self.seen_order = deque()

    def bind(
        self, *, transport, user_aggregator, assistant_aggregator, stt=None, realtime_service=None, tts_services=()
    ):
        """Bind one supported pipeline before it starts; does not alter routing."""
        import importlib.metadata

        if importlib.metadata.version("pipecat-ai") != "1.12.0":
            raise ValueError("Experimental voice hooks currently support Pipecat 1.12.0")
        if getattr(self, "_bound", False):
            raise ValueError("Voice observer is already bound to a pipeline")
        if (stt is None) == (realtime_service is None):
            raise ValueError("Bind either segmented STT or an OpenAI realtime service")
        self.turns.install(user_aggregator, assistant_aggregator)
        from .tts_metrics import TTSRequests

        self.tts_requests = TTSRequests(self.hooks, tts_services)
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
        identity = getattr(frame, "_braintrust_realtime_context", None)
        if identity and self.realtime and identity[0] is self.realtime.frame_token:
            return identity[1]
        return getattr(frame, "context_id", None)

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
                self.synthesis.capture(state, frame)
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
            request = self.tts_requests.take(frame, data.source) if self.tts_requests else None
            if request is not None:
                for metric in frame.data:
                    self.ttfb.capture_request(metric, data.source, request)
                return
            # Pipeline startup broadcasts zero placeholders for every service.
            if frame.data and all(
                getattr(metric, "value", None) == 0
                and getattr(metric, "model", None) is None
                and getattr(metric, "processor", None) != data.source.name
                for metric in frame.data
            ):
                return
            for metric in frame.data:
                metric_type = type(metric).__name__
                if metric_type in TURN_METRIC_TYPES and data.source is self.user_aggregator and self.turns.user:
                    state = self.turns.user
                    log_turn_metric(state["span"], state, metric)  # pylint: disable=unsubscriptable-object
                elif metric_type == "TTFBMetricsData":
                    owner = self.ttfb.capture(metric, data.source)
                    if owner == "llm":
                        self.llm.log(metrics={"time_to_first_token": metric.value})
                elif (
                    metric_type == "LLMUsageMetricsData"
                    and self.ttfb.owner(metric, data.source, operation="llm")[0] == "llm"
                ):
                    self.llm.log(
                        metrics=_llm_usage_metrics(metric.value),
                        metadata={
                            "contrib.pipecat.usage": native_value(metric),
                            **_metadata_from_processor(data.source),
                            **_metadata_from_metric(metric),
                        },
                    )
                else:
                    self.ttfb.capture_measurement(metric, data.source)
            return
        if kind == "StartFrame":
            self.root.log(
                metadata={
                    f"contrib.pipecat.{name}": getattr(frame, name)
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
                    metadata={"contrib.pipecat.start_frame": kind},
                )
                event_owner = self.user
        elif kind in {"UserStoppedSpeakingFrame", "VADUserStoppedSpeakingFrame"}:
            if kind == "VADUserStoppedSpeakingFrame" and self.user:
                event_owner = self.user
                self.user.log(metadata={"contrib.pipecat.end_frame": kind})
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
                metadata={f"contrib.pipecat.{key}": value for key, value in fields.items()},
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
            self.llm_tool_calls = []
            self.llm_text = []
            # A new model operation may arrive before the previous aggregator
            # callback task finishes. Keep each native response lifecycle distinct.
            self.turns.assistant = None
            self.llm_turn = self.turns.start("assistant", kind, self.context_reply_to)
            self.llm = self.llm_turn["span"].start_span(
                name="llm_response",
                type="llm",
                input=self.last_context,
                metadata={
                    **self.turns.metadata(self.llm_turn),
                    "continuation.tool_call_ids": self.context_tool_results,
                    **_metadata_from_processor(data.source),
                },
                set_current=False,
                internal={"instrumentation": "pipecat-auto"},
            )
            self.ttfb.start("llm", data.source, self.llm.log)
        elif kind == "LLMTextFrame" and self.llm:
            self.llm_text.append(frame.text)
        elif kind == "FunctionCallsStartedFrame":
            self.llm_tool_calls = [
                {
                    "id": call.tool_call_id,
                    "type": "function",
                    "function": {"name": call.function_name, "arguments": json.dumps(native_value(call.arguments))},
                }
                for call in frame.function_calls
            ]
            if self.llm_turn:
                self.llm_turn["tool_calls"] = list(self.llm_tool_calls)
                self.turns.log_message(self.llm_turn, "")
            for call in frame.function_calls:
                self.requests.setdefault(
                    call.tool_call_id,
                    (
                        self.llm or self.root,
                        self.turns.metadata(self.llm_turn)
                        or ({"turn.reply_to": self.context_reply_to} if self.context_reply_to else {}),
                    ),
                )
            if self.llm:
                self.llm.log(metadata={"contrib.pipecat.function_calls": native_value(frame.function_calls)})
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
                    "contrib.pipecat.tool_call_id": frame.tool_call_id,
                    "contrib.pipecat.function_name": frame.function_name,
                    "contrib.pipecat.arguments": frame.arguments,
                    "contrib.pipecat.group_id": frame.group_id,
                    "contrib.pipecat.cancel_on_interruption": frame.cancel_on_interruption,
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
                        self.result_origins[frame.tool_call_id] = request[1].get("turn.reply_to")
                    tool.log(
                        output=native_value(frame.result),
                        metadata={"contrib.pipecat.result": native_value(frame.result)},
                    )
                    if frame.error:
                        tool.log(error=frame.error)
                else:
                    tool.log(metadata={"contrib.pipecat.cancelled": True})
                tool.end()
        elif kind == "LLMFullResponseEndFrame" and self.llm:
            text = "".join(self.llm_text)
            message = {"role": "assistant", "content": text or None}
            if self.llm_tool_calls:
                message["tool_calls"] = list(self.llm_tool_calls)
            self.llm.log(output=[message], metadata={"contrib.pipecat.text": text} if text else {})
            self.ttfb.end("llm")
            self.llm.end()
            self.llm = None
        elif kind == "TTSStartedFrame":
            context = self.frame_context(frame)
            if context in self.tts:
                return
            turn = self.turns.assistant or self.turns.start("assistant", kind)
            span = turn["span"].start_span(
                name="pipecat.audio_output" if self.realtime else "tts",
                type="task",
                metadata={
                    **self.turns.metadata(turn),
                    "contrib.pipecat.context_id": getattr(frame, "context_id", None),
                    **({"openai.response.id": context} if self.realtime and context else {}),
                    "contrib.pipecat.append_to_context": frame.append_to_context,
                    **_metadata_from_processor(data.source),
                },
                set_current=False,
                internal={"instrumentation": "pipecat-auto"},
            )
            self.ttfb.start(("tts", context), data.source, span.log)
            self.tts[context] = Synthesis(span)
            if self.capture_agent_audio:
                self.alignment.begin_output(context, [span, turn["span"]])
        elif kind == "TTSTextFrame":
            state = self.tts.get(self.frame_context(frame))
            if state:
                state.text.append(frame.text)
                text = "".join(state.text)
                state.span.log(input={"text": text}, metadata={"contrib.pipecat.text": text})
        elif kind == "TTSStoppedFrame":
            self.ttfb.end(("tts", self.frame_context(frame)))
            state = self.tts.pop(self.frame_context(frame), None)
            if state:
                state.span.end()
                self.synthesis.complete(state)
        elif kind == "InterruptionFrame":
            for state in self.tts.values():
                state.span.log(metadata={"contrib.pipecat.end_frame": kind})
                state.span.end()
                self.synthesis.complete(state)
            for context in self.tts:
                self.ttfb.end(("tts", context))
            self.tts.clear()
        elif kind in {"ErrorFrame", "FatalErrorFrame"}:
            self.root.log(error=str(frame.error))

        if kind in CONFIG_TYPES:
            fields = compact_frame(fields)
            identity = (kind, data.source.name)
            initial = identity not in self.configuration
            if not initial and self.configuration[identity] == fields:
                return
            if initial and len(self.configuration) >= 32:
                self.omitted_events += 1
                return
            self.configuration[identity] = fields
            self.root.log(
                metadata={
                    "contrib.pipecat.configuration": [
                        {"type": frame_type, "source": source, "fields": values}
                        for (frame_type, source), values in self.configuration.items()
                    ]
                }
            )
            # Explicit update commands are changes even on their first observation.
            if initial and kind in {"SpeechControlParamsFrame", "STTMetadataFrame", "LLMServiceMetadataFrame"}:
                return
        if kind in EVENT_TYPES and kind not in {
            "StartFrame",
            "EndFrame",
            "ClientConnectedFrame",
            "OutputTransportReadyFrame",
        }:
            if kind.startswith("User"):
                turn = self.turns.user
                event_owner = turn["span"] if turn else self.root
            elif kind.startswith("Bot") or kind == "InterruptionFrame":
                turn = self.turns.assistant
                event_owner = turn["span"] if turn else self.root
            self.capture_event(data, fields, event_owner)

    def capture_event(self, data, fields, owner):
        if len(self.events) >= 256:
            self.omitted_events += 1
            return
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
            "contrib.pipecat.frame.type": type(data.frame).__name__,
            "contrib.pipecat.observer.timestamp": data.timestamp,
            "contrib.pipecat.observer.source": data.source.name,
            "contrib.pipecat.frame": compact_frame(fields),
        }
        size = len(json.dumps(event, ensure_ascii=False).encode("utf-8"))
        if len(self.events) < 256 and self.event_bytes + size < 65536:
            self.events.append(event)
            self.event_bytes += size
            group = self.event_groups.setdefault(owner.span_id, (owner, []))
            group[1].append(event)
        else:
            self.omitted_events += 1

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
            self.call_recording.release_idle_buffer()
            await self.synthesis.drain()
            for state in self.tts.values():
                self.synthesis.release(state)
            if self.user_capture:
                await self.user_capture.close()

    async def _finalize(self):
        self.ttfb.clear()
        self.turns.finish()
        for span in [self.llm, self.user, *self.tools.values()]:
            if span:
                span.end()
        for state in self.tts.values():
            state.span.end()
            state.omitted = True
            state.reason = "pipeline_closed_before_tts_stop"
            self.synthesis.complete(state)
        for owner, events in self.event_groups.values():
            owner.log(metadata={"contrib.pipecat.events": events})
        self.root.log(
            metadata={
                "braintrust.capture.events_omitted": self.omitted_events,
                "braintrust.capture.recordings_omitted": self.synthesis.omitted,
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
        await self.synthesis.drain()
        if self.capture_transport:
            await self.call_recording.finish()
            self._publish_call_manifest()
            self.alignment.publish()
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

    def _pending_call_segment(self, segment):
        self.call_descriptors[segment.segment_id] = {
            "id": segment.segment_id,
            "recording_group_id": "call",
            "state": "pending",
            "sources": self._call_sources(),
            "timeline": {
                "origin_unix_ms": segment.origin_unix_ms,
                "recording_start_offset_ms": segment.start_ms,
                "basis": "input_sample_clock_and_output_write_observation",
            },
        }
        self._publish_call_manifest()

    async def _publish_call_segment(self, segment, encoded):
        if encoded is None:
            self._publish_call_manifest()
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
