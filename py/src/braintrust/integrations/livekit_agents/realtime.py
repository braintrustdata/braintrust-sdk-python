"""Qualified Realtime observations, owned by the existing LiveKit session."""

import asyncio
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from braintrust.logger import Span


if TYPE_CHECKING:
    from livekit.agents import llm
    from livekit.agents.voice.agent_activity import AgentActivity

    from .tracing import Session


_ACTIVITY = ContextVar("braintrust_livekit_realtime_activity", default=None)
_RESPONSE = ContextVar("braintrust_livekit_realtime_response", default=None)
_INPUT = ContextVar("braintrust_livekit_realtime_input", default=None)


@dataclass
class InputClock:
    """Map OpenAI's 24 kHz PCM buffer offsets to positioned recorder nanoseconds."""

    origin: Fraction | None = None
    end: Fraction | None = None
    appended: int = 0
    valid: bool = True

    def push(self, start: Fraction, duration: Fraction) -> None:
        if self.origin is None:
            self.origin = start
        if self.end is not None and start != self.end:
            self.valid = False
        self.end = start + duration

    def offset(self, milliseconds: int | float | None) -> Fraction | None:
        if not self.valid or self.origin is None or not isinstance(milliseconds, (int, float)):
            return None
        if milliseconds < 0 or milliseconds > self.appended / 24 + 1:
            return None
        return self.origin + Fraction(str(milliseconds)) * 1_000_000


@dataclass
class InputItem:
    turn: Span | None = None
    speech: Span | None = None
    receipt: Span | None = None
    start: Fraction | None = None
    end: Fraction | None = None
    stopped: bool = False


@dataclass
class Response:
    controller: "Realtime"
    identifier: str
    accepted: bool = False
    span: Span | None = None
    handle: Any = None
    created: float | None = None
    ended: float | None = None
    status: str | None = None
    outputs: dict = field(default_factory=dict)
    calls: set = field(default_factory=set)
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    span_ended: bool = False
    usage_present: bool | None = None
    modality: str | None = None
    audio_sources: dict = field(default_factory=dict)
    context: list = field(default_factory=list)
    consumed: list = field(default_factory=list)

    def bind(self, handle=None):
        from .tracing import _new, link

        owner = self.controller.owner
        if handle is not None:
            self.handle = handle
        if self.span is None and self.accepted:
            parent = owner.turns.get(self.handle, owner.span)
            self.span = _new(
                parent,
                "llm_response",
                kind="llm",
                start_time=self.created,
                metadata={
                    "model": self.controller.model.model,
                    "provider": self.controller.model.provider,
                    "contrib.livekit.response_id": self.identifier,
                },
            )
            if self.context:
                owner.log(self.span, input=self.context)
                self.span.log(metadata={"contrib.livekit.input_partial": True})
                self.context.clear()
            relationships = [value for target in self.consumed if (value := link(target, "consumes_tool_result"))]
            if relationships:
                self.span.log(metadata={"links": relationships})
        if self.span is None:
            return
        if self.handle is not None:
            owner.requests.setdefault(self.handle, {}).update({call: self.span for call in self.calls})
        self.publish()

    def publish(self):
        if self.span is None:
            return
        owner = self.controller.owner
        if self.outputs:
            owner.log(
                self.span, output=[value for _, value in sorted(self.outputs.values(), key=lambda pair: pair[0])]
            )
            if self.ended is not None:
                self.outputs.clear()
        if self.metadata:
            self.span.log(metadata=self.metadata)
        event = dict(self.metrics)
        if self.created is not None:
            event["start"] = self.created
        if self.ended is not None:
            event["end"] = self.ended
        if event:
            self.span.log(metrics=event)
        if self.status:
            self.span.log(metadata={"contrib.livekit.response_status": self.status})
            if self.status == "cancelled":
                self.span.log(metadata={"contrib.livekit.cancelled": True})
            elif self.status in ("failed", "incomplete"):
                owner.log(self.span, error="LiveKit Realtime response " + self.status)
        if self.ended is not None and not self.span_ended:
            self.span.end(self.ended)
            self.span_ended = True


class Realtime:
    def __init__(
        self, owner: "Session", activity: "AgentActivity", native: "llm.RealtimeSession", model: "llm.RealtimeModel"
    ):
        self.owner, self.activity, self.native, self.model = owner, activity, native, model
        self.clock = InputClock()
        self.epoch = 0
        self.inputs: dict[tuple[int, str], InputItem] = {}
        self.responses: dict[tuple[int, str], Response] = {}
        self.tools = {}
        self.sent_items = set()
        self.listeners = []
        self.closed = False
        self.connected = False
        self.raw_events = False
        self.original_push = native.push_audio

        def push_audio(frame):
            position = _INPUT.get()
            if position is not None and position[0] is self.owner:
                self.input_frame(position[1], position[2])
            try:
                return self.original_push(frame)
            except BaseException:
                self.clock.valid = False
                raise

        self.push_audio = push_audio
        native.push_audio = push_audio
        for name, callback in (
            ("openai_server_event_received", self.server),
            ("openai_client_event_queued", self.client),
            ("generation_created", self.generation),
            ("input_audio_transcription_completed", self.input_transcription),
            ("metrics_collected", self.metrics),
            ("session_reconnected", self.reconnected),
        ):
            native.on(name, callback)
            self.listeners.append((name, callback))

    def permits(self):
        allowed = self.owner.permits()
        if not allowed:
            for record in self.responses.values():
                record.outputs.clear()
                record.context.clear()
        return allowed

    def response(self, identifier):
        key = (self.epoch, identifier)
        if key not in self.responses:
            self.responses[key] = Response(self, identifier)
        return self.responses[key]

    def input_frame(self, start, duration):
        if not self.closed:
            self.clock.push(start, duration)

    def client(self, event):
        if self.closed:
            return
        kind = event.get("type")
        if kind == "input_audio_buffer.append":
            audio = event.get("audio", "")
            self.clock.appended += (len(audio) * 3 // 4 - (len(audio) - len(audio.rstrip("=")))) // 2
        elif kind == "input_audio_buffer.clear":
            self.clock.valid = False
        elif kind == "conversation.item.create":
            item = event.get("item", {})
            if item.get("id"):
                self.sent_items.add(item["id"])
        self.permits()

    def server(self, event):
        if self.closed:
            return
        self.permits()
        self.raw_events = True
        kind = event.get("type", "")
        now = time.time()
        if kind == "session.created":
            if self.connected:
                self.reconnected(None)
            self.connected = True
        elif kind in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"):
            self.speech(event, now)
        elif kind == "conversation.item.input_audio_transcription.completed":
            self.transcript(event, now)
        elif kind == "response.created":
            native = event["response"]
            record = self.response(native["id"])
            record.created = now
            # Only explicitly sent text/tool items from the actual remote mirror.
            if self.permits():
                from .compat import chat_messages

                context = self.native.chat_ctx.copy()
                context.items[:] = [item for item in context.items if item.id in self.sent_items]
                record.context = chat_messages(context)
                for item in context.items:
                    if (
                        item.type == "function_call_output"
                        and (target := self.tools.get((self.epoch, item.call_id))) is not None
                    ):
                        record.consumed.append(target)
        elif kind == "response.output_item.done":
            self.output(self.response(event["response_id"]), event["item"], event.get("output_index", 0))
        elif kind == "response.done":
            native = event["response"]
            record = self.response(native["id"])
            record.status = native.get("status", "incomplete")
            record.ended = now
            record.usage_present = isinstance(native.get("usage"), dict)
            details = native.get("status_details") or {}
            for name, value in (
                ("type", details.get("type")),
                ("reason", details.get("reason")),
                ("code", (details.get("error") or {}).get("code")),
            ):
                if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_]{1,80}", value):
                    record.metadata["contrib.livekit.response_" + name] = value
            for index, item in enumerate(native.get("output", [])):
                self.output(record, item, index, publish=False)
            record.publish()
            if not record.accepted:
                record.outputs.clear()
                record.context.clear()

    def speech(self, event, now):
        from .tracing import _new

        identifier = event.get("item_id")
        if not identifier:
            return
        key = (self.epoch, identifier)
        item = self.inputs.setdefault(key, InputItem())
        started = event["type"].endswith("speech_started")
        position = self.clock.offset(event.get("audio_start_ms" if started else "audio_end_ms"))
        if started and item.turn is None:
            item.start = position
            start = (self.owner.audio.wall_ns + position) / 1e9 if position is not None else now
            meta = self.owner.source_metadata("session-input")
            meta["contrib.livekit.item_id"] = identifier
            meta["contrib.livekit.timing"] = "openai_audio_buffer_bounds" if position is not None else "unavailable"
            item.turn = _new(self.owner.span, "user_turn", start_time=float(start), metadata=meta)
            item.speech = _new(item.turn, "user_speech", start_time=float(start), metadata=meta)
        elif not started and not item.stopped:
            item.stopped = True
            end = now
            metadata = {"contrib.livekit.timing": "unavailable"}
            if item.start is not None and position is not None and position >= item.start:
                item.end = position
                end = float((self.owner.audio.wall_ns + position) / 1e9)
                metadata = {
                    "audio.selections": {
                        "selection-0": {
                            "start_offset_ms": round(item.start / 1e6),
                            "end_offset_ms": round(position / 1e6),
                        }
                    }
                }
            for span in (item.speech, item.turn):
                if span is not None:
                    span.log(metadata=metadata)
                    span.end(end)

    def transcript(self, event, now):
        from .tracing import _new, link

        key = (self.epoch, event["item_id"])
        item = self.inputs.setdefault(key, InputItem())
        if item.receipt is None:
            item.receipt = _new(
                item.turn or self.owner.span,
                "transcript_received",
                start_time=now,
                metadata={**self.owner.source_metadata("session-input"), "contrib.livekit.item_id": event["item_id"]},
            )
            if item.speech is not None and (relationship := link(item.speech, "transcribes")):
                item.receipt.log(metadata={"links": [relationship]})
            item.receipt.end(now)
        if self.permits():
            message = [{"role": "user", "content": event.get("transcript", "")}]
            self.owner.log(item.receipt, output=message)
            if item.turn is not None:
                self.owner.log(item.turn, input=message)
        usage = event.get("usage")
        if isinstance(usage, dict):
            safe = numeric_usage(usage)
            options = getattr(self.native, "_opts", None)
            transcription = getattr(options, "input_audio_transcription", None)
            identity = {
                "contrib.livekit.transcription_usage": safe,
                "contrib.livekit.transcription_provider": self.model.provider,
            }
            if (model := getattr(transcription, "model", None)) is not None:
                identity["contrib.livekit.transcription_model"] = model
            item.receipt.log(metadata=identity)

    def input_transcription(self, event):
        # The qualified bridge already handled this before native dispatch.
        # Other providers can at least preserve an identified transcript receipt.
        if not self.closed and not self.raw_events and event.is_final and event.transcript:
            self.transcript({"item_id": event.item_id, "transcript": event.transcript}, time.time())

    def generation(self, event):
        if self.closed or not event.response_id:
            return
        record = self.response(event.response_id)
        record.accepted = True
        if record.created is None:
            record.created = time.time()

    def metrics(self, event):
        if self.closed or getattr(event, "type", None) != "realtime_model_metrics" or not event.request_id:
            return
        record = self.response(event.request_id)
        record.created = event.timestamp
        record.ended = event.timestamp + event.duration
        if record.status is None:
            record.status = "cancelled" if event.cancelled else "completed"
        if record.usage_present is not False:
            record.metrics.update(
                {
                    "prompt_tokens": event.input_tokens,
                    "completion_tokens": event.output_tokens,
                    "tokens": event.total_tokens,
                    "prompt_audio_tokens": event.input_token_details.audio_tokens,
                    "completion_audio_tokens": event.output_token_details.audio_tokens,
                    "prompt_cached_tokens": event.input_token_details.cached_tokens,
                }
            )
        if event.ttft >= 0 and record.modality is not None:
            record.metrics["time_to_first_audio" if record.modality == "audio" else "time_to_first_token"] = event.ttft
        if event.input_token_details.cached_tokens_details is not None:
            record.metadata["contrib.livekit.cached_tokens"] = (
                event.input_token_details.cached_tokens_details.model_dump()
            )
        record.publish()

    def output(self, record, item, index, publish=True):
        identifier = item.get("id", str(index))
        kind = item.get("type")
        if kind == "function_call":
            record.calls.add(item["call_id"])
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": item.get("arguments", "")},
                    }
                ],
            }
        elif kind == "message":
            parts = item.get("content", [])
            record.modality = (
                "audio" if any(part.get("type") in ("audio", "output_audio") for part in parts) else "text"
            )
            message = {
                "role": item.get("role", "assistant"),
                "content": "".join(part.get("transcript", part.get("text", "")) or "" for part in parts),
            }
        else:
            return
        if self.permits():
            record.outputs[identifier] = (index, message)
        if record.span is not None and record.handle is not None:
            self.owner.requests.setdefault(record.handle, {}).update({call: record.span for call in record.calls})
        if publish:
            record.publish()

    def reconnected(self, _):
        self.clock.valid = False
        self.epoch += 1
        self.sent_items.clear()

    def close(self):
        if self.closed:
            return
        self.closed = True
        now = time.time()
        for record in self.responses.values():
            if record.accepted:
                record.bind()
                if record.ended is None and record.span is not None:
                    record.span.log(metadata={"contrib.livekit.incomplete": True})
                    record.span.end(now)
        for item in self.inputs.values():
            if item.turn is not None and not item.stopped:
                item.turn.log(metadata={"turn.incomplete": True})
                item.turn.end(now)
                if item.speech is not None:
                    item.speech.log(metadata={"turn.incomplete": True})
                    item.speech.end(now)
        for name, callback in self.listeners:
            self.native.off(name, callback)
        self.listeners.clear()
        if self.native.push_audio is self.push_audio:
            self.native.push_audio = self.original_push
        self.inputs.clear()
        self.responses.clear()
        self.tools.clear()
        self.sent_items.clear()
        self.native = self.activity = self.model = None


def prepare(activity: "AgentActivity") -> None:
    from livekit.agents import llm

    model = activity.llm
    if not isinstance(model, llm.RealtimeModel):
        return
    factory = model.session
    if not getattr(factory, "_braintrust_realtime", False):

        def session(*args, **kwargs):
            native = factory(*args, **kwargs)
            current = _ACTIVITY.get()
            if current is not None:
                current_owner, current_activity = current
                bind(current_owner, current_activity, native, model)
            return native

        session._braintrust_realtime = True
        model.session = session


def bind(
    owner: "Session", activity: "AgentActivity", native: "llm.RealtimeSession", model: "llm.RealtimeModel"
) -> Realtime:
    if native not in owner.realtime:
        owner.realtime[native] = Realtime(owner, activity, native, model)
    else:
        owner.realtime[native].activity = activity
    return owner.realtime[native]


def numeric_usage(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: numeric_usage(item) if isinstance(item, dict) else item
        for key, item in value.items()
        if isinstance(item, dict) or (isinstance(item, (int, float)) and not isinstance(item, bool))
    }


async def generate(wrapped, instance, args, kwargs):
    from .compat import arguments, state_for

    owner = state_for(instance)
    if owner is None:
        return await wrapped(*args, **kwargs)
    controller = owner.realtime.get(instance._rt_session)
    values = arguments(wrapped, args, kwargs)
    event = values["generation_ev"]
    if controller is None or not event.response_id:
        return await wrapped(*args, **kwargs)
    record = controller.response(event.response_id)
    record.accepted = True
    record.bind(values["speech_handle"])
    token = _RESPONSE.set(record)
    try:
        return await wrapped(*args, **kwargs)
    except asyncio.CancelledError:
        if record.ended is None:
            record.status = "cancelled"
        raise
    except Exception:
        if record.ended is None:
            record.status = "failed"
        raise
    finally:
        _RESPONSE.reset(token)
        record.audio_sources.clear()
        if record.status in ("failed", "cancelled") and record.ended is None:
            record.ended = time.time()
        record.publish()
