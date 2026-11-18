"""Prototype GPT-Live tracing: continuous conversation and hosted Responses.

The voice service's LLM/TTS frames are transcript grouping notifications, not
provider calls. Public frames own transcript groups; native events own hosted
Responses. The ordinary observer still owns transport audio and speech. This
intentionally supports one Live service per pipeline and hosted Responses
delegation first.
"""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from braintrust._audio.clock import SessionClock
from braintrust._audio.schema import SOURCE_ID
from braintrust.integrations.openai.tracing import _parse_metrics_from_usage
from braintrust.logger import Span, SpanTypeAttribute

from ._common import SpanFactory, metadata_from_processor, tools_schema_to_openai_tools


# Prototype pricing bridge until the backend prices duration-based models.
# Public USD/minute rate verified 2026-11-16. Do not price unknown models.
_LIVE_USD_PER_MINUTE = {"gpt-live-1": 0.05}
_LIVE_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-live-1"


def is_live_service(processor: Any) -> bool:
    return any(base.__name__ == "OpenAILiveLLMService" for base in type(processor).__mro__)


@dataclass
class _Transcript:
    span: Span
    text: list[str] = field(default_factory=list)
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass
class _Response:
    wrapper: Span
    llm: Span
    start_ms: float
    output: dict[str, Any] = field(default_factory=dict)
    first_content: bool = False
    llm_finished: bool = False
    processing_finished: bool = False
    pending_calls: set[str] = field(default_factory=set)
    pending_tool_spans: set[str] = field(default_factory=set)

    def record_output(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind in ("response.output_item.added", "response.output_item.done"):
            item = event.get("item") or {}
            key = item.get("id") or str(event.get("output_index", len(self.output)))
            # Deltas update our snapshot, never the event Pipecat is handling.
            self.output[key] = deepcopy(item)
            return
        item = self.output.get(event.get("item_id") or str(event.get("output_index")))
        if item is None:
            return
        if kind == "response.function_call_arguments.delta":
            item["arguments"] = item.get("arguments", "") + event["delta"]
        elif kind == "response.function_call_arguments.done":
            item["arguments"] = event["arguments"]
        elif kind in ("response.content_part.added", "response.content_part.done"):
            content = item.setdefault("content", [])
            index = event["content_index"]
            if index == len(content):
                content.append(deepcopy(event["part"]))
            elif 0 <= index < len(content):
                content[index] = deepcopy(event["part"])
        elif kind in ("response.output_text.delta", "response.output_text.done"):
            content = item.get("content", [])
            index = event["content_index"]
            if 0 <= index < len(content):
                part = content[index]
                part["text"] = part.get("text", "") + event["delta"] if kind.endswith(".delta") else event["text"]


class LiveTracker:
    def __init__(self, session: Any, service: Any) -> None:
        self.session, self.service = session, service
        session.agent.report_uninterrupted_playback = False
        # Native callbacks and frame callbacks run in different tasks. Do not
        # share the frame observer's temporarily pinned event clock.
        self.clock = SessionClock(session.recorder)
        self.spans = SpanFactory(session.span, self.clock)
        self.frame_spans = SpanFactory(session.span, session.clock)
        self.origin_ms = self.clock.now_ms()
        self.turns: dict[str, Span] = {}
        self.transcripts: dict[str, _Transcript] = {}
        self.responses: dict[str, _Response] = {}
        self.delegation_responses: dict[str, str] = {}
        self.delegation_offsets: dict[str, int] = {}
        self.tool_responses: dict[str, str] = {}
        self.closed = False
        self.model_closed = False
        self.inputs: list[Any] = []
        self.outputs: list[dict[str, str]] = []
        self.first_content = False
        self.model_name: str | None = None
        self.model = self.spans.start(
            "pipecat.live",
            type=SpanTypeAttribute.LLM,
            metadata={**metadata_from_processor(service), "audio.scope": "session"},
        )
        session.span.log(metadata={"contrib.pipecat.mode": "live"})

    def event(self, event: Any) -> None:
        if self.closed:
            return
        kind = event.type
        if kind == "session.started":
            self.origin_ms = self.clock.now_ms()
            resource = event.session
            self.model_name = resource.model
            metadata = {
                "provider": "openai",
                "model": resource.model,
                "contrib.pipecat.live.session_id": resource.id,
            }
            self.model.log(
                metadata={
                    **metadata,
                    "contrib.pipecat.live.input_partial": True,
                    "contrib.pipecat.live.transcript_grouping": "pipecat_transcript_gap",
                }
            )
            self.session.span.log(metadata=metadata)
            context = getattr(self.service, "_context", None)
            self.inputs = list(context.get_messages()) if context is not None else []
            if resource.instructions:
                # Pipecat consumes a leading system message as instructions;
                # the server reports the effective value after settings override it.
                if self.inputs and isinstance(self.inputs[0], dict) and self.inputs[0].get("role") == "system":
                    self.inputs.pop(0)
                self.inputs.insert(0, {"role": "system", "content": resource.instructions})
            self.model.log(input=self.inputs)
        elif kind == "session.output_audio.delta":
            if event.delta:
                self._first_content()
        elif kind in ("session.input_transcript.delta", "session.output_transcript.delta"):
            if kind == "session.output_transcript.delta" and event.delta:
                self._first_content()
        elif kind == "session.delegation.created":
            delegation = event.delegation
            self.delegation_offsets[delegation.id] = event.offset_ms
            if delegation.response_id:
                self.delegation_responses[delegation.id] = delegation.response_id
        elif kind == "response.event":
            self._response(event)
        elif kind in ("session.usage.updated", "session.closed"):
            usage = getattr(event, "usage", None)
            if usage is not None and usage.seconds is not None:
                self._usage(usage.seconds)
            if kind == "session.closed":
                self.model.log(metadata={"contrib.pipecat.live.close_reason": event.reason})
                # Frame observations are queued separately. Closing the remote
                # model must not discard transcript frames already in flight.
                self._end_model(confirmed=True)
        elif kind == "error":
            self.model.log(error=event.error.message)

    def _first_content(self) -> None:
        if not self.first_content:
            self.first_content = True
            self.model.log(metrics={"time_to_first_token": (self.clock.now_ms() - self.origin_ms) / 1000})

    def _usage(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            return
        # Each provider update is cumulative: replace the estimate, never sum
        # updates or substitute local wall time for the provider's billed time.
        metadata: dict[str, Any] = {"contrib.pipecat.live.usage_seconds": seconds}
        rate = _LIVE_USD_PER_MINUTE.get(self.model_name or "")
        if rate is not None:
            metadata.update(
                {
                    "contrib.pipecat.live.pricing": {
                        "usd_per_minute": rate,
                        "source": _LIVE_PRICING_SOURCE,
                        "as_of": "2026-11-16",
                    }
                }
            )
            self.model.log(metrics={"estimated_cost": seconds * rate / 60})
        self.model.log(metadata=metadata)

    def frame(self, frame: Any) -> None:
        if self.closed:
            return
        kind = type(frame).__name__
        if kind == "ProposedUserStartedSpeakingFrame":
            self._open_turn("user")
        elif kind == "LLMFullResponseStartFrame":
            self._open_turn("assistant")
        elif kind in ("InterimTranscriptionFrame", "TranscriptionFrame"):
            # User frames contain the accumulated group, not a text delta.
            self._transcript("user", frame.text, replace=True, event=getattr(frame, "result", None))
        elif kind == "LLMTextFrame":
            self._transcript("assistant", frame.text)
        elif kind == "ProposedUserStoppedSpeakingFrame":
            self._end_turn("user")
        elif kind == "LLMFullResponseEndFrame":
            self._end_turn("assistant")

    def _open_turn(self, role: str) -> None:
        if role in self.turns:
            return
        turn = self.turns[role] = self.frame_spans.turn("user_turn" if role == "user" else "agent_turn")
        turn.log(metadata={"contrib.pipecat.live.grouping": "pipecat_transcript_gap"})

    def _transcript(self, role: str, text: str, *, replace: bool = False, event: Any = None) -> None:
        if not text or role not in self.turns:
            return
        transcript = self.transcripts.get(role)
        if transcript is None:
            transcript = self.transcripts[role] = _Transcript(
                self.frame_spans.start(
                    "transcript_received",
                    turn=self.turns[role],
                    metadata={
                        SOURCE_ID: "session-input" if role == "user" else "agent-output",
                        "contrib.pipecat.live.grouping": "pipecat_transcript_gap",
                    },
                )
            )
        if replace:
            transcript.text = [text]
        else:
            transcript.text.append(text)
        # User interim frames carry the original event; assistant text frames
        # do not. Retain provider timestamps only when attached to this frame.
        start_ms, end_ms = getattr(event, "start_ms", None), getattr(event, "end_ms", None)
        if start_ms is not None:
            transcript.start_ms = start_ms if transcript.start_ms is None else min(transcript.start_ms, start_ms)
        if end_ms is not None:
            transcript.end_ms = end_ms if transcript.end_ms is None else max(transcript.end_ms, end_ms)
        transcript.span.log(output=[{"role": role, "content": "".join(transcript.text)}])
        if transcript.start_ms is not None and transcript.end_ms is not None:
            transcript.span.log(
                metadata={
                    "contrib.pipecat.live.start_ms": transcript.start_ms,
                    "contrib.pipecat.live.end_ms": transcript.end_ms,
                }
            )

    def _end_turn(self, role: str, *, incomplete: bool = False, end_ms: float | None = None) -> None:
        end_time = self.session.clock.wall(end_ms)
        transcript = self.transcripts.pop(role, None)
        if transcript is not None:
            message = {"role": role, "content": "".join(transcript.text)}
            if role == "assistant":
                self.outputs.append(message)
                self.model.log(output=self.outputs)
            else:
                self.inputs.append(message)
                self.model.log(input=self.inputs)
            transcript.span.end(end_time=end_time)
        turn = self.turns.pop(role, None)
        if turn is not None:
            if incomplete:
                turn.log(metadata={"turn.incomplete": True})
            turn.end(end_time=end_time)

    def client_event(self, event: Any) -> None:
        if self.closed or event.type != "response.item.create":
            return
        item = event.item
        kind = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if kind == "function_call_output":
            call_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
            # Observed submission, not server acknowledgment: Live doesn't
            # acknowledge these items. Never infer which continuation uses it.
            self.tool_result_submitted(call_id)

    def _response(self, envelope: Any) -> None:
        event = envelope.event
        kind = event.get("type", "")
        snapshot = event.get("response") or {}
        delegation = envelope.delegation_id or "uncorrelated"
        response_id = snapshot.get("id") or event.get("response_id") or self.delegation_responses.get(delegation)
        if not response_id:
            return
        self.delegation_responses[delegation] = response_id
        response = self.responses.get(response_id)
        if response is None:
            # Missing an observed start is preferable to inventing one.
            if kind != "response.created":
                return
            settings = getattr(getattr(self.service, "_delegation", None), "settings", None)
            metadata = {
                "provider": "openai",
                "model": snapshot.get("model") or getattr(settings, "model", None),
                "contrib.pipecat.live.response_id": response_id,
                "contrib.pipecat.live.delegation_id": envelope.delegation_id,
                "contrib.pipecat.live.input_partial": True,
            }
            if snapshot.get("previous_response_id"):
                metadata["contrib.pipecat.live.previous_response_id"] = snapshot["previous_response_id"]
            if delegation in self.delegation_offsets:
                # This locates delegation creation on the provider's session
                # timeline, including for subsequent responses in that delegation.
                metadata["contrib.pipecat.live.delegation_offset_ms"] = self.delegation_offsets[delegation]
            context = getattr(self.service, "_context", None)
            tools = tools_schema_to_openai_tools(getattr(context, "tools", None))
            if tools:
                metadata["tools"] = tools
            instruction = getattr(settings, "system_instruction", None)
            inputs = [{"role": "system", "content": instruction}] if isinstance(instruction, str) else None
            wrapper = self.spans.start("llm_response", input=inputs, metadata=metadata)
            llm = self.spans.start(
                "pipecat.live.backend_response",
                parent=wrapper,
                type=SpanTypeAttribute.LLM,
                input=inputs,
                metadata=metadata,
            )
            response = self.responses[response_id] = _Response(wrapper, llm, self.clock.now_ms())
        response.record_output(event)
        if not response.first_content and (
            (kind.endswith(".delta") and event.get("delta")) or kind == "response.output_item.done"
        ):
            response.first_content = True
            response.llm.log(metrics={"time_to_first_token": (self.clock.now_ms() - response.start_ms) / 1000})
        if kind == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call" and item.get("status") == "completed" and item.get("call_id"):
                call_id = item["call_id"]
                response.pending_calls.add(call_id)
                response.pending_tool_spans.add(call_id)
                self.tool_responses[call_id] = response_id
                self.session.agent.link_tool_request(call_id, response.wrapper)
        elif kind in ("response.completed", "response.failed", "response.incomplete"):
            if snapshot.get("model"):
                response.llm.log(metadata={"model": snapshot["model"]})
            response.llm.log(metrics=_parse_metrics_from_usage(snapshot.get("usage")))
            if kind != "response.completed":
                error = snapshot.get("error") or snapshot.get("incomplete_details") or {"status": kind}
                response.llm.log(error=error)
                response.wrapper.log(error=error)
            # Provider inference ends here. Pipecat may still be executing
            # tools and submitting their results as part of this response.
            response.llm.log(output=list(response.output.values()))
            response.llm.end(end_time=self.clock.wall())
            response.llm_finished = True

    def event_processed(self, envelope: Any) -> None:
        if self.closed or envelope.type != "response.event":
            return
        event = envelope.event
        if event.get("type") not in ("response.completed", "response.failed", "response.incomplete"):
            return
        response_id = (event.get("response") or {}).get("id") or self.delegation_responses.get(envelope.delegation_id)
        if response_id is None:
            return
        response = self.responses.get(response_id)
        if response is not None:
            response.processing_finished = True
            self._finish_if_ready(response_id, response)

    def tool_result_submitted(self, call_id: str) -> None:
        response_id = self.tool_responses.get(call_id)
        if response_id is None:
            return
        response = self.responses.get(response_id)
        if response is not None:
            response.pending_calls.discard(call_id)
            if call_id not in response.pending_tool_spans:
                self.tool_responses.pop(call_id, None)
            self._finish_if_ready(response_id, response)

    def tool_finished(self, call_id: str) -> None:
        response_id = self.tool_responses.get(call_id)
        if response_id is None:
            return
        response = self.responses.get(response_id)
        if response is not None:
            response.pending_tool_spans.discard(call_id)
            if call_id not in response.pending_calls:
                self.tool_responses.pop(call_id, None)
            self._finish_if_ready(response_id, response)

    def _finish_if_ready(self, response_id: str, response: _Response) -> None:
        if (
            response.llm_finished
            and response.processing_finished
            and not response.pending_calls
            and not response.pending_tool_spans
        ):
            self._end_response(response_id)

    def _end_response(self, response_id: str, *, incomplete: bool = False, end_ms: float | None = None) -> None:
        response = self.responses.pop(response_id)
        output = list(response.output.values())
        spans = (response.wrapper,) if response.llm_finished else (response.llm, response.wrapper)
        for span in spans:
            span.log(output=output)
            if incomplete:
                span.log(metadata={"contrib.pipecat.incomplete": True})
            span.end(end_time=self.clock.wall(end_ms))
        for call_id in response.pending_calls | response.pending_tool_spans:
            self.tool_responses.pop(call_id, None)

    def _end_model(self, *, confirmed: bool = False, end_ms: float | None = None) -> None:
        if self.model_closed:
            return
        self.model_closed = True
        self.model.log(metadata={"contrib.pipecat.live.finalization_confirmed": confirmed})
        self.model.end(end_time=self.clock.wall(end_ms))

    def close(self, end_ms: float) -> None:
        if self.closed:
            return
        self.closed = True
        for role in list(self.turns):
            self._end_turn(role, incomplete=True, end_ms=end_ms)
        for response_id in list(self.responses):
            self._end_response(response_id, incomplete=True, end_ms=end_ms)
        self._end_model(end_ms=end_ms)
