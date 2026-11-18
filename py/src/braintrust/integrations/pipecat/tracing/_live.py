"""Prototype GPT-Live tracing: continuous conversation and hosted Responses.

The voice service's LLM/TTS frames are transcript grouping notifications, not
provider calls. Public frames own transcript groups; native events own hosted
Responses. The ordinary observer still owns transport audio and speech. This
intentionally supports one Live service per pipeline and hosted Responses
delegation first.
"""

import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

from braintrust._audio.clock import SessionClock
from braintrust._audio.schema import SOURCE_ID
from braintrust.integrations.openai.tracing import _parse_metrics_from_usage
from braintrust.logger import Span, SpanTypeAttribute

from ._common import MAX_PENDING, SpanFactory, metadata_from_processor, tools_schema_to_openai_tools


# Prototype pricing bridge until the backend prices duration-based models.
# Public USD/minute rate verified 2026-11-16. Do not price unknown models.
_LIVE_USD_PER_MINUTE = {"gpt-live-1": 0.05}
_LIVE_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-live-1"


def is_live_service(processor: Any) -> bool:
    return any(base.__name__ == "OpenAILiveLLMService" for base in type(processor).__mro__)


class _LivePhase(Enum):
    ACTIVE = auto()
    DRAINING = auto()  # Remote model ended; queued transcript frames may still arrive.
    CLOSED = auto()  # Pipeline cleanup finished.


class _ResponsePhase(Enum):
    STREAMING = auto()
    PROCESSING = auto()  # Inference ended; Pipecat is handling its terminal event.
    WAITING_FOR_TOOLS = auto()


_Role = Literal["user", "assistant"]
_ToolSignal = Literal["submission", "span"]


@dataclass
class _Turn:
    span: Span
    transcript: Span | None = None
    text: list[str] = field(default_factory=list)
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass
class _Delegation:
    response_id: str | None = None
    offset_ms: int | None = None


@dataclass
class _PendingTool:
    response_id: str
    # Submission and observation run in different tasks and can arrive in
    # either order. Neither alone proves that the response task can close.
    waiting_for: set[_ToolSignal] = field(default_factory=lambda: {"submission", "span"})


@dataclass
class _Response:
    wrapper: Span
    llm: Span
    start_ms: float
    output: dict[str, Any] = field(default_factory=dict)
    first_content: bool = False
    phase: _ResponsePhase = _ResponsePhase.STREAMING
    pending_tools: set[str] = field(default_factory=set)

    @property
    def ready(self) -> bool:
        return self.phase is _ResponsePhase.WAITING_FOR_TOOLS and not self.pending_tools

    def finish_inference(self, kind: str, snapshot: dict[str, Any], end_time: float) -> None:
        if self.phase is not _ResponsePhase.STREAMING:
            return
        if snapshot.get("model"):
            self.llm.log(metadata={"model": snapshot["model"]})
        self.llm.log(metrics=_parse_metrics_from_usage(snapshot.get("usage")))
        if kind != "response.completed":
            error = snapshot.get("error") or snapshot.get("incomplete_details") or {"status": kind}
            self.llm.log(error=error)
            self.wrapper.log(error=error)
        self.llm.log(output=list(self.output.values()))
        self.llm.end(end_time=end_time)
        self.phase = _ResponsePhase.PROCESSING

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
    """Observe two independent lifecycles without making either one current.

    The remote model goes ACTIVE -> DRAINING on session.closed; only pipeline
    cleanup makes it CLOSED, after queued transcript frames have drained.

    Each hosted response goes STREAMING -> PROCESSING -> WAITING_FOR_TOOLS,
    then is removed once both tool submissions and tool spans have finished.
    Cleanup ends any remaining work as incomplete. Response/tool indexes own
    active work, and at most two transcript turns are open. Delegation history
    keeps recent IDs/offsets plus active responses' routing. Full session
    transcripts grow until cleanup by design.
    """

    def __init__(self, session: Any, service: Any) -> None:
        self.session, self.service = session, service
        session.agent.report_uninterrupted_playback = False
        # Native callbacks and frame callbacks run in different tasks. Do not
        # share the frame observer's temporarily pinned event clock.
        self.clock = SessionClock(session.recorder)
        self.spans = SpanFactory(session.span, self.clock)
        self.frame_spans = SpanFactory(session.span, session.clock)
        self.origin_ms = self.clock.now_ms()
        self.turns: dict[_Role, _Turn] = {}
        self.responses: dict[str, _Response] = {}
        self.delegations: OrderedDict[str, _Delegation] = OrderedDict()
        self.tools: dict[str, _PendingTool] = {}
        self.phase = _LivePhase.ACTIVE
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
        if self.phase is _LivePhase.CLOSED:
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
        elif kind in ("session.output_audio.delta", "session.output_transcript.delta"):
            if event.delta:
                self._first_content()
        elif kind == "session.delegation.created":
            delegation = self._delegation(event.delegation.id)
            delegation.offset_ms = event.offset_ms
            delegation.response_id = event.delegation.response_id
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

    def _delegation(self, delegation_id: str | None) -> _Delegation:
        key = delegation_id or "uncorrelated"
        if key not in self.delegations:
            self.delegations[key] = _Delegation()
        self.delegations.move_to_end(key)
        self._trim_delegations(keep=key)
        return self.delegations[key]

    def _trim_delegations(self, *, keep: str | None = None) -> None:
        # Live has no delegation-finished notification. Bound historical IDs
        # and offsets, while keeping routing needed by an active response.
        if len(self.delegations) <= MAX_PENDING:
            return
        for key, delegation in list(self.delegations.items()):
            if key != keep and delegation.response_id not in self.responses:
                del self.delegations[key]
                if len(self.delegations) <= MAX_PENDING:
                    break

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
        if self.phase is _LivePhase.CLOSED:
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

    def _open_turn(self, role: _Role) -> None:
        if role in self.turns:
            return
        turn = self.turns[role] = _Turn(self.frame_spans.turn("user_turn" if role == "user" else "agent_turn"))
        turn.span.log(metadata={"contrib.pipecat.live.grouping": "pipecat_transcript_gap"})

    def _transcript(self, role: _Role, text: str, *, replace: bool = False, event: Any = None) -> None:
        turn = self.turns.get(role)
        if not text or turn is None:
            return
        if turn.transcript is None:
            turn.transcript = self.frame_spans.start(
                "transcript_received",
                turn=turn.span,
                metadata={
                    SOURCE_ID: "session-input" if role == "user" else "agent-output",
                    "contrib.pipecat.live.grouping": "pipecat_transcript_gap",
                },
            )
        if replace:
            turn.text = [text]
        else:
            turn.text.append(text)
        # User interim frames carry the original event; assistant text frames
        # do not. Retain provider timestamps only when attached to this frame.
        start_ms, end_ms = getattr(event, "start_ms", None), getattr(event, "end_ms", None)
        if start_ms is not None:
            turn.start_ms = start_ms if turn.start_ms is None else min(turn.start_ms, start_ms)
        if end_ms is not None:
            turn.end_ms = end_ms if turn.end_ms is None else max(turn.end_ms, end_ms)
        turn.transcript.log(output=[{"role": role, "content": "".join(turn.text)}])
        if turn.start_ms is not None and turn.end_ms is not None:
            turn.transcript.log(
                metadata={
                    "contrib.pipecat.live.start_ms": turn.start_ms,
                    "contrib.pipecat.live.end_ms": turn.end_ms,
                }
            )

    def _end_turn(self, role: _Role, *, incomplete: bool = False, end_ms: float | None = None) -> None:
        turn = self.turns.pop(role, None)
        if turn is None:
            return
        end_time = self.session.clock.wall(end_ms)
        if turn.transcript is not None:
            message = {"role": role, "content": "".join(turn.text)}
            if role == "assistant":
                self.outputs.append(message)
                self.model.log(output=self.outputs)
            else:
                self.inputs.append(message)
                self.model.log(input=self.inputs)
            turn.transcript.end(end_time=end_time)
        if incomplete:
            turn.span.log(metadata={"turn.incomplete": True})
        turn.span.end(end_time=end_time)

    def client_event(self, event: Any) -> None:
        if self.phase is _LivePhase.CLOSED or event.type != "response.item.create":
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
        delegation = self._delegation(envelope.delegation_id)
        response_id = snapshot.get("id") or event.get("response_id") or delegation.response_id
        if not response_id:
            return
        delegation.response_id = response_id
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
            if delegation.offset_ms is not None:
                # This locates delegation creation on the provider's session
                # timeline, including for subsequent responses in that delegation.
                metadata["contrib.pipecat.live.delegation_offset_ms"] = delegation.offset_ms
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
                response.pending_tools.add(call_id)
                self.tools[call_id] = _PendingTool(response_id)
                self.session.agent.link_tool_request(call_id, response.wrapper)
        elif kind in ("response.completed", "response.failed", "response.incomplete"):
            # Provider inference ends here. Pipecat may still be executing
            # tools and submitting their results as part of this response.
            response.finish_inference(kind, snapshot, self.clock.wall())

    def event_processed(self, envelope: Any) -> None:
        if self.phase is _LivePhase.CLOSED or envelope.type != "response.event":
            return
        event = envelope.event
        if event.get("type") not in ("response.completed", "response.failed", "response.incomplete"):
            return
        response_id = (event.get("response") or {}).get("id") or self._delegation(envelope.delegation_id).response_id
        if response_id is None:
            return
        response = self.responses.get(response_id)
        if response is not None and response.phase is _ResponsePhase.PROCESSING:
            response.phase = _ResponsePhase.WAITING_FOR_TOOLS
            self._finish_if_ready(response_id, response)

    def tool_result_submitted(self, call_id: str) -> None:
        self._tool_progress(call_id, "submission")

    def tool_finished(self, call_id: str) -> None:
        self._tool_progress(call_id, "span")

    def _tool_progress(self, call_id: str, signal: _ToolSignal) -> None:
        tool = self.tools.get(call_id)
        if tool is None:
            return
        tool.waiting_for.discard(signal)
        if tool.waiting_for:
            return
        del self.tools[call_id]
        response = self.responses.get(tool.response_id)
        if response is not None:
            response.pending_tools.discard(call_id)
            self._finish_if_ready(tool.response_id, response)

    def _finish_if_ready(self, response_id: str, response: _Response) -> None:
        if response.ready:
            self._end_response(response_id)

    def _end_response(self, response_id: str, *, incomplete: bool = False, end_ms: float | None = None) -> None:
        response = self.responses.pop(response_id)
        output = list(response.output.values())
        spans = (response.llm, response.wrapper) if response.phase is _ResponsePhase.STREAMING else (response.wrapper,)
        for span in spans:
            span.log(output=output)
            if incomplete:
                span.log(metadata={"contrib.pipecat.incomplete": True})
            span.end(end_time=self.clock.wall(end_ms))
        for call_id in response.pending_tools:
            self.tools.pop(call_id, None)
        self._trim_delegations()

    def _end_model(self, *, confirmed: bool = False, end_ms: float | None = None) -> None:
        if self.phase is not _LivePhase.ACTIVE:
            return
        self.phase = _LivePhase.DRAINING
        self.model.log(metadata={"contrib.pipecat.live.finalization_confirmed": confirmed})
        self.model.end(end_time=self.clock.wall(end_ms))

    def close(self, end_ms: float) -> None:
        if self.phase is _LivePhase.CLOSED:
            return
        for role in list(self.turns):
            self._end_turn(role, incomplete=True, end_ms=end_ms)
        for response_id in list(self.responses):
            self._end_response(response_id, incomplete=True, end_ms=end_ms)
        self._end_model(end_ms=end_ms)
        self.phase = _LivePhase.CLOSED
        self.delegations.clear()
        self.tools.clear()
        self.inputs.clear()
        self.outputs.clear()
