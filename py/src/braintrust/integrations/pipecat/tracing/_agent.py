"""Agent turns and the generation, tool, synthesis and playback work they own."""

import json
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from braintrust._audio.clock import SessionClock
from braintrust._audio.schema import (
    PARTICIPANT_IDS,
    SELECTIONS,
    SOURCE_ID,
    SPEAKER_IDS,
    TURN_INCOMPLETE,
    TURN_REPLY_TO,
    set_links,
    span_id,
)
from braintrust._audio.session import AudioSessionRecorder
from braintrust._audio.types import selection
from braintrust.logger import Span, SpanTypeAttribute

from ._common import (
    MAX_PENDING,
    ProcessorKind,
    SpanFactory,
    SpeechText,
    metadata_from_processor,
    request_settings_from_processor,
    settings_from_processor,
)
from ._context import ContextLinker
from ._model import ModelOperation


if TYPE_CHECKING:
    from pipecat.frames.frames import (
        AudioRawFrame,
        Frame,
        FunctionCallInProgressFrame,
        FunctionCallResultFrame,
        TTSStartedFrame,
        TTSTextFrame,
    )
    from pipecat.processors.frame_processor import FrameProcessor


class _FunctionCall(TypedDict):
    name: str
    arguments: str


class _ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: _FunctionCall


# Work an agent turn owns. A turn ends only once no work remains open.
_Work = tuple[Literal["response", "tool", "tts", "playback", "followup"], str]
_RESPONSE: _Work = ("response", "")
_PLAYBACK: _Work = ("playback", "")
_FOLLOWUP: _Work = ("followup", "")


@dataclass
class _AgentTurn:
    span: Span
    work: set[_Work] = field(default_factory=set)
    completion_requested: bool = False


@dataclass
class _ModelResponse:
    span: Span
    metadata: dict[str, Any]
    processor: "FrameProcessor | None"
    text: list[str] = field(default_factory=list)
    tool_calls: list[_ToolCall] = field(default_factory=list)
    model: ModelOperation | None = None


@dataclass
class _Synthesis:
    span: Span
    speaker_ids: list[str]
    processor: "FrameProcessor | None"
    text: SpeechText = field(default_factory=SpeechText)
    ended: bool = False
    start_observed: bool = False
    models: list[ModelOperation] = field(default_factory=list)

    @property
    def work(self) -> _Work:
        return ("tts", span_id(self.span))


@dataclass
class _Playback:
    # BotStartedSpeakingFrame precedes the write attempt. Create the speech
    # span and set its start only after the transport confirms an audio frame.
    span: Span | None = None
    start_ms: float = 0.0
    audio_end_ms: float = 0.0
    text: SpeechText = field(default_factory=SpeechText)
    generators: dict[str, _Synthesis] = field(default_factory=dict)


_DEFAULT_CONTEXT = "__default__"


class AgentTracker:
    """Group work on routes with a native Pipecat assistant aggregator.

    That aggregator groups full-response frames and publishes completion.
    Outstanding tools/followups and synthesis/playback keep their established
    group open. Bare response boundaries never invent conversational turns.
    Interruption ends all work as interrupted and the turn as incomplete; audio
    the transport still plays from the interrupted reply opens no speech.
    """

    def __init__(self, spans: SpanFactory, clock: SessionClock, recorder: AudioSessionRecorder) -> None:
        self._spans, self._clock, self._recorder = spans, clock, recorder
        self.report_uninterrupted_playback = True
        self._turn: _AgentTurn | None = None
        self.native_turn_processors: set[int] = set()
        self._response: _ModelResponse | None = None
        self._playback: _Playback | None = None
        # An interruption ended the agent turn and no agent work has started
        # since. Audio the transport plays now is left over from that reply.
        self._interrupted_reply = False
        # TTS context ID -> synthesis. Several IDs may alias one synthesis.
        self._synthesis: dict[str, _Synthesis] = {}
        self._tools: dict[str, Span] = {}
        self._requesters: dict[str, Span] = {}
        # Pipecat broadcasts tool frames; ignore copies of finished calls.
        self._finished_tools: deque[str] = deque(maxlen=MAX_PENDING)

    # Model responses

    def start_response(
        self,
        processor: "FrameProcessor | None",
        linker: ContextLinker,
        *,
        covered: bool | None = None,
        pending_span: Span | None = None,
    ) -> Span | None:
        """Start a model response; return its span, or None if one is active.

        The task contains either a provider-owned model span or an uncovered
        client's Pipecat model span. Model telemetry belongs to that child.
        """
        if self._response is not None:
            return None
        self._interrupted_reply = False
        turn = self._turn
        if turn is None and id(processor) in self.native_turn_processors:
            turn = self._turn = _AgentTurn(self._spans.turn("agent_turn"))
        # A response supersedes completion from an earlier utterance in a tool loop.
        if turn is not None:
            turn.completion_requested = False
            turn.work.discard(_FOLLOWUP)
        metadata = {
            **linker.metadata,
            **request_settings_from_processor(processor),
            **metadata_from_processor(processor),
        }
        messages = linker.messages
        instruction = getattr(settings_from_processor(processor), "system_instruction", None)
        if isinstance(instruction, str) and instruction:
            # Service settings are separate from LLMContext in current Pipecat.
            # Keep the linker's context/cursor untouched when adding them.
            messages = [{"role": "system", "content": instruction}, *(messages or [])]
        span = self._spans.start(
            "llm_response",
            turn=turn.span if turn else None,
            input=messages,
            metadata=metadata,
            can_set_current=True,
            **({"id": pending_span.id, "internal": {"span_id": span_id(pending_span)}} if pending_span else {}),
        )
        links = linker.consume()
        if turn is not None and linker.reply_to is not None:
            turn.span.log(metadata={TURN_REPLY_TO: span_id(linker.reply_to)})
        if links:
            set_links(span, links)
        model = ModelOperation.start(self._spans, span, processor, "llm_response", covered=covered, input=messages)
        model.request(linker.metadata)
        self._response = _ModelResponse(span, metadata, processor, model=model)
        self._open(_RESPONSE)
        return span

    def response_text(self, text: str) -> None:
        if self._response is not None:
            self._response.text.append(text)
            if self._response.model is not None:
                self._response.model.content()

    def request_tools(self, calls: list[Any]) -> None:
        response = self._response
        if response is None:
            return
        for call in calls:
            # broadcast_frame creates distinct upstream/downstream frames with
            # the same call IDs, so frame-level deduplication cannot catch this.
            if any(existing["id"] == call.tool_call_id for existing in response.tool_calls):
                continue
            response.tool_calls.append(
                {
                    "id": call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": call.function_name,
                        "arguments": call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments),
                    },
                }
            )
            self._requesters[call.tool_call_id] = response.span
            self._open(("tool", call.tool_call_id))

    def end_response(self) -> None:
        response = self._response
        if response is None:
            return
        self._end_response(response)
        self._close(_RESPONSE)

    # Tools

    def link_tool_request(self, call_id: str, requester: Span) -> None:
        """Associate a native Live backend request with its later tool frame."""
        # Pipecat owns the tool's execution and result submission as part of
        # handling this response, even when execution uses another task.
        self._requesters[call_id] = requester

    def start_tool(self, frame: "FunctionCallInProgressFrame | FunctionCallResultFrame") -> None:
        key = frame.tool_call_id
        if key in self._tools or key in self._finished_tools:
            return
        requester = self._requesters.pop(key, None)
        span = self._spans.start(
            frame.function_name,
            turn=self._turn.span if self._turn is not None else None,
            parent=requester,
            type=SpanTypeAttribute.TOOL,
            input=frame.arguments,
            metadata={"gen_ai.tool.call.id": key},
        )
        if requester is not None:
            set_links(span, [("requested_by", requester)])
        self._tools[key] = span
        self._open(("tool", key))

    def finish_tool(self, frame: "FunctionCallResultFrame", linker: ContextLinker) -> None:
        key = frame.tool_call_id
        if key in self._finished_tools:
            return
        self.start_tool(frame)
        span = self._tools[key]
        span.log(output=frame.result)
        error = getattr(frame, "error", None)
        if error is not None:
            span.log(error=error)
        properties = getattr(frame, "properties", None)
        if not getattr(properties, "is_final", True):
            return
        del self._tools[key]
        span.end(end_time=self._clock.wall())
        self._finished_tools.append(key)
        linker.add_tool_result(key, span)
        run_llm = getattr(properties, "run_llm", None)
        if run_llm is None:
            run_llm = getattr(frame, "run_llm", None)
        if frame.result and run_llm is not False and self._turn is not None:
            # Open the follow-up before closing the tool, so the turn survives.
            self._turn.completion_requested = False
            self._open(_FOLLOWUP)
        self._close(("tool", key))

    def cancel_tool(self, tool_call_id: str) -> None:
        self._requesters.pop(tool_call_id, None)
        span = self._tools.pop(tool_call_id, None)
        self._finished_tools.append(tool_call_id)
        if span is not None:
            span.log(metadata={"contrib.pipecat.cancelled": True})
            span.end(end_time=self._clock.wall())
        self._close(("tool", tool_call_id))

    # Synthesis

    def start_tts(
        self,
        frame: "TTSStartedFrame",
        processor: "FrameProcessor | None",
        *,
        generator: Span | None = None,
        observed: bool = True,
    ) -> _Synthesis:
        self._interrupted_reply = False
        key = frame.context_id or _DEFAULT_CONTEXT
        existing = self._synthesis.get(key)
        if existing is not None and (not existing.ended or (observed and not existing.start_observed)):
            # The queued start can arrive after the executing service has
            # completed. Consume that observation once, without reopening it.
            existing.start_observed |= observed
            return existing
        metadata = {**metadata_from_processor(processor), "contrib.pipecat.context_id": frame.context_id}
        voice = getattr(settings_from_processor(processor), "voice", None)
        speaker_ids: list[str] = []
        if isinstance(voice, str):
            speaker = self._recorder.speaker("agent-output", voice, {"type": "agent", "label": voice})
            if speaker is not None:
                speaker_ids.append(speaker)
                metadata[SPEAKER_IDS] = speaker_ids
        span = self._spans.start(
            "tts", turn=self._turn.span if self._turn else None, metadata=metadata, can_set_current=True
        )
        if generator is not None:
            set_links(span, [("generated_by", generator)])
        synthesis = _Synthesis(span, speaker_ids, processor, start_observed=observed)
        self._synthesis[key] = synthesis
        # Bare pipelines have no native turn completion at which to prune.
        # Keep active work and the current playback's observed associations;
        # only completed routing history is bounded.
        if len(self._synthesis) > MAX_PENDING:
            playing = self._playback.generators if self._playback is not None else {}
            for old_key, old in list(self._synthesis.items()):
                if len(self._synthesis) <= MAX_PENDING:
                    break
                if old.ended and span_id(old.span) not in playing:
                    del self._synthesis[old_key]
        self._open(synthesis.work)
        return synthesis

    def tts_text(self, frame: "TTSTextFrame") -> None:
        synthesis = self._resolve_synthesis(frame)
        if synthesis is not None:
            synthesis.text.append(frame)
            synthesis.span.log(input={"text": str(synthesis.text)})

    def stop_tts(self, frame: "Frame") -> None:
        synthesis = self._resolve_synthesis(frame)
        if synthesis is not None:
            for model in synthesis.models:
                if model.deferred:
                    model.complete = model.error is None
            self._end_synthesis(synthesis)
            self._close(synthesis.work)

    def _resolve_synthesis(self, frame: "Frame") -> _Synthesis | None:
        """Resolve only an observed native context, including the no-ID context."""
        synthesis = self._synthesis.get(getattr(frame, "context_id", None) or _DEFAULT_CONTEXT)
        return synthesis if synthesis is not None and not synthesis.ended else None

    # Playback

    def start_playback(self) -> None:
        if self._playback is not None:
            return
        if self._interrupted_reply and self._turn is None:
            # Pipecat keeps its audio task running through an interruption and
            # still writes the chunk it had dequeued, restarting bot speech.
            # That audio belongs to the interrupted reply, whose speech ended
            # at the interruption; it must not become speech without a turn.
            return
        self._playback = _Playback()
        self._open(_PLAYBACK)

    def played_audio(self, frame: "AudioRawFrame") -> None:
        playback = self._playback
        if playback is None:
            return
        playback.audio_end_ms = max(playback.audio_end_ms, self._clock.now_ms()) + (
            len(frame.audio) * 1000 / (2 * frame.num_channels * frame.sample_rate)
        )
        span = playback.span
        if span is None:
            playback.start_ms = self._clock.now_ms()
            participant = self._recorder.participant(
                "agent-connection", {"type": "agent", "label": "Agent connection"}
            )
            metadata: dict[str, Any] = {SOURCE_ID: "agent-output"}
            if participant is not None:
                metadata[PARTICIPANT_IDS] = [participant]
            span = playback.span = self._spans.start(
                "agent_speech",
                turn=self._turn.span if self._turn else None,
                start_ms=playback.start_ms,
                metadata=metadata,
            )
        context_id = getattr(frame, "context_id", None)
        synthesis = self._synthesis.get(context_id) if context_id else None
        self._link_playback_synthesis(playback, synthesis)

    def _link_playback_synthesis(self, playback: _Playback, synthesis: _Synthesis | None) -> None:
        span = playback.span
        if span is None or synthesis is None or span_id(synthesis.span) in playback.generators:
            return
        playback.generators[span_id(synthesis.span)] = synthesis
        speakers = sorted({speaker for item in playback.generators.values() for speaker in item.speaker_ids})
        if speakers:
            span.log(metadata={SPEAKER_IDS: speakers})
        # Links are replaced on each update: always send the complete set.
        set_links(span, [("generated_by", item.span) for item in playback.generators.values()])

    def played_text(self, frame: "TTSTextFrame") -> None:
        if self._playback is not None:
            self._playback.text.append(frame)
            # The transport forwards the context-tagged text only as it plays
            # that context. Its rechunked audio can have lost the original ID.
            self._link_playback_synthesis(self._playback, self._synthesis.get(getattr(frame, "context_id", None)))

    def stop_playback(self) -> None:
        if self._playback is not None:
            self._end_playback(self._playback, interrupted=False)
        self._close(_PLAYBACK)

    # Turn completion

    def request_completion(self) -> None:
        if self._turn is not None:
            self._turn.completion_requested = True
            self._try_finish()

    def fail(self, processor: "FrameProcessor | None", kind: ProcessorKind | None, error: object) -> None:
        response = self._response
        if kind == "llm" and response is not None and response.processor is processor:
            response.span.log(error=error)
            if response.model is not None:
                response.model.error = error
            self._end_response(response)
            self._close(_RESPONSE)
        elif kind == "tts":
            for synthesis in {id(item): item for item in self._synthesis.values()}.values():
                if not synthesis.ended and synthesis.processor is processor:
                    synthesis.span.log(error=error)
                    for model in synthesis.models:
                        model.error = error
                    self._end_synthesis(synthesis)
                    self._close(synthesis.work)
        elif kind == "output" and self._playback is not None and self._playback.span is not None:
            self._playback.span.log(error=error)

    def interrupt(self, *, interrupted: bool = True, end_ms: float | None = None) -> None:
        """End all agent work: interrupted by the user, or incomplete at shutdown."""
        status = "contrib.pipecat.interrupted" if interrupted else "contrib.pipecat.incomplete"
        end_time = self._clock.wall(end_ms)
        # Without an assistant aggregator a reply has no turn, but the transport
        # still writes its dequeued audio after an interruption just the same.
        ended_reply = (
            self._turn is not None
            or self._playback is not None
            or self._response is not None
            or any(not synthesis.ended for synthesis in self._synthesis.values())
        )
        if ended_reply:
            self._interrupted_reply = interrupted
        if self._playback is not None:
            if self._playback.span is not None:
                self._playback.span.log(metadata={status: True})
            self._end_playback(
                self._playback, interrupted=True if interrupted else None, end_ms=end_ms, shutdown=not interrupted
            )
        if self._response is not None:
            self._response.span.log(metadata={status: True})
            self._end_response(self._response, end_ms=end_ms)
        for synthesis in self._synthesis.values():
            if not synthesis.ended:
                synthesis.span.log(metadata={status: True})
                self._end_synthesis(synthesis, end_ms=end_ms)
        self._synthesis.clear()
        for span in self._tools.values():
            span.log(metadata={status: True})
            span.end(end_time=end_time)
        self._tools.clear()
        self._requesters.clear()
        self._finished_tools.clear()
        if self._turn is not None:
            self._finish_turn(self._turn, incomplete=True, end_ms=end_ms)

    def _open(self, work: _Work) -> None:
        if self._turn is not None:
            self._turn.work.add(work)

    def _close(self, work: _Work) -> None:
        if self._turn is not None:
            self._turn.work.discard(work)
            self._try_finish()

    def _try_finish(self) -> None:
        turn = self._turn
        if turn is not None and turn.completion_requested and not turn.work:
            self._finish_turn(turn, incomplete=False)

    def _finish_turn(self, turn: _AgentTurn, *, incomplete: bool, end_ms: float | None = None) -> None:
        turn.span.log(metadata={TURN_INCOMPLETE: incomplete})
        turn.span.end(end_time=self._clock.wall(end_ms))
        self._turn = None
        self._synthesis = {key: item for key, item in self._synthesis.items() if not item.ended}

    def _end_response(self, response: _ModelResponse, *, end_ms: float | None = None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(response.text) or None}
        if response.tool_calls:
            message["tool_calls"] = response.tool_calls
        response.span.log(
            output=[message],
            metadata=response.metadata,
        )
        if response.model is not None:
            response.model.finish(end_time=self._clock.wall(end_ms), output=[message], error=response.model.error)
        response.span.end(end_time=self._clock.wall(end_ms))
        self._response = None

    def _end_synthesis(self, synthesis: _Synthesis, *, end_ms: float | None = None) -> None:
        for model in synthesis.models:
            model.finish(end_time=self._clock.wall(end_ms), output={"content": []}, error=model.error)
        synthesis.span.end(end_time=self._clock.wall(end_ms))
        synthesis.ended = True
        synthesis.text.parts.clear()

    def _end_playback(
        self, playback: _Playback, *, interrupted: bool | None, end_ms: float | None = None, shutdown: bool = False
    ) -> None:
        if playback.span is None:
            self._playback = None
            return
        end_ms = self._clock.now_ms() if end_ms is None else end_ms
        # Cleanup may wait for remote closure after the transport stops sending.
        # Keep the observed span lifetime, but don't select teardown as speech.
        selection_end = min(end_ms, playback.audio_end_ms) if shutdown else end_ms
        metadata: dict[str, Any] = {SELECTIONS: {"selection-0": selection(playback.start_ms, selection_end)}}
        # Live handles automatic interruption remotely without notifying
        # Pipecat. An explicit local interruption is still observable.
        if interrupted is not None and (interrupted or self.report_uninterrupted_playback):
            metadata["contrib.pipecat.interrupted"] = interrupted
        playback.span.log(metadata=metadata)
        if playback.text.parts:
            playback.span.log(output=[{"role": "assistant", "content": str(playback.text)}])
        playback.span.end(end_time=self._clock.wall(end_ms))
        self._playback = None
