"""Agent turns and the generation, tool, synthesis and playback work they own."""

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
    metadata_from_processor,
    processor_kind,
    provider_from_name,
    request_settings_from_processor,
)
from ._context import ContextLinker


if TYPE_CHECKING:
    from pipecat.frames.frames import (
        AudioRawFrame,
        Frame,
        FunctionCallInProgressFrame,
        FunctionCallResultFrame,
        MetricsFrame,
        TTSStartedFrame,
    )
    from pipecat.metrics import metrics
    from pipecat.processors.frame_processor import FrameProcessor
else:
    try:
        from pipecat.metrics import metrics
    except ImportError:  # pragma: no cover - Pipecat is an optional dependency.
        metrics = None


class _FunctionCall(TypedDict):
    name: str
    arguments: object


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
    metrics: dict[str, int | float] = field(default_factory=dict)


@dataclass
class _Synthesis:
    span: Span
    speaker_ids: list[str]
    processor: "FrameProcessor | None"
    text: list[str] = field(default_factory=list)
    characters: int = 0
    ended: bool = False

    @property
    def work(self) -> _Work:
        return ("tts", span_id(self.span))


@dataclass
class _Playback:
    # BotStartedSpeakingFrame precedes the write attempt. Create the speech
    # span and set its start only after the transport confirms an audio frame.
    span: Span | None = None
    start_ms: float = 0.0
    text: list[str] = field(default_factory=list)
    generators: dict[str, _Synthesis] = field(default_factory=dict)


_DEFAULT_CONTEXT = "__default__"


class AgentTracker:
    """One agent turn at a time, owning all concurrent agent work.

    A turn opens with its first model response. Each response, tool call,
    synthesis and playback opens work on the turn and closes it on completion.
    The turn ends once its work has drained after a native completion boundary:

    - assistant context completion (``LLMContextAssistantTurnFrame``),
    - playback stop (``BotStoppedSpeakingFrame``), or
    - a model response end, in a pipeline without an output transport.

    A tool result that runs the model opens follow-up work and withdraws any
    earlier completion; the next response consumes it. Interruption ends all
    work as interrupted and the turn as incomplete. User turns are independent:
    starting one does not imply an interruption.
    """

    def __init__(self, spans: SpanFactory, clock: SessionClock, recorder: AudioSessionRecorder) -> None:
        self._spans, self._clock, self._recorder = spans, clock, recorder
        self._turn: _AgentTurn | None = None
        self._response: _ModelResponse | None = None
        self._last_response: Span | None = None
        self._playback: _Playback | None = None
        # TTS context ID -> synthesis. Several IDs may alias one synthesis.
        self._synthesis: dict[str, _Synthesis] = {}
        self._latest_synthesis: _Synthesis | None = None
        # TTS service -> characters it reported before its synthesis started.
        self._pending_characters: dict[int, int] = {}
        self._tools: dict[str, Span] = {}
        self._requesters: dict[str, Span] = {}
        # Pipecat broadcasts tool frames; ignore copies of finished calls.
        self._finished_tools: deque[str] = deque(maxlen=MAX_PENDING)

    # Model responses

    def start_response(
        self, processor: "FrameProcessor | None", linker: ContextLinker, reply_to: Span | None
    ) -> Span | None:
        """Start a model response; return its span, or None if one is active.

        The span is a task: the provider integration's span nests under it as
        the only LLM span and owns token usage, so trace totals count each
        model call, its tokens and its duration once.
        """
        if self._response is not None:
            return None
        turn = self._turn
        if turn is None:
            turn = self._turn = _AgentTurn(self._spans.turn("agent_turn"))
            if reply_to is not None:
                turn.span.log(metadata={TURN_REPLY_TO: span_id(reply_to)})
        # A response supersedes completion from an earlier utterance in a tool loop.
        turn.completion_requested = False
        turn.work.discard(_FOLLOWUP)
        metadata = {
            **linker.metadata,
            **request_settings_from_processor(processor),
            **metadata_from_processor(processor),
        }
        span = self._spans.start(
            "llm_response", turn=turn.span, input=linker.messages, metadata=metadata, can_set_current=True
        )
        links = linker.consume()
        if links:
            set_links(span, links)
        self._response = _ModelResponse(span, metadata, processor)
        turn.work.add(_RESPONSE)
        return span

    def response_text(self, text: str) -> None:
        if self._response is not None:
            self._response.text.append(text)

    def request_tools(self, calls: list[Any]) -> None:
        response = self._response
        if response is None:
            return
        for call in calls:
            response.tool_calls.append(
                {
                    "id": call.tool_call_id,
                    "type": "function",
                    "function": {"name": call.function_name, "arguments": call.arguments},
                }
            )
            self._requesters[call.tool_call_id] = response.span
            self._open(("tool", call.tool_call_id))

    def end_response(self, *, completes_turn: bool) -> None:
        response = self._response
        if response is None:
            if completes_turn:
                self.request_completion()
            return
        self._end_response(response)
        if completes_turn:
            self.request_completion()
        self._close(_RESPONSE)

    def record_metrics(self, frame: "MetricsFrame", processor: "FrameProcessor | None") -> None:
        if processor_kind(type(processor)) == "tts":
            self._record_synthesis_usage(frame, processor)
            return
        response = self._response
        if response is None or processor_kind(type(processor)) != "llm" or response.processor is not processor:
            return
        for metric in frame.data or []:
            if getattr(metric, "processor", None) != processor.name:
                continue
            if isinstance(metric, metrics.LLMUsageMetricsData):
                # Token usage belongs to the provider span nested under this
                # one; logging it here would double count trace totals.
                if metric.model is not None:
                    response.metadata["model"] = metric.model
                provider = provider_from_name(metric.processor)
                if provider is not None:
                    response.metadata["provider"] = provider
            elif isinstance(metric, metrics.TTFBMetricsData):
                response.metrics["time_to_first_token"] = metric.value
        response.metadata.update(metadata_from_processor(processor))

    # Tools

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

    def start_tts(self, frame: "TTSStartedFrame", processor: "FrameProcessor | None") -> None:
        key = frame.context_id or _DEFAULT_CONTEXT
        existing = self._synthesis.get(key)
        if existing is not None and not existing.ended:
            return
        metadata = {**metadata_from_processor(processor), "contrib.pipecat.context_id": frame.context_id}
        voice = getattr(getattr(processor, "_settings", None), "voice", None)
        speaker_ids: list[str] = []
        if isinstance(voice, str):
            speaker = self._recorder.speaker("agent-output", voice, {"type": "agent", "label": voice})
            if speaker is not None:
                speaker_ids.append(speaker)
                metadata[SPEAKER_IDS] = speaker_ids
        span = self._spans.start("tts", turn=self._turn.span if self._turn else None, metadata=metadata)
        generator = self._response.span if self._response is not None else self._last_response
        if generator is not None:
            set_links(span, [("generated_by", generator)])
        synthesis = _Synthesis(span, speaker_ids, processor)
        self._synthesis[key] = self._latest_synthesis = synthesis
        self._open(synthesis.work)
        pending = self._pending_characters.pop(id(processor), 0)
        if pending:
            self._add_characters(synthesis, pending)

    def tts_text(self, frame: "Frame") -> None:
        synthesis = self._resolve_synthesis(frame, adopt=True)
        if synthesis is not None:
            synthesis.text.append(frame.text)
            synthesis.span.log(input={"text": "".join(synthesis.text)})

    def stop_tts(self, frame: "Frame") -> None:
        synthesis = self._resolve_synthesis(frame)
        if synthesis is not None:
            self._end_synthesis(synthesis)
            self._close(synthesis.work)

    def _record_synthesis_usage(self, frame: "MetricsFrame", processor: "FrameProcessor | None") -> None:
        """Count the characters a TTS service reports for its active synthesis.

        Services report usage per chunk of text they synthesize, so counts
        accumulate. Pipecat queues a synthesis's start frame behind its audio,
        so usage can arrive first; it then waits for the service's next
        synthesis. No provider integration traces speech synthesis, so this
        span is the only place the usage is recorded.
        """
        name = getattr(processor, "name", None)
        characters = sum(
            metric.value
            for metric in frame.data or []
            if isinstance(metric, metrics.TTSUsageMetricsData)
            and getattr(metric, "processor", None) == name
            and isinstance(metric.value, int)
            and not isinstance(metric.value, bool)
        )
        if not characters:
            return
        active = [item for item in self._synthesis.values() if not item.ended and item.processor is processor]
        if not active:
            self._pending_characters[id(processor)] = self._pending_characters.get(id(processor), 0) + characters
            return
        self._add_characters(active[-1], characters)

    def _add_characters(self, synthesis: _Synthesis, characters: int) -> None:
        synthesis.characters += characters
        synthesis.span.log(metrics={"characters": synthesis.characters})

    def _resolve_synthesis(self, frame: "Frame", *, adopt: bool = False) -> _Synthesis | None:
        """Resolve a frame's synthesis; a known but ended context never falls back.

        Some services tag only part of a synthesis's frames with a context ID.
        An untagged frame resolves to the sole active synthesis; with ``adopt``,
        an unknown ID is bound to the sole active synthesis for later frames.
        """
        context_id = getattr(frame, "context_id", None)
        key = context_id or _DEFAULT_CONTEXT
        if key in self._synthesis:
            synthesis = self._synthesis[key]
            return None if synthesis.ended else synthesis
        if context_id and not adopt:
            return None
        active = {id(item): item for item in self._synthesis.values() if not item.ended}
        if len(active) != 1:
            return None
        synthesis = next(iter(active.values()))
        if context_id:
            self._synthesis[context_id] = synthesis
        return synthesis

    # Playback

    def start_playback(self) -> None:
        if self._playback is not None:
            return
        self._playback = _Playback()
        self._open(_PLAYBACK)

    def played_audio(self, frame: "AudioRawFrame") -> None:
        playback = self._playback
        if playback is None:
            return
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
        synthesis = self._synthesis.get(context_id) if context_id else self._latest_synthesis
        if synthesis is None or span_id(synthesis.span) in playback.generators:
            return
        playback.generators[span_id(synthesis.span)] = synthesis
        speakers = sorted({speaker for item in playback.generators.values() for speaker in item.speaker_ids})
        if speakers:
            span.log(metadata={SPEAKER_IDS: speakers})
        # Links are replaced on each update: always send the complete set.
        set_links(span, [("generated_by", item.span) for item in playback.generators.values()])

    def played_text(self, text: str) -> None:
        if self._playback is not None:
            self._playback.text.append(text)

    def stop_playback(self) -> None:
        if self._playback is not None:
            self._end_playback(self._playback, interrupted=False)
        self.request_completion()
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
            self._end_response(response)
            self._close(_RESPONSE)
        elif kind == "tts":
            for synthesis in {id(item): item for item in self._synthesis.values()}.values():
                if not synthesis.ended and synthesis.processor is processor:
                    synthesis.span.log(error=error)
                    self._end_synthesis(synthesis)
                    self._close(synthesis.work)
        elif kind == "output" and self._playback is not None and self._playback.span is not None:
            self._playback.span.log(error=error)

    def interrupt(self, *, interrupted: bool = True, end_ms: float | None = None) -> None:
        """End all agent work: interrupted by the user, or incomplete at shutdown."""
        status = "contrib.pipecat.interrupted" if interrupted else "contrib.pipecat.incomplete"
        end_time = self._clock.wall(end_ms)
        if self._playback is not None:
            self._end_playback(self._playback, interrupted=interrupted, end_ms=end_ms)
        if self._response is not None:
            self._response.span.log(metadata={status: True})
            self._end_response(self._response, end_ms=end_ms)
        for synthesis in self._synthesis.values():
            if not synthesis.ended:
                synthesis.span.log(metadata={status: True})
                self._end_synthesis(synthesis, end_ms=end_ms)
        self._synthesis.clear()
        self._latest_synthesis = None
        self._pending_characters.clear()
        for span in self._tools.values():
            span.log(metadata={status: True})
            span.end(end_time=end_time)
        self._tools.clear()
        self._requesters.clear()
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
        self._last_response = None
        self._synthesis = {key: item for key, item in self._synthesis.items() if not item.ended}
        if self._latest_synthesis is not None and self._latest_synthesis.ended:
            self._latest_synthesis = None

    def _end_response(self, response: _ModelResponse, *, end_ms: float | None = None) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(response.text) or None}
        if response.tool_calls:
            message["tool_calls"] = response.tool_calls
        response.span.log(
            output=[message],
            metadata=response.metadata,
            metrics=response.metrics,
        )
        response.span.end(end_time=self._clock.wall(end_ms))
        self._last_response, self._response = response.span, None

    def _end_synthesis(self, synthesis: _Synthesis, *, end_ms: float | None = None) -> None:
        synthesis.span.end(end_time=self._clock.wall(end_ms))
        synthesis.ended = True
        synthesis.text.clear()

    def _end_playback(self, playback: _Playback, *, interrupted: bool, end_ms: float | None = None) -> None:
        if playback.span is None:
            self._playback = None
            return
        end_ms = self._clock.now_ms() if end_ms is None else end_ms
        playback.span.log(
            metadata={
                SELECTIONS: {"selection-0": selection(playback.start_ms, end_ms)},
                "contrib.pipecat.interrupted": interrupted,
            }
        )
        if playback.text:
            playback.span.log(output=[{"role": "assistant", "content": "".join(playback.text)}])
        playback.span.end(end_time=self._clock.wall(end_ms))
        self._playback = None
