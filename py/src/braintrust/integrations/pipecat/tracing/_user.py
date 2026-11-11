"""User turns, speech intervals and speech recognition."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from braintrust._audio.clock import SessionClock
from braintrust._audio.schema import PARTICIPANT_IDS, SELECTIONS, SOURCE_ID, TURN_INCOMPLETE, set_links
from braintrust._audio.session import AudioSessionRecorder
from braintrust._audio.types import Selection, selection
from braintrust.logger import Span

from ._common import MAX_PENDING, SpanFactory, metadata_from_processor
from ._context import ContextLinker


if TYPE_CHECKING:
    from pipecat.frames.frames import Frame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameProcessor


@dataclass
class _UserTurn:
    span: Span
    selections: list[Selection] = field(default_factory=list)


@dataclass
class _UserSpeech:
    start_ms: float
    # Created lazily: VAD detects speech before the aggregator starts its turn.
    span: Span | None = None
    participants: set[str] = field(default_factory=set)


class _RecognitionPhase(Enum):
    AWAITING_TURN = "awaiting_turn"
    ACTIVE = "active"
    TRANSCRIBED = "transcribed"


@dataclass
class _Recognition:
    processor: "FrameProcessor"
    start_ms: float
    speech: _UserSpeech | None
    turn_index: int
    phase: _RecognitionPhase = _RecognitionPhase.AWAITING_TURN
    span: Span | None = None
    transcript: "tuple[TranscriptionFrame, float] | None" = None


class UserTracker:
    """At most one user turn and one speech interval; neither owns agent work.

    VAD usually detects speech before the user aggregator starts its turn, so
    speech and recognition spans are created once their turn is known: when
    it starts, or when they end without one. Provider execution may open a
    recognition earlier so its spans have an active parent.

    A recognition is per STT processor. It starts when the processor consumes
    audio during VAD speech (or, in a pipeline without VAD, during a user turn)
    and ends with its transcript. Its span waits for a native user turn, the
    aggregator may only start that turn from the transcript itself:

        AWAITING_TURN --turn starts--> ACTIVE --transcript--> emitted
        AWAITING_TURN --transcript--> TRANSCRIBED --turn or model request--> emitted
        AWAITING_TURN | ACTIVE --speech without a transcript, then new speech--> retired

    Pauses within one turn keep a recognition; speech that produced no
    transcript (e.g. a cough) does not leak into the next utterance.
    """

    def __init__(
        self, spans: SpanFactory, clock: SessionClock, recorder: AudioSessionRecorder, linker: ContextLinker
    ) -> None:
        self._spans, self._clock, self._recorder, self._linker = spans, clock, recorder, linker
        self._turn: _UserTurn | None = None
        self._speech: _UserSpeech | None = None
        self._last_turn: Span | None = None
        self._last_speech: Span | None = None
        self._turn_index = 0
        self._vad_seen = False
        self._recognitions: dict[int, _Recognition] = {}
        self._transcribed: deque[_Recognition] = deque()
        # Per STT processor: the speech its last transcript covered, and when.
        self._recognized_until: dict[int, tuple[_UserSpeech | None, float]] = {}

    @property
    def reply_target(self) -> Span | None:
        """The user turn an agent response replies to; ambiguous while one is open."""
        return self._last_turn if self._turn is None else None

    @property
    def _turn_span(self) -> Span | None:
        return self._turn.span if self._turn is not None else None

    @property
    def _context_turn(self) -> Span | None:
        return self._turn_span or self._last_turn

    def start_turn(self) -> None:
        if self._turn is None:
            self._turn = _UserTurn(self._spans.turn("user_turn"))
            if self._speech is not None:
                self._speech_span(self._speech)
            self._resolve_recognitions()

    def stop_turn(self) -> None:
        self._resolve_recognitions()
        # Recognition without a final transcript must not keep timing through
        # the agent's reply or into the next user turn. A later transcript can
        # still be represented as transcript_received.
        for recognition in list(self._recognitions.values()):
            self._retire(recognition)
        self._finish_turn(incomplete=False)

    def log_input(self, message: dict[str, Any]) -> None:
        turn = self._context_turn
        if turn is not None:
            turn.log(input=[message])

    def start_speech(self, start_ms: float) -> None:
        self._vad_seen = True
        if self._speech is None:
            self._speech = _UserSpeech(start_ms)
            if self._turn is not None:
                self._speech_span(self._speech)

    def stop_speech(self, end_ms: float | None = None) -> None:
        speech = self._speech
        if speech is None:
            return
        end_ms = max(speech.start_ms, self._clock.now_ms() if end_ms is None else end_ms)
        interval = selection(speech.start_ms, end_ms)
        span = self._speech_span(speech)
        span.log(metadata={SELECTIONS: {"selection-0": interval}})
        span.end(end_time=self._clock.wall(end_ms))
        if self._turn is not None:
            self._turn.selections.append(interval)
        self._last_speech, self._speech = span, None

    def identify(self, frame: "Frame", span: Span | None = None) -> None:
        user_id = getattr(frame, "user_id", None)
        if not user_id:
            return
        participant = self._recorder.participant(user_id, {"type": "user", "label": user_id})
        if participant is None:
            return
        if span is not None:
            span.log(metadata={PARTICIPANT_IDS: [participant]})
        speech = self._speech
        if speech is not None and participant not in speech.participants:
            speech.participants.add(participant)
            if speech.span is not None:
                speech.span.log(metadata={PARTICIPANT_IDS: sorted(speech.participants)})

    def consume_audio(self, processor: "FrameProcessor") -> None:
        # Streaming STT also consumes silence: speech gates recognition.
        if self._speech is None and (self._vad_seen or self._turn is None):
            return
        speech = self._speech
        existing = self._recognitions.get(id(processor))
        if existing is not None:
            if existing.turn_index == self._turn_index and (existing.span is not None or existing.speech is speech):
                return
            self._retire(existing)
        start_ms = speech.start_ms if speech is not None else self._clock.now_ms()
        previous = self._recognized_until.get(id(processor))
        if previous is not None and previous[0] is speech:
            # A later final within the same speech covers only subsequent audio.
            start_ms = max(start_ms, previous[1])
        recognition = _Recognition(processor, start_ms, speech, self._turn_index)
        self._recognitions[id(processor)] = recognition
        if self._turn is not None:
            self._open(recognition)

    def transcribe(self, frame: "TranscriptionFrame", processor: "FrameProcessor | None") -> None:
        end_ms = self._clock.now_ms()
        recognition = self._recognitions.pop(id(processor), None)
        speech = recognition.speech if recognition is not None else self._speech
        self._recognized_until[id(processor)] = (speech, end_ms)
        if recognition is not None and recognition.phase is _RecognitionPhase.AWAITING_TURN and self._turn is None:
            recognition.phase, recognition.transcript = _RecognitionPhase.TRANSCRIBED, (frame, end_ms)
            self._transcribed.append(recognition)
            if len(self._transcribed) >= MAX_PENDING:
                # A pipeline may never establish native turns.
                self.flush_transcribed()
            return
        self._emit(frame, recognition, end_ms)

    def recognition_span(self, processor: "FrameProcessor") -> Span | None:
        """The observed recognition whose provider work is about to execute."""
        recognition = self._recognitions.get(id(processor))
        return self._open(recognition) if recognition is not None else None

    def flush_transcribed(self) -> None:
        while self._transcribed:
            recognition = self._transcribed.popleft()
            if recognition.transcript is not None:
                frame, end_ms = recognition.transcript
                self._emit(frame, recognition, end_ms)

    def fail(self, processor: "FrameProcessor | None", error: object) -> None:
        recognition = self._recognitions.pop(id(processor), None)
        if recognition is not None:
            span = self._open(recognition)
            span.log(error=error)
            span.end(end_time=self._clock.wall())

    def close(self, end_ms: float) -> None:
        self.flush_transcribed()
        self.stop_speech(end_ms)
        for recognition in self._recognitions.values():
            span = self._open(recognition)
            span.log(metadata={"contrib.pipecat.incomplete": True})
            span.end(end_time=self._clock.wall(end_ms))
        self._recognitions.clear()
        self._recognized_until.clear()
        self._finish_turn(incomplete=True, end_ms=end_ms)

    def _finish_turn(self, *, incomplete: bool, end_ms: float | None = None) -> None:
        turn = self._turn
        if turn is None:
            return
        turn.span.log(
            metadata={
                SELECTIONS: {f"selection-{index}": item for index, item in enumerate(turn.selections)},
                TURN_INCOMPLETE: incomplete,
            }
        )
        turn.span.end(end_time=self._clock.wall(end_ms))
        self._last_turn, self._turn = turn.span, None
        self._turn_index += 1

    def _resolve_recognitions(self) -> None:
        self.flush_transcribed()
        for recognition in self._recognitions.values():
            self._open(recognition)

    def _speech_span(self, speech: _UserSpeech) -> Span:
        if speech.span is None:
            metadata: dict[str, Any] = {SOURCE_ID: "session-input"}
            if speech.participants:
                metadata[PARTICIPANT_IDS] = sorted(speech.participants)
            speech.span = self._spans.start(
                "user_speech", turn=self._turn_span, start_ms=speech.start_ms, metadata=metadata
            )
        return speech.span

    def _open(self, recognition: _Recognition) -> Span:
        if recognition.span is None:
            recognition.span = self._spans.start(
                "stt",
                turn=self._turn_span,
                start_ms=recognition.start_ms,
                metadata={SOURCE_ID: "session-input", **metadata_from_processor(recognition.processor)},
                can_set_current=True,
            )
            recognition.phase = _RecognitionPhase.ACTIVE
        return recognition.span

    def _retire(self, recognition: _Recognition) -> None:
        self._recognitions.pop(id(recognition.processor), None)
        if recognition.span is not None:
            recognition.span.log(metadata={"contrib.pipecat.incomplete": True})
            recognition.span.end(end_time=self._clock.wall())

    def _emit(self, frame: "TranscriptionFrame", recognition: _Recognition | None, end_ms: float) -> None:
        speech: Span | None
        if recognition is not None:
            span = self._open(recognition)
            speech = self._speech_span(recognition.speech) if recognition.speech is not None else None
        else:
            span = self._spans.start(
                "transcript_received", turn=self._context_turn, metadata={SOURCE_ID: "session-input"}
            )
            speech = self._speech_span(self._speech) if self._speech is not None else self._last_speech
        metadata = {
            f"contrib.pipecat.{key}": getattr(frame, key, None)
            for key in ("user_id", "timestamp", "language", "finalized")
        }
        self.identify(frame, span)
        span.log(
            output=[{"role": "user", "content": frame.text}],
            metadata={key: value for key, value in metadata.items() if value is not None},
        )
        if speech is not None:
            set_links(span, [("transcribes", speech)])
        span.end(end_time=self._clock.wall(end_ms))
        self._linker.add_transcript(frame.text, span)
