"""Waits observed at the local turn controller, rather than inferred from VAD.

Qualified built-in stop strategies consume the pause and start a real decision
wait. Other modes (notably externally managed realtime turns) are omitted.
"""

import math
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from braintrust._audio.schema import PARTICIPANT_IDS, SOURCE_ID, TURN_ID

from ._common import INSTRUMENTATION


CONDITION: ContextVar[tuple[Any, str] | None] = ContextVar("braintrust_pipecat_eou_condition", default=None)
SPEECH_TIMEOUT_STRATEGY = (
    "pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy",
    "SpeechTimeoutUserTurnStopStrategy",
)
_LOCAL_STRATEGIES = {
    SPEECH_TIMEOUT_STRATEGY,
    ("pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy", "TurnAnalyzerUserTurnStopStrategy"),
}

_AUXILIARY_STRATEGIES = {
    # Pipecat < 1.17.
    ("pipecat.turns.user_stop.llm_turn_completion_user_turn_stop_strategy", "LLMTurnCompletionUserTurnStopStrategy"),
    # Pipecat >= 1.17 moved the class into a subpackage; the old module re-exports it.
    (
        "pipecat.turns.user_stop.llm_turn_completion.llm_turn_completion_user_turn_stop_strategy",
        "LLMTurnCompletionUserTurnStopStrategy",
    ),
}


def stop_strategies(controller):
    configured = getattr(controller, "user_turn_strategies", None)
    if configured is None:  # Pipecat 1.3.0 has no public property.
        configured = getattr(controller, "_user_turn_strategies", None)
    return tuple(getattr(configured, "stop", None) or ())


def speech_time(frame, correction):
    stamp, delay = getattr(frame, "timestamp", None), getattr(frame, correction, None)
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (stamp, delay)):
        return None
    if stamp <= 0 or delay < 0 or stamp - delay <= 0 or stamp > time.time():
        return None
    return stamp - delay


@dataclass
class ControllerWait:
    controller: Any
    session: Any
    enqueue: Callable[[Callable[[], None]], None] | None
    turn: Any = None
    span: Any = None
    start: float | None = None
    # A resumed event can arrive through both the pipeline and controller.
    last_stop: float | None = None

    def observe(self, frame):
        # Capture framework state before it changes, then update all trace
        # trackers in the native observer queue's FIFO order.
        if self.enqueue is None:
            return
        kind = type(frame).__name__
        if kind not in ("VADUserStartedSpeakingFrame", "VADUserStoppedSpeakingFrame"):
            return
        speech_at = speech_time(frame, "start_secs" if kind == "VADUserStartedSpeakingFrame" else "stop_secs")
        strategies = ()
        if kind == "VADUserStoppedSpeakingFrame" and getattr(self.controller, "_user_turn", False):
            strategies = stop_strategies(self.controller)
        self.enqueue(lambda: self._observe(frame, strategies, speech_at))

    def _observe(self, frame, strategies, speech_at):
        kind = type(frame).__name__
        if kind == "VADUserStartedSpeakingFrame":
            resumed = speech_at
            self.finish("user_resumed" if resumed is not None else None, resumed)
            self.last_stop = None
        elif kind == "VADUserStoppedSpeakingFrame":
            stopped = speech_at
            # Unknown/custom strategies may not consume this frame, or may use
            # FIRST/ALL semantics this spike cannot fully establish. Do not
            # infer a local completion wait from a configured detector alone.
            if (
                stopped is None
                or self.span is not None
                or stopped == self.last_stop
                or not any((type(s).__module__, type(s).__name__) in _LOCAL_STRATEGIES for s in strategies)
                or not all(
                    (type(s).__module__, type(s).__name__) in _LOCAL_STRATEGIES | _AUXILIARY_STRATEGIES
                    for s in strategies
                )
            ):
                return
            self.start = self.last_stop = stopped
            metadata = {
                SOURCE_ID: "session-input",
                "eou.speech_end_source": "vad",
                "contrib.pipecat.configured_stop_strategies": [type(s).__name__ for s in strategies],
                "contrib.pipecat.vad_timestamp": frame.timestamp,
                "contrib.pipecat.vad_stop_secs": frame.stop_secs,
            }
            if self.turn is not None:
                metadata[TURN_ID] = self.turn.span_id
            participants = self.session.user.speech_participants
            if participants:
                metadata[PARTICIPANT_IDS] = participants
            # Strategy identity/configuration snapshots cannot prove the full
            # applied gate history: public settings may change in place and
            # change back between observations. Omit detection_methods.
            self.span = (self.turn or self.session.span).start_span(
                name="eou_wait",
                type="task",
                start_time=stopped,
                metadata=metadata,
                internal={"instrumentation": INSTRUMENTATION},
            )

    def event(self, name, args):
        if self.enqueue is None:
            return
        reason = CONDITION.get()
        end_reason = reason[1] if reason is not None and args and args[0] is reason[0] else None
        observed_at = time.time()
        self.enqueue(lambda: self._event(name, observed_at, end_reason))

    def _event(self, name, observed_at, end_reason):
        # LiveTracker owns turns from the service's proposed speaking frames.
        # Universal aggregator callbacks must not also update cascade turns.
        if self.session.live_mode:
            return
        if name == "on_user_turn_started":
            self.session.user.start_turn(self.session.clock.from_wall_ms(observed_at))
            self.turn = self.session.user.turn_span
        elif name == "on_user_turn_stopped":
            # The causal condition must belong to the strategy which actually
            # completed this controller's turn. No inference-event shortcut.
            self.finish("committed", ended=observed_at, end_reason=end_reason)
            self.session.user.stop_turn(self.session.clock.from_wall_ms(observed_at))
            self.turn = None

    def finish(self, outcome=None, ended=None, end_reason=None):
        span, self.span = self.span, None
        if span is None:
            return
        ended = time.time() if ended is None else ended
        metadata = {}
        if self.start is not None and ended < self.start:
            ended, outcome = time.time(), None
        if outcome is not None:
            metadata["eou.outcome"] = outcome
        if outcome == "committed" and end_reason is not None:
            metadata["eou.end_reason"] = end_reason
        span.log(metadata=metadata)
        span.end(end_time=max(ended, self.start or ended))

    def close(self, ended):
        self.finish(ended=ended)
        self.turn = self.controller = self.session = self.enqueue = None
        self.start = self.last_stop = None
