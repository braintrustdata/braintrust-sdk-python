"""Pure reducer-based Harbor lifecycle state machines."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    NEW = "new"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    RECONCILING = "reconciling"
    CLOSED = "closed"
    DISABLED = "disabled"
    FAILED = "failed"


class JobEvent(str, Enum):
    INITIALIZE = "initialize"
    READY = "ready"
    RECONCILE = "reconcile"
    CLOSE = "close"
    DISABLE = "disable"
    FAIL = "fail"


@dataclass(frozen=True)
class JobMachine:
    status: JobStatus = JobStatus.NEW
    warnings: tuple[str, ...] = ()


def can_reconcile(state: JobMachine) -> bool:
    """Report whether reconciliation may start, keeping the rule with the transitions.

    DISABLED and FAILED both reach RECONCILE illegally, so callers must ask rather
    than enumerate terminal statuses at each site.
    """
    return state.status == JobStatus.ACTIVE


def accepts_trial_events(state: JobMachine) -> bool:
    return state.status in {JobStatus.ACTIVE, JobStatus.RECONCILING}


def reduce_job(state: JobMachine, event: JobEvent, *, strict: bool = False) -> JobMachine:
    transitions = {
        (JobStatus.NEW, JobEvent.INITIALIZE): JobStatus.INITIALIZING,
        (JobStatus.INITIALIZING, JobEvent.READY): JobStatus.ACTIVE,
        (JobStatus.ACTIVE, JobEvent.RECONCILE): JobStatus.RECONCILING,
        (JobStatus.RECONCILING, JobEvent.CLOSE): JobStatus.CLOSED,
    }
    if event == JobEvent.DISABLE and state.status not in {JobStatus.CLOSED, JobStatus.FAILED}:
        return replace(state, status=JobStatus.DISABLED)
    if event == JobEvent.FAIL and state.status not in {JobStatus.CLOSED, JobStatus.DISABLED}:
        return replace(state, status=JobStatus.FAILED)
    target = transitions.get((state.status, event))
    if target is not None:
        return replace(state, status=target)
    message = f"illegal job transition {state.status.value} + {event.value}"
    if strict:
        raise ValueError(message)
    return replace(state, warnings=(*state.warnings, message))


class TrialStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    WAITING_RETRY = "waiting_retry"
    FINAL_CANDIDATE = "final_candidate"
    CANCELLED = "cancelled"
    FINALIZING = "finalizing"
    SYNCED = "synced"
    OMITTED = "omitted"


class TrialPhase(str, Enum):
    STARTED = "started"
    ENVIRONMENT = "environment"
    AGENT = "agent"
    AGENT_DONE = "agent_done"
    VERIFICATION = "verification"


_PHASE_ORDER = {
    TrialPhase.STARTED: 0,
    TrialPhase.ENVIRONMENT: 1,
    TrialPhase.AGENT: 2,
    TrialPhase.AGENT_DONE: 3,
    TrialPhase.VERIFICATION: 4,
}


class TrialEventKind(str, Enum):
    START = "start"
    ENVIRONMENT_START = "environment-start"
    AGENT_START = "agent-start"
    AGENT_END = "agent-end"
    VERIFICATION_START = "verification-start"
    END = "end"
    CANCEL = "cancel"
    FINAL_RESULT = "final_result"
    OMIT = "omit"
    SYNCED = "synced"
    SYNC_FAILED = "sync_failed"


@dataclass(frozen=True)
class TrialEvent:
    kind: TrialEventKind
    timestamp: float | None = None
    payload: Any = None
    retry_predicted: bool = False


class EffectKind(str, Enum):
    STAGE_RESULT = "stage_result"
    RECORD_RETRY = "record_retry"
    RECORD_CANCELLATION = "record_cancellation"
    SYNC_FINAL = "sync_final"
    CLOSE_OMITTED = "close_omitted"


@dataclass(frozen=True)
class Effect:
    kind: EffectKind
    payload: Any = None


@dataclass(frozen=True)
class TrialMachine:
    identity: str
    status: TrialStatus = TrialStatus.PENDING
    phase: TrialPhase | None = None
    retry_index: int = 0
    completed_attempts: int = 0
    warnings: tuple[str, ...] = ()
    final_result: Any = None


def _warn(state: TrialMachine, message: str, strict: bool) -> tuple[TrialMachine, tuple[Effect, ...]]:
    if strict:
        raise ValueError(message)
    return replace(state, warnings=(*state.warnings, message)), ()


def reduce_trial(
    state: TrialMachine,
    event: TrialEvent,
    *,
    strict: bool = False,
) -> tuple[TrialMachine, tuple[Effect, ...]]:
    """Reduce one lifecycle event without performing I/O."""
    kind = event.kind
    if kind == TrialEventKind.START:
        if state.status == TrialStatus.ACTIVE:
            return state, ()
        if state.status in {
            TrialStatus.PENDING,
            TrialStatus.WAITING_RETRY,
            TrialStatus.FINAL_CANDIDATE,
            TrialStatus.CANCELLED,
        }:
            retry_index = state.retry_index
            effects: tuple[Effect, ...] = ()
            if state.status != TrialStatus.PENDING:
                retry_index += 1
                effects = (Effect(EffectKind.RECORD_RETRY, retry_index),)
            return replace(
                state, status=TrialStatus.ACTIVE, phase=TrialPhase.STARTED, retry_index=retry_index
            ), effects
        return _warn(state, f"START after terminal state {state.status.value}", strict)

    phase_for_event = {
        TrialEventKind.ENVIRONMENT_START: TrialPhase.ENVIRONMENT,
        TrialEventKind.AGENT_START: TrialPhase.AGENT,
        TrialEventKind.AGENT_END: TrialPhase.AGENT_DONE,
        TrialEventKind.VERIFICATION_START: TrialPhase.VERIFICATION,
    }.get(kind)
    if phase_for_event is not None:
        if state.status != TrialStatus.ACTIVE or state.phase is None:
            return _warn(state, f"{kind.value} while {state.status.value}", strict)
        current_order = _PHASE_ORDER[state.phase]
        next_order = _PHASE_ORDER[phase_for_event]
        if next_order == current_order:
            return state, ()
        if next_order < current_order:
            return _warn(state, f"backward phase {state.phase.value} -> {phase_for_event.value}", strict)
        return replace(state, phase=phase_for_event), ()

    if kind == TrialEventKind.END:
        if state.status in {TrialStatus.WAITING_RETRY, TrialStatus.FINAL_CANDIDATE, TrialStatus.CANCELLED}:
            return state, ()
        if state.status != TrialStatus.ACTIVE:
            return _warn(state, f"END while {state.status.value}", strict)
        status = TrialStatus.WAITING_RETRY if event.retry_predicted else TrialStatus.FINAL_CANDIDATE
        return (
            replace(state, status=status, completed_attempts=state.completed_attempts + 1),
            (Effect(EffectKind.STAGE_RESULT, event.payload),),
        )

    if kind == TrialEventKind.CANCEL:
        if state.status == TrialStatus.CANCELLED:
            return state, ()
        if state.status in {TrialStatus.SYNCED, TrialStatus.OMITTED}:
            return _warn(state, f"CANCEL after terminal state {state.status.value}", strict)
        return replace(state, status=TrialStatus.CANCELLED), (Effect(EffectKind.RECORD_CANCELLATION),)

    if kind == TrialEventKind.FINAL_RESULT:
        if state.status == TrialStatus.SYNCED:
            return state, ()
        if state.status == TrialStatus.OMITTED:
            return _warn(state, "FINAL_RESULT after OMIT", strict)
        return replace(state, status=TrialStatus.FINALIZING, final_result=event.payload), (
            Effect(EffectKind.SYNC_FINAL, event.payload),
        )

    if kind == TrialEventKind.SYNCED:
        if state.status != TrialStatus.FINALIZING:
            return _warn(state, f"SYNCED while {state.status.value}", strict)
        return replace(state, status=TrialStatus.SYNCED), ()

    if kind == TrialEventKind.SYNC_FAILED:
        if state.status != TrialStatus.FINALIZING:
            return _warn(state, f"SYNC_FAILED while {state.status.value}", strict)
        return _warn(replace(state, status=TrialStatus.FINAL_CANDIDATE), str(event.payload), False)

    if kind == TrialEventKind.OMIT:
        if state.status in {TrialStatus.SYNCED, TrialStatus.OMITTED}:
            return state, ()
        return replace(state, status=TrialStatus.OMITTED), (Effect(EffectKind.CLOSE_OMITTED),)

    return _warn(state, f"unknown trial event {kind}", strict)
