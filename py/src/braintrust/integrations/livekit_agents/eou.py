"""Observe native EOU lifetimes without turning evaluation triggers into gates.

Private hooks are qualified against 1.8.9 and 1.8.12. Native detector cleanup
can outlive resumed speech, so its execution is a separate framework operation.
"""

import contextlib
import math
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from braintrust.logger import Span

from .compat import arguments, state_for


SNAPSHOT_WAIT: ContextVar["tuple[Any, Wait] | None"] = ContextVar("braintrust_livekit_snapshot_wait", default=None)


def observed_arguments(wrapped, args, kwargs):
    try:
        return arguments(wrapped, args, kwargs)
    except (TypeError, ValueError):
        return {}


def timestamp(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def native_metadata(native):
    return {
        f"contrib.livekit.{key}": value
        for key, value in (getattr(native, "attributes", None) or {}).items()
        if key.startswith("lk.eou.")
    }


@dataclass
class Wait:
    parent: Span
    start: float
    span: Span
    committed_at: float | None = None


def begin(owner, recognition, parent, anchor, source):
    wait = owner.eou_active.get(recognition)
    if wait is None:
        from .tracing import _new

        span = _new(
            parent,
            "eou_wait",
            start_time=anchor,
            metadata={**owner.source_metadata("session-input"), "eou.speech_end_source": source},
        )
        wait = Wait(parent, anchor, span)
        owner.eou_active[recognition] = wait
    elif anchor != wait.start:
        wait.start = anchor
        wait.span.log(metrics={"start": anchor}, metadata={"eou.speech_end_source": source})
    return wait


def finish(wait, end, outcome=None, native=None):
    if not timestamp(end) or end < wait.start:
        # A backwards terminal stamp cannot describe a canonical interval.
        # Close at observation time, without asserting an outcome at that stamp.
        end, outcome = max(wait.start, time.time()), None
    metadata = native_metadata(native)
    if outcome in ("committed", "user_resumed", "dropped"):
        metadata.update({"eou.outcome": outcome, "contrib.livekit.outcome": outcome})
    if timestamp(getattr(native, "end_time", None)):
        metadata["contrib.livekit.native_end_time_ns"] = native.end_time
    wait.span.log(metadata=metadata)
    wait.span.end(end)


def ensure(wrapped, instance, args, kwargs):
    values = observed_arguments(wrapped, args, kwargs)
    native = wrapped(*args, **kwargs)
    owner = state_for(instance)
    if owner is None or owner.closed:
        return native
    anchor = values.get("last_speaking_time")
    observed = owner.speech_ends.get(instance)
    # Native falls back to now, and STT sometimes substitutes receipt time.
    # Neither is evidence of speech end. Only use a separately observed anchor.
    if not timestamp(anchor) or anchor > time.time() or observed is None or anchor != observed[0]:
        return native
    parent = owner.native_users.get((instance, values.get("user_turn_span"))) or owner.span
    wait = begin(owner, instance, parent, anchor, observed[1])
    wait.span.log(
        metadata={
            **native_metadata(native),
            "contrib.livekit.trigger": values.get("trigger"),
            "contrib.livekit.endpointing_delay": values.get("endpointing_delay"),
        }
    )
    return native


def snapshot(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    owner = state_for(instance)
    wait = owner.eou_active.pop(instance, None) if owner is not None else None
    if wait is not None:
        # A manual commit owns this wait independently of the current buffer.
        # Retain it for the eventual snapshot terminal callback.
        # No-op OTel tracers can return the same native span for every turn.
        # Snapshot identity remains unique even when native span identity does not.
        owner.eou_snapshots[(instance, id(result))] = wait
        SNAPSHOT_WAIT.set((instance, wait))
        if wait.parent is not owner.span:
            owner.pending_users.add(wait.parent)
    return result


def run_detection(wrapped, instance, args, kwargs):
    # Native creates the manual snapshot and its delivery task synchronously.
    # Only that task inherits the owning wait, including with native OTel off.
    token = SNAPSHOT_WAIT.set(None)
    try:
        return wrapped(*args, **kwargs)
    finally:
        SNAPSHOT_WAIT.reset(token)


def end(wrapped, instance, args, kwargs):
    values = observed_arguments(wrapped, args, kwargs)
    snapshot = values.get("turn_snapshot")
    native = (
        getattr(snapshot, "eou_wait_span", None) if snapshot is not None else getattr(instance, "_eou_wait_span", None)
    )
    requested = values.get("end_time")
    ended = time.time() if requested is None else requested
    result = wrapped(*args, **kwargs)
    owner = state_for(instance)
    if owner is None:
        return result
    wait = (
        owner.eou_active.pop(instance, None)
        if snapshot is None
        else owner.eou_snapshots.pop((instance, id(snapshot)), None)
    )
    if wait is not None:
        outcome = values.get("outcome")
        # Native labels teardown as dropped; shutdown itself proves no outcome.
        if owner.closing or ((closing := getattr(instance, "_closing", None)) is not None and closing.is_set()):
            outcome = None
        finish(
            wait,
            wait.committed_at if outcome == "committed" and wait.committed_at is not None else ended,
            outcome,
            native,
        )
    return result


async def vad_event(wrapped, instance, args, kwargs):
    event = args[0] if args else kwargs["ev"]
    result = await wrapped(*args, **kwargs)
    owner = state_for(instance)
    if owner is not None and not owner.closed and not owner.closing:
        kind = event.type.value
        if kind == "end_of_speech" and timestamp(instance._last_speaking_time):
            owner.speech_ends[instance] = (instance._last_speaking_time, "vad")
            # In these native branches EOS enters _run_eou_detection even when
            # it cannot schedule a detector yet because STT has no transcript.
            # That is an observed completion wait too, not just STT execution.
            if instance._vad_base_turn_detection or (
                instance._turn_detection_mode == "stt" and instance._user_turn_committed
            ):
                parent = owner.native_users.get((instance, instance._user_turn_span)) or owner.span
                begin(owner, instance, parent, instance._last_speaking_time, "vad")
        elif kind == "start_of_speech":
            owner.speech_ends.pop(instance, None)
    return result


def stt_anchor(instance, event, owner):
    if event.type.value == "start_of_speech":
        owner.speech_ends.pop(instance, None)
        return
    anchor = instance._last_speaking_time
    reported = getattr(event, "speech_end_time", None)
    if event.type.value == "end_of_speech" and timestamp(reported) and anchor == reported:
        owner.speech_ends[instance] = (anchor, "stt")
    elif event.alternatives and timestamp(instance._input_started_at):
        offset = event.alternatives[0].end_time
        if timestamp(offset) and anchor == instance._input_started_at + offset:
            owner.speech_ends[instance] = (anchor, "stt")


def detection(wrapped, instance, args, kwargs):
    native = args[0] if args else kwargs["span"]
    owner = state_for(instance)
    wait = owner.eou_active.get(instance) if owner is not None else None

    @contextlib.contextmanager
    def context():
        try:
            with wrapped(*args, **kwargs):
                yield
        finally:
            start, ended = getattr(native, "start_time", None), getattr(native, "end_time", None)
            if wait is not None and timestamp(start) and timestamp(ended) and ended >= start:
                from .tracing import _new

                operation = _new(
                    wait.parent, "livekit.eou_detection", start_time=start / 1e9, metadata=native_metadata(native)
                )
                operation.end(ended / 1e9)

    return context()
