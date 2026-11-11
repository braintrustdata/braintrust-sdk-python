"""Braintrust-owned cascade spans from qualified LiveKit framework hooks."""

import asyncio
import contextlib
import inspect
import logging
import re
import time
import weakref
from contextvars import ContextVar
from fractions import Fraction

from braintrust._audio.config import AudioRecordingOptions
from braintrust._audio.session import AudioSessionRecorder
from braintrust._instrumentation import provider_content_allowed
from braintrust.logger import Span, start_span

from .compat import STATE_ATTR, arguments, chat_messages, llm_metadata, provider_llm_expected, state_for
from .privacy import Privacy, PrivacyCustomizer, opted_in


_ACTIVE: ContextVar[object] = ContextVar("braintrust_livekit_owner", default=None)
_HANDLE: ContextVar[object] = ContextVar("braintrust_livekit_speech", default=None)
_REPLY_TO: ContextVar[object] = ContextVar("braintrust_livekit_reply_to", default=None)
_SYNTHESIS: ContextVar[object] = ContextVar("braintrust_livekit_synthesis", default=None)
_OPTIONS = AudioRecordingOptions()
_warned_links = False


def configure(audio):
    global _OPTIONS
    if audio is not None:
        if audio.encoding is not None:
            from braintrust._audio.middleware import encoding_stack

            directions = sum(opted_in(direction) for direction in ("USER", "AGENT"))
            encoding_stack(audio.encoding, channels=directions or None)
        _OPTIONS = audio


def _new(parent, name, *, kind="task", start_time=None, metadata=None, privacy=None):
    turn_id = getattr(parent, "_braintrust_turn_id", None)
    if turn_id:
        metadata = {"turn.id": turn_id, **(metadata or {})}
    filters = (PrivacyCustomizer(privacy),) if privacy is not None else getattr(parent, "_export_customizers", ())
    kwargs = dict(
        name=name,
        type=kind,
        start_time=start_time,
        set_current=True,
        internal={"instrumentation": "livekit-agents-auto", "export_customizers": filters},
        # AudioSessionRecorder emits the schema marker with the session timeline.
        metadata=metadata or {},
    )
    span = parent.start_span(**kwargs) if parent is not None else start_span(**kwargs)
    if name in ("user_turn", "agent_turn") and span.span_id:
        turn_id = span.span_id
        span.log(metadata={"turn.id": turn_id})
    if turn_id:
        span._braintrust_turn_id = turn_id
    return span


@contextlib.contextmanager
def _operation_context(span: Span):
    """Activate an owned operation without changing its explicit lifetime."""
    # An active customer span takes precedence over a propagated parent slug.
    # Set the operation itself current, then restore that customer's context.
    span.set_current()
    try:
        yield
    finally:
        span.unset_current()


def link(target, relationship):
    global _warned_links
    trace_id, span_id = getattr(target, "root_span_id", ""), getattr(target, "span_id", "")
    if re.fullmatch("[0-9a-f]{32}", trace_id) and re.fullmatch("[0-9a-f]{16}", span_id):
        return {
            "context": {"traceId": trace_id, "spanId": span_id, "traceFlags": 1},
            "attributes": {"braintrust.relationship.type": relationship},
        }
    if not _warned_links:
        _warned_links = True
        logging.getLogger("braintrust").debug("LiveKit links omitted: span IDs do not use W3C format")
    return None


class Session:
    def __init__(self, native, record):
        self.native = native
        self.privacy = Privacy(native, record)
        self.options = _OPTIONS
        self.span = _new(None, "session", privacy=self.privacy)
        self.directions = tuple(direction for direction in ("USER", "AGENT") if opted_in(direction))
        sources = tuple({"USER": "session-input", "AGENT": "agent-output"}[d] for d in self.directions)
        try:
            self.audio = AudioSessionRecorder(
                self.span,
                sources=sources if self.privacy.allows_content() else (),
                options=self.options,
                allowed=self.audio_allowed,
            )
        except BaseException:
            self.span.end()
            raise
        self.audio.source("session-input")
        self.audio.source("agent-output")
        self.tap = None
        self.input_end: Fraction | None = None
        self.user = None
        self.user_started = None
        self.pending_users = set()
        self.user_speech = None
        self.user_selection = None
        self.user_start = None
        self.user_end = None
        self.turns: dict[object, object] = {}
        self.requests: dict[object, dict[str, object]] = {}
        self.tool_results: dict[object, dict[str, object]] = {}
        self.participants = {}
        self.stt = None
        self.realtime = {}
        self.closed = False
        self.closing = False
        self._finishing = None
        if self.privacy.job is not None:
            reference = weakref.ref(self)

            async def shutdown():
                if (owner := reference()) is not None:
                    # LiveKit runs job callbacks concurrently. Settle its media
                    # lifecycle before sealing, even when its own close callback
                    # is running alongside this idempotent fallback.
                    if owner.native is not None:
                        await owner.native.aclose()
                    else:
                        await owner.finish()

            self.privacy.job.add_shutdown_callback(shutdown)

    def permits(self):
        allowed = self.privacy.allows_content()
        if not allowed:
            if self.tap is not None:
                self.tap.discard_private()
            if self.audio.recorder is not None:
                self.audio.recorder.discard("privacy_restricted")
        return allowed

    def audio_allowed(self):
        return self.privacy.allows_content() and all(opted_in(d) for d in self.directions)

    def capture_output(self):
        return not self.closed and "AGENT" in self.directions and self.permits() and opted_in("AGENT")

    def log(self, span, **event):
        # Per-span privacy customizers recheck at lazy export.
        for field in ("input", "output", "error"):
            if field in event:
                if not self.permits():
                    del event[field]
        span.log(**event)

    def ensure_user(self, start=None):
        if self.user is None:
            self.user_started = start if start is not None else time.time()
            self.user = _new(self.span, "user_turn", start_time=self.user_started)
            self.user.log(metadata={"turn.id": self.user.span_id})
            self.user_start = self.user_end = None
        elif start is not None and start < self.user_started:
            self.user_started = start
            self.user.log(metrics={"start": start})
        return self.user

    def agent_parent(self):
        return self.turns.get(_HANDLE.get(), self.span)

    def source_metadata(self, source):
        metadata = {"audio.source_id": source}
        if self.permits() and (participant := self.participants.get(source)):
            metadata["audio.participant_ids"] = [participant]
        return metadata

    def attach_media(self):
        from .playback import PlaybackTap

        if self.tap is None and self.native.output.audio is not None:
            self.tap = PlaybackTap(self.native.output.audio, self)
            self.native.output.audio = self.tap
        room_io = self.native._room_io
        if room_io is not None and self.permits():
            # session.start() joins the room concurrently with the first activity
            # start, and LiveKit raises when local_participant is read before the
            # room connects. Skip the agent label until then; session_start calls
            # attach_media again once the room has connected.
            room = room_io.room
            local = room.local_participant if room.isconnected() else None
            self.label("session-input", room_io.linked_participant, "user")
            self.label("agent-output", local, "agent")

    def label(self, source, participant, kind):
        # Labels are keyed by sid, so repeated calls for one participant reuse it.
        if participant is not None and participant.sid:
            identifier = self.audio.participant(participant.sid, {"type": kind})
            if identifier:
                self.participants[source] = identifier

    def input(self, frame):
        if self.closed:
            return
        duration = Fraction(frame.samples_per_channel * 1_000_000_000, frame.sample_rate)
        if self.input_end is None:
            self.input_end = max(0, self.audio.elapsed_ns() - duration)
        start = self.input_end
        self.input_end += duration
        if "USER" in self.directions and self.permits() and opted_in("USER"):
            self.audio.record(
                "session-input",
                bytes(frame.data),
                sample_rate_hz=frame.sample_rate,
                channels=frame.num_channels,
                offset_ns=start,
            )
        self.advance()
        return start

    def advance(self):
        watermark = self.audio.elapsed_ns()
        if "USER" in self.directions:
            watermark = min(watermark, max(0, (self.input_end or 0) - 1_000_000))
        if self.tap is not None:
            watermark = min(watermark, self.tap.watermark())
        self.audio.advance(watermark)

    def played(self, run, start, end):
        if run.speech is None:
            run.speech = _new(
                run.parent,
                "agent_speech",
                start_time=(self.audio.wall_ns + start) / 1e9,
                metadata=self.source_metadata("agent-output"),
            )
            target = getattr(run, "synthesis", None)
            if target is not None and (relationship := link(target, "generated_by")):
                run.speech.log(metadata={"links": [relationship]})
        left, right = round(start / 1_000_000), round(end / 1_000_000)
        last = next(reversed(run.selections.values()), None) if run.selections else None
        if last is not None and last["end_offset_ms"] == left:
            last["end_offset_ms"] = right
        else:
            run.selections[f"selection-{len(run.selections)}"] = {"start_offset_ms": left, "end_offset_ms": right}
        run.speech.log(metadata={"audio.selections": run.selections})

    def terminal_metadata(self, *, interrupted: bool = False, incomplete: bool = False) -> dict[str, bool]:
        if self.closing:
            return {"contrib.livekit.incomplete": True}
        if interrupted:
            return {"contrib.livekit.interrupted": True}
        if incomplete:
            return {"contrib.livekit.incomplete": True}
        return {}

    def end_playback(self, run):
        if run.speech is None:
            return
        if run.finished is None or getattr(run.finished, "interrupted", False):
            run.speech.log(metadata=self.terminal_metadata(interrupted=True, incomplete=run.finished is None))
        transcript = getattr(run.finished, "synchronized_transcript", None)
        if transcript is not None:
            self.log(run.speech, output=[{"role": "assistant", "content": transcript}])
        run.speech.end((self.audio.wall_ns + (run.played_end or run.queued_ns)) / 1e9)
        run.speech = None

    async def finish(self):
        if self._finishing is None:
            self._finishing = asyncio.create_task(self._finish())
        await asyncio.shield(self._finishing)

    async def _finish(self):
        if self.closed:
            return
        self.closed = True
        end = time.time()
        for controller in self.realtime.values():
            controller.close()
        self.realtime.clear()
        if self.tap is not None:
            self.tap.close()
            self.tap = None
        for span in (self.user_speech, self.stt):
            if span is not None:
                span.log(metadata={"contrib.livekit.incomplete": True})
                span.end(end)
        for span in (self.user, *self.pending_users, *self.turns.values()):
            if span is not None:
                span.log(metadata={"turn.incomplete": True})
                span.end(end)
        self.span.end(end)
        self.turns.clear()
        self.pending_users.clear()
        self.requests.clear()
        self.tool_results.clear()
        self.user = self.stt = self.user_speech = None
        try:
            await self.audio.finish()
        finally:
            self.native = None
            self.privacy.session = None
            self.privacy.job = None


async def session_start(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    if owner is not None and not owner.closed:
        return await wrapped(*args, **kwargs)
    owner = Session(instance, kwargs.get("record"))
    setattr(instance, STATE_ATTR, owner)
    token = _ACTIVE.set(owner)
    policy_token = provider_content_allowed.set(owner.privacy.allows_content)
    try:
        result = await wrapped(*args, **kwargs)
        owner.attach_media()
        return result
    except BaseException:
        await owner.finish()
        delattr(instance, STATE_ATTR)
        raise
    finally:
        provider_content_allowed.reset(policy_token)
        _ACTIVE.reset(token)


async def session_close(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    if owner is not None:
        owner.closing = True
    try:
        return await wrapped(*args, **kwargs)
    finally:
        if owner is not None:
            await owner.finish()
            if getattr(instance, STATE_ATTR, None) is owner:
                delattr(instance, STATE_ATTR)


async def activity_start(wrapped, instance, args, kwargs):
    from . import realtime

    owner = state_for(instance)
    if owner is None:
        return await wrapped(*args, **kwargs)
    owner.attach_media()
    realtime.prepare(instance)
    resources = kwargs.get("reuse_resources")
    native = getattr(resources, "rt_session", None)
    if native is not None:
        realtime.bind(owner, instance, native, instance.llm)
    token = realtime._ACTIVITY.set((owner, instance))
    try:
        return await wrapped(*args, **kwargs)
    finally:
        realtime._ACTIVITY.reset(token)


def participant_linked(wrapped, instance, args, kwargs):
    # RoomIO links the caller in a background task, usually after
    # session.start() returns, and again when a caller joins or rejoins later.
    result = wrapped(*args, **kwargs)
    owner = state_for(instance)
    if owner is not None and not owner.closed and owner.permits():
        owner.label("session-input", args[0] if args else kwargs["participant"], "user")
    return result


def push_audio(wrapped, instance, args, kwargs):
    from . import realtime

    owner = state_for(instance)
    if owner is None or owner.closed or not instance._started:
        return wrapped(*args, **kwargs)
    frame = args[0] if args else kwargs["frame"]
    start = owner.input(frame)
    duration = Fraction(frame.samples_per_channel * 1_000_000_000, frame.sample_rate)
    token = realtime._INPUT.set((owner, start, duration))
    try:
        return wrapped(*args, **kwargs)
    finally:
        realtime._INPUT.reset(token)


def create_speech_task(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    handle = kwargs.get("speech_handle")
    if owner is not None and handle is not None:
        setattr(handle, STATE_ATTR, owner)
        setattr(handle, "_braintrust_reply_to", _REPLY_TO.get())
    token, handle_token = _ACTIVE.set(owner), _HANDLE.set(handle)
    policy_token = provider_content_allowed.set(owner.privacy.allows_content if owner is not None else None)
    try:
        return wrapped(*args, **kwargs)
    finally:
        provider_content_allowed.reset(policy_token)
        _HANDLE.reset(handle_token)
        _ACTIVE.reset(token)


def agent_turn(wrapped, instance, args, kwargs):
    handle = args[0]
    owner = getattr(handle, STATE_ATTR, None)

    @contextlib.contextmanager
    def context():
        if owner is not None and handle not in owner.turns:
            span = _new(owner.span, "agent_turn", metadata={"contrib.livekit.speech_id": handle.id})
            owner.turns[handle] = span
            owner.requests[handle] = {}
            metadata = {"turn.id": span.span_id}
            if (reply := getattr(handle, "_braintrust_reply_to", None)) is not None:
                metadata["turn.reply_to"] = reply.span_id
            span.log(metadata=metadata)
        with wrapped(*args, **kwargs) as native:
            yield native

    return context()


def speech_done(wrapped, instance, args, kwargs):
    owner = getattr(instance, STATE_ATTR, None)
    try:
        return wrapped(*args, **kwargs)
    finally:
        if owner is not None:
            if (span := owner.turns.pop(instance, None)) is not None:
                if instance.interrupted or instance.exception() is not None:
                    span.log(
                        metadata={
                            "turn.incomplete": True,
                            **owner.terminal_metadata(
                                interrupted=instance.interrupted, incomplete=instance.exception() is not None
                            ),
                        }
                    )
                span.end()
            owner.requests.pop(instance, None)
            owner.tool_results.pop(instance, None)
            if hasattr(instance, STATE_ATTR):
                delattr(instance, STATE_ATTR)
            if hasattr(instance, "_braintrust_reply_to"):
                delattr(instance, "_braintrust_reply_to")


def continue_turn(wrapped, instance, args, kwargs):
    discarded, successor = args[:2]
    owner = getattr(discarded, STATE_ATTR, None)
    if owner is not None and discarded in owner.turns:
        owner.turns[successor] = owner.turns.pop(discarded)
        owner.requests[successor] = owner.requests.pop(discarded, {})
        owner.tool_results[successor] = owner.tool_results.pop(discarded, {})
        setattr(successor, STATE_ATTR, owner)
    return wrapped(*args, **kwargs)


def ensure_user(wrapped, instance, args, kwargs):
    if (owner := state_for(instance)) is not None:
        owner.ensure_user(arguments(wrapped, args, kwargs).get("start_time"))
    return wrapped(*args, **kwargs)


def user_speech_start(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    values = arguments(wrapped, args, kwargs)
    event = values.get("ev")
    if owner is not None:
        parent = owner.ensure_user()
        if event is not None and owner.user_speech is None:
            start = values["speech_start_time"]
            owner.user_speech = _new(
                parent, "user_speech", start_time=start, metadata=owner.source_metadata("session-input")
            )
            owner.user_selection = owner.audio.from_wall(start)
            owner.user_start = owner.user_start if owner.user_start is not None else owner.user_selection
    return wrapped(*args, **kwargs)


def user_speech_end(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    event = arguments(wrapped, args, kwargs).get("ev")
    if owner is not None and owner.user_speech is not None:
        end = time.time() - event.silence_duration - event.inference_duration if event is not None else time.time()
        if event is not None:
            end_ns = owner.audio.from_wall(end)
            owner.user_end = end_ns
            selection = {
                "selection-0": {
                    "start_offset_ms": round(owner.user_selection / 1e6),
                    "end_offset_ms": round(max(end_ns, owner.user_selection) / 1e6),
                }
            }
            owner.user_speech.log(metadata={"audio.selections": selection})
        owner.user_speech.end(end)
        owner.user_speech = None
    return wrapped(*args, **kwargs)


def process_stt(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    event = args[0] if args else kwargs["ev"]
    kind = event.type.value
    if (
        owner is not None
        and kind in ("interim_transcript", "preflight_transcript", "final_transcript")
        and event.alternatives
    ):
        parent = owner.ensure_user()
        received_at = time.time()
        result = event.alternatives[0]
        interval = getattr(result, "_braintrust_recognition_interval", None)
        recognized = getattr(result, "_braintrust_recognition_span", None)
        if recognized is None and owner.stt is None:
            metadata = owner.source_metadata("session-input")
            metadata.update(
                {
                    key: value
                    for key, value in (("model", instance._stt_model), ("provider", instance._stt_provider))
                    if value
                }
            )
            owner.stt = _new(parent, "stt", start_time=interval[0] if interval else received_at, metadata=metadata)
        if kind == "final_transcript":
            span = recognized or owner.stt
            if owner.stt is span:
                owner.stt = None
            scope = getattr(event, "_braintrust_source_run", None)
            # Native timestamp alignment can replace the event and drop this
            # attribute. Without recognition ownership, omit attribution.
            if scope is not None and result.speaker_id is not None and owner.permits():
                speaker = owner.audio.speaker("session-input", (scope, result.speaker_id), {"type": "user"})
                if speaker is not None:
                    span.log(metadata={"audio.speaker_ids": [speaker]})
            owner.log(span, output=[{"role": "user", "content": result.text}])
            span.end(interval[1] if interval else received_at)
    return wrapped(*args, **kwargs)


def end_of_turn(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    info = args[0] if args else kwargs["info"]
    if owner is not None:
        user = owner.ensure_user()
        setattr(info, "_braintrust_turn", user)
        setattr(info, "_braintrust_owner", owner)
        owner.log(user, input=[{"role": "user", "content": info.new_transcript}])
        if owner.user_start is not None and owner.user_end is not None:
            user.log(
                metadata={
                    "audio.selections": {
                        "selection-0": {
                            "start_offset_ms": round(owner.user_start / 1e6),
                            "end_offset_ms": round(owner.user_end / 1e6),
                        }
                    },
                    "audio.source_id": "session-input",
                }
            )
        token = _REPLY_TO.set(user)
        try:
            committed = wrapped(*args, **kwargs)
            if committed:
                owner.user = None
                if info.user_turn_span_adopted:
                    owner.pending_users.add(user)
                else:
                    user.end()
            return committed
        finally:
            _REPLY_TO.reset(token)
    return wrapped(*args, **kwargs)


def preemptive(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    token = _REPLY_TO.set(owner.ensure_user() if owner is not None else None)
    try:
        return wrapped(*args, **kwargs)
    finally:
        _REPLY_TO.reset(token)


async def user_completed(wrapped, instance, args, kwargs):
    info = arguments(wrapped, args, kwargs).get("info")
    token = _REPLY_TO.set(getattr(info, "_braintrust_turn", None))
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _REPLY_TO.reset(token)


def end_user_turn(wrapped, instance, args, kwargs):
    info = args[0] if args else kwargs["info"]
    owner = getattr(info, "_braintrust_owner", None)
    span = getattr(info, "_braintrust_turn", None)
    try:
        return wrapped(*args, **kwargs)
    finally:
        if owner is not None and span in owner.pending_users:
            owner.pending_users.remove(span)
            span.end()


def recognition_end(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    try:
        return wrapped(*args, **kwargs)
    finally:
        if owner is not None:
            for field in ("stt", "user_speech", "user"):
                span = getattr(owner, field)
                if span is not None:
                    span.log(
                        metadata={"turn.incomplete": True} if field == "user" else {"contrib.livekit.incomplete": True}
                    )
                    span.end()
                    setattr(owner, field, None)


def session_run(wrapped, instance, args, kwargs):
    owner = state_for(instance)
    user_input = arguments(wrapped, args, kwargs).get("user_input")
    if owner is None or not isinstance(user_input, str):
        return wrapped(*args, **kwargs)
    user = _new(owner.span, "user_turn")
    user.log(metadata={"turn.id": user.span_id})
    owner.log(user, input=[{"role": "user", "content": user_input}])
    user.end()
    token = _REPLY_TO.set(user)
    try:
        return wrapped(*args, **kwargs)
    finally:
        _REPLY_TO.reset(token)


async def llm_operation(wrapped, instance, args, kwargs):
    owner = _ACTIVE.get()
    if owner is None:
        return await wrapped(*args, **kwargs)
    values = arguments(wrapped, args, kwargs)
    span = _new(
        owner.agent_parent(),
        "llm_response",
        metadata=llm_metadata(values)
        if owner.permits()
        else {key: values[key] for key in ("model", "provider") if values.get(key)},
    )
    if owner.permits():
        owner.log(span, input=chat_messages(values["chat_ctx"]))
    data, node, usage, handle = values["data"], values["node"], None, _HANDLE.get()
    fallback = None
    if not provider_llm_expected(node):
        fallback = _new(span, "LLM request", kind="llm", metadata=llm_metadata(values) if owner.permits() else {})
        if owner.permits():
            owner.log(fallback, input=chat_messages(values["chat_ctx"]))
    relationships = []
    for item in values["chat_ctx"].items:
        if getattr(item, "type", None) == "function_call_output":
            target = owner.tool_results.get(handle, {}).get(item.call_id)
            if target is not None and (relationship := link(target, "consumes_tool_result")):
                relationships.append(relationship)
    if relationships:
        span.log(metadata={"links": relationships})

    async def observed_node(*node_args, **node_kwargs):
        result = node(*node_args, **node_kwargs)
        if inspect.isawaitable(result):
            result = await result
        if not hasattr(result, "__aiter__"):
            return result

        async def chunks():
            nonlocal usage
            try:
                async for chunk in result:
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    delta = getattr(chunk, "delta", None)
                    for call in getattr(delta, "tool_calls", ()) or ():
                        owner.requests.setdefault(handle, {})[call.call_id] = span
                    yield chunk
            finally:
                if hasattr(result, "aclose"):
                    await result.aclose()

        return chunks()

    values["node"] = observed_node
    completed = False
    try:
        with _operation_context(fallback or span):
            result = await wrapped(**values)
        completed = True
        return result
    except asyncio.CancelledError:
        status = {
            "contrib.livekit.cancelled": True,
            **owner.terminal_metadata(interrupted=bool(getattr(handle, "interrupted", False)), incomplete=True),
        }
        span.log(metadata=status)
        if fallback is not None:
            fallback.log(metadata=status)
        raise
    except Exception:
        if fallback is not None:
            fallback.log(error="LiveKit LLM operation failed")
        span.log(error="LiveKit LLM operation failed")
        raise
    finally:
        output = [{"role": "assistant", "content": data.generated_text}]
        if functions := data.generated_functions:
            output[0]["tool_calls"] = [
                {
                    "id": fn.call_id,
                    "type": "function",
                    "function": {
                        "name": fn.name,
                        "arguments": fn.arguments,
                    },
                }
                for fn in functions
            ]
            owner.requests.setdefault(handle, {}).update({fn.call_id: span for fn in functions})
        owner.log(span, output=output)
        metrics = {}
        if usage is not None:
            for native, target in (
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "tokens"),
                ("prompt_cached_tokens", "prompt_cached_tokens"),
            ):
                value = getattr(usage, native, None)
                if native in usage.model_fields_set and value is not None:
                    metrics[target] = value
            if "cache_creation_tokens" in usage.model_fields_set:
                metrics["prompt_cache_creation_tokens"] = usage.cache_creation_tokens
            if "cache_read_tokens" in usage.model_fields_set:
                metrics["prompt_cached_tokens"] = max(metrics.get("prompt_cached_tokens", 0), usage.cache_read_tokens)
            if "reasoning_tokens" in usage.model_fields_set:
                metrics["completion_reasoning_tokens"] = usage.reasoning_tokens
        if data.ttft is not None:
            metrics["time_to_first_token"] = data.ttft
        if fallback is not None:
            # LiveKit exposes tool calls but not the provider's terminal reason.
            # Keep an unknown text-only finish reason null rather than guessing stop.
            owner.log(
                fallback,
                output=[
                    {
                        "index": 0,
                        "message": {**output[0], "content": data.generated_text or None},
                        "finish_reason": "tool_calls" if completed and functions else None,
                    }
                ],
            )
            fallback.log(metrics=metrics)
            fallback.end()
        span.end()


async def tts_operation(wrapped, instance, args, kwargs):
    owner = _ACTIVE.get()
    if owner is None:
        return await wrapped(*args, **kwargs)
    values = arguments(wrapped, args, kwargs)
    span = _new(
        owner.agent_parent(), "tts", metadata={key: values[key] for key in ("model", "provider") if values.get(key)}
    )
    data = values["data"]
    setattr(data, "_braintrust_span", span)
    stream, text = values["input"], []

    async def observed_input():
        async for item in stream:
            if owner.permits() and isinstance(item, str):
                text.append(item)
            yield item

    values["input"] = observed_input()
    try:
        with _operation_context(span):
            return await wrapped(**values)
    except asyncio.CancelledError:
        span.log(
            metadata={
                "contrib.livekit.cancelled": True,
                **owner.terminal_metadata(
                    interrupted=bool(getattr(_HANDLE.get(), "interrupted", False)), incomplete=True
                ),
            }
        )
        raise
    except Exception:
        span.log(error="LiveKit TTS operation failed")
        raise
    finally:
        owner.log(span, input="".join(text))
        if data.ttfb is not None:
            span.log(metrics={"time_to_first_audio": data.ttfb})
        span.end()


async def stt_recognize(wrapped, instance, args, kwargs):
    from livekit.agents.voice.audio_recognition import _STTPipelineContextVar

    owner = _ACTIVE.get()
    if owner is None or _STTPipelineContextVar.get(None) is None:
        return await wrapped(*args, **kwargs)
    start = time.time()
    span = _new(
        owner.ensure_user(),
        "stt",
        start_time=start,
        metadata={
            **owner.source_metadata("session-input"),
            "model": instance.model,
            "provider": instance.provider,
        },
    )
    owner.stt = span
    try:
        with _operation_context(span):
            event = await wrapped(*args, **kwargs)
        end = time.time()
        # Alternatives survive StreamAdapter and timestamp-alignment copies.
        for alternative in event.alternatives:
            alternative._braintrust_recognition_interval = (start, end)
            alternative._braintrust_recognition_span = span
        if event.alternatives:
            owner.log(span, output=[{"role": "user", "content": event.alternatives[0].text}])
        return event
    except asyncio.CancelledError:
        span.log(metadata={"contrib.livekit.cancelled": True, "contrib.livekit.incomplete": True})
        raise
    except Exception:
        span.log(error="LiveKit STT operation failed")
        raise
    finally:
        span.end()
        if owner.stt is span:
            owner.stt = None


async def stt_pump(wrapped, instance, args, kwargs):
    _wrap_stt_node(instance)
    return await wrapped(*args, **kwargs)


def _wrap_stt_node(pipeline):
    node = pipeline._stt_node
    if getattr(node, "_braintrust_stt", False):
        return

    async def observed(*args, **kwargs):
        scope = object()
        result = node(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if not hasattr(result, "__aiter__"):
            return result

        async def events():
            try:
                async for event in result:
                    event._braintrust_source_run = scope
                    yield event
            finally:
                if hasattr(result, "aclose"):
                    await result.aclose()

        return events()

    observed._braintrust_stt = True
    pipeline._stt_node = observed


def rebind_stt(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    _wrap_stt_node(instance)
    return result


async def audio_forward(wrapped, instance, args, kwargs):
    from .realtime import _RESPONSE

    values = arguments(wrapped, args, kwargs)
    data = values.get("tts_data")
    if data is None and (record := _RESPONSE.get()) is not None:
        data = record.audio_sources.get(values["tts_output"], record.span)
    token = _SYNTHESIS.set(data)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _SYNTHESIS.reset(token)


def tts_created(wrapped, instance, args, kwargs):
    from .realtime import _RESPONSE

    result = wrapped(*args, **kwargs)
    if (record := _RESPONSE.get()) is not None:
        record.audio_sources[result[1].audio_ch] = result[1]
    return result


async def execute_tool(wrapped, instance, args, kwargs):
    values = arguments(wrapped, args, kwargs)
    context = values["run_ctx"]
    owner = state_for(context.session)
    if owner is None:
        return await wrapped(*args, **kwargs)
    handle, call = context.speech_handle, context.function_call
    span = _new(
        owner.turns.get(handle, owner.span), "tool", kind="tool", metadata={"gen_ai.tool.call.id": call.call_id}
    )
    from .realtime import _RESPONSE

    if (record := _RESPONSE.get()) is not None:
        record.controller.tools[(record.controller.epoch, call.call_id)] = span
    if owner.permits():
        span.log(span_attributes={"name": call.name})
        owner.log(span, input=values["raw_arguments"])
    if (target := owner.requests.get(handle, {}).get(call.call_id)) is not None:
        if relationship := link(target, "requested_by"):
            span.log(metadata={"links": [relationship]})
    try:
        with _operation_context(span):
            result = await wrapped(*args, **kwargs)
        owner.log(span, output=result)
        owner.tool_results.setdefault(handle, {})[call.call_id] = span
        return result
    except asyncio.CancelledError:
        span.log(
            metadata={
                "contrib.livekit.cancelled": True,
                **owner.terminal_metadata(
                    interrupted=bool(getattr(_HANDLE.get(), "interrupted", False)), incomplete=True
                ),
            }
        )
        raise
    except Exception:
        span.log(error="LiveKit tool operation failed")
        raise
    finally:
        span.end()
