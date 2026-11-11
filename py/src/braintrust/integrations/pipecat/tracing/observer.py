"""Route Pipecat observer callbacks to the session trackers."""

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from weakref import WeakValueDictionary

from braintrust._audio.clock import SessionClock
from braintrust._audio.config import AudioRecordingOptions
from braintrust._audio.session import AudioSessionRecorder
from braintrust._audio.types import RecordingResult
from braintrust.integrations.utils import _resolve_audio_attachment_options
from braintrust.integrations.versioning import detect_module_version, version_satisfies
from braintrust.logger import NOOP_SPAN, Span, SpanTypeAttribute, current_span

from ._agent import AgentTracker
from ._common import LOG, SpanFactory, is_pipeline_sink, processor_kind, start_span
from ._context import ContextLinker
from ._user import UserTracker


if TYPE_CHECKING:
    from pipecat.frames import frames
    from pipecat.frames.frames import AudioRawFrame, Frame
    from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
    from pipecat.processors.frame_processor import FrameProcessor

    _USES_NATIVE_FRAME_DEDUPLICATION = True
else:
    try:
        import pipecat
        from pipecat.frames import frames
        from pipecat.observers.base_observer import BaseObserver
    except ImportError:  # pragma: no cover - Pipecat is an optional dependency.
        _USES_NATIVE_FRAME_DEDUPLICATION = False

        class BaseObserver:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def cleanup(self) -> None:
                pass
    else:
        _USES_NATIVE_FRAME_DEDUPLICATION = version_satisfies(detect_module_version(pipecat, ("pipecat",)), ">=1.12.0")


_AudioSource = Literal["session-input", "agent-output"]
# Upper bound on how long an LLM service waits for its llm_response span.
_RESPONSE_SPAN_WAIT_SECONDS = 0.05
_TERMINAL_FRAME_TYPES = {"EndFrame", "StopFrame", "CancelFrame"}
# Pipecat broadcasts these in both directions; observe the downstream copy.
_UPSTREAM_IGNORED_FRAME_TYPES = {
    "UserStartedSpeakingFrame",
    "UserStoppedSpeakingFrame",
    "VADUserStartedSpeakingFrame",
    "VADUserStoppedSpeakingFrame",
    "BotStartedSpeakingFrame",
    "BotStoppedSpeakingFrame",
    "InputAudioRawFrame",
    "UserAudioRawFrame",
    "TTSTextFrame",
}


class _SessionPhase(Enum):
    NEW = "new"
    RUNNING = "running"
    CLOSED = "closed"


@dataclass
class _Session:
    span: Span
    recorder: AudioSessionRecorder
    clock: SessionClock
    linker: ContextLinker
    user: UserTracker
    agent: AgentTracker
    # Without playback, a model response end is the agent completion boundary.
    has_output_transport: bool = False


class _FrameAdmission:
    """Admit each frame once across every push and process callback.

    Pipecat 1.12+ reports a frame's first push natively. Older versions need
    local deduplication, kept only while the frame itself is alive.
    """

    def __init__(self, native: bool) -> None:
        self.native = native
        self._seen: WeakValueDictionary[int, Frame] = WeakValueDictionary()

    def admit(self, frame: "Frame") -> bool:
        if self.native:
            return True
        if frame.id in self._seen:
            return False
        self._seen[frame.id] = frame
        return True

    def clear(self) -> None:
        self._seen.clear()


_FrameAction = Callable[
    ["BraintrustPipecatObserver", _Session, Any, "FrameProcessor | None", bool], Awaitable[None] | None
]


class BraintrustPipecatObserver(BaseObserver):
    """Observe cascade sessions, native turns, speech, STT, LLM, tools and TTS.

    Audio capture is opt-in per direction. Only audio pushed *by* an output
    transport is recorded as agent playback; synthesized buffers are not proof
    that the user heard them. Encoding and upload run in the shared recorder.
    """

    def __init__(
        self,
        *,
        capture_audio_attachments: bool | None = None,
        capture_user_audio_attachments: bool | None = None,
        capture_agent_audio_attachments: bool | None = None,
        audio_recording_options: AudioRecordingOptions | None = None,
        **kwargs: Any,
    ) -> None:
        self._admission = _FrameAdmission(_USES_NATIVE_FRAME_DEDUPLICATION)
        if self._admission.native:
            kwargs["observe_every_push"] = True
        super().__init__(**kwargs)
        self.capture_user_audio_attachments, self.capture_agent_audio_attachments = _resolve_audio_attachment_options(
            capture_audio_attachments=capture_audio_attachments,
            capture_user_audio_attachments=capture_user_audio_attachments,
            capture_agent_audio_attachments=capture_agent_audio_attachments,
        )
        self.capture_audio_attachments = self.capture_user_audio_attachments or self.capture_agent_audio_attachments
        self._recording_options = audio_recording_options
        parent = current_span()
        self._parent = parent.export() if parent != NOOP_SPAN else None
        self._phase = _SessionPhase.NEW
        self._session: _Session | None = None
        self._finish_task: asyncio.Task[RecordingResult] | None = None
        # Response start frame ID -> its llm_response span, for the service.
        self._response_waiters: dict[int, asyncio.Future[Span | None]] = {}

    # Pipecat observer callbacks. Tracing failures never reach the pipeline.

    async def on_pipeline_started(self) -> None:
        if self._phase is _SessionPhase.NEW:
            self._start_session()

    async def on_process_frame(self, data: "FrameProcessed") -> None:
        if self._phase is _SessionPhase.CLOSED:
            return
        try:
            frame, processor = data.frame, data.processor
            session = self._start_session()
            kind = processor_kind(type(processor))
            if kind == "output":
                session.has_output_transport = True
            # Recognition starts when STT consumes audio, not when its transcript
            # arrives; the sink alone sees terminal frames after all services.
            stt_audio = kind == "stt" and isinstance(frame, frames.InputAudioRawFrame)
            terminal = type(frame).__name__ in _TERMINAL_FRAME_TYPES and is_pipeline_sink(processor)
            if self._admission.native and not (stt_audio or terminal):
                return
            with session.clock.event(data.timestamp):
                if stt_audio:
                    session.user.consume_audio(processor)
                if terminal or not self._admission.native:
                    await self._dispatch(session, frame, processor, played=False)
        except Exception:
            LOG.warning("Unable to trace Pipecat processing", exc_info=True)

    async def on_push_frame(self, data: "FramePushed") -> None:
        if self._phase is _SessionPhase.CLOSED:
            return
        try:
            frame = data.frame
            upstream = data.direction.name == "UPSTREAM"
            if upstream and type(frame).__name__ in _UPSTREAM_IGNORED_FRAME_TYPES:
                return
            session = self._start_session()
            output = processor_kind(type(data.source)) == "output"
            if output:
                session.has_output_transport = True
            played = output and not upstream
            with session.clock.event(data.timestamp):
                # A transport can forward a previously observed generated frame:
                # capture playback before admission, once it has been sent.
                if played:
                    self._observe_playback(session, frame)
                if self._admission.native and not getattr(data, "first_push", True):
                    return
                await self._dispatch(session, frame, data.source, played=played)
        except Exception:
            LOG.warning("Unable to trace Pipecat frame push", exc_info=True)

    async def cleanup(self) -> None:
        await self._finish()
        await super().cleanup()

    async def _response_span(self, frame: "Frame", push: Callable[[], Awaitable[Any]]) -> tuple[Any, Span | None]:
        """Push a service's response start frame, then return its span.

        Called from the LLM service's own task, which sends the provider
        request next. Observer callbacks run in order, so once this frame is
        traced, so is every frame before it. Waiting is capped so tracing never
        holds a request back noticeably.
        """
        loop = asyncio.get_running_loop()
        waiter = self._response_waiters[frame.id] = loop.create_future()
        try:
            result = await push()
            try:
                span = await asyncio.wait_for(asyncio.shield(waiter), _RESPONSE_SPAN_WAIT_SECONDS)
            except asyncio.TimeoutError:
                span = None
            return result, span
        finally:
            self._response_waiters.pop(frame.id, None)

    def _resolve_response(self, frame: "Frame", span: Span | None) -> None:
        waiter = self._response_waiters.get(frame.id)
        if waiter is not None and not waiter.done():
            waiter.set_result(span)

    @contextmanager
    def _current_recognition(self, processor: "FrameProcessor") -> Iterator[None]:
        """Scope STT work in its executing task; the observer owns span timing."""
        span = None
        try:
            if self._phase is _SessionPhase.RUNNING and self._session is not None:
                span = self._session.user.recognition_span(processor)
            # Reentrant setup/wrapping must not overwrite a span's scope token.
            if span is not None and current_span() is span:
                span = None
            if span is not None:
                span.set_current()
        except Exception:
            LOG.warning("Unable to activate Pipecat recognition", exc_info=True)
            span = None
        try:
            yield
        finally:
            if span is not None:
                span.unset_current()

    @contextmanager
    def _current_session(self) -> Iterator[None]:
        """Make the session span current while a worker runs.

        Pipecat creates every processor task from ``PipelineWorker.run``, and
        tasks inherit the context they are created in. Spans that application,
        tool and provider code start without a parent then join the session's
        trace. The observer's own spans keep their explicit parents.
        """
        if self._phase is not _SessionPhase.NEW:
            yield
            return
        span = self._start_session().span
        span.set_current()
        try:
            yield
        finally:
            span.unset_current()

    # Session lifecycle

    def _start_session(self) -> _Session:
        if self._session is not None:
            return self._session
        # Only made current explicitly, around a worker run.
        span = start_span(
            name="session",
            type=SpanTypeAttribute.TASK,
            parent=self._parent,
            set_current=True,
            metadata={"framework": "pipecat"},
        )
        sources = tuple(
            source
            for source, enabled in (
                ("session-input", self.capture_user_audio_attachments),
                ("agent-output", self.capture_agent_audio_attachments),
            )
            if enabled
        )
        recorder = AudioSessionRecorder(span, sources=sources, options=self._recording_options)
        recorder.source("session-input", {"label": "Agent input audio"})
        recorder.source("agent-output", {"label": "Agent audio"})
        clock = SessionClock(recorder)
        spans = SpanFactory(span, clock)
        linker = ContextLinker()
        self._session = _Session(
            span,
            recorder,
            clock,
            linker,
            UserTracker(spans, clock, recorder, linker),
            AgentTracker(spans, clock, recorder),
        )
        self._phase = _SessionPhase.RUNNING
        return self._session

    async def _finish(self) -> None:
        if self._phase is _SessionPhase.CLOSED:
            if self._finish_task is not None:
                await asyncio.shield(self._finish_task)
            return
        self._phase = _SessionPhase.CLOSED
        session, self._session = self._session, None
        if session is None:
            return
        end_ms = session.clock.now_ms()
        closers: tuple[Callable[[], None], ...] = (
            lambda: session.user.close(end_ms),
            lambda: session.agent.interrupt(interrupted=False, end_ms=end_ms),
            lambda: session.span.end(end_time=session.clock.wall(end_ms)),
        )
        for close in closers:
            try:
                close()
            except Exception:
                # Still end the session and upload audio already recorded.
                LOG.warning("Unable to close Pipecat trace", exc_info=True)
        self._admission.clear()
        # WorkerObserver cancels its callback worker before calling cleanup.
        # Shield the recorder so cleanup can finish the same upload exactly once.
        self._finish_task = asyncio.create_task(session.recorder.finish())
        await asyncio.shield(self._finish_task)

    # Frame routing

    async def _dispatch(
        self, session: _Session, frame: "Frame", processor: "FrameProcessor | None", *, played: bool
    ) -> None:
        kind = type(frame).__name__
        action = self._FRAME_ACTIONS.get(kind)
        if action is None or (kind in _TERMINAL_FRAME_TYPES and not is_pipeline_sink(processor)):
            return
        if not self._admission.admit(frame):
            return
        pending = action(self, session, frame, processor, played)
        if pending is not None:
            await pending

    def _observe_playback(self, session: _Session, frame: "Frame") -> None:
        if isinstance(frame, frames.OutputAudioRawFrame):
            self._record_audio(session, "agent-output", frame)
            session.agent.played_audio(frame)
        elif isinstance(frame, frames.TTSTextFrame):
            session.agent.played_text(frame)

    def _record_audio(self, session: _Session, source: _AudioSource, frame: "AudioRawFrame") -> None:
        enabled = (
            self.capture_user_audio_attachments if source == "session-input" else self.capture_agent_audio_attachments
        )
        if not enabled:
            return
        try:
            session.recorder.record_stream(
                source,
                frame.audio,
                sample_rate_hz=frame.sample_rate,
                channels=frame.num_channels,
                now_ns=session.clock.now_ns(),
            )
        except ValueError:
            LOG.warning("Unable to record Pipecat audio frame", exc_info=True)

    # Frame actions: translate native frames into tracker events.

    def _on_session_started(self, s: _Session, frame: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.span.log(metadata={"contrib.pipecat.start_metadata": getattr(frame, "metadata", {}) or {}})

    def _on_session_stopped(
        self, s: _Session, frame: "Frame", _p: "FrameProcessor | None", _played: bool
    ) -> Awaitable[None]:
        s.span.log(
            metadata={
                "contrib.pipecat.terminal_frame": type(frame).__name__,
                "contrib.pipecat.reason": getattr(frame, "reason", None),
            }
        )
        return self._finish()

    def _on_error(
        self, s: _Session, frame: "frames.ErrorFrame", processor: "FrameProcessor | None", _played: bool
    ) -> Awaitable[None] | None:
        error = frame.exception or frame.error
        s.span.log(error=error, metadata={"contrib.pipecat.fatal": frame.fatal})
        # A forwarded error names its origin; otherwise it is the emitter.
        origin = getattr(frame, "processor", None) or processor
        kind = processor_kind(type(origin))
        if kind == "stt":
            s.user.fail(origin, error)
        else:
            s.agent.fail(origin, kind, error)
        return self._finish() if frame.fatal else None

    def _on_user_turn_started(self, s: _Session, _f: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.user.start_turn()

    def _on_user_turn_stopped(self, s: _Session, _f: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.user.stop_turn()

    def _on_vad_started(self, s: _Session, frame: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.user.start_speech(s.clock.from_wall_ms(frame.timestamp - getattr(frame, "start_secs", 0)))

    def _on_vad_stopped(self, s: _Session, frame: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.user.stop_speech(s.clock.from_wall_ms(frame.timestamp - getattr(frame, "stop_secs", 0)))

    def _on_user_audio(
        self, s: _Session, frame: "frames.InputAudioRawFrame", _p: "FrameProcessor | None", _played: bool
    ) -> None:
        self._record_audio(s, "session-input", frame)
        s.user.identify(frame)

    def _on_transcription(
        self, s: _Session, frame: "frames.TranscriptionFrame", processor: "FrameProcessor | None", _played: bool
    ) -> None:
        s.user.transcribe(frame, processor)

    def _on_user_context(self, s: _Session, frame: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        if not getattr(frame, "speculation", False):
            s.user.log_input({"role": "user", "content": frame.text})

    def _on_model_context(
        self, s: _Session, frame: "frames.LLMContextFrame", _p: "FrameProcessor | None", _played: bool
    ) -> None:
        message = s.linker.submit(frame.context)
        if message is not None:
            s.user.log_input(message)

    def _on_response_started(
        self, s: _Session, frame: "Frame", processor: "FrameProcessor | None", _played: bool
    ) -> None:
        # Emit transcripts still waiting for a user turn before linking them.
        s.user.flush_transcribed()
        span = s.agent.start_response(processor, s.linker, reply_to=s.user.reply_target)
        self._resolve_response(frame, span)

    def _on_response_text(
        self, s: _Session, frame: "frames.LLMTextFrame", _p: "FrameProcessor | None", _played: bool
    ) -> None:
        s.agent.response_text(frame.text)

    def _on_tool_requests(
        self, s: _Session, frame: "frames.FunctionCallsStartedFrame", _p: "FrameProcessor | None", _played: bool
    ) -> None:
        s.agent.request_tools(frame.function_calls or [])

    def _on_response_stopped(self, s: _Session, _f: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        # Without playback the response end completes the turn: older Pipecat
        # has no assistant-context notification, and pipelines without an
        # assistant aggregator never send one.
        s.agent.end_response(completes_turn=not s.has_output_transport)

    def _on_tool_started(
        self, s: _Session, frame: "frames.FunctionCallInProgressFrame", _p: "FrameProcessor | None", _played: bool
    ) -> None:
        s.agent.start_tool(frame)

    def _on_tool_result(
        self, s: _Session, frame: "frames.FunctionCallResultFrame", _p: "FrameProcessor | None", _played: bool
    ) -> None:
        s.agent.finish_tool(frame, s.linker)

    def _on_tool_cancelled(
        self, s: _Session, frame: "frames.FunctionCallCancelFrame", _p: "FrameProcessor | None", _played: bool
    ) -> None:
        s.agent.cancel_tool(frame.tool_call_id)

    def _on_tts_started(
        self, s: _Session, frame: "frames.TTSStartedFrame", processor: "FrameProcessor | None", _played: bool
    ) -> None:
        s.agent.start_tts(frame, processor)

    def _on_tts_text(self, s: _Session, frame: "Frame", _p: "FrameProcessor | None", played: bool) -> None:
        if not played:
            s.agent.tts_text(frame)

    def _on_tts_stopped(self, s: _Session, frame: "Frame", _p: "FrameProcessor | None", played: bool) -> None:
        if not played:
            s.agent.stop_tts(frame)

    def _on_playback_started(self, s: _Session, _f: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.has_output_transport = True
        s.agent.start_playback()

    def _on_playback_stopped(self, s: _Session, _f: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.agent.stop_playback()

    def _on_agent_turn_stopped(self, s: _Session, _f: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.agent.request_completion()

    def _on_interruption(self, s: _Session, _f: "Frame", _p: "FrameProcessor | None", _played: bool) -> None:
        s.agent.interrupt()

    def _on_metrics(
        self, s: _Session, frame: "frames.MetricsFrame", processor: "FrameProcessor | None", _played: bool
    ) -> None:
        s.agent.record_metrics(frame, processor)

    # Frame names preserve compatibility with version-specific Pipecat frames.
    _FRAME_ACTIONS: ClassVar[dict[str, _FrameAction]] = {
        "StartFrame": _on_session_started,
        "EndFrame": _on_session_stopped,
        "StopFrame": _on_session_stopped,
        "CancelFrame": _on_session_stopped,
        "ErrorFrame": _on_error,
        "UserStartedSpeakingFrame": _on_user_turn_started,
        "UserStoppedSpeakingFrame": _on_user_turn_stopped,
        "VADUserStartedSpeakingFrame": _on_vad_started,
        "VADUserStoppedSpeakingFrame": _on_vad_stopped,
        "InputAudioRawFrame": _on_user_audio,
        "UserAudioRawFrame": _on_user_audio,
        "TranscriptionFrame": _on_transcription,
        "LLMContextUserTurnFrame": _on_user_context,
        "LLMContextFrame": _on_model_context,
        "LLMFullResponseStartFrame": _on_response_started,
        "LLMTextFrame": _on_response_text,
        "FunctionCallsStartedFrame": _on_tool_requests,
        "LLMFullResponseEndFrame": _on_response_stopped,
        "FunctionCallInProgressFrame": _on_tool_started,
        "FunctionCallResultFrame": _on_tool_result,
        "FunctionCallCancelFrame": _on_tool_cancelled,
        "TTSStartedFrame": _on_tts_started,
        "TTSTextFrame": _on_tts_text,
        "TTSStoppedFrame": _on_tts_stopped,
        "BotStartedSpeakingFrame": _on_playback_started,
        "BotStoppedSpeakingFrame": _on_playback_stopped,
        "LLMContextAssistantTurnFrame": _on_agent_turn_stopped,
        "InterruptionFrame": _on_interruption,
        "UserTurnResponseCancelFrame": _on_interruption,
        "EagerEndOfTurnCancelFrame": _on_interruption,
        "MetricsFrame": _on_metrics,
    }
