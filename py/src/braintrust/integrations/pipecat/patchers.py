"""Pipecat integration patchers."""

import inspect
from contextvars import ContextVar
from typing import Any

from braintrust._audio.config import AudioRecordingOptions
from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import BraintrustPipecatObserver
from .tracing._live import is_live_service


_DEFAULT_OBSERVER_OPTIONS: dict[str, Any] = {
    "capture_audio_attachments": None,
    "capture_user_audio_attachments": None,
    "capture_agent_audio_attachments": None,
    "audio_recording_options": None,
}


def set_default_observer_options(**options: Any) -> None:
    """Set options used when global patching injects a Braintrust observer."""
    _DEFAULT_OBSERVER_OPTIONS.update(options)


_OBSERVER_ATTR = "_braintrust_pipecat_observer"
_RESPONSE_SPAN_ATTR = "_braintrust_pipecat_response_span"
# The observer of the running worker. Every processor task inherits it.
_ACTIVE_OBSERVER: ContextVar[BraintrustPipecatObserver | None] = ContextVar(
    "braintrust_pipecat_observer", default=None
)


def _with_braintrust_observer(observers: Any) -> tuple[list[Any], BraintrustPipecatObserver]:
    ret = list(observers or [])
    for observer in ret:
        if isinstance(observer, BraintrustPipecatObserver):
            return ret, observer
    observer = BraintrustPipecatObserver(**_DEFAULT_OBSERVER_OPTIONS)
    ret.append(observer)
    return ret, observer


def traced_pipeline_worker_init(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Inject the Braintrust Pipecat observer into PipelineWorker construction."""
    kwargs = dict(kwargs)
    kwargs["observers"], observer = _with_braintrust_observer(kwargs.get("observers"))
    result = wrapped(*args, **kwargs)
    setattr(instance, _OBSERVER_ATTR, observer)
    return result


async def traced_pipeline_worker_run(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Run a worker with its session span current, so nested spans join it."""
    observer = getattr(instance, _OBSERVER_ATTR, None)
    if observer is None:
        return await wrapped(*args, **kwargs)
    token = _ACTIVE_OBSERVER.set(observer)
    try:
        with observer._current_session():
            return await wrapped(*args, **kwargs)
    finally:
        _ACTIVE_OBSERVER.reset(token)


def _release_response_span(service: Any) -> None:
    span = getattr(service, _RESPONSE_SPAN_ATTR, None)
    if span is not None:
        setattr(service, _RESPONSE_SPAN_ATTR, None)
        span.unset_current()


async def traced_llm_push_frame(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Make an LLM service's llm_response span current while it makes the call.

    Cascade LLM services push their response start frame, send the provider
    request and push their response end frame from the same task. Spans the
    provider integration starts in between nest under llm_response.
    """
    observer = _ACTIVE_OBSERVER.get()
    frame = args[0] if args else kwargs.get("frame")
    kind = type(frame).__name__
    if (
        observer is None
        or kind not in ("LLMFullResponseStartFrame", "LLMFullResponseEndFrame")
        or is_live_service(instance)
    ):
        return await wrapped(*args, **kwargs)
    _release_response_span(instance)
    if kind == "LLMFullResponseEndFrame":
        return await wrapped(*args, **kwargs)
    result, span = await observer._response_span(frame, lambda: wrapped(*args, **kwargs))
    if span is not None:
        span.set_current()
        setattr(instance, _RESPONSE_SPAN_ATTR, span)
    return result


async def traced_stt_process_generator(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Activate recognition in the service task that actually runs STT.

    STT services consume their run_stt generator here, including segmented
    services that execute it in a separate task. Merely constructing that
    generator, or activating a span in the observer, cannot scope its work.
    """
    observer = _ACTIVE_OBSERVER.get()
    if observer is None:
        return await wrapped(*args, **kwargs)
    with observer._current_recognition(instance):
        return await wrapped(*args, **kwargs)


def _controller_wait(instance):
    observer = _ACTIVE_OBSERVER.get()
    return observer._controller(instance) if observer is not None else None


def traced_observer_proxy(wrapped, instance, args, kwargs):
    proxy = wrapped(*args, **kwargs)
    if isinstance(proxy.observer, BraintrustPipecatObserver):
        proxy.observer._controller_queue = proxy.queue
    return proxy


class WorkerObserverProxyPatcher(FunctionWrapperPatcher):
    name = "worker_observer_proxy"
    target_module = "pipecat.pipeline.worker_observer"
    target_path = "WorkerObserver._create_proxy"
    wrapper = staticmethod(traced_observer_proxy)


async def traced_controller_frame(wrapped, instance, args, kwargs):
    from .tracing._eou import CONDITION, SPEECH_TIMEOUT_STRATEGY, stop_strategies

    condition = None
    try:
        wait = _controller_wait(instance)
        if wait is not None:
            frame = args[0] if args else kwargs["frame"]
            wait.observe(frame)
            if type(frame).__name__ == "TranscriptionFrame" and frame.finalized:
                strategies = stop_strategies(instance)
                strategy = strategies[0] if len(strategies) == 1 else None
                if (
                    (type(strategy).__module__, type(strategy).__name__) == SPEECH_TIMEOUT_STRATEGY
                    and strategy._user_speech_wait_done
                    and not strategy._stt_wait_done
                ):
                    condition = (strategy, "transcription_finalized")
    except Exception:
        from .tracing._common import LOG

        LOG.warning("Unable to observe Pipecat turn wait", exc_info=True)
    token = CONDITION.set(condition)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        CONDITION.reset(token)


async def traced_controller_event(wrapped, instance, args, kwargs):
    name = args[0] if args else kwargs.get("event_name")
    try:
        if name in ("on_user_turn_started", "on_user_turn_stopped"):
            wait = _controller_wait(instance)
            if wait is not None:
                wait.event(name, args[1:])
    except Exception:
        from .tracing._common import LOG

        LOG.warning("Unable to observe Pipecat turn decision", exc_info=True)
    return await wrapped(*args, **kwargs)


async def traced_speech_timeout(wrapped, instance, args, kwargs):
    from .tracing._eou import CONDITION

    token = CONDITION.set((instance, "silence_timeout"))
    try:
        return await wrapped(*args, **kwargs)
    finally:
        CONDITION.reset(token)


class UserTurnControllerFramePatcher(FunctionWrapperPatcher):
    name = "user_turn_controller_frame"
    target_module = "pipecat.turns.user_turn_controller"
    target_path = "UserTurnController.process_frame"
    wrapper = staticmethod(traced_controller_frame)


class UserTurnControllerEventPatcher(FunctionWrapperPatcher):
    name = "user_turn_controller_event"
    target_module = "pipecat.turns.user_turn_controller"
    target_path = "UserTurnController._call_event_handler"
    wrapper = staticmethod(traced_controller_event)


class SpeechTimeoutPatcher(FunctionWrapperPatcher):
    name = "speech_timeout_eou_condition"
    target_module = "pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy"
    target_path = "SpeechTimeoutUserTurnStopStrategy._user_speech_timeout_handler"
    wrapper = staticmethod(traced_speech_timeout)


class EOUWaitPatcher(CompositeFunctionWrapperPatcher):
    """Install queue routing, lifecycle and causal observations together."""

    name = "eou_wait"
    sub_patchers = (
        WorkerObserverProxyPatcher,
        UserTurnControllerFramePatcher,
        UserTurnControllerEventPatcher,
        SpeechTimeoutPatcher,
    )

    @classmethod
    def applies(cls, module, version, *, target=None):
        for patcher, asynchronous, arguments in zip(
            cls.sub_patchers,
            (False, True, True, True),
            ({"observer": None}, {"frame": None}, {"event_name": "on_user_turn_stopped"}, {"timeout": 0}),
        ):
            method = patcher.resolve_target(module, version, target=target)
            if not callable(method) or inspect.iscoroutinefunction(method) != asynchronous:
                return False
            try:
                inspect.signature(method).bind(None, **arguments)
            except (TypeError, ValueError):
                return False
        return super().applies(module, version, target=target)


class PipelineWorkerInitPatcher(FunctionWrapperPatcher):
    """Patch ``PipelineWorker.__init__`` to add a Braintrust observer."""

    name = "pipeline_worker_init"
    target_module = "pipecat.pipeline.worker"
    target_path = "PipelineWorker.__init__"
    wrapper = staticmethod(traced_pipeline_worker_init)


class PipelineWorkerRunPatcher(FunctionWrapperPatcher):
    """Patch ``PipelineWorker.run`` to make the session span current."""

    name = "pipeline_worker_run"
    target_module = "pipecat.pipeline.worker"
    target_path = "PipelineWorker.run"
    wrapper = staticmethod(traced_pipeline_worker_run)


class LLMServicePushFramePatcher(FunctionWrapperPatcher):
    """Patch ``LLMService.push_frame`` to nest provider spans under responses."""

    name = "llm_service_push_frame"
    target_module = "pipecat.services.llm_service"
    target_path = "LLMService.push_frame"
    wrapper = staticmethod(traced_llm_push_frame)


class STTServiceProcessGeneratorPatcher(FunctionWrapperPatcher):
    """Scope provider transcription spans under their Pipecat recognition."""

    name = "stt_service_process_generator"
    target_module = "pipecat.services.stt_service"
    target_path = "STTService.process_generator"
    wrapper = staticmethod(traced_stt_process_generator)


class SegmentedSTTPushSegmentFramesPatcher(FunctionWrapperPatcher):
    """Scope segmented STT transcription spans under their Pipecat recognition.

    Since Pipecat 1.16, ``SegmentedSTTService`` consumes ``run_stt`` in
    ``_push_segment_frames`` instead of ``process_generator``. Older versions
    lack the method, so the patcher skips them.
    """

    name = "segmented_stt_push_segment_frames"
    target_module = "pipecat.services.stt_service"
    target_path = "SegmentedSTTService._push_segment_frames"
    wrapper = staticmethod(traced_stt_process_generator)


class PipelineWorkerPatcher(CompositeFunctionWrapperPatcher):
    """Instrument ``PipelineWorker`` construction and runs together."""

    name = "pipeline_worker"
    sub_patchers = (PipelineWorkerInitPatcher, PipelineWorkerRunPatcher)


async def traced_live_event(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    event = args[0] if args else kwargs.get("evt")
    if observer is not None:
        observer._live_event(instance, event)
    result = await wrapped(*args, **kwargs)
    if observer is not None:
        observer._live_event_processed(event)
    return result


async def traced_live_client_event(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    result = await wrapped(*args, **kwargs)
    if observer is not None:
        observer._live_client_event(args[0] if args else kwargs.get("event"))
    return result


class OpenAILiveEventPatcher(FunctionWrapperPatcher):
    name = "openai_live_event"
    target_module = "pipecat.services.openai.live.llm"
    target_path = "OpenAILiveLLMService._handle_server_event"
    wrapper = staticmethod(traced_live_event)


class OpenAILiveClientEventPatcher(FunctionWrapperPatcher):
    name = "openai_live_client_event"
    target_module = "pipecat.services.openai.live.llm"
    target_path = "OpenAILiveLLMService.send_client_event"
    wrapper = staticmethod(traced_live_client_event)


class OpenAILivePatcher(CompositeFunctionWrapperPatcher):
    name = "openai_live"
    sub_patchers = (OpenAILiveEventPatcher, OpenAILiveClientEventPatcher)

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        # Both hooks are needed to track a response through tool submission.
        # Let the composite own patch markers and idempotency after this gate.
        for patcher, argument in ((OpenAILiveEventPatcher, "evt"), (OpenAILiveClientEventPatcher, "event")):
            method = patcher.resolve_target(module, version, target=target)
            if not inspect.iscoroutinefunction(method):
                return False
            try:
                inspect.signature(method).bind(None, **{argument: None})
            except (TypeError, ValueError):
                return False
        return super().applies(module, version, target=target)


def wrap_pipeline_worker(
    PipelineWorker: Any,
    *,
    capture_audio_attachments: bool | None = None,
    capture_user_audio_attachments: bool | None = None,
    capture_agent_audio_attachments: bool | None = None,
    audio_recording_options: AudioRecordingOptions | None = None,
) -> Any:
    """Instrument a Pipecat ``PipelineWorker`` class directly.

    Every worker the class constructs afterwards is traced. Options match
    ``setup_pipecat()``. Options omitted here keep their current values, so
    wrapping again never discards earlier configuration; unset options fall
    back to the generic audio capture environment variables.
    """
    set_default_observer_options(
        **{
            name: value
            for name, value in {
                "capture_audio_attachments": capture_audio_attachments,
                "capture_user_audio_attachments": capture_user_audio_attachments,
                "capture_agent_audio_attachments": capture_agent_audio_attachments,
                "audio_recording_options": audio_recording_options,
            }.items()
            if value is not None
        }
    )
    from pipecat.services.llm_service import LLMService  # pylint: disable=import-error
    from pipecat.services.stt_service import SegmentedSTTService, STTService  # pylint: disable=import-error

    # The same gate and composite own global markers for every entrypoint.
    if EOUWaitPatcher.applies(None, None):
        EOUWaitPatcher.patch(None, None)
    LLMServicePushFramePatcher.wrap_target(LLMService)
    STTServiceProcessGeneratorPatcher.wrap_target(STTService)
    SegmentedSTTPushSegmentFramesPatcher.wrap_target(SegmentedSTTService)
    # Optional on older Pipecat versions; the patchers feature-detect the module.
    if OpenAILivePatcher.applies(None, None):
        OpenAILivePatcher.patch(None, None)
    return PipelineWorkerPatcher.wrap_target(PipelineWorker)
