"""Pipecat integration patchers."""

from contextvars import ContextVar
from typing import Any

from braintrust._audio.config import AudioRecordingOptions
from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import BraintrustPipecatObserver


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

    Every LLM service pushes its response start frame, sends the provider
    request and pushes its response end frame from the same task. Spans the
    provider integration starts in between nest under llm_response.
    """
    observer = _ACTIVE_OBSERVER.get()
    frame = args[0] if args else kwargs.get("frame")
    kind = type(frame).__name__
    if observer is None or kind not in ("LLMFullResponseStartFrame", "LLMFullResponseEndFrame"):
        return await wrapped(*args, **kwargs)
    _release_response_span(instance)
    if kind == "LLMFullResponseEndFrame":
        return await wrapped(*args, **kwargs)
    result, span = await observer._response_span(frame, lambda: wrapped(*args, **kwargs))
    if span is not None:
        span.set_current()
        setattr(instance, _RESPONSE_SPAN_ATTR, span)
    return result


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


class PipelineWorkerPatcher(CompositeFunctionWrapperPatcher):
    """Instrument ``PipelineWorker`` construction and runs together."""

    name = "pipeline_worker"
    sub_patchers = (PipelineWorkerInitPatcher, PipelineWorkerRunPatcher)


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

    LLMServicePushFramePatcher.wrap_target(LLMService)
    return PipelineWorkerPatcher.wrap_target(PipelineWorker)
