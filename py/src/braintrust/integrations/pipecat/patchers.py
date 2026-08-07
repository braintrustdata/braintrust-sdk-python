"""Pipecat integration patchers."""

from typing import Any

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import BraintrustPipecatObserver


_DEFAULT_OBSERVER_OPTIONS: dict[str, Any] = {
    "capture_audio_attachments": None,
    "capture_user_audio_attachments": None,
    "capture_agent_audio_attachments": None,
    "trace_turns": True,
}


def set_default_observer_options(**options: Any) -> None:
    """Set options used when global patching injects a Braintrust observer."""
    _DEFAULT_OBSERVER_OPTIONS.update(options)


def _has_braintrust_observer(observers: list[Any]) -> bool:
    return any(isinstance(observer, BraintrustPipecatObserver) for observer in observers)


def _with_braintrust_observer(observers: Any) -> list[Any]:
    ret = list(observers or [])
    if not _has_braintrust_observer(ret):
        ret.append(BraintrustPipecatObserver(**_DEFAULT_OBSERVER_OPTIONS))
    return ret


def traced_pipeline_worker_init(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Inject the Braintrust Pipecat observer into PipelineWorker construction."""
    kwargs = dict(kwargs)
    kwargs["observers"] = _with_braintrust_observer(kwargs.get("observers"))
    return wrapped(*args, **kwargs)


class PipelineWorkerInitPatcher(FunctionWrapperPatcher):
    """Patch ``PipelineWorker.__init__`` to add a Braintrust observer."""

    name = "pipeline_worker_init"
    target_module = "pipecat.pipeline.worker"
    target_path = "PipelineWorker.__init__"
    wrapper = staticmethod(traced_pipeline_worker_init)


# Alias used by the implementation plan and public helper.
PipelineWorkerPatcher = PipelineWorkerInitPatcher


def wrap_pipeline_worker(PipelineWorker: Any) -> Any:
    """Instrument a Pipecat ``PipelineWorker`` class directly."""
    return PipelineWorkerInitPatcher.wrap_target(PipelineWorker)
