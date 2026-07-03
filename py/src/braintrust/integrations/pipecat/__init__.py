"""Braintrust integration for Pipecat AI."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import PipecatIntegration
from .patchers import set_default_observer_options, wrap_pipeline_worker
from .tracing import BraintrustPipecatObserver


__all__ = [
    "BraintrustPipecatObserver",
    "PipecatIntegration",
    "setup_pipecat",
    "wrap_pipeline_worker",
]


def setup_pipecat(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
    *,
    capture_audio_attachments: bool | None = None,
    capture_user_audio_attachments: bool | None = None,
    capture_agent_audio_attachments: bool | None = None,
    trace_turns: bool = True,
) -> bool:
    """Set up Braintrust tracing for Pipecat ``PipelineWorker`` instances.

    The setup hook patches ``PipelineWorker.__init__`` so newly constructed
    workers receive a ``BraintrustPipecatObserver`` unless one was already
    provided explicitly. Pass ``capture_audio_attachments=True`` to attach
    both user and agent audio, or configure them independently with
    ``capture_user_audio_attachments`` and ``capture_agent_audio_attachments``.
    When arguments are omitted, the generic
    environment variables ``BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS`` and
    ``BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS`` apply. Explicit arguments
    take precedence.
    """
    set_default_observer_options(
        capture_audio_attachments=capture_audio_attachments,
        capture_user_audio_attachments=capture_user_audio_attachments,
        capture_agent_audio_attachments=capture_agent_audio_attachments,
        trace_turns=trace_turns,
    )

    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return PipecatIntegration.setup()
