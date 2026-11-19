"""Braintrust integration for Pipecat AI."""

from braintrust.audio import AudioRecordingOptions
from braintrust.logger import NOOP_SPAN, current_span, init_logger

from ._model_clients import set_model_client
from .integration import PipecatIntegration
from .patchers import set_default_observer_options, wrap_pipeline_worker


__all__ = [
    "PipecatIntegration",
    "setup_pipecat",
    "set_model_client",
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
    audio_recording_options: AudioRecordingOptions | None = None,
) -> bool:
    """Enable Braintrust tracing for newly created Pipecat ``PipelineWorker`` instances.

    Set ``capture_audio_attachments=True`` to record both sides, or use
    ``capture_user_audio_attachments`` and ``capture_agent_audio_attachments``.
    Omitted capture options use ``BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS``
    and ``BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS`` environment defaults.
    Explicit arguments take precedence.

    Use ``braintrust.audio.AudioRecordingOptions`` with ``audio_recording_options``
    to configure recording format, segmentation, and shutdown.
    """
    set_default_observer_options(
        capture_audio_attachments=capture_audio_attachments,
        capture_user_audio_attachments=capture_user_audio_attachments,
        capture_agent_audio_attachments=capture_agent_audio_attachments,
        audio_recording_options=audio_recording_options,
    )

    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return PipecatIntegration.setup()
