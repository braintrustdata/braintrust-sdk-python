"""Braintrust integration for Pipecat AI."""

from braintrust._audio.config import AudioRecordingOptions
from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import PipecatIntegration
from .patchers import set_default_observer_options, wrap_pipeline_worker


__all__ = [
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
    audio_recording_options: AudioRecordingOptions | None = None,
) -> bool:
    """Set up Braintrust tracing for Pipecat ``PipelineWorker`` instances.

    Setup instruments ``PipelineWorker`` so every newly constructed worker is
    traced, the same as ``braintrust.auto_instrument()`` or
    ``wrap_pipeline_worker()``. Pass ``capture_audio_attachments=True`` to attach
    both user and agent audio, or configure them independently with
    ``capture_user_audio_attachments`` and ``capture_agent_audio_attachments``.
    When arguments are omitted, the generic
    environment variables ``BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS`` and
    ``BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS`` apply. Explicit arguments
    take precedence.

    Emits the audio v3 schema: a ``session`` with shared recordings, native
    ``user_turn`` / ``agent_turn`` groups, speech, STT, LLM, tools and TTS.
    Turn containers follow Pipecat's native boundaries. Agent recordings
    contain transport-confirmed playback.
    ``audio_recording_options`` configures segmentation, encoding and shutdown
    using ``braintrust._audio.config.AudioRecordingOptions``. Recordings default
    to Opus/Ogg with ``braintrust[audio]`` installed, otherwise raw PCM.
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
