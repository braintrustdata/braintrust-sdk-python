from .integration import LiveKitAgentsIntegration
from .patchers import wrap_livekit_agents


def setup_livekit_agents() -> bool:
    """Set up LiveKit Agents tracing.

    Set ``BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=false`` to omit agent
    playback audio attachments while preserving the ``agent_speaking`` spans
    and transcripts.
    """
    return LiveKitAgentsIntegration.setup()


__all__ = ["LiveKitAgentsIntegration", "setup_livekit_agents", "wrap_livekit_agents"]
