from .integration import LiveKitAgentsIntegration
from .patchers import wrap_livekit_agents


def setup_livekit_agents() -> bool:
    """Set up LiveKit Agents tracing.

    Agent playback audio attachments are disabled by default. Set
    ``BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true`` to include them on
    ``agent_speaking`` spans.
    """
    return LiveKitAgentsIntegration.setup()


__all__ = ["LiveKitAgentsIntegration", "setup_livekit_agents", "wrap_livekit_agents"]
