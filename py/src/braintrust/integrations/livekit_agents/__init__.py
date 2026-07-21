from .integration import LiveKitAgentsIntegration
from .patchers import wrap_livekit_agents


def setup_livekit_agents() -> bool:
    return LiveKitAgentsIntegration.setup()


__all__ = ["LiveKitAgentsIntegration", "setup_livekit_agents", "wrap_livekit_agents"]
