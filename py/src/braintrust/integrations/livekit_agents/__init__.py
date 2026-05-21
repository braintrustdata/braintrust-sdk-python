"""Braintrust integration for LiveKit Agents."""

from .integration import LiveKitAgentsIntegration
from .patchers import wrap_livekit_agents


def setup_livekit_agents() -> bool:
    """Register the Braintrust LiveKit Agents OpenTelemetry span processor."""
    return LiveKitAgentsIntegration.setup()


__all__ = ["LiveKitAgentsIntegration", "setup_livekit_agents", "wrap_livekit_agents"]
