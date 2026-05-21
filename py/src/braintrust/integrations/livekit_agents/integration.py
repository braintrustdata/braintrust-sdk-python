"""LiveKit Agents integration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import AgentSessionPatcher, MetricsEmitterPatcher


class LiveKitAgentsIntegration(BaseIntegration):
    name = "livekit_agents"
    import_names = ("livekit.agents",)
    distribution_names = ("livekit-agents",)
    min_version = "1.3.1"
    patchers = (AgentSessionPatcher, MetricsEmitterPatcher)
