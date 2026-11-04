from braintrust.integrations.base import BaseIntegration

from .patchers import AgentSessionPatcher


class LiveKitAgentsIntegration(BaseIntegration):
    name = "livekit_agents"
    import_names = ("livekit.agents",)
    distribution_names = ("livekit-agents",)
    min_version = "1.8.10"
    max_version = "1.8.10"
    patchers = (AgentSessionPatcher,)
