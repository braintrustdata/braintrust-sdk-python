"""AutoGen integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import ChatAgentPatcher, FunctionToolRunPatcher, TeamPatcher


class AutoGenIntegration(BaseIntegration):
    """Braintrust instrumentation for Microsoft AutoGen AgentChat."""

    name = "autogen"
    import_names = ("autogen_agentchat",)
    min_version = "0.7.0"
    patchers = (ChatAgentPatcher, TeamPatcher, FunctionToolRunPatcher)
