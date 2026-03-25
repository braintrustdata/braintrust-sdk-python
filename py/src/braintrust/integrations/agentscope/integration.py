"""AgentScope integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AgentCallPatcher,
    ChatModelPatcher,
    FanoutPipelinePatcher,
    SequentialPipelinePatcher,
    ToolkitCallToolFunctionPatcher,
)


class AgentScopeIntegration(BaseIntegration):
    """Braintrust instrumentation for AgentScope. Requires AgentScope v1.0.0 or higher."""

    name = "agentscope"
    import_names = ("agentscope",)
    min_version = "1.0.0"
    patchers = (
        AgentCallPatcher,
        SequentialPipelinePatcher,
        FanoutPipelinePatcher,
        ToolkitCallToolFunctionPatcher,
        ChatModelPatcher,
    )
