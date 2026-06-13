"""AgentScope integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AgentCallPatcher,
    AgentReplyPatcher,
    ChatModelPatcher,
    FanoutPipelinePatcher,
    SequentialPipelinePatcher,
    ToolkitCallToolFunctionPatcher,
    ToolkitCallToolPatcher,
)


class AgentScopeIntegration(BaseIntegration):
    """Braintrust instrumentation for AgentScope. Requires AgentScope v1.0.0 or higher."""

    name = "agentscope"
    import_names = ("agentscope",)
    min_version = "1.0.0"
    patchers = (
        AgentCallPatcher,
        AgentReplyPatcher,
        SequentialPipelinePatcher,
        FanoutPipelinePatcher,
        ToolkitCallToolFunctionPatcher,
        ToolkitCallToolPatcher,
        ChatModelPatcher,
    )
