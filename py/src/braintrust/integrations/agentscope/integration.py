"""AgentScope integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AgentCallPatcher,
    AgentReplyPatcher,
    ChatModelPatcher,
    FanoutPipelinePatcher,
    SequentialPipelinePatcher,
    TeamPipelineReplyStreamPatcher,
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
        TeamPipelineReplyStreamPatcher,
        ToolkitCallToolFunctionPatcher,
        ToolkitCallToolPatcher,
        ChatModelPatcher,
    )
