"""AgentScope integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AgentCallPatcher,
    ChatModelPatcher,
    FanoutPipelinePatcher,
    GeneralEvaluatorPatcher,
    MetricCallPatcher,
    RayEvaluatorRunPatcher,
    SequentialPipelinePatcher,
    TaskEvaluatePatcher,
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
        GeneralEvaluatorPatcher,
        RayEvaluatorRunPatcher,
        TaskEvaluatePatcher,
        MetricCallPatcher,
    )

    eval_patchers = (
        GeneralEvaluatorPatcher,
        RayEvaluatorRunPatcher,
        TaskEvaluatePatcher,
        MetricCallPatcher,
    )

    @classmethod
    def setup(
        cls,
        *,
        target=None,
        instrument_evals: bool = True,
    ) -> bool:
        patchers = cls.patchers if instrument_evals else tuple(p for p in cls.patchers if p not in cls.eval_patchers)
        return super().setup(target=target, patchers=patchers)
