"""Agno integration — orchestration class and setup entry-point."""

import logging

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AccuracyEvalPatcher,
    AgentAsJudgeEvalPatcher,
    AgentPatcher,
    EvalSuitePatcher,
    FunctionCallPatcher,
    ModelPatcher,
    PerformanceEvalPatcher,
    ReliabilityEvalPatcher,
    TeamPatcher,
    WorkflowPatcher,
)


logger = logging.getLogger(__name__)


class AgnoIntegration(BaseIntegration):
    """Braintrust instrumentation for Agno."""

    name = "agno"
    import_names = ("agno",)
    min_version = "2.1.0"
    patchers = (
        AgentPatcher,
        TeamPatcher,
        ModelPatcher,
        FunctionCallPatcher,
        WorkflowPatcher,
        AccuracyEvalPatcher,
        AgentAsJudgeEvalPatcher,
        ReliabilityEvalPatcher,
        PerformanceEvalPatcher,
        EvalSuitePatcher,
    )
