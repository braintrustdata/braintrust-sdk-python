"""Braintrust integration for AgentScope."""

from typing import Any

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import AgentScopeIntegration
from .patchers import (
    GeneralEvaluatorPatcher,
    MetricCallPatcher,
    RayEvaluatorRunPatcher,
    TaskEvaluatePatcher,
)


__all__ = ["AgentScopeIntegration", "setup_agentscope", "wrap_evaluator"]


def setup_agentscope(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
    instrument_evals: bool = True,
) -> bool:
    """Setup Braintrust integration with AgentScope."""
    if current_span() == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return AgentScopeIntegration.setup(instrument_evals=instrument_evals)


def wrap_evaluator(Evaluator: Any) -> Any:
    """Manually patch an AgentScope evaluator class for tracing.

    This helper patches the evaluator class itself and, when available, also
    enables task and metric tracing from the exported ``agentscope.evaluate``
    module so ``GeneralEvaluator`` produces nested evaluation spans even when
    global setup is not used.
    """
    class_name = getattr(Evaluator, "__name__", "")
    if class_name == "RayEvaluator":
        RayEvaluatorRunPatcher.wrap_target(Evaluator)
    else:
        GeneralEvaluatorPatcher.wrap_target(Evaluator)

    try:
        import agentscope.evaluate as agentscope_evaluate
    except ImportError:
        return Evaluator

    task_cls = getattr(agentscope_evaluate, "Task", None)
    if task_cls is not None:
        TaskEvaluatePatcher.wrap_target(task_cls)

    metric_cls = getattr(agentscope_evaluate, "MetricBase", None)
    if metric_cls is not None:
        MetricCallPatcher.wrap_target(metric_cls)

    return Evaluator
