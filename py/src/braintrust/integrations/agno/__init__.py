"""Braintrust integration for Agno."""

import logging

from braintrust.logger import NOOP_SPAN, current_experiment, current_span, init_logger

from .eval_experiments import configure as _configure_eval_experiments
from .integration import AgnoIntegration
from .patchers import (
    wrap_accuracy_eval,
    wrap_agent,
    wrap_agent_as_judge_eval,
    wrap_eval_suite,
    wrap_function_call,
    wrap_model,
    wrap_performance_eval,
    wrap_reliability_eval,
    wrap_team,
    wrap_workflow,
)


logger = logging.getLogger(__name__)

__all__ = [
    "AgnoIntegration",
    "setup_agno",
    "wrap_accuracy_eval",
    "wrap_agent",
    "wrap_agent_as_judge_eval",
    "wrap_eval_suite",
    "wrap_function_call",
    "wrap_model",
    "wrap_performance_eval",
    "wrap_reliability_eval",
    "wrap_team",
    "wrap_workflow",
]


def setup_agno(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
    eval_experiments: bool | None = None,
) -> bool:
    """
    Setup Braintrust integration with Agno. Will automatically patch Agno agents, models,
    function calls, and evals (``agno.eval``) for tracing.

    Args:
        api_key: Braintrust API key (optional, can use env var BRAINTRUST_API_KEY)
        project_id: Braintrust project ID (optional)
        project_name: Braintrust project name (optional; defaults to the Global project)
        eval_experiments: Whether an eval suite run should open a Braintrust experiment,
            so its cases land as experiment rows rather than logs. Defaults to the
            BRAINTRUST_AGNO_EVAL_EXPERIMENTS env var, which itself defaults to true.
            Individual evals (AccuracyEval and friends) always log to whatever is
            current, so pass eval_experiments=False to keep suite runs in logs too.

    Returns:
        True if setup was successful, False otherwise
    """
    _configure_eval_experiments(eval_experiments)

    # An experiment opened by the caller is the destination for eval rows, so don't
    # install a logger that would only shadow it for non-eval tracing.
    if current_span() == NOOP_SPAN and current_experiment() is None:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return AgnoIntegration.setup()
