"""Braintrust integration for Agno."""

import logging

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import AgnoIntegration
from .patchers import (
    wrap_agent,
    wrap_function_call,
    wrap_model,
    wrap_team,
    wrap_workflow,
)


logger = logging.getLogger(__name__)

__all__ = [
    "AgnoIntegration",
    "setup_agno",
    "wrap_agent",
    "wrap_function_call",
    "wrap_model",
    "wrap_team",
    "wrap_workflow",
]


def setup_agno(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """
    Setup Braintrust integration with Agno. Will automatically patch Agno agents, models, and function calls for tracing.

    Args:
        api_key: Braintrust API key (optional, can use env var BRAINTRUST_API_KEY)
        project_id: Braintrust project ID (optional)
        project_name: Braintrust project name (optional, can use env var BRAINTRUST_PROJECT)

    Returns:
        True if setup was successful, False otherwise
    """
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return AgnoIntegration.setup()
