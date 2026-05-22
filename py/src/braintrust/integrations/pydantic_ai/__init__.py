"""Braintrust integration for Pydantic AI."""

import logging

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import PydanticAIIntegration
from .patchers import wrap_agent, wrap_model_classes
from .tracing import (
    wrap_model_request,
    wrap_model_request_stream,
    wrap_model_request_stream_sync,
    wrap_model_request_sync,
)


logger = logging.getLogger(__name__)

__all__ = [
    "PydanticAIIntegration",
    "setup_pydantic_ai",
    "wrap_agent",
    "wrap_model_classes",
    "wrap_model_request",
    "wrap_model_request_sync",
    "wrap_model_request_stream",
    "wrap_model_request_stream_sync",
]


def setup_pydantic_ai(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """
    Setup Braintrust integration with Pydantic AI. Will automatically patch Pydantic AI
    agents and direct API functions for automatic tracing.

    Args:
        api_key: Braintrust API key.
        project_id: Braintrust project ID.
        project_name: Braintrust project name.

    Returns:
        True if setup was successful, False otherwise.
    """
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return PydanticAIIntegration.setup()
