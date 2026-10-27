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
    """Set up Braintrust tracing for Pydantic AI.

    :func:`wrap_model_classes` is deprecated.

    Returns:
        True if patched or already patched, False if Pydantic AI is missing or too old.
    """
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return PydanticAIIntegration.setup()
