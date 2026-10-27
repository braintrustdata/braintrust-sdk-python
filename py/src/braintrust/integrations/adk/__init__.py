"""Braintrust integration for Google ADK."""

import logging
import warnings

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import ADKIntegration
from .patchers import (
    wrap_agent,
    wrap_flow,
    wrap_mcp_tool,
    wrap_runner,
    wrap_workflow,
)


logger = logging.getLogger(__name__)

__all__ = [
    "ADKIntegration",
    "setup_adk",
    "setup_braintrust",
    "wrap_agent",
    "wrap_runner",
    "wrap_workflow",
    "wrap_flow",
    "wrap_mcp_tool",
]


def setup_braintrust(*args, **kwargs):
    """Deprecated. Call :func:`setup_adk` instead.

    Forwards ``*args`` and ``**kwargs`` to :func:`setup_adk`.
    """
    warnings.warn("setup_braintrust is deprecated, use setup_adk instead", DeprecationWarning, stacklevel=2)
    return setup_adk(*args, **kwargs)


def setup_adk(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
    SpanProcessor: type | None = None,
) -> bool:
    """Set up Braintrust tracing for Google ADK.

    Args:
        api_key (Optional[str]): Braintrust API key.
        project_id (Optional[str]): Braintrust project ID.
        project_name (Optional[str]): Braintrust project name.
        SpanProcessor (Optional[type]): Deprecated parameter.

    Returns:
        True if patched or already patched, False if ADK is missing or too old.
    """
    if SpanProcessor is not None:
        warnings.warn("SpanProcessor parameter is deprecated and will be ignored", DeprecationWarning, stacklevel=2)

    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    from braintrust.integrations.google_genai import GoogleGenAIIntegration

    GoogleGenAIIntegration.setup()
    return ADKIntegration.setup()
