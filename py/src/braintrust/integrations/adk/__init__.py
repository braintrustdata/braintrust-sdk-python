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
    warnings.warn("setup_braintrust is deprecated, use setup_adk instead", DeprecationWarning, stacklevel=2)
    return setup_adk(*args, **kwargs)


def setup_adk(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
    SpanProcessor: type | None = None,
) -> bool:
    """
    Setup Braintrust integration with Google ADK. Will automatically patch Google ADK agents, runners, flows, and MCP tools for automatic tracing.

    Also enables the Google GenAI integration, which ADK's Gemini models call
    through. It owns the ``llm`` span and token usage for each Gemini request;
    ADK's ``llm_call`` spans are ``task`` spans so usage is counted once. For
    other ADK models (e.g. LiteLLM or Anthropic), enable that provider's
    integration, or use ``braintrust.auto_instrument()``, to record usage.

    If you prefer manual patching take a look at `wrap_agent`, `wrap_runner`, `wrap_workflow`, `wrap_flow`, and `wrap_mcp_tool`.

    Args:
        api_key (Optional[str]): Braintrust API key.
        project_id (Optional[str]): Braintrust project ID.
        project_name (Optional[str]): Braintrust project name.
        SpanProcessor (Optional[type]): Deprecated parameter.

    Returns:
        bool: True if setup was successful, False otherwise.
    """
    if SpanProcessor is not None:
        warnings.warn("SpanProcessor parameter is deprecated and will be ignored", DeprecationWarning, stacklevel=2)

    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    from braintrust.integrations.google_genai import GoogleGenAIIntegration

    GoogleGenAIIntegration.setup()
    return ADKIntegration.setup()
