"""Braintrust integration for Google GenAI."""

import logging

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import GoogleGenAIIntegration


logger = logging.getLogger(__name__)

__all__ = [
    "GoogleGenAIIntegration",
    "setup_genai",
]


def setup_genai(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Setup Braintrust integration with Google GenAI.

    Will automatically patch Google GenAI models for automatic tracing.

    Args:
        api_key: Braintrust API key.
        project_id: Braintrust project ID.
        project_name: Braintrust project name.

    Returns:
        True if setup was successful, False if google-genai is not installed.
    """
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return GoogleGenAIIntegration.setup()
