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
    """Set up Braintrust tracing for the Google GenAI SDK.

    Returns:
        True if patched or already patched, False if google-genai is missing or too old.
    """
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return GoogleGenAIIntegration.setup()
