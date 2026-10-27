"""Braintrust integration for the OpenAI Python SDK and OpenAI-compatible gateways."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import OpenAIIntegration
from .patchers import wrap_openai


__all__ = [
    "OpenAIIntegration",
    "setup_openai",
    "wrap_openai",
]


def setup_openai(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Set up Braintrust tracing for the OpenAI Python SDK.

    Returns:
        True if patched or already patched, False if OpenAI is not installed.
    """
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return OpenAIIntegration.setup()
