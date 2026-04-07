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
    """Setup Braintrust integration with OpenAI.

    Patches OpenAI resource classes at the module level so that all clients
    produce Braintrust tracing spans.

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

    return OpenAIIntegration.setup()
