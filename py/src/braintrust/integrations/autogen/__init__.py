"""Braintrust AutoGen integration."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import AutoGenIntegration


def setup_autogen(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Set up Braintrust tracing for AutoGen.

    Returns:
        True if patched or already patched, False if AutoGen is missing or too old.
    """
    if current_span() == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return AutoGenIntegration.setup()


__all__ = ["AutoGenIntegration", "setup_autogen"]
