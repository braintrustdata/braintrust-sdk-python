"""Braintrust integration for Vercel AI SDK for Python."""

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import AISDKIntegration
from .patchers import unregister_adapter
from .tracing import BraintrustAISDKAdapter


__all__ = [
    "AISDKIntegration",
    "BraintrustAISDKAdapter",
    "patch_ai_sdk",
    "setup_ai_sdk",
    "unpatch_ai_sdk",
]


def setup_ai_sdk(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    """Set up Braintrust tracing for the Vercel AI SDK.

    Returns:
        True if patched or already patched, False if the SDK is missing or too old.
    """
    if current_span() == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)
    return AISDKIntegration.setup()


def patch_ai_sdk() -> bool:
    """Register the telemetry adapter without initializing a Braintrust logger."""
    return AISDKIntegration.setup()


def unpatch_ai_sdk() -> bool:
    """Unregister the Braintrust telemetry adapter. Safe to call repeatedly."""
    return unregister_adapter()
