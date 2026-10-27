"""Braintrust integration for google-cloud-discoveryengine v1."""

from .integration import GoogleDiscoveryEngineIntegration
from .patchers import wrap_google_discoveryengine


__all__ = ["GoogleDiscoveryEngineIntegration", "setup_google_discoveryengine", "wrap_google_discoveryengine"]


def setup_google_discoveryengine() -> bool:
    """Set up Braintrust tracing for Discovery Engine.

    Returns True if patched or already patched, False if the library is missing or too old.
    """
    return GoogleDiscoveryEngineIntegration.setup()
