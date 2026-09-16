"""Braintrust integration for google-cloud-discoveryengine v1."""

from .integration import GoogleDiscoveryEngineIntegration
from .patchers import wrap_google_discoveryengine


__all__ = ["GoogleDiscoveryEngineIntegration", "setup_google_discoveryengine", "wrap_google_discoveryengine"]


def setup_google_discoveryengine() -> bool:
    """Instrument supported v1 clients in this process."""
    return GoogleDiscoveryEngineIntegration.setup()
