"""Braintrust integration for google-cloud-discoveryengine v1."""

from .integration import DiscoveryEngineIntegration
from .patchers import wrap_discoveryengine


__all__ = ["DiscoveryEngineIntegration", "setup_discoveryengine", "wrap_discoveryengine"]


def setup_discoveryengine() -> bool:
    """Instrument supported v1 clients in this process."""
    return DiscoveryEngineIntegration.setup()
