"""Discovery Engine integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import PATCHERS


class DiscoveryEngineIntegration(BaseIntegration):
    name = "discoveryengine"
    import_names = ("google.cloud.discoveryengine_v1",)
    distribution_names = ("google-cloud-discoveryengine",)
    min_version = "0.20.3"
    patchers = PATCHERS
