"""Cohere integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import V2ChatPatcher, V2EmbedPatcher, V2RerankPatcher


class CohereIntegration(BaseIntegration):
    """Braintrust instrumentation for the Cohere Python SDK."""

    name = "cohere"
    import_names = ("cohere",)
    min_version = "5.10.0"
    patchers = (
        V2ChatPatcher,
        V2EmbedPatcher,
        V2RerankPatcher,
    )
