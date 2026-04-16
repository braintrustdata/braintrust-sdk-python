"""Cohere integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    CohereChatPatcher,
    CohereChatStreamPatcher,
    CohereEmbedPatcher,
    CohereRerankPatcher,
)


class CohereIntegration(BaseIntegration):
    """Braintrust instrumentation for the Cohere Python SDK."""

    name = "cohere"
    import_names = ("cohere",)
    min_version = "5.0.0"
    patchers = (
        CohereChatPatcher,
        CohereChatStreamPatcher,
        CohereEmbedPatcher,
        CohereRerankPatcher,
    )
