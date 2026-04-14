"""OpenRouter integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import ChatPatcher, EmbeddingsPatcher, ResponsesPatcher


class OpenRouterIntegration(BaseIntegration):
    """Braintrust instrumentation for the OpenRouter Python SDK."""

    name = "openrouter"
    import_names = ("openrouter",)
    min_version = "0.6.0"
    patchers = (
        ChatPatcher,
        EmbeddingsPatcher,
        ResponsesPatcher,
    )
