from braintrust.integrations.base import BaseIntegration

from .patchers import AnthropicAsyncInitPatcher, AnthropicSyncInitPatcher


class AnthropicIntegration(BaseIntegration):
    """Braintrust instrumentation for the Anthropic Python SDK on anthropic>=0.48.0."""

    name = "anthropic"
    import_names = ("anthropic",)
    min_version = "0.48.0"
    patchers = (AnthropicSyncInitPatcher, AnthropicAsyncInitPatcher)
