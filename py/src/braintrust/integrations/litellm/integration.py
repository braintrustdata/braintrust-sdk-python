"""LiteLLM integration definition."""

from braintrust.integrations.base import BaseIntegration

from .patchers import _ALL_LITELLM_PATCHERS


class LiteLLMIntegration(BaseIntegration):
    """Braintrust instrumentation for the LiteLLM Python SDK."""

    name = "litellm"
    import_names = ("litellm",)
    patchers = _ALL_LITELLM_PATCHERS
