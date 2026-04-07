"""LangChain integration definition."""

from braintrust.integrations.base import BaseIntegration

from .patchers import GlobalHandlerPatcher


class LangChainIntegration(BaseIntegration):
    """Braintrust instrumentation for LangChain via a global callback handler."""

    name = "langchain"
    import_names = ("langchain_core",)
    patchers = (GlobalHandlerPatcher,)
