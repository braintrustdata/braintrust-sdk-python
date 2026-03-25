"""LangChain integration orchestration."""

from typing import Any

from braintrust.integrations.base import BaseIntegration, BasePatcher


class LangChainCallbackPatcher(BasePatcher):
    """Patcher that registers a global BraintrustCallbackHandler with LangChain."""

    name = "langchain_callback"
    _patched: bool = False

    @classmethod
    def is_patched(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        return cls._patched

    @classmethod
    def patch(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        from .tracing import BraintrustCallbackHandler, _ensure_hook_registered, set_global_handler

        _ensure_hook_registered()
        handler = BraintrustCallbackHandler()
        set_global_handler(handler)
        cls._patched = True
        return True


class LangChainIntegration(BaseIntegration):
    """Braintrust instrumentation for LangChain."""

    name = "langchain"
    import_names = ("langchain_core",)
    patchers = (LangChainCallbackPatcher,)
