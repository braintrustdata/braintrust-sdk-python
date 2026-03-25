"""Braintrust integration for LangChain."""

from .integration import LangChainIntegration
from .tracing import BraintrustCallbackHandler, BraintrustTracer, clear_global_handler, set_global_handler


def setup_langchain() -> bool:
    """
    Auto-instrument LangChain for Braintrust tracing.

    Registers a global BraintrustCallbackHandler with LangChain's callback system
    so that all chains, LLMs, tools, and retrievers are automatically traced.

    This is called automatically by braintrust.auto_instrument(). It is safe to
    call multiple times – subsequent calls are no-ops.

    Returns:
        True if setup succeeded, False if langchain_core is not installed.
    """
    return LangChainIntegration.setup()


__all__ = [
    "LangChainIntegration",
    "BraintrustCallbackHandler",
    "BraintrustTracer",
    "set_global_handler",
    "clear_global_handler",
    "setup_langchain",
]
