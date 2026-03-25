"""Braintrust integration for LangChain."""

from .integration import LangChainIntegration


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


# Lazily imported to avoid circular imports at module load time
# (tracing.py imports from braintrust, which must be fully initialized first)
_LAZY_ATTRS = frozenset(
    ["BraintrustCallbackHandler", "BraintrustTracer", "set_global_handler", "clear_global_handler"]
)


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        from .tracing import BraintrustCallbackHandler, BraintrustTracer, clear_global_handler, set_global_handler

        g = globals()
        g["BraintrustCallbackHandler"] = BraintrustCallbackHandler
        g["BraintrustTracer"] = BraintrustTracer
        g["set_global_handler"] = set_global_handler
        g["clear_global_handler"] = clear_global_handler
        return g[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LangChainIntegration",
    "BraintrustCallbackHandler",
    "BraintrustTracer",
    "set_global_handler",
    "clear_global_handler",
    "setup_langchain",
]
