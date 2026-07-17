"""Braintrust LiteLLM integration."""

from .integration import LiteLLMIntegration
from .patchers import wrap_litellm


def patch_litellm() -> bool:
    """Patch LiteLLM top-level callables to emit Braintrust spans.

    Returns ``True`` if LiteLLM was patched (or already patched), ``False``
    if LiteLLM is not installed.
    """
    return LiteLLMIntegration.setup()


__all__ = [
    "LiteLLMIntegration",
    "patch_litellm",
    "wrap_litellm",
]
