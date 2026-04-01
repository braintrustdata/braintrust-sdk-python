"""Braintrust LiteLLM integration."""

from .integration import LiteLLMIntegration
from .patchers import wrap_litellm


def patch_litellm() -> bool:
    """Patch LiteLLM to add Braintrust tracing.

    This wraps litellm.completion, litellm.acompletion, litellm.responses,
    litellm.aresponses, litellm.embedding, litellm.aembedding, and
    litellm.moderation to automatically create Braintrust spans with
    detailed token metrics,
    timing, and costs.

    Returns:
        True if LiteLLM was patched (or already patched), False if LiteLLM is not installed.

    Example:
        ```python
        import braintrust
        braintrust.integrations.litellm.patch_litellm()

        import litellm
        from braintrust import init_logger

        logger = init_logger(project="my-project")
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}]
        )
        ```
    """
    return LiteLLMIntegration.setup()


__all__ = [
    "LiteLLMIntegration",
    "patch_litellm",
    "wrap_litellm",
]
