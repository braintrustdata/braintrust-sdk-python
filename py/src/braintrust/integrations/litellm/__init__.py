"""Braintrust LiteLLM integration."""

import importlib

from .integration import LiteLLMIntegration
from .patchers import LiteLLMAcompletionPatcher, LiteLLMCompletionPatcher, wrap_litellm


def patch_litellm() -> bool:
    """Patch LiteLLM to add Braintrust tracing.

    This wraps litellm.completion, litellm.acompletion, litellm.responses,
    litellm.aresponses, litellm.image_generation, litellm.aimage_generation,
    litellm.embedding, litellm.aembedding, litellm.moderation,
    litellm.speech, litellm.aspeech, litellm.transcription,
    litellm.atranscription, litellm.rerank, and litellm.arerank to automatically
    create Braintrust spans with detailed token metrics, timing, and costs.

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


def _is_litellm_patched() -> bool:
    """Return ``True`` when the LiteLLM completion entry points are patched.

    Internal helper used by other Braintrust integrations (e.g. CrewAI,
    which delegates its LLM calls to LiteLLM) to decide whether to emit
    token metrics on their own LLM spans.  When LiteLLM is already
    patched, the Braintrust ``Completion`` span it produces is the leaf
    span and must own token accounting; non-leaf wrappers should emit
    timing-only metrics to avoid double-counting during trace-tree
    rollup.

    Not part of the public API — cross-integration callers import this
    directly by its underscore-prefixed name.

    Returns ``False`` if ``litellm`` is not importable or neither of the
    ``completion`` / ``acompletion`` entry points has been patched.
    """
    try:
        litellm = importlib.import_module("litellm")
    except ImportError:
        return False

    completion = getattr(litellm, "completion", None)
    acompletion = getattr(litellm, "acompletion", None)
    return bool(
        LiteLLMCompletionPatcher.has_patch_marker(completion)
        or LiteLLMCompletionPatcher.has_patch_marker(litellm)
        or LiteLLMAcompletionPatcher.has_patch_marker(acompletion)
        or LiteLLMAcompletionPatcher.has_patch_marker(litellm)
    )


__all__ = [
    "LiteLLMIntegration",
    "patch_litellm",
    "wrap_litellm",
]
