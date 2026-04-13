"""LiteLLM patchers — FunctionWrapperPatcher subclasses for each patch target."""

from typing import Any

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import (
    _acompletion_wrapper_async,
    _aembedding_wrapper_async,
    _aimage_generation_wrapper_async,
    _aresponses_wrapper_async,
    _atranscription_wrapper_async,
    _completion_wrapper,
    _embedding_wrapper,
    _image_generation_wrapper,
    _moderation_wrapper,
    _responses_wrapper,
    _transcription_wrapper,
)


# ---------------------------------------------------------------------------
# Individual patchers
# ---------------------------------------------------------------------------


class LiteLLMCompletionPatcher(FunctionWrapperPatcher):
    name = "litellm.completion"
    target_path = "completion"
    wrapper = _completion_wrapper


class LiteLLMAcompletionPatcher(FunctionWrapperPatcher):
    name = "litellm.acompletion"
    target_path = "acompletion"
    wrapper = _acompletion_wrapper_async


class LiteLLMResponsesPatcher(FunctionWrapperPatcher):
    name = "litellm.responses"
    target_path = "responses"
    wrapper = _responses_wrapper


class LiteLLMAresponsesPatcher(FunctionWrapperPatcher):
    name = "litellm.aresponses"
    target_path = "aresponses"
    wrapper = _aresponses_wrapper_async


class LiteLLMImageGenerationPatcher(FunctionWrapperPatcher):
    name = "litellm.image_generation"
    target_path = "image_generation"
    wrapper = _image_generation_wrapper


class LiteLLMAimageGenerationPatcher(FunctionWrapperPatcher):
    name = "litellm.aimage_generation"
    target_path = "aimage_generation"
    wrapper = _aimage_generation_wrapper_async


class LiteLLMEmbeddingPatcher(FunctionWrapperPatcher):
    name = "litellm.embedding"
    target_path = "embedding"
    wrapper = _embedding_wrapper


class LiteLLMAembeddingPatcher(FunctionWrapperPatcher):
    name = "litellm.aembedding"
    target_path = "aembedding"
    wrapper = _aembedding_wrapper_async


class LiteLLMModerationPatcher(FunctionWrapperPatcher):
    name = "litellm.moderation"
    target_path = "moderation"
    wrapper = _moderation_wrapper


class LiteLLMTranscriptionPatcher(FunctionWrapperPatcher):
    name = "litellm.transcription"
    target_path = "transcription"
    wrapper = _transcription_wrapper


class LiteLLMATranscriptionPatcher(FunctionWrapperPatcher):
    name = "litellm.atranscription"
    target_path = "atranscription"
    wrapper = _atranscription_wrapper_async


# ---------------------------------------------------------------------------
# All patchers, in declaration order
# ---------------------------------------------------------------------------

_ALL_LITELLM_PATCHERS = (
    LiteLLMCompletionPatcher,
    LiteLLMAcompletionPatcher,
    LiteLLMResponsesPatcher,
    LiteLLMAresponsesPatcher,
    LiteLLMImageGenerationPatcher,
    LiteLLMAimageGenerationPatcher,
    LiteLLMEmbeddingPatcher,
    LiteLLMAembeddingPatcher,
    LiteLLMModerationPatcher,
    LiteLLMTranscriptionPatcher,
    LiteLLMATranscriptionPatcher,
)


# ---------------------------------------------------------------------------
# Manual wrapping helper
# ---------------------------------------------------------------------------


def wrap_litellm(litellm: Any) -> Any:
    """Wrap a LiteLLM module to add Braintrust tracing.

    Unlike :func:`patch_litellm`, which patches the globally-imported ``litellm``
    module, this function instruments a specific module object (or any object
    that exposes the same top-level callables such as ``completion``,
    ``acompletion``, ``responses``, ``aresponses``, ``image_generation``,
    ``aimage_generation``, ``embedding``, ``aembedding``, ``moderation``,
    ``transcription``, and ``atranscription``). Each patcher is applied
    idempotently — calling
    ``wrap_litellm`` twice on the same object is safe.

    Args:
        litellm: The ``litellm`` module or a module-like object that exposes
            the standard LiteLLM top-level functions.

    Returns:
        The same *litellm* object, with tracing wrappers applied in-place.

    Example::

        import litellm
        from braintrust.integrations.litellm import wrap_litellm

        wrap_litellm(litellm)

        # All subsequent calls are automatically traced.
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
    for patcher in _ALL_LITELLM_PATCHERS:
        patcher.wrap_target(litellm)
    return litellm
