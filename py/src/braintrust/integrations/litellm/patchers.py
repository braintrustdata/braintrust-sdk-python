"""LiteLLM patchers — FunctionWrapperPatcher subclasses for each patch target."""

from typing import Any

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import (
    _acompletion_wrapper_async,
    _aembedding_wrapper_async,
    _aimage_generation_wrapper_async,
    _amoderation_wrapper_async,
    _arerank_wrapper_async,
    _aresponses_wrapper_async,
    _aspeech_wrapper_async,
    _atext_completion_wrapper_async,
    _atranscription_wrapper_async,
    _completion_wrapper,
    _embedding_wrapper,
    _image_generation_wrapper,
    _moderation_wrapper,
    _rerank_wrapper,
    _responses_wrapper,
    _speech_wrapper,
    _text_completion_wrapper,
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


class LiteLLMTextCompletionPatcher(FunctionWrapperPatcher):
    name = "litellm.text_completion"
    target_path = "text_completion"
    wrapper = _text_completion_wrapper


class LiteLLMATextCompletionPatcher(FunctionWrapperPatcher):
    name = "litellm.atext_completion"
    target_path = "atext_completion"
    wrapper = _atext_completion_wrapper_async


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


class LiteLLMAModerationPatcher(FunctionWrapperPatcher):
    name = "litellm.amoderation"
    target_path = "amoderation"
    wrapper = _amoderation_wrapper_async


class LiteLLMSpeechPatcher(FunctionWrapperPatcher):
    name = "litellm.speech"
    target_path = "speech"
    wrapper = _speech_wrapper


class LiteLLMAspeechPatcher(FunctionWrapperPatcher):
    name = "litellm.aspeech"
    target_path = "aspeech"
    wrapper = _aspeech_wrapper_async


class LiteLLMTranscriptionPatcher(FunctionWrapperPatcher):
    name = "litellm.transcription"
    target_path = "transcription"
    wrapper = _transcription_wrapper


class LiteLLMATranscriptionPatcher(FunctionWrapperPatcher):
    name = "litellm.atranscription"
    target_path = "atranscription"
    wrapper = _atranscription_wrapper_async


class LiteLLMRerankPatcher(FunctionWrapperPatcher):
    name = "litellm.rerank"
    target_path = "rerank"
    wrapper = _rerank_wrapper


class LiteLLMArerankPatcher(FunctionWrapperPatcher):
    name = "litellm.arerank"
    target_path = "arerank"
    wrapper = _arerank_wrapper_async


# ---------------------------------------------------------------------------
# All patchers, in declaration order
# ---------------------------------------------------------------------------

_ALL_LITELLM_PATCHERS = (
    LiteLLMCompletionPatcher,
    LiteLLMAcompletionPatcher,
    LiteLLMTextCompletionPatcher,
    LiteLLMATextCompletionPatcher,
    LiteLLMResponsesPatcher,
    LiteLLMAresponsesPatcher,
    LiteLLMImageGenerationPatcher,
    LiteLLMAimageGenerationPatcher,
    LiteLLMEmbeddingPatcher,
    LiteLLMAembeddingPatcher,
    LiteLLMModerationPatcher,
    LiteLLMAModerationPatcher,
    LiteLLMSpeechPatcher,
    LiteLLMAspeechPatcher,
    LiteLLMTranscriptionPatcher,
    LiteLLMATranscriptionPatcher,
    LiteLLMRerankPatcher,
    LiteLLMArerankPatcher,
)


# ---------------------------------------------------------------------------
# Manual wrapping helper
# ---------------------------------------------------------------------------


def wrap_litellm(litellm: Any) -> Any:
    """Wrap a LiteLLM module-like object in-place with Braintrust tracing.

    Idempotent; safe to call twice on the same object.
    """
    for patcher in _ALL_LITELLM_PATCHERS:
        patcher.wrap_target(litellm)
    return litellm
