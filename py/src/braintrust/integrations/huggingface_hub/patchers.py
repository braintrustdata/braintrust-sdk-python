"""HuggingFace Hub patchers.

Each composite patcher groups the sync and async variants of one
``InferenceClient`` surface so they can be wired up under a single patcher
name. The sync surface lives on ``huggingface_hub.inference._client.InferenceClient``;
the async surface lives on
``huggingface_hub.inference._generated._async_client.AsyncInferenceClient``.
"""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _async_chat_completion_wrapper,
    _async_feature_extraction_wrapper,
    _async_sentence_similarity_wrapper,
    _async_text_generation_wrapper,
    _chat_completion_wrapper,
    _feature_extraction_wrapper,
    _sentence_similarity_wrapper,
    _text_generation_wrapper,
)


_SYNC_MODULE = "huggingface_hub.inference._client"
_ASYNC_MODULE = "huggingface_hub.inference._generated._async_client"


# ---------------------------------------------------------------------------
# chat_completion
#
# ``client.chat.completions.create`` is a thin proxy to ``chat_completion``,
# so patching the method covers both the OpenAI-compatible alias and the
# native HuggingFace surface.
# ---------------------------------------------------------------------------


class ChatCompletionPatcher(FunctionWrapperPatcher):
    name = "huggingface_hub.chat_completion"
    target_module = _SYNC_MODULE
    target_path = "InferenceClient.chat_completion"
    wrapper = _chat_completion_wrapper


class AsyncChatCompletionPatcher(FunctionWrapperPatcher):
    name = "huggingface_hub.async.chat_completion"
    target_module = _ASYNC_MODULE
    target_path = "AsyncInferenceClient.chat_completion"
    wrapper = _async_chat_completion_wrapper


class HuggingFaceChatCompletionPatcher(CompositeFunctionWrapperPatcher):
    name = "huggingface_hub.chat_completion.all"
    sub_patchers = (
        ChatCompletionPatcher,
        AsyncChatCompletionPatcher,
    )


# ---------------------------------------------------------------------------
# text_generation
# ---------------------------------------------------------------------------


class TextGenerationPatcher(FunctionWrapperPatcher):
    name = "huggingface_hub.text_generation"
    target_module = _SYNC_MODULE
    target_path = "InferenceClient.text_generation"
    wrapper = _text_generation_wrapper


class AsyncTextGenerationPatcher(FunctionWrapperPatcher):
    name = "huggingface_hub.async.text_generation"
    target_module = _ASYNC_MODULE
    target_path = "AsyncInferenceClient.text_generation"
    wrapper = _async_text_generation_wrapper


class HuggingFaceTextGenerationPatcher(CompositeFunctionWrapperPatcher):
    name = "huggingface_hub.text_generation.all"
    sub_patchers = (
        TextGenerationPatcher,
        AsyncTextGenerationPatcher,
    )


# ---------------------------------------------------------------------------
# feature_extraction
# ---------------------------------------------------------------------------


class FeatureExtractionPatcher(FunctionWrapperPatcher):
    name = "huggingface_hub.feature_extraction"
    target_module = _SYNC_MODULE
    target_path = "InferenceClient.feature_extraction"
    wrapper = _feature_extraction_wrapper


class AsyncFeatureExtractionPatcher(FunctionWrapperPatcher):
    name = "huggingface_hub.async.feature_extraction"
    target_module = _ASYNC_MODULE
    target_path = "AsyncInferenceClient.feature_extraction"
    wrapper = _async_feature_extraction_wrapper


class HuggingFaceFeatureExtractionPatcher(CompositeFunctionWrapperPatcher):
    name = "huggingface_hub.feature_extraction.all"
    sub_patchers = (
        FeatureExtractionPatcher,
        AsyncFeatureExtractionPatcher,
    )


# ---------------------------------------------------------------------------
# sentence_similarity
# ---------------------------------------------------------------------------


class SentenceSimilarityPatcher(FunctionWrapperPatcher):
    name = "huggingface_hub.sentence_similarity"
    target_module = _SYNC_MODULE
    target_path = "InferenceClient.sentence_similarity"
    wrapper = _sentence_similarity_wrapper


class AsyncSentenceSimilarityPatcher(FunctionWrapperPatcher):
    name = "huggingface_hub.async.sentence_similarity"
    target_module = _ASYNC_MODULE
    target_path = "AsyncInferenceClient.sentence_similarity"
    wrapper = _async_sentence_similarity_wrapper


class HuggingFaceSentenceSimilarityPatcher(CompositeFunctionWrapperPatcher):
    name = "huggingface_hub.sentence_similarity.all"
    sub_patchers = (
        SentenceSimilarityPatcher,
        AsyncSentenceSimilarityPatcher,
    )
