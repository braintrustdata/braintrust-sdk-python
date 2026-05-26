"""HuggingFace Hub integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    HuggingFaceChatCompletionPatcher,
    HuggingFaceFeatureExtractionPatcher,
    HuggingFaceSentenceSimilarityPatcher,
    HuggingFaceTextGenerationPatcher,
)


class HuggingFaceHubIntegration(BaseIntegration):
    """Braintrust instrumentation for the HuggingFace Hub Python SDK.

    Patches the inference methods on both ``InferenceClient`` (sync) and
    ``AsyncInferenceClient`` (async):

    - ``chat_completion`` — chat completions (also covers the OpenAI alias
      ``client.chat.completions.create`` which is a thin proxy to this method).
    - ``text_generation`` — prompt-based text generation.
    - ``feature_extraction`` — text embeddings as a numpy array.
    - ``sentence_similarity`` — semantic similarity between sentences.

    The ``min_version`` floor of ``0.32.0`` is the earliest release that exposes
    the stable multi-provider ``InferenceClient(provider=...)`` surface,
    including the ``provider="auto"`` routing mode commonly used by callers,
    while still keeping the same ``chat_completion`` / ``text_generation`` /
    ``feature_extraction`` / ``sentence_similarity`` method names this
    integration patches today.
    """

    name = "huggingface_hub"
    import_names = ("huggingface_hub",)
    distribution_names = ("huggingface_hub", "huggingface-hub")
    min_version = "0.32.0"
    patchers = (
        HuggingFaceChatCompletionPatcher,
        HuggingFaceTextGenerationPatcher,
        HuggingFaceFeatureExtractionPatcher,
        HuggingFaceSentenceSimilarityPatcher,
    )
