"""OpenAI integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AudioSpeechPatcher,
    AudioTranscriptionsPatcher,
    AudioTranslationsPatcher,
    ChatCompletionsPatcher,
    EmbeddingsPatcher,
    ModerationsPatcher,
    ResponsesPatcher,
)


class OpenAIIntegration(BaseIntegration):
    """Braintrust instrumentation for the OpenAI Python SDK."""

    name = "openai"
    import_names = ("openai",)
    patchers = (
        ChatCompletionsPatcher,
        EmbeddingsPatcher,
        ModerationsPatcher,
        AudioSpeechPatcher,
        AudioTranscriptionsPatcher,
        AudioTranslationsPatcher,
        ResponsesPatcher,
    )
