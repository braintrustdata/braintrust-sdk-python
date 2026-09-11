"""OpenAI integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AgentsSessionsPatcher,
    AudioSpeechPatcher,
    AudioTranscriptionsPatcher,
    AudioTranslationsPatcher,
    ChatCompletionsPatcher,
    EmbeddingsPatcher,
    ImagesPatcher,
    ModerationsPatcher,
    ResponsesPatcher,
)


class OpenAIIntegration(BaseIntegration):
    """Braintrust instrumentation for the OpenAI Python SDK."""

    name = "openai"
    import_names = ("openai",)
    patchers = (
        AgentsSessionsPatcher,
        ChatCompletionsPatcher,
        EmbeddingsPatcher,
        ModerationsPatcher,
        AudioSpeechPatcher,
        AudioTranscriptionsPatcher,
        AudioTranslationsPatcher,
        ImagesPatcher,
        ResponsesPatcher,
    )
