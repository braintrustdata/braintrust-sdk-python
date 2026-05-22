"""Mistral integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AgentsPatcher,
    ChatPatcher,
    ConversationsPatcher,
    EmbeddingsPatcher,
    FimPatcher,
    OcrPatcher,
    SpeechPatcher,
    TranscriptionsPatcher,
)


class MistralIntegration(BaseIntegration):
    """Braintrust instrumentation for the Mistral Python SDK."""

    name = "mistral"
    import_names = ("mistralai", "mistralai.client")
    min_version = "1.12.4"
    patchers = (
        ChatPatcher,
        EmbeddingsPatcher,
        FimPatcher,
        AgentsPatcher,
        ConversationsPatcher,
        OcrPatcher,
        TranscriptionsPatcher,
        SpeechPatcher,
    )
