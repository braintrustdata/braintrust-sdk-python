"""Google GenAI integration — orchestration class and setup entry-point."""

import logging

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AsyncModelsEmbedContentPatcher,
    AsyncModelsGenerateContentPatcher,
    AsyncModelsGenerateContentStreamPatcher,
    AsyncModelsGenerateImagesPatcher,
    ModelsEmbedContentPatcher,
    ModelsGenerateContentPatcher,
    ModelsGenerateContentStreamPatcher,
    ModelsGenerateImagesPatcher,
)


logger = logging.getLogger(__name__)


class GoogleGenAIIntegration(BaseIntegration):
    """Braintrust instrumentation for the Google GenAI Python SDK."""

    name = "google_genai"
    import_names = ("google.genai",)
    min_version = "1.30.0"
    patchers = (
        ModelsGenerateContentPatcher,
        ModelsGenerateContentStreamPatcher,
        ModelsEmbedContentPatcher,
        ModelsGenerateImagesPatcher,
        AsyncModelsGenerateContentPatcher,
        AsyncModelsGenerateContentStreamPatcher,
        AsyncModelsEmbedContentPatcher,
        AsyncModelsGenerateImagesPatcher,
    )
