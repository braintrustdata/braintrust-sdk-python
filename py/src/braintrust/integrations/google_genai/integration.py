"""Google GenAI integration — orchestration class and setup entry-point."""

import logging

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AsyncModelsEmbedContentPatcher,
    AsyncModelsGenerateContentPatcher,
    AsyncModelsGenerateContentStreamPatcher,
    ModelsEmbedContentPatcher,
    ModelsGenerateContentPatcher,
    ModelsGenerateContentStreamPatcher,
)


logger = logging.getLogger(__name__)


class GoogleGenAIIntegration(BaseIntegration):
    """Braintrust instrumentation for the Google GenAI Python SDK."""

    name = "google_genai"
    import_names = ("google.genai",)
    patchers = (
        ModelsGenerateContentPatcher,
        ModelsGenerateContentStreamPatcher,
        ModelsEmbedContentPatcher,
        AsyncModelsGenerateContentPatcher,
        AsyncModelsGenerateContentStreamPatcher,
        AsyncModelsEmbedContentPatcher,
    )
