"""Google GenAI integration — orchestration class and setup entry-point."""

import logging

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AsyncGAOSInteractionsCancelPatcher,
    AsyncGAOSInteractionsCreatePatcher,
    AsyncGAOSInteractionsDeletePatcher,
    AsyncGAOSInteractionsGetPatcher,
    AsyncInteractionsCancelPatcher,
    AsyncInteractionsCreatePatcher,
    AsyncInteractionsDeletePatcher,
    AsyncInteractionsGetPatcher,
    AsyncModelsEmbedContentPatcher,
    AsyncModelsGenerateContentPatcher,
    AsyncModelsGenerateContentStreamPatcher,
    AsyncModelsGenerateImagesPatcher,
    GAOSInteractionsCancelPatcher,
    GAOSInteractionsCreatePatcher,
    GAOSInteractionsDeletePatcher,
    GAOSInteractionsGetPatcher,
    GenerateContentResponseFromMldevPatcher,
    GenerateContentResponseFromVertexPatcher,
    InteractionsCancelPatcher,
    InteractionsCreatePatcher,
    InteractionsDeletePatcher,
    InteractionsGetPatcher,
    ModelsEmbedContentPatcher,
    ModelsGenerateContentContinuationPatcher,
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
        ModelsGenerateContentContinuationPatcher,
        ModelsGenerateContentPatcher,
        ModelsGenerateContentStreamPatcher,
        GenerateContentResponseFromMldevPatcher,
        GenerateContentResponseFromVertexPatcher,
        ModelsEmbedContentPatcher,
        ModelsGenerateImagesPatcher,
        InteractionsCreatePatcher,
        InteractionsGetPatcher,
        InteractionsCancelPatcher,
        InteractionsDeletePatcher,
        GAOSInteractionsCreatePatcher,
        GAOSInteractionsGetPatcher,
        GAOSInteractionsCancelPatcher,
        GAOSInteractionsDeletePatcher,
        AsyncModelsGenerateContentPatcher,
        AsyncModelsGenerateContentStreamPatcher,
        AsyncModelsEmbedContentPatcher,
        AsyncModelsGenerateImagesPatcher,
        AsyncInteractionsCreatePatcher,
        AsyncInteractionsGetPatcher,
        AsyncInteractionsCancelPatcher,
        AsyncInteractionsDeletePatcher,
        AsyncGAOSInteractionsCreatePatcher,
        AsyncGAOSInteractionsGetPatcher,
        AsyncGAOSInteractionsCancelPatcher,
        AsyncGAOSInteractionsDeletePatcher,
    )
