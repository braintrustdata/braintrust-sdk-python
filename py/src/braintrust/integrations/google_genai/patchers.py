"""Google GenAI patchers — one patcher per coherent patch target."""

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import (
    _async_embed_content_wrapper,
    _async_generate_content_stream_wrapper,
    _async_generate_content_wrapper,
    _async_generate_images_wrapper,
    _async_interactions_cancel_wrapper,
    _async_interactions_create_wrapper,
    _async_interactions_delete_wrapper,
    _async_interactions_get_wrapper,
    _embed_content_wrapper,
    _generate_content_stream_wrapper,
    _generate_content_wrapper,
    _generate_images_wrapper,
    _interactions_cancel_wrapper,
    _interactions_create_wrapper,
    _interactions_delete_wrapper,
    _interactions_get_wrapper,
)


# ---------------------------------------------------------------------------
# Sync Models patchers
# ---------------------------------------------------------------------------


class ModelsGenerateContentPatcher(FunctionWrapperPatcher):
    """Patch ``Models._generate_content`` for tracing."""

    name = "google_genai.models.generate_content"
    target_module = "google.genai.models"
    target_path = "Models._generate_content"
    wrapper = _generate_content_wrapper


class ModelsGenerateContentStreamPatcher(FunctionWrapperPatcher):
    """Patch ``Models.generate_content_stream`` for tracing."""

    name = "google_genai.models.generate_content_stream"
    target_module = "google.genai.models"
    target_path = "Models.generate_content_stream"
    wrapper = _generate_content_stream_wrapper


class ModelsEmbedContentPatcher(FunctionWrapperPatcher):
    """Patch ``Models.embed_content`` for tracing."""

    name = "google_genai.models.embed_content"
    target_module = "google.genai.models"
    target_path = "Models.embed_content"
    wrapper = _embed_content_wrapper


class ModelsGenerateImagesPatcher(FunctionWrapperPatcher):
    """Patch ``Models.generate_images`` for tracing."""

    name = "google_genai.models.generate_images"
    target_module = "google.genai.models"
    target_path = "Models.generate_images"
    wrapper = _generate_images_wrapper


# ---------------------------------------------------------------------------
# Sync Interactions patchers
# ---------------------------------------------------------------------------


class InteractionsCreatePatcher(FunctionWrapperPatcher):
    """Patch ``InteractionsResource.create`` for tracing."""

    name = "google_genai.interactions.create"
    target_module = "google.genai._interactions.resources.interactions"
    target_path = "InteractionsResource.create"
    wrapper = _interactions_create_wrapper


class InteractionsGetPatcher(FunctionWrapperPatcher):
    """Patch ``InteractionsResource.get`` for tracing."""

    name = "google_genai.interactions.get"
    target_module = "google.genai._interactions.resources.interactions"
    target_path = "InteractionsResource.get"
    wrapper = _interactions_get_wrapper


class InteractionsCancelPatcher(FunctionWrapperPatcher):
    """Patch ``InteractionsResource.cancel`` for tracing."""

    name = "google_genai.interactions.cancel"
    target_module = "google.genai._interactions.resources.interactions"
    target_path = "InteractionsResource.cancel"
    wrapper = _interactions_cancel_wrapper


class InteractionsDeletePatcher(FunctionWrapperPatcher):
    """Patch ``InteractionsResource.delete`` for tracing."""

    name = "google_genai.interactions.delete"
    target_module = "google.genai._interactions.resources.interactions"
    target_path = "InteractionsResource.delete"
    wrapper = _interactions_delete_wrapper


# ---------------------------------------------------------------------------
# Async Models patchers
# ---------------------------------------------------------------------------


class AsyncModelsGenerateContentPatcher(FunctionWrapperPatcher):
    """Patch ``AsyncModels.generate_content`` for tracing."""

    name = "google_genai.async_models.generate_content"
    target_module = "google.genai.models"
    target_path = "AsyncModels.generate_content"
    wrapper = _async_generate_content_wrapper


class AsyncModelsGenerateContentStreamPatcher(FunctionWrapperPatcher):
    """Patch ``AsyncModels.generate_content_stream`` for tracing."""

    name = "google_genai.async_models.generate_content_stream"
    target_module = "google.genai.models"
    target_path = "AsyncModels.generate_content_stream"
    wrapper = _async_generate_content_stream_wrapper


class AsyncModelsEmbedContentPatcher(FunctionWrapperPatcher):
    """Patch ``AsyncModels.embed_content`` for tracing."""

    name = "google_genai.async_models.embed_content"
    target_module = "google.genai.models"
    target_path = "AsyncModels.embed_content"
    wrapper = _async_embed_content_wrapper


class AsyncModelsGenerateImagesPatcher(FunctionWrapperPatcher):
    """Patch ``AsyncModels.generate_images`` for tracing."""

    name = "google_genai.async_models.generate_images"
    target_module = "google.genai.models"
    target_path = "AsyncModels.generate_images"
    wrapper = _async_generate_images_wrapper


# ---------------------------------------------------------------------------
# Async Interactions patchers
# ---------------------------------------------------------------------------


class AsyncInteractionsCreatePatcher(FunctionWrapperPatcher):
    """Patch ``AsyncInteractionsResource.create`` for tracing."""

    name = "google_genai.async_interactions.create"
    target_module = "google.genai._interactions.resources.interactions"
    target_path = "AsyncInteractionsResource.create"
    wrapper = _async_interactions_create_wrapper


class AsyncInteractionsGetPatcher(FunctionWrapperPatcher):
    """Patch ``AsyncInteractionsResource.get`` for tracing."""

    name = "google_genai.async_interactions.get"
    target_module = "google.genai._interactions.resources.interactions"
    target_path = "AsyncInteractionsResource.get"
    wrapper = _async_interactions_get_wrapper


class AsyncInteractionsCancelPatcher(FunctionWrapperPatcher):
    """Patch ``AsyncInteractionsResource.cancel`` for tracing."""

    name = "google_genai.async_interactions.cancel"
    target_module = "google.genai._interactions.resources.interactions"
    target_path = "AsyncInteractionsResource.cancel"
    wrapper = _async_interactions_cancel_wrapper


class AsyncInteractionsDeletePatcher(FunctionWrapperPatcher):
    """Patch ``AsyncInteractionsResource.delete`` for tracing."""

    name = "google_genai.async_interactions.delete"
    target_module = "google.genai._interactions.resources.interactions"
    target_path = "AsyncInteractionsResource.delete"
    wrapper = _async_interactions_delete_wrapper
