"""Google GenAI patchers — one patcher per coherent patch target."""

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import (
    _async_embed_content_wrapper,
    _async_generate_content_stream_wrapper,
    _async_generate_content_wrapper,
    _async_generate_images_wrapper,
    _embed_content_wrapper,
    _generate_content_stream_wrapper,
    _generate_content_wrapper,
    _generate_images_wrapper,
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
