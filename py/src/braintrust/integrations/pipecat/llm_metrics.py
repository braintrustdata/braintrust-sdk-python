"""Shared conversion of native Pipecat usage into Braintrust LLM metrics."""

from typing import Any


def _metadata_from_processor(processor: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    settings = getattr(processor, "_settings", None)
    model = getattr(settings, "model", None) or getattr(processor, "model", None)
    if isinstance(model, str):
        metadata["model"] = model
    provider = _provider_from_processor(processor)
    if provider:
        metadata["provider"] = provider
    return metadata


def _metadata_from_metric(metric: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    model = getattr(metric, "model", None)
    if isinstance(model, str):
        metadata["model"] = model
    processor = getattr(metric, "processor", None)
    if isinstance(processor, str):
        provider = _provider_from_name(processor)
        if provider:
            metadata["provider"] = provider
    return metadata


def _provider_from_processor(processor: Any) -> str | None:
    module = getattr(type(processor), "__module__", "")
    return _provider_from_name(module)


def _provider_from_name(name: str) -> str | None:
    lowered = name.lower()
    providers = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
        "gemini": "google",
        "mistral": "mistral",
        "cohere": "cohere",
        "bedrock": "bedrock",
        "aws": "bedrock",
        "openrouter": "openrouter",
    }
    for needle, provider in providers.items():
        if needle in lowered:
            return provider
    return None


def _llm_usage_metrics(usage: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)
    reasoning_tokens = getattr(usage, "reasoning_tokens", None)
    for key, value in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("tokens", total_tokens),
        ("cache_read_input_tokens", cache_read),
        ("cache_creation_input_tokens", cache_creation),
        ("reasoning_tokens", reasoning_tokens),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = value
    return metrics
