"""Shared utilities for Anthropic API wrappers."""

from typing import Any


class Wrapper:
    """Base wrapper class with __getattr__ delegation to preserve original types."""

    def __init__(self, wrapped: Any):
        self.__wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__wrapped, name)


def extract_anthropic_usage(usage: Any) -> dict[str, float]:
    """Extract and normalize usage metrics from Anthropic usage object or dict.

    Converts Anthropic's usage format to Braintrust's standard token metric names.
    Handles both object attributes and dictionary keys.

    Args:
        usage: Anthropic usage object (from Message.usage) or dict

    Returns:
        Dictionary with normalized metric names:
        - prompt_tokens (from input_tokens)
        - completion_tokens (from output_tokens)
        - prompt_cached_tokens (from cache_read_input_tokens)
        - prompt_cache_creation_tokens (from cache_creation_input_tokens)
    """
    metrics: dict[str, float] = {}

    if not usage:
        return metrics

    def get_value(key: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(key)
        return getattr(usage, key, None)

    input_tokens = get_value("input_tokens")
    if input_tokens is not None:
        try:
            metrics["prompt_tokens"] = float(input_tokens)
        except (ValueError, TypeError):
            pass

    output_tokens = get_value("output_tokens")
    if output_tokens is not None:
        try:
            metrics["completion_tokens"] = float(output_tokens)
        except (ValueError, TypeError):
            pass

    cache_read_tokens = get_value("cache_read_input_tokens")
    if cache_read_tokens is not None:
        try:
            metrics["prompt_cached_tokens"] = float(cache_read_tokens)
        except (ValueError, TypeError):
            pass

    cache_creation_tokens = get_value("cache_creation_input_tokens")
    if cache_creation_tokens is not None:
        try:
            metrics["prompt_cache_creation_tokens"] = float(cache_creation_tokens)
        except (ValueError, TypeError):
            pass

    return metrics


def finalize_anthropic_tokens(metrics: dict[str, float]) -> dict[str, float]:
    """Finalize Anthropic token calculations."""
    total_prompt_tokens = (
        metrics.get("prompt_tokens", 0)
        + metrics.get("prompt_cached_tokens", 0)
        + metrics.get("prompt_cache_creation_tokens", 0)
    )

    return {
        **metrics,
        "prompt_tokens": total_prompt_tokens,
        "tokens": total_prompt_tokens + metrics.get("completion_tokens", 0),
    }
