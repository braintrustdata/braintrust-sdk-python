"""Shared utilities for Anthropic API wrappers."""

from typing import Any

from braintrust.util import is_numeric


class Wrapper:
    """Base wrapper class with __getattr__ delegation to preserve original types."""

    def __init__(self, wrapped: Any):
        self.__wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__wrapped, name)


_ANTHROPIC_USAGE_METRIC_FIELDS = (
    ("input_tokens", "prompt_tokens"),
    ("output_tokens", "completion_tokens"),
    ("cache_read_input_tokens", "prompt_cached_tokens"),
    ("cache_creation_input_tokens", "prompt_cache_creation_tokens"),
)

_ANTHROPIC_CACHE_CREATION_METRIC_FIELDS = (
    ("ephemeral_5m_input_tokens", "prompt_cache_creation_ephemeral_5m_tokens"),
    ("ephemeral_1h_input_tokens", "prompt_cache_creation_ephemeral_1h_tokens"),
)

_ANTHROPIC_SERVER_TOOL_USE_METRIC_FIELDS = (
    ("web_search_requests", "server_tool_use_web_search_requests"),
    ("web_fetch_requests", "server_tool_use_web_fetch_requests"),
)

_ANTHROPIC_USAGE_METADATA_FIELDS = frozenset(
    {
        "service_tier",
        "inference_geo",
    }
)


def _try_to_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        try:
            candidate = obj.model_dump(mode="python")
        except TypeError:
            candidate = obj.model_dump()
        return candidate if isinstance(candidate, dict) else None

    if hasattr(obj, "to_dict"):
        candidate = obj.to_dict()
        return candidate if isinstance(candidate, dict) else None

    if hasattr(obj, "dict"):
        candidate = obj.dict()
        return candidate if isinstance(candidate, dict) else None

    if hasattr(obj, "__dict__"):
        return vars(obj)

    return None


def _set_numeric_metric(metrics: dict[str, float], name: str, value: Any) -> None:
    if is_numeric(value):
        metrics[name] = float(value)


def extract_anthropic_usage(usage: Any) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract normalized metrics and allowlisted metadata from Anthropic usage.

    Numeric usage fields are converted into Braintrust metrics. Allowlisted
    non-numeric fields are attached as span metadata with a ``usage_`` prefix.
    """
    usage = _try_to_dict(usage)
    if usage is None:
        return {}, {}

    metrics: dict[str, float] = {}
    for source_name, metric_name in _ANTHROPIC_USAGE_METRIC_FIELDS:
        _set_numeric_metric(metrics, metric_name, usage.get(source_name))

    cache_creation = _try_to_dict(usage.get("cache_creation"))
    cache_creation_breakdown: list[float] = []
    if cache_creation is not None:
        for source_name, metric_name in _ANTHROPIC_CACHE_CREATION_METRIC_FIELDS:
            value = cache_creation.get(source_name)
            _set_numeric_metric(metrics, metric_name, value)
            if is_numeric(value):
                cache_creation_breakdown.append(float(value))

    server_tool_use = _try_to_dict(usage.get("server_tool_use"))
    if server_tool_use is not None:
        for source_name, metric_name in _ANTHROPIC_SERVER_TOOL_USE_METRIC_FIELDS:
            _set_numeric_metric(metrics, metric_name, server_tool_use.get(source_name))

    if "prompt_cache_creation_tokens" not in metrics and cache_creation_breakdown:
        metrics["prompt_cache_creation_tokens"] = sum(cache_creation_breakdown)

    if metrics:
        total_prompt_tokens = (
            metrics.get("prompt_tokens", 0)
            + metrics.get("prompt_cached_tokens", 0)
            + metrics.get("prompt_cache_creation_tokens", 0)
        )
        metrics["prompt_tokens"] = total_prompt_tokens
        metrics["tokens"] = total_prompt_tokens + metrics.get("completion_tokens", 0)

    metadata = {
        f"usage_{name}": value
        for name, value in usage.items()
        if name in _ANTHROPIC_USAGE_METADATA_FIELDS and value is not None
    }
    return metrics, metadata
