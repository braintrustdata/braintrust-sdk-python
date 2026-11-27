import warnings

from .batch import (
    complete_anthropic_batch_trace,
    complete_anthropic_batch_trace_async,
    start_anthropic_batch_trace,
)
from .integration import AnthropicIntegration
from .tracing import _wrap_anthropic


wrap_anthropic = _wrap_anthropic


def wrap_anthropic_client(client):
    warnings.warn(
        "wrap_anthropic_client() is deprecated. Use wrap_anthropic() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _wrap_anthropic(client)


__all__ = [
    "AnthropicIntegration",
    "complete_anthropic_batch_trace",
    "complete_anthropic_batch_trace_async",
    "start_anthropic_batch_trace",
    "wrap_anthropic",
    "wrap_anthropic_client",
]
