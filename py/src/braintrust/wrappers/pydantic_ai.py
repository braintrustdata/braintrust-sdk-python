"""Compatibility re-exports for the migrated Pydantic AI integration."""

from braintrust.integrations.pydantic_ai import (
    PydanticAIIntegration,
    setup_pydantic_ai,
    wrap_agent,
    wrap_model_classes,
    wrap_model_request,
    wrap_model_request_stream,
    wrap_model_request_stream_sync,
    wrap_model_request_sync,
)


__all__ = [
    "PydanticAIIntegration",
    "setup_pydantic_ai",
    "wrap_agent",
    "wrap_model_classes",
    "wrap_model_request",
    "wrap_model_request_sync",
    "wrap_model_request_stream",
    "wrap_model_request_stream_sync",
]
