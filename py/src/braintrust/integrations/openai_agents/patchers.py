"""OpenAI Agents SDK patchers."""

from braintrust.integrations.base import CallbackPatcher


def _setup_openai_agents_tracing() -> None:
    from .tracing import _setup_openai_agents_tracing as setup_openai_agents_tracing

    setup_openai_agents_tracing()


def _has_braintrust_tracing_processor() -> bool:
    from .tracing import _has_braintrust_tracing_processor as has_braintrust_tracing_processor

    return has_braintrust_tracing_processor()


class OpenAIAgentsTracingPatcher(CallbackPatcher):
    """Register the Braintrust tracing processor with the OpenAI Agents SDK."""

    name = "openai_agents.tracing"
    target_module = "agents"
    callback = _setup_openai_agents_tracing
    state_getter = _has_braintrust_tracing_processor
