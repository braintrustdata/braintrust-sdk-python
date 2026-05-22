"""OpenAI Agents SDK integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import OpenAIAgentsTracingPatcher


class OpenAIAgentsIntegration(BaseIntegration):
    """Braintrust instrumentation for the OpenAI Agents SDK."""

    name = "openai_agents"
    import_names = ("agents",)
    min_version = "0.0.19"
    patchers = (OpenAIAgentsTracingPatcher,)
