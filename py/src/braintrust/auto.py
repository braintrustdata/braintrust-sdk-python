"""
Auto-instrumentation for AI/ML libraries.

Provides one-line instrumentation for supported libraries.
"""

import logging
from contextlib import contextmanager

from braintrust.integrations import (
    ADKIntegration,
    AgentScopeIntegration,
    AgnoIntegration,
    AnthropicIntegration,
    AutoGenIntegration,
    ClaudeAgentSDKIntegration,
    CohereIntegration,
    CrewAIIntegration,
    DSPyIntegration,
    GoogleGenAIIntegration,
    LangChainIntegration,
    LiteLLMIntegration,
    MistralIntegration,
    OpenAIAgentsIntegration,
    OpenAIIntegration,
    OpenRouterIntegration,
    PydanticAIIntegration,
)
from braintrust.integrations.base import BaseIntegration


__all__ = ["auto_instrument"]

logger = logging.getLogger(__name__)


@contextmanager
def _try_patch():
    """Context manager that suppresses ImportError and logs other exceptions."""
    try:
        yield
    except ImportError:
        pass
    except Exception:
        logger.exception("Failed to instrument")


def auto_instrument(
    *,
    openai: bool = True,
    anthropic: bool = True,
    litellm: bool = True,
    pydantic_ai: bool = True,
    google_genai: bool = True,
    openrouter: bool = True,
    mistral: bool = True,
    agno: bool = True,
    agentscope: bool = True,
    claude_agent_sdk: bool = True,
    dspy: bool = True,
    adk: bool = True,
    langchain: bool = True,
    openai_agents: bool = True,
    cohere: bool = True,
    autogen: bool = True,
    crewai: bool = True,
) -> dict[str, bool]:
    """
    Auto-instrument supported AI/ML libraries for Braintrust tracing.

    Safe to call multiple times - already instrumented libraries are skipped.

    Note on import order: If you use `from openai import OpenAI` style imports,
    call auto_instrument() first. If you use `import openai` style imports,
    order doesn't matter since attribute lookup happens dynamically.

    Args:
        openai: Enable OpenAI instrumentation (default: True)
        anthropic: Enable Anthropic instrumentation (default: True)
        litellm: Enable LiteLLM instrumentation (default: True)
        pydantic_ai: Enable Pydantic AI instrumentation (default: True)
        google_genai: Enable Google GenAI instrumentation (default: True)
        openrouter: Enable OpenRouter instrumentation (default: True)
        mistral: Enable Mistral instrumentation (default: True)
        agno: Enable Agno instrumentation (default: True)
        agentscope: Enable AgentScope instrumentation (default: True)
        claude_agent_sdk: Enable Claude Agent SDK instrumentation (default: True)
        dspy: Enable DSPy instrumentation (default: True)
        adk: Enable Google ADK instrumentation (default: True)
        langchain: Enable LangChain instrumentation (default: True)
        openai_agents: Enable OpenAI Agents SDK instrumentation (default: True)
        cohere: Enable Cohere instrumentation (default: True)
        autogen: Enable AutoGen instrumentation (default: True)
        crewai: Enable CrewAI instrumentation (default: True)

    Returns:
        Dict mapping integration name to whether it was successfully instrumented.

    Example:
        ```python
        import braintrust
        braintrust.auto_instrument()

        # OpenAI
        import openai
        client = openai.OpenAI()
        client.chat.completions.create(model="gpt-4o-mini", messages=[...])

        # Anthropic
        import anthropic
        client = anthropic.Anthropic()
        client.messages.create(model="claude-sonnet-4-20250514", messages=[...])

        # LiteLLM
        import litellm
        litellm.completion(model="gpt-4o-mini", messages=[...])

        # DSPy
        import dspy
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)

        # Google ADK
        from google.adk import Agent
        from google.adk.runners import Runner
        agent = Agent(name="my_agent", model="gemini-2.0-flash")
        runner = Runner(agent=agent, app_name="my_app")

        # Pydantic AI
        from pydantic_ai import Agent
        agent = Agent("openai:gpt-4o-mini")
        result = agent.run_sync("Hello!")

        # Google GenAI
        from google.genai import Client
        client = Client()
        client.models.generate_content(model="gemini-2.0-flash", contents="Hello!")
        ```
    """
    results: dict[str, bool] = {}

    if openai:
        results["openai"] = _instrument_integration(OpenAIIntegration)
    if anthropic:
        results["anthropic"] = _instrument_integration(AnthropicIntegration)
    if litellm:
        results["litellm"] = _instrument_integration(LiteLLMIntegration)
    if pydantic_ai:
        results["pydantic_ai"] = _instrument_integration(PydanticAIIntegration)
    if google_genai:
        results["google_genai"] = _instrument_integration(GoogleGenAIIntegration)
    if openrouter:
        results["openrouter"] = _instrument_integration(OpenRouterIntegration)
    if mistral:
        results["mistral"] = _instrument_integration(MistralIntegration)
    if agno:
        results["agno"] = _instrument_integration(AgnoIntegration)
    if agentscope:
        results["agentscope"] = _instrument_integration(AgentScopeIntegration)
    if claude_agent_sdk:
        results["claude_agent_sdk"] = _instrument_integration(ClaudeAgentSDKIntegration)
    if dspy:
        results["dspy"] = _instrument_integration(DSPyIntegration)
    if adk:
        results["adk"] = _instrument_integration(ADKIntegration)
    if langchain:
        results["langchain"] = _instrument_integration(LangChainIntegration)
    if openai_agents:
        results["openai_agents"] = _instrument_integration(OpenAIAgentsIntegration)
    if cohere:
        results["cohere"] = _instrument_integration(CohereIntegration)
    if autogen:
        results["autogen"] = _instrument_integration(AutoGenIntegration)
    if crewai:
        results["crewai"] = _instrument_integration(CrewAIIntegration)

    return results


def _instrument_integration(integration: type[BaseIntegration]) -> bool:
    with _try_patch():
        return integration.setup()
    return False
