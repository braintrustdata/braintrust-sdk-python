"""Public integration classes, loaded only when their attributes are accessed."""

from importlib import import_module
from typing import TYPE_CHECKING, Any


_INTEGRATION_MODULES = {
    "ADKIntegration": "adk",
    "AgentScopeIntegration": "agentscope",
    "AgnoIntegration": "agno",
    "AISDKIntegration": "ai_sdk",
    "AnthropicIntegration": "anthropic",
    "AutoGenIntegration": "autogen",
    "BedrockRuntimeIntegration": "bedrock_runtime",
    "ClaudeAgentSDKIntegration": "claude_agent_sdk",
    "CohereIntegration": "cohere",
    "CrewAIIntegration": "crewai",
    "CursorSDKIntegration": "cursor_sdk",
    "DSPyIntegration": "dspy",
    "GoogleDiscoveryEngineIntegration": "google_discoveryengine",
    "GoogleGenAIIntegration": "google_genai",
    "HuggingFaceHubIntegration": "huggingface_hub",
    "InstructorIntegration": "instructor",
    "LangChainIntegration": "langchain",
    "LiteLLMIntegration": "litellm",
    "LiveKitAgentsIntegration": "livekit_agents",
    "LlamaIndexIntegration": "llamaindex",
    "MistralIntegration": "mistral",
    "OpenAIIntegration": "openai",
    "OpenAIAgentsIntegration": "openai_agents",
    "OpenRouterIntegration": "openrouter",
    "PipecatIntegration": "pipecat",
    "PydanticAIIntegration": "pydantic_ai",
    "StrandsIntegration": "strands",
    "TemporalIntegration": "temporal",
    "TransformersIntegration": "transformers",
    "TypeSafeIntegration": "typesafe",
}


if TYPE_CHECKING:
    from .adk import ADKIntegration as ADKIntegration
    from .agentscope import AgentScopeIntegration as AgentScopeIntegration
    from .agno import AgnoIntegration as AgnoIntegration
    from .ai_sdk import AISDKIntegration as AISDKIntegration
    from .anthropic import AnthropicIntegration as AnthropicIntegration
    from .autogen import AutoGenIntegration as AutoGenIntegration
    from .bedrock_runtime import BedrockRuntimeIntegration as BedrockRuntimeIntegration
    from .claude_agent_sdk import ClaudeAgentSDKIntegration as ClaudeAgentSDKIntegration
    from .cohere import CohereIntegration as CohereIntegration
    from .crewai import CrewAIIntegration as CrewAIIntegration
    from .cursor_sdk import CursorSDKIntegration as CursorSDKIntegration
    from .dspy import DSPyIntegration as DSPyIntegration
    from .google_discoveryengine import GoogleDiscoveryEngineIntegration as GoogleDiscoveryEngineIntegration
    from .google_genai import GoogleGenAIIntegration as GoogleGenAIIntegration
    from .huggingface_hub import HuggingFaceHubIntegration as HuggingFaceHubIntegration
    from .instructor import InstructorIntegration as InstructorIntegration
    from .langchain import LangChainIntegration as LangChainIntegration
    from .litellm import LiteLLMIntegration as LiteLLMIntegration
    from .livekit_agents import LiveKitAgentsIntegration as LiveKitAgentsIntegration
    from .llamaindex import LlamaIndexIntegration as LlamaIndexIntegration
    from .mistral import MistralIntegration as MistralIntegration
    from .openai import OpenAIIntegration as OpenAIIntegration
    from .openai_agents import OpenAIAgentsIntegration as OpenAIAgentsIntegration
    from .openrouter import OpenRouterIntegration as OpenRouterIntegration
    from .pipecat import PipecatIntegration as PipecatIntegration
    from .pydantic_ai import PydanticAIIntegration as PydanticAIIntegration
    from .strands import StrandsIntegration as StrandsIntegration
    from .temporal import TemporalIntegration as TemporalIntegration
    from .transformers import TransformersIntegration as TransformersIntegration
    from .typesafe import TypeSafeIntegration as TypeSafeIntegration


__all__ = [
    "ADKIntegration",
    "AgentScopeIntegration",
    "AgnoIntegration",
    "AISDKIntegration",
    "AnthropicIntegration",
    "AutoGenIntegration",
    "BedrockRuntimeIntegration",
    "ClaudeAgentSDKIntegration",
    "CohereIntegration",
    "CrewAIIntegration",
    "CursorSDKIntegration",
    "DSPyIntegration",
    "GoogleDiscoveryEngineIntegration",
    "GoogleGenAIIntegration",
    "HuggingFaceHubIntegration",
    "InstructorIntegration",
    "LiteLLMIntegration",
    "LiveKitAgentsIntegration",
    "LangChainIntegration",
    "LlamaIndexIntegration",
    "MistralIntegration",
    "OpenAIIntegration",
    "OpenAIAgentsIntegration",
    "OpenRouterIntegration",
    "PipecatIntegration",
    "PydanticAIIntegration",
    "StrandsIntegration",
    "TemporalIntegration",
    "TransformersIntegration",
    "TypeSafeIntegration",
]


def __getattr__(name: str) -> Any:
    module_name = _INTEGRATION_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
