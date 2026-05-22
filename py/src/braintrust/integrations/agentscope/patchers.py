"""AgentScope patchers."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agent_call_wrapper,
    _fanout_pipeline_wrapper,
    _model_call_wrapper,
    _sequential_pipeline_wrapper,
    _toolkit_call_tool_function_wrapper,
)


class AgentCallPatcher(FunctionWrapperPatcher):
    """Patch AgentScope agent execution."""

    name = "agentscope.agent.call"
    target_module = "agentscope.agent"
    target_path = "AgentBase.__call__"
    wrapper = _agent_call_wrapper


class SequentialPipelinePatcher(FunctionWrapperPatcher):
    """Patch AgentScope sequential pipeline execution."""

    name = "agentscope.pipeline.sequential"
    target_module = "agentscope.pipeline"
    target_path = "sequential_pipeline"
    wrapper = _sequential_pipeline_wrapper


class FanoutPipelinePatcher(FunctionWrapperPatcher):
    """Patch AgentScope fanout pipeline execution."""

    name = "agentscope.pipeline.fanout"
    target_module = "agentscope.pipeline"
    target_path = "fanout_pipeline"
    wrapper = _fanout_pipeline_wrapper


class ToolkitCallToolFunctionPatcher(FunctionWrapperPatcher):
    """Patch AgentScope toolkit execution."""

    name = "agentscope.tool.call_tool_function"
    target_module = "agentscope.tool"
    target_path = "Toolkit.call_tool_function"
    wrapper = _toolkit_call_tool_function_wrapper


class _OpenAIChatModelPatcher(FunctionWrapperPatcher):
    name = "agentscope.model.openai"
    target_module = "agentscope.model"
    target_path = "OpenAIChatModel.__call__"
    wrapper = _model_call_wrapper


class _DashScopeChatModelPatcher(FunctionWrapperPatcher):
    name = "agentscope.model.dashscope"
    target_module = "agentscope.model"
    target_path = "DashScopeChatModel.__call__"
    wrapper = _model_call_wrapper


class _AnthropicChatModelPatcher(FunctionWrapperPatcher):
    name = "agentscope.model.anthropic"
    target_module = "agentscope.model"
    target_path = "AnthropicChatModel.__call__"
    wrapper = _model_call_wrapper


class _OllamaChatModelPatcher(FunctionWrapperPatcher):
    name = "agentscope.model.ollama"
    target_module = "agentscope.model"
    target_path = "OllamaChatModel.__call__"
    wrapper = _model_call_wrapper


class _GeminiChatModelPatcher(FunctionWrapperPatcher):
    name = "agentscope.model.gemini"
    target_module = "agentscope.model"
    target_path = "GeminiChatModel.__call__"
    wrapper = _model_call_wrapper


class _TrinityChatModelPatcher(FunctionWrapperPatcher):
    name = "agentscope.model.trinity"
    target_module = "agentscope.model"
    target_path = "TrinityChatModel.__call__"
    wrapper = _model_call_wrapper


class ChatModelPatcher(CompositeFunctionWrapperPatcher):
    """Patch the built-in AgentScope chat model implementations."""

    name = "agentscope.model"
    sub_patchers = (
        _OpenAIChatModelPatcher,
        _DashScopeChatModelPatcher,
        _AnthropicChatModelPatcher,
        _OllamaChatModelPatcher,
        _GeminiChatModelPatcher,
        _TrinityChatModelPatcher,
    )
