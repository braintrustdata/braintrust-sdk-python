"""AgentScope patchers."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agent_call_wrapper,
    _fanout_pipeline_wrapper,
    _general_evaluator_run_evaluation_wrapper,
    _general_evaluator_run_solution_wrapper,
    _general_evaluator_run_wrapper,
    _metric_call_wrapper,
    _model_call_wrapper,
    _ray_evaluator_run_wrapper,
    _sequential_pipeline_wrapper,
    _task_evaluate_wrapper,
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


class _GeneralEvaluatorRunPatcher(FunctionWrapperPatcher):
    """Patch AgentScope GeneralEvaluator root execution."""

    name = "agentscope.evaluate.general.run"
    target_module = "agentscope.evaluate"
    target_path = "GeneralEvaluator.run"
    wrapper = _general_evaluator_run_wrapper


class _GeneralEvaluatorRunSolutionPatcher(FunctionWrapperPatcher):
    """Patch AgentScope GeneralEvaluator solution execution."""

    name = "agentscope.evaluate.general.run_solution"
    target_module = "agentscope.evaluate"
    target_path = "GeneralEvaluator.run_solution"
    wrapper = _general_evaluator_run_solution_wrapper


class _GeneralEvaluatorRunEvaluationPatcher(FunctionWrapperPatcher):
    """Patch AgentScope GeneralEvaluator evaluation execution."""

    name = "agentscope.evaluate.general.run_evaluation"
    target_module = "agentscope.evaluate"
    target_path = "GeneralEvaluator.run_evaluation"
    wrapper = _general_evaluator_run_evaluation_wrapper


class GeneralEvaluatorPatcher(CompositeFunctionWrapperPatcher):
    """Patch AgentScope GeneralEvaluator for Braintrust eval tracing."""

    name = "agentscope.evaluate.general"
    sub_patchers = (
        _GeneralEvaluatorRunPatcher,
        _GeneralEvaluatorRunSolutionPatcher,
        _GeneralEvaluatorRunEvaluationPatcher,
    )


class RayEvaluatorRunPatcher(FunctionWrapperPatcher):
    """Patch AgentScope RayEvaluator root execution."""

    name = "agentscope.evaluate.ray"
    target_module = "agentscope.evaluate"
    target_path = "RayEvaluator.run"
    wrapper = _ray_evaluator_run_wrapper


class TaskEvaluatePatcher(FunctionWrapperPatcher):
    """Patch AgentScope task evaluation."""

    name = "agentscope.evaluate.task"
    target_module = "agentscope.evaluate"
    target_path = "Task.evaluate"
    wrapper = _task_evaluate_wrapper


class MetricCallPatcher(FunctionWrapperPatcher):
    """Patch AgentScope metric execution."""

    name = "agentscope.evaluate.metric"
    target_module = "agentscope.evaluate"
    target_path = "MetricBase.__call__"
    wrapper = _metric_call_wrapper
