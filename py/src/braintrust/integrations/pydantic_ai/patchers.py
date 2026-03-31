"""Pydantic AI patchers."""

import warnings
from typing import Any, ClassVar

from braintrust.integrations.base import ClassScanPatcher, CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agent_get_model_wrapper,
    _agent_run_stream_events_wrapper,
    _agent_run_stream_sync_wrapper,
    _agent_run_stream_wrapper,
    _agent_run_sync_wrapper,
    _agent_run_wrapper,
    _agent_to_cli_sync_wrapper,
    _create_direct_model_request_stream_sync_wrapper,
    _create_direct_model_request_stream_wrapper,
    _create_direct_model_request_sync_wrapper,
    _create_direct_model_request_wrapper,
    _create_start_producer_wrapper,
    _direct_prepare_model_wrapper,
    _tool_manager_call_function_tool_wrapper,
    _tool_manager_execute_function_tool_wrapper,
    _wrap_concrete_model_class,
)


class _AgentRunPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.agent.run"
    target_module = "pydantic_ai.agent.abstract"
    target_path = "AbstractAgent.run"
    wrapper = _agent_run_wrapper


class _AgentRunSyncPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.agent.run_sync"
    target_module = "pydantic_ai.agent.abstract"
    target_path = "AbstractAgent.run_sync"
    wrapper = _agent_run_sync_wrapper


class _AgentToCliSyncPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.agent.to_cli_sync"
    target_module = "pydantic_ai.agent.abstract"
    target_path = "AbstractAgent.to_cli_sync"
    wrapper = _agent_to_cli_sync_wrapper


class _AgentRunStreamPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.agent.run_stream"
    target_module = "pydantic_ai.agent.abstract"
    target_path = "AbstractAgent.run_stream"
    wrapper = _agent_run_stream_wrapper


class _AgentRunStreamSyncPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.agent.run_stream_sync"
    target_module = "pydantic_ai.agent.abstract"
    target_path = "AbstractAgent.run_stream_sync"
    wrapper = _agent_run_stream_sync_wrapper


class _AgentRunStreamEventsPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.agent.run_stream_events"
    target_module = "pydantic_ai.agent.abstract"
    target_path = "AbstractAgent.run_stream_events"
    wrapper = _agent_run_stream_events_wrapper


class _AgentGetModelPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.agent.get_model"
    target_module = "pydantic_ai"
    target_path = "Agent._get_model"
    wrapper = _agent_get_model_wrapper


class AgentPatcher(CompositeFunctionWrapperPatcher):
    """Patch Pydantic AI agent entrypoints for tracing."""

    name = "pydantic_ai.agent"
    sub_patchers = (
        _AgentRunPatcher,
        _AgentRunSyncPatcher,
        _AgentToCliSyncPatcher,
        _AgentRunStreamPatcher,
        _AgentRunStreamSyncPatcher,
        _AgentRunStreamEventsPatcher,
        _AgentGetModelPatcher,
    )


class DirectPrepareModelPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.direct.prepare_model"
    target_module = "pydantic_ai.direct"
    target_path = "_prepare_model"
    wrapper = _direct_prepare_model_wrapper


class DirectModelRequestPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.direct.model_request"
    target_module = "pydantic_ai.direct"
    target_path = "model_request"
    wrapper = _create_direct_model_request_wrapper()


class DirectModelRequestSyncPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.direct.model_request_sync"
    target_module = "pydantic_ai.direct"
    target_path = "model_request_sync"
    wrapper = _create_direct_model_request_sync_wrapper()


class DirectModelRequestStreamPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.direct.model_request_stream"
    target_module = "pydantic_ai.direct"
    target_path = "model_request_stream"
    wrapper = _create_direct_model_request_stream_wrapper()


class DirectModelRequestStreamSyncPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.direct.model_request_stream_sync"
    target_module = "pydantic_ai.direct"
    target_path = "model_request_stream_sync"
    wrapper = _create_direct_model_request_stream_sync_wrapper()


class StreamedResponseSyncStartProducerPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.direct.streamed_response_sync.start_producer"
    target_module = "pydantic_ai.direct"
    target_path = "StreamedResponseSync._start_producer"
    wrapper = _create_start_producer_wrapper()
    priority: ClassVar[int] = 50


class _ToolManagerExecuteFunctionToolPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.tool_manager.execute_function_tool"
    target_module = "pydantic_ai._tool_manager"
    target_path = "ToolManager._execute_function_tool_call"
    wrapper = _tool_manager_execute_function_tool_wrapper


class _ToolManagerCallFunctionToolPatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.tool_manager.call_function_tool"
    target_module = "pydantic_ai._tool_manager"
    target_path = "ToolManager._call_function_tool"
    wrapper = _tool_manager_call_function_tool_wrapper
    superseded_by = (_ToolManagerExecuteFunctionToolPatcher,)


class ToolManagerFunctionToolPatcher(CompositeFunctionWrapperPatcher):
    name = "pydantic_ai.tool_manager"
    sub_patchers = (
        _ToolManagerExecuteFunctionToolPatcher,
        _ToolManagerCallFunctionToolPatcher,
    )


def wrap_agent(Agent: Any) -> Any:
    return AgentPatcher.wrap_target(Agent)


class ModelClassesPatcher(ClassScanPatcher):
    """Deprecated compatibility fallback for model subclass scanning.

    Normal setup now wraps resolved models via ``Agent._get_model`` and
    ``pydantic_ai.direct._prepare_model`` instead of relying on subclass scans.
    """

    name = "pydantic_ai.models"
    priority: ClassVar[int] = 200
    target_module = "pydantic_ai.models"
    root_class_path = "Model"

    patch_class = staticmethod(_wrap_concrete_model_class)


def wrap_model_class(model_class: Any) -> Any:
    warnings.warn(
        "wrap_model_class() is deprecated and no longer needed for normal setup. "
        "setup_pydantic_ai() now wraps models at runtime via model resolution seams.",
        DeprecationWarning,
        stacklevel=2,
    )
    if ModelClassesPatcher.has_patch_marker(model_class):
        return model_class
    _wrap_concrete_model_class(model_class)
    ModelClassesPatcher.mark_patched(model_class)
    return model_class


def wrap_model_classes() -> bool:
    """Deprecated compatibility shim for scanning currently loaded model subclasses."""
    warnings.warn(
        "wrap_model_classes() is deprecated and no longer needed. "
        "setup_pydantic_ai() now wraps models at runtime via model resolution seams.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not ModelClassesPatcher.applies(None, None):
        return False
    return ClassScanPatcher.patch.__func__(ModelClassesPatcher, None, None)
