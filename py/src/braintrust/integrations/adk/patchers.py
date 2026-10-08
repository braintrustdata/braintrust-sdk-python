"""ADK patchers — one patcher per coherent patch target."""

from importlib import import_module
from typing import Any, ClassVar

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agent_run_async_wrapper,
    _create_thread_wrapper,
    _flow_call_llm_async_wrapper,
    _flow_run_async_wrapper,
    _mcp_tool_run_async_wrapper_async,
    _runner_run_async_wrapper,
    _tool_call_async_wrapper,
    _workflow_node_run_wrapper,
    _workflow_tool_node_run_impl_wrapper,
)


# ---------------------------------------------------------------------------
# Agent patcher
# ---------------------------------------------------------------------------


class AgentRunAsyncPatcher(FunctionWrapperPatcher):
    """Patch the ``BaseAgent.run_async`` agent execution entry point."""

    name = "adk.agent.run_async"
    target_module = "google.adk.agents"
    target_path = "BaseAgent.run_async"
    wrapper = _agent_run_async_wrapper


# ---------------------------------------------------------------------------
# Runner patchers
# ---------------------------------------------------------------------------


class _RunnerRunAsyncSubPatcher(FunctionWrapperPatcher):
    """Patch ``Runner.run_async`` (async generator)."""

    name = "adk.runner.run.async"
    target_module = "google.adk.runners"
    target_path = "Runner.run_async"
    wrapper = _runner_run_async_wrapper


class RunnerRunPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``Runner.run_async`` for tracing.

    ``Runner.run()`` already delegates to ``run_async()`` in supported ADK
    versions, so tracing the async surface alone gives sync and async callers a
    single invocation span with the same child structure.
    """

    name = "adk.runner.run"
    sub_patchers = (_RunnerRunAsyncSubPatcher,)


# ---------------------------------------------------------------------------
# Flow patchers
# ---------------------------------------------------------------------------


class _FlowRunAsyncSubPatcher(FunctionWrapperPatcher):
    """Patch ``BaseLlmFlow.run_async``."""

    name = "adk.flow.run_async.run"
    target_module = "google.adk.flows.llm_flows.base_llm_flow"
    target_path = "BaseLlmFlow.run_async"
    wrapper = _flow_run_async_wrapper


class _FlowCallLlmAsyncSubPatcher(FunctionWrapperPatcher):
    """Patch ``BaseLlmFlow._call_llm_async``."""

    name = "adk.flow.run_async.call_llm"
    target_module = "google.adk.flows.llm_flows.base_llm_flow"
    target_path = "BaseLlmFlow._call_llm_async"
    wrapper = _flow_call_llm_async_wrapper


class FlowRunAsyncPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``BaseLlmFlow.run_async`` and ``_call_llm_async`` for tracing."""

    name = "adk.flow.run_async"
    sub_patchers = (_FlowRunAsyncSubPatcher, _FlowCallLlmAsyncSubPatcher)


# ---------------------------------------------------------------------------
# ADK 2.x workflow patchers
# ---------------------------------------------------------------------------


class WorkflowNodeRunPatcher(FunctionWrapperPatcher):
    """Patch ``BaseNode.run`` to trace workflow and graph node execution."""

    name = "adk.workflow.node.run"
    target_module = "google.adk.workflow._base_node"
    target_path = "BaseNode.run"
    wrapper = _workflow_node_run_wrapper


class WorkflowToolNodeRunImplPatcher(FunctionWrapperPatcher):
    """Patch ``_ToolNode._run_impl`` to trace direct workflow tool execution."""

    name = "adk.workflow.tool_node.run_impl"
    target_module = "google.adk.workflow._tool_node"
    target_path = "_ToolNode._run_impl"
    wrapper = _workflow_tool_node_run_impl_wrapper


# ---------------------------------------------------------------------------
# Tool patcher
# ---------------------------------------------------------------------------


class _ToolsCallerCallToolAsyncSubPatcher(FunctionWrapperPatcher):
    """Patch ``tools._caller._call_tool_async`` (ADK >= 2.10.0).

    ADK 2.10.0 moved ``llm_flows._tool_caller`` into the ``llm_flows.tools``
    package as ``_caller``. As in 2.9.0, the call sites resolve the
    module-global name, so the wrapper has to be installed on ``_caller``.
    """

    name = "adk.tool.call_async.tools_caller"
    target_module = "google.adk.flows.llm_flows.tools._caller"
    target_path = "_call_tool_async"
    wrapper = _tool_call_async_wrapper


class _ToolCallerCallToolAsyncSubPatcher(FunctionWrapperPatcher):
    """Patch ``_tool_caller._call_tool_async`` (ADK 2.9.x).

    ADK 2.9.0 moved tool execution out of ``llm_flows.functions`` into the
    ``llm_flows._tool_caller`` module. ``functions`` re-exports the helper, but
    the call sites live in ``_tool_caller`` and resolve the module-global name,
    so the wrapper has to be installed on ``_tool_caller`` itself.
    """

    name = "adk.tool.call_async.tool_caller"
    target_module = "google.adk.flows.llm_flows._tool_caller"
    target_path = "_call_tool_async"
    wrapper = _tool_call_async_wrapper
    superseded_by = (_ToolsCallerCallToolAsyncSubPatcher,)


class _FunctionsCallToolAsyncSubPatcher(FunctionWrapperPatcher):
    """Patch ``functions.__call_tool_async`` (ADK < 2.9.0)."""

    name = "adk.tool.call_async.functions"
    target_module = "google.adk.flows.llm_flows.functions"
    target_path = "__call_tool_async"
    wrapper = _tool_call_async_wrapper
    # Yield to the ``_tool_caller`` target when both exist so a single tool
    # execution never produces two tool spans.
    superseded_by = (_ToolsCallerCallToolAsyncSubPatcher, _ToolCallerCallToolAsyncSubPatcher)


class ToolCallAsyncPatcher(CompositeFunctionWrapperPatcher):
    """Patch ADK's central async tool execution helper for tracing."""

    name = "adk.tool.call_async"
    sub_patchers = (
        _ToolsCallerCallToolAsyncSubPatcher,
        _ToolCallerCallToolAsyncSubPatcher,
        _FunctionsCallToolAsyncSubPatcher,
    )


# ---------------------------------------------------------------------------
# Thread-bridge patchers
# ---------------------------------------------------------------------------


class _ThreadBridgePlatformSubPatcher(FunctionWrapperPatcher):
    """Patch ``google.adk.platform.thread.create_thread`` for context propagation."""

    name = "adk.thread_bridge.platform"
    target_module = "google.adk.platform.thread"
    target_path = "create_thread"
    wrapper = _create_thread_wrapper


class _ThreadBridgeRunnersSubPatcher(FunctionWrapperPatcher):
    """Patch ``google.adk.runners.create_thread`` for context propagation."""

    name = "adk.thread_bridge.runners"
    target_module = "google.adk.runners"
    target_path = "create_thread"
    wrapper = _create_thread_wrapper


class ThreadBridgePatcher(CompositeFunctionWrapperPatcher):
    """Patch ``create_thread`` in ADK platform and runners for context propagation."""

    name = "adk.thread_bridge"
    priority: ClassVar[int] = 50  # run before other patchers so context propagates
    sub_patchers = (_ThreadBridgePlatformSubPatcher, _ThreadBridgeRunnersSubPatcher)


# ---------------------------------------------------------------------------
# MCP tool patcher
# ---------------------------------------------------------------------------


class McpToolPatcher(FunctionWrapperPatcher):
    """Patch ``McpTool.run_async`` for tracing (optional – MCP may not be installed)."""

    name = "adk.mcp_tool"
    target_module = "google.adk.tools.mcp_tool.mcp_tool"
    target_path = "McpTool.run_async"
    wrapper = _mcp_tool_run_async_wrapper_async


# ---------------------------------------------------------------------------
# Public wrap_*() helpers — thin wrappers around patcher.wrap_target()
# ---------------------------------------------------------------------------


def wrap_agent(Agent: Any) -> Any:
    """Manually patch an agent class for tracing."""
    return AgentRunAsyncPatcher.wrap_target(Agent)


def wrap_runner(Runner: Any) -> Any:
    """Manually patch a runner class for tracing."""
    return RunnerRunPatcher.wrap_target(Runner)


def wrap_workflow(Workflow: Any) -> Any:
    """Manually patch ADK 2.x workflow nodes for tracing.

    Pass the ``Workflow`` class from ``google.adk.workflow``. The base node
    patch covers workflow and graph nodes; the tool-node patch adds spans for
    tools executed directly by the workflow graph.
    """
    base_node = next(
        (
            cls
            for cls in Workflow.__mro__
            if cls.__name__ == "BaseNode" and cls.__module__.startswith("google.adk.workflow")
        ),
        None,
    )
    if base_node is None:
        raise TypeError("wrap_workflow expects an ADK Workflow class")

    WorkflowNodeRunPatcher.wrap_target(base_node)
    try:
        tool_node_module = import_module("google.adk.workflow._tool_node")
    except ImportError:
        return Workflow

    tool_node = getattr(tool_node_module, "_ToolNode", None)
    if tool_node is not None:
        WorkflowToolNodeRunImplPatcher.wrap_target(tool_node)
    return Workflow


def wrap_flow(Flow: Any) -> Any:
    """Manually patch a flow class for tracing."""
    return FlowRunAsyncPatcher.wrap_target(Flow)


def wrap_mcp_tool(McpTool: Any) -> Any:
    """Manually patch an MCP tool class for tracing.

    Creates Braintrust spans for each MCP tool call, capturing:
    - Tool name
    - Input arguments
    - Output results
    - Execution time
    - Errors if they occur
    """
    return McpToolPatcher.wrap_target(McpTool)
