"""ADK patchers — one patcher per coherent patch target."""

from typing import Any, ClassVar

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _agent_run_async_wrapper,
    _create_thread_wrapper,
    _flow_call_llm_async_wrapper,
    _flow_run_async_wrapper,
    _mcp_tool_run_async_wrapper_async,
    _runner_run_async_wrapper,
    _runner_run_wrapper,
)


# ---------------------------------------------------------------------------
# Agent patcher
# ---------------------------------------------------------------------------


class AgentRunAsyncPatcher(FunctionWrapperPatcher):
    """Patch ``BaseAgent.run_async`` for tracing."""

    name = "adk.agent.run_async"
    target_module = "google.adk.agents"
    target_path = "BaseAgent.run_async"
    wrapper = _agent_run_async_wrapper


# ---------------------------------------------------------------------------
# Runner patchers (sync + async)
# ---------------------------------------------------------------------------


class _RunnerRunSyncSubPatcher(FunctionWrapperPatcher):
    """Patch ``Runner.run`` (sync generator)."""

    name = "adk.runner.run.sync"
    target_module = "google.adk.runners"
    target_path = "Runner.run"
    wrapper = _runner_run_wrapper


class _RunnerRunAsyncSubPatcher(FunctionWrapperPatcher):
    """Patch ``Runner.run_async`` (async generator)."""

    name = "adk.runner.run.async"
    target_module = "google.adk.runners"
    target_path = "Runner.run_async"
    wrapper = _runner_run_async_wrapper


class RunnerRunSyncPatcher(CompositeFunctionWrapperPatcher):
    """Patch ``Runner.run`` (sync) and ``Runner.run_async`` for tracing."""

    name = "adk.runner.run"
    sub_patchers = (_RunnerRunSyncSubPatcher, _RunnerRunAsyncSubPatcher)


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
    return RunnerRunSyncPatcher.wrap_target(Runner)


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
