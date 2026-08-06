"""Patchers for Cursor SDK agent runs and stream lifecycles."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _wrap_apply_event_state,
    _wrap_async_cancel,
    _wrap_async_close,
    _wrap_async_observe,
    _wrap_async_send,
    _wrap_async_wait,
    _wrap_sync_cancel,
    _wrap_sync_close,
    _wrap_sync_observe,
    _wrap_sync_send,
    _wrap_sync_wait,
    _wrap_tool_dispatch,
)


class AgentSendPatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.agent.send"
    target_path = "Agent.send"
    wrapper = staticmethod(_wrap_sync_send)


class AsyncAgentSendPatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.async_agent.send"
    target_path = "AsyncAgent.send"
    wrapper = staticmethod(_wrap_async_send)


class AgentClosePatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.agent.close"
    target_path = "Agent.close"
    wrapper = staticmethod(_wrap_sync_close)


class AsyncAgentClosePatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.async_agent.close"
    target_path = "AsyncAgent.close"
    wrapper = staticmethod(_wrap_async_close)


class AgentLifecyclePatcher(CompositeFunctionWrapperPatcher):
    name = "cursor_sdk.agent"
    sub_patchers = (AgentSendPatcher, AsyncAgentSendPatcher, AgentClosePatcher, AsyncAgentClosePatcher)


class RunApplyEventStatePatcher(FunctionWrapperPatcher):
    """Trace stream events at the one point `Run` and `AsyncRun` share.

    Both `_handle_event` implementations are thin shims over this synchronous
    base method, and every consumption path (`wait`, `events`, `stream`,
    `messages`, `iter_text`, iteration) routes through them.
    """

    name = "cursor_sdk.run.apply_event_state"
    target_module = "cursor_sdk._run_base"
    target_path = "_RunBase._apply_event_state"
    wrapper = staticmethod(_wrap_apply_event_state)


class RunWaitPatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.run.wait"
    target_module = "cursor_sdk._run"
    target_path = "Run.wait"
    wrapper = staticmethod(_wrap_sync_wait)


class AsyncRunWaitPatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.async_run.wait"
    target_module = "cursor_sdk._async_run"
    target_path = "AsyncRun.wait"
    wrapper = staticmethod(_wrap_async_wait)


class RunObservePatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.run.observe"
    target_module = "cursor_sdk._run"
    target_path = "Run.observe"
    wrapper = staticmethod(_wrap_sync_observe)


class AsyncRunObservePatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.async_run.observe"
    target_module = "cursor_sdk._async_run"
    target_path = "AsyncRun.observe"
    wrapper = staticmethod(_wrap_async_observe)


class RunCancelPatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.run.cancel"
    target_module = "cursor_sdk._run"
    target_path = "Run.cancel"
    wrapper = staticmethod(_wrap_sync_cancel)


class AsyncRunCancelPatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.async_run.cancel"
    target_module = "cursor_sdk._async_run"
    target_path = "AsyncRun.cancel"
    wrapper = staticmethod(_wrap_async_cancel)


class RunLifecyclePatcher(CompositeFunctionWrapperPatcher):
    name = "cursor_sdk.run"
    sub_patchers = (
        RunApplyEventStatePatcher,
        RunWaitPatcher,
        AsyncRunWaitPatcher,
        RunObservePatcher,
        AsyncRunObservePatcher,
        RunCancelPatcher,
        AsyncRunCancelPatcher,
    )


class CustomToolDispatchPatcher(FunctionWrapperPatcher):
    name = "cursor_sdk.custom_tool"
    target_module = "cursor_sdk._tool_callback"
    target_path = "_dispatch_tool_call"
    wrapper = staticmethod(_wrap_tool_dispatch)
