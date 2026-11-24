"""Braintrust tracing integration for the Cursor Python SDK."""

import inspect
from typing import Any

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import CursorSDKIntegration
from .patchers import (
    AgentClosePatcher,
    AgentSendPatcher,
    AgentSummarizePatcher,
    AsyncAgentClosePatcher,
    AsyncAgentSendPatcher,
    AsyncAgentSummarizePatcher,
)


__all__ = ["CursorSDKIntegration", "setup_cursor_sdk", "wrap_cursor_sdk_agent"]


def setup_cursor_sdk(
    api_key: str | None = None,
    project_id: str | None = None,
    project: str | None = None,
) -> bool:
    """Patch Cursor SDK agent runs for Braintrust tracing."""
    if current_span() == NOOP_SPAN:
        init_logger(project=project, api_key=api_key, project_id=project_id)
    return CursorSDKIntegration.setup()


def wrap_cursor_sdk_agent(agent_class: Any) -> Any:
    """Instrument a Cursor ``Agent`` or ``AsyncAgent`` class in place.

    Cursor traces model turns off ``Run``, not ``Agent``, so this also runs
    the full integration setup to patch the run lifecycle.
    """
    CursorSDKIntegration.setup()
    # Both flavors use the same method names when wrapping a class directly.
    is_async = inspect.iscoroutinefunction(getattr(agent_class, "send", None))
    patchers = (
        (AsyncAgentSendPatcher, AsyncAgentSummarizePatcher, AsyncAgentClosePatcher)
        if is_async
        else (AgentSendPatcher, AgentSummarizePatcher, AgentClosePatcher)
    )
    for patcher in patchers:
        method_name = patcher.target_path.rsplit(".", 1)[-1]
        if not patcher.has_patch_marker(getattr(agent_class, method_name, None)):
            patcher.wrap_target(agent_class)
    return agent_class
