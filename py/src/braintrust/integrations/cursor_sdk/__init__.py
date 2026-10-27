"""Braintrust tracing integration for the Cursor Python SDK."""

import inspect
from typing import Any

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import CursorSDKIntegration
from .patchers import AgentClosePatcher, AgentSendPatcher, AsyncAgentClosePatcher, AsyncAgentSendPatcher


__all__ = ["CursorSDKIntegration", "setup_cursor_sdk", "wrap_cursor_sdk_agent"]


def setup_cursor_sdk(
    api_key: str | None = None,
    project_id: str | None = None,
    project: str | None = None,
) -> bool:
    """Set up Braintrust tracing for Cursor SDK agent runs.

    ``project`` is the project name.

    Returns:
        True if patched or already patched, False if the Cursor SDK is missing or too old.
    """
    if current_span() == NOOP_SPAN:
        init_logger(project=project, api_key=api_key, project_id=project_id)
    return CursorSDKIntegration.setup()


def wrap_cursor_sdk_agent(agent_class: Any) -> Any:
    """Instrument a Cursor ``Agent`` or ``AsyncAgent`` class in place."""
    CursorSDKIntegration.setup()
    # Both flavors spell these methods `send`/`close`, so the composite
    # patcher cannot tell them apart when wrapping a class directly.
    is_async = inspect.iscoroutinefunction(getattr(agent_class, "send", None))
    send_patcher = AsyncAgentSendPatcher if is_async else AgentSendPatcher
    close_patcher = AsyncAgentClosePatcher if is_async else AgentClosePatcher
    send_patcher.wrap_target(agent_class)
    close_patcher.wrap_target(agent_class)
    return agent_class
