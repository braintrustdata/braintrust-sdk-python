"""ADK integration — orchestration class and setup entry-point."""

import logging

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    AgentRunAsyncPatcher,
    FlowRunAsyncPatcher,
    McpToolPatcher,
    RunnerRunPatcher,
    ThreadBridgePatcher,
    ToolCallAsyncPatcher,
)


logger = logging.getLogger(__name__)


class ADKIntegration(BaseIntegration):
    """Braintrust instrumentation for Google ADK (Agent Development Kit)."""

    name = "adk"
    import_names = ("google.adk",)
    min_version = "1.14.1"
    patchers = (
        ThreadBridgePatcher,
        AgentRunAsyncPatcher,
        RunnerRunPatcher,
        FlowRunAsyncPatcher,
        ToolCallAsyncPatcher,
        McpToolPatcher,
    )
