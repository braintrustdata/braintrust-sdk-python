"""Cursor SDK integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import AgentLifecyclePatcher, CustomToolDispatchPatcher, RunLifecyclePatcher


class CursorSDKIntegration(BaseIntegration):
    """Braintrust instrumentation for Cursor's Python agent SDK."""

    name = "cursor_sdk"
    import_names = ("cursor_sdk",)
    distribution_names = ("cursor-sdk",)
    min_version = "1.0.25"
    patchers = (RunLifecyclePatcher, AgentLifecyclePatcher, CustomToolDispatchPatcher)
