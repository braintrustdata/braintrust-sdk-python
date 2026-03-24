"""Claude Agent SDK integration — orchestration class and setup entry-point."""

from braintrust.integrations.base import BaseIntegration

from .patchers import ClaudeSDKClientPatcher, SdkMcpToolPatcher


class ClaudeAgentSDKIntegration(BaseIntegration):
    """Braintrust instrumentation for the Claude Agent SDK."""

    name = "claude_agent_sdk"
    import_names = ("claude_agent_sdk",)
    min_version = "0.1.10"
    patchers = (ClaudeSDKClientPatcher, SdkMcpToolPatcher)
