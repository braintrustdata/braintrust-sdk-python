"""Claude Agent SDK integration — orchestration class and setup entry-point."""

from braintrust.integrations.base import BaseIntegration

from .patchers import (
    ClaudeSDKClientPatcher,
    ClaudeSDKMessageReaderPatcher,
    ClaudeSDKQueryPatcher,
    SdkMcpToolPatcher,
)


class ClaudeAgentSDKIntegration(BaseIntegration):
    """Braintrust instrumentation for the Claude Agent SDK."""

    name = "claude_agent_sdk"
    import_names = ("claude_agent_sdk",)
    # 0.1.9-0.1.17 were deleted from PyPI, so CI tests 0.1.18 as the oldest
    # version (see [tool.braintrust.matrix.claude-agent-sdk] in pyproject.toml).
    min_version = "0.1.10"
    patchers = (
        ClaudeSDKMessageReaderPatcher,
        ClaudeSDKClientPatcher,
        ClaudeSDKQueryPatcher,
        SdkMcpToolPatcher,
    )
