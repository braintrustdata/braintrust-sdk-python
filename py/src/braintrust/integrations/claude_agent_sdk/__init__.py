"""Braintrust integration for Claude Agent SDK with automatic tracing.

Usage (imports can be before or after setup):
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    from braintrust.integrations.claude_agent_sdk import setup_claude_agent_sdk

    setup_claude_agent_sdk(project="my-project")

    # Use normally - all calls are automatically traced
    options = ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Hello!")
        async for message in client.receive_response():
            print(message)
"""

import logging

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import ClaudeAgentSDKIntegration


logger = logging.getLogger(__name__)

__all__ = ["ClaudeAgentSDKIntegration", "setup_claude_agent_sdk"]


def setup_claude_agent_sdk(
    api_key: str | None = None,
    project_id: str | None = None,
    project: str | None = None,
) -> bool:
    """
    Setup Braintrust integration with Claude Agent SDK. Will automatically patch the SDK for automatic tracing.

    Args:
        api_key (Optional[str]): Braintrust API key.
        project_id (Optional[str]): Braintrust project ID.
        project (Optional[str]): Braintrust project name.

    Returns:
        bool: True if setup was successful, False otherwise.

    Example:
        ```python
        import claude_agent_sdk
        from braintrust.integrations.claude_agent_sdk import setup_claude_agent_sdk

        setup_claude_agent_sdk(project="my-project")

        # Now use claude_agent_sdk normally - all calls automatically traced
        options = claude_agent_sdk.ClaudeAgentOptions(model="claude-sonnet-4-5-20250929")
        async with claude_agent_sdk.ClaudeSDKClient(options=options) as client:
            await client.query("Hello!")
            async for message in client.receive_response():
                print(message)
        ```
    """
    span = current_span()
    if span == NOOP_SPAN:
        init_logger(project=project, api_key=api_key, project_id=project_id)

    return ClaudeAgentSDKIntegration.setup()
