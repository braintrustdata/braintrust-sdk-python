"""
Braintrust integration for Claude Agent SDK with automatic tracing.

Usage (imports can be before or after setup):
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    from braintrust.wrappers.claude_agent_sdk import setup_claude_agent_sdk

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

from ._wrapper import _create_client_wrapper_class, _create_tool_wrapper_class


logger = logging.getLogger(__name__)

__all__ = ["setup_claude_agent_sdk"]


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
        from braintrust.wrappers.claude_agent_sdk import setup_claude_agent_sdk

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

    try:
        import sys

        import claude_agent_sdk

        original_client = claude_agent_sdk.ClaudeSDKClient if hasattr(claude_agent_sdk, "ClaudeSDKClient") else None
        original_tool_class = claude_agent_sdk.SdkMcpTool if hasattr(claude_agent_sdk, "SdkMcpTool") else None

        if original_client:
            wrapped_client = _create_client_wrapper_class(original_client)
            claude_agent_sdk.ClaudeSDKClient = wrapped_client

            for module in list(sys.modules.values()):
                if module and hasattr(module, "ClaudeSDKClient"):
                    if getattr(module, "ClaudeSDKClient", None) is original_client:
                        setattr(module, "ClaudeSDKClient", wrapped_client)

        if original_tool_class:
            wrapped_tool_class = _create_tool_wrapper_class(original_tool_class)
            claude_agent_sdk.SdkMcpTool = wrapped_tool_class

            for module in list(sys.modules.values()):
                if module and hasattr(module, "SdkMcpTool"):
                    if getattr(module, "SdkMcpTool", None) is original_tool_class:
                        setattr(module, "SdkMcpTool", wrapped_tool_class)

        return True
    except ImportError:
        # Not installed - this is expected when using auto_instrument()
        return False
