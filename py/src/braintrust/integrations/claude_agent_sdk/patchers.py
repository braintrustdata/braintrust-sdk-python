"""Claude Agent SDK patchers — class-replacement patchers for ClaudeSDKClient and SdkMcpTool."""

from braintrust.integrations.base import ClassReplacementPatcher

from .tracing import _create_client_wrapper_class, _create_tool_wrapper_class


class ClaudeSDKClientPatcher(ClassReplacementPatcher):
    """Replace ``claude_agent_sdk.ClaudeSDKClient`` with a tracing wrapper class.

    This integration needs class replacement because the wrapper keeps
    per-instance state across ``query()`` and ``receive_response()`` and must
    update modules that imported ``ClaudeSDKClient`` before setup.
    """

    name = "claude_agent_sdk.client"
    target_attr = "ClaudeSDKClient"
    replacement_factory = staticmethod(_create_client_wrapper_class)


class SdkMcpToolPatcher(ClassReplacementPatcher):
    """Replace ``claude_agent_sdk.SdkMcpTool`` with a tracing wrapper class.

    This integration needs class replacement because the tool wrapper must
    intercept construction and replace the handler before the SDK stores it.
    """

    name = "claude_agent_sdk.tool"
    target_attr = "SdkMcpTool"
    replacement_factory = staticmethod(_create_tool_wrapper_class)
