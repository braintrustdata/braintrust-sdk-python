"""Tests for MCP tool tracing integration.

Behavioral coverage of the MCP tool span is provided by the VCR-backed suite in
``test_adk.py`` — when an ADK agent invokes a real tool through the flow, the
``tool [...]`` / ``mcp_tool [...]`` span is exercised end-to-end. The tests here
only cover the patcher wiring, which does not require a live MCP server.
"""

import pytest
from braintrust.integrations.adk import setup_adk, wrap_mcp_tool
from braintrust.integrations.adk.patchers import McpToolPatcher


@pytest.mark.asyncio
async def test_wrap_mcp_tool_marks_as_patched():
    """wrap_mcp_tool marks the class via the patcher marker (idempotence signal)."""

    class MockMcpTool:
        async def run_async(self, *, args, tool_context):
            return {"result": "success"}

    wrapped_class = wrap_mcp_tool(MockMcpTool)
    assert getattr(wrapped_class, McpToolPatcher.patch_marker_attr(), False)


@pytest.mark.asyncio
async def test_setup_adk_patches_mcp_tool():
    """setup_adk patches the real McpTool class via ADKIntegration."""
    assert setup_adk(project_name="test") is True
    assert McpToolPatcher.is_patched(None, None), "McpTool should be patched"
