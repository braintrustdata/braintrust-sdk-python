"""Test auto_instrument for Claude Agent SDK (no uninstrument available)."""

import asyncio

from braintrust.integrations.claude_agent_sdk._test_transport import make_cassette_transport
from braintrust.integrations.test_utils import run_auto_smoke


def _call(memory_logger):
    import claude_agent_sdk  # pylint: disable=import-error

    options = claude_agent_sdk.ClaudeAgentOptions(
        model="claude-3-5-haiku-20241022",
        permission_mode="bypassPermissions",
    )
    transport = make_cassette_transport(
        cassette_name="test_auto_claude_agent_sdk",
        prompt="",
        options=options,
    )

    async def run_agent():
        async with claude_agent_sdk.ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query("Say hi")
            async for message in client.receive_response():
                if type(message).__name__ == "ResultMessage":
                    return message
        return None

    result = asyncio.run(run_agent())
    assert result is not None

    spans = memory_logger.pop()
    assert len(spans) >= 1, f"Expected at least 1 span, got {len(spans)}"


run_auto_smoke("claude_agent_sdk", use_vcr=False, run=_call)
print("SUCCESS")
