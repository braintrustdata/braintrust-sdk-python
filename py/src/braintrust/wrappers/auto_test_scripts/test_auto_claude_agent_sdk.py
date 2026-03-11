"""Test auto_instrument for Claude Agent SDK (no uninstrument available)."""

from braintrust.auto import auto_instrument
from braintrust.wrappers.claude_agent_sdk._test_transport import make_cassette_transport
from braintrust.wrappers.test_utils import autoinstrument_test_context

# 1. Instrument
results = auto_instrument()
assert results.get("claude_agent_sdk") == True

# 2. Idempotent
results2 = auto_instrument()
assert results2.get("claude_agent_sdk") == True

# 3. Make API call and verify span
with autoinstrument_test_context("test_auto_claude_agent_sdk", use_vcr=False) as memory_logger:
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

    import asyncio

    result = asyncio.run(run_agent())
    assert result is not None

    spans = memory_logger.pop()
    assert len(spans) >= 1, f"Expected at least 1 span, got {len(spans)}"

print("SUCCESS")
