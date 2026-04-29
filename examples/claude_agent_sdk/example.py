#!/usr/bin/env python
"""Claude Agent SDK traced via braintrust.auto_instrument()."""

import asyncio

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-claude-agent-sdk")

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient


async def main() -> None:
    options = ClaudeAgentOptions(model="claude-haiku-4-5-20251001")

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Say hello in exactly three words.")
        async for message in client.receive_response():
            print(message)


if __name__ == "__main__":
    asyncio.run(main())
