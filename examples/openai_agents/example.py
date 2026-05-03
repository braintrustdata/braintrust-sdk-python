#!/usr/bin/env python
"""OpenAI Agents SDK traced via braintrust.auto_instrument()."""

import asyncio

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-openai-agents")

from agents import Agent, Runner


async def main() -> None:
    agent = Agent(
        name="assistant",
        instructions="You are concise. Answer in one sentence.",
        model="gpt-4o-mini",
    )

    result = await Runner.run(agent, "What is the capital of Australia?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
