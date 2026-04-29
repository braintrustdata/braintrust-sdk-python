#!/usr/bin/env python
"""AutoGen AssistantAgent traced via braintrust.auto_instrument()."""

import asyncio

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-autogen")

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:
    agent = AssistantAgent(
        "assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0),
        system_message="You are concise. Answer directly.",
    )

    result = await agent.run(task="What is the capital of Australia?")
    for message in result.messages:
        print(f"{message.source}: {getattr(message, 'content', message)}")


if __name__ == "__main__":
    asyncio.run(main())
