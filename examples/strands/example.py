#!/usr/bin/env python
"""Strands Agent traced via braintrust.auto_instrument()."""

import asyncio

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-strands")

from strands import Agent
from strands.models.openai import OpenAIModel


async def main() -> None:
    model = OpenAIModel(model_id="gpt-4o-mini", params={"temperature": 0, "max_tokens": 64})
    agent = Agent(
        model=model,
        name="assistant",
        system_prompt="Answer with one short sentence.",
    )

    result = await agent.invoke_async("What is the capital of Australia?")
    print(result.message["content"][0]["text"])


if __name__ == "__main__":
    asyncio.run(main())
