#!/usr/bin/env python
"""Async Anthropic client traced via braintrust.auto_instrument()."""

import asyncio

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-anthropic-app")

from anthropic import AsyncAnthropic


client = AsyncAnthropic()


async def stream():
    async with client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write me a haiku about a stream."}],
        model="claude-haiku-4-5",
    ) as stream:
        async for event in stream:
            pass

        msg = await stream.get_final_message()
        print(msg.to_json())


async def create():
    msg = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write me a haiku about creation."}],
    )
    print(msg.to_json())


async def create_with_stream():
    stream = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write me a haiku about creation."}],
        stream=True,
    )

    async for event in stream:
        print(event.to_json())


async def main() -> None:
    promises = []
    for target in [stream, create, create_with_stream]:
        print(f"Running {target.__name__}")
        promises.append(target())

    for promise in promises:
        await promise


if __name__ == "__main__":
    asyncio.run(main())
