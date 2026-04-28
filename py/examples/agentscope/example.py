#!/usr/bin/env python
"""AgentScope ReAct agent traced via braintrust.auto_instrument()."""

import asyncio

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-agentscope")

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit


async def main() -> None:
    agent = ReActAgent(
        name="Friday",
        sys_prompt="You are a concise assistant. Answer in one sentence.",
        model=OpenAIChatModel(model_name="gpt-4o-mini", stream=False),
        formatter=OpenAIChatFormatter(),
        toolkit=Toolkit(),
        memory=InMemoryMemory(),
    )

    response = await agent(Msg(name="user", content="Say hello in exactly two words.", role="user"))
    print(response.content if response is not None else "(no response)")


if __name__ == "__main__":
    asyncio.run(main())
