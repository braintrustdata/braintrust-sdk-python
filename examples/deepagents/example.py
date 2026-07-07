#!/usr/bin/env python
"""Deep Agents traced via Braintrust's LangChain auto-instrumentation.

Deep Agents is built on LangChain/LangGraph, so `braintrust.auto_instrument()`
installs the LangChain callback handler that traces Deep Agents runs.
"""

import importlib
from pathlib import Path

import braintrust
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

braintrust.auto_instrument()
braintrust.init_logger(project="example-deepagents")


@tool
def get_weather(city: str) -> str:
    """Return the current demo weather for a city."""
    return f"The weather in {city} is sunny and 72°F."


deepagents = importlib.import_module("deepagents")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = deepagents.create_deep_agent(model=model, tools=[get_weather], name="example-deep-agent")

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Use the get_weather tool to answer: What is the weather in Paris?",
            }
        ]
    }
)
print(result["messages"][-1].content)
braintrust.flush()
