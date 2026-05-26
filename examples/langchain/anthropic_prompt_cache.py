#!/usr/bin/env python
"""Verify LangChain Anthropic prompt-cache metrics in Braintrust.

This sends two Anthropic requests through LangChain with a cacheable system
prompt. The resulting Braintrust spans should show Anthropic cache reads and
cache writes, including TTL-specific cache creation metrics when Anthropic
returns them.
"""

import os
import uuid
from pathlib import Path

import braintrust
from braintrust.integrations.langchain import BraintrustCallbackHandler
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

PROJECT_NAME = os.environ.get("BRAINTRUST_PROJECT", "py-sdk-demo-langchain-anthropic-cache")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# Anthropic prompt caching requires a sufficiently long cacheable prefix.
CACHEABLE_SYSTEM_PROMPT = "\n".join(
    [
        "You are helping validate prompt-cache accounting in an SDK integration.",
        "Always answer briefly and mention the requested section title.",
        "",
        "Reference document:",
        *[
            f"Section {i}: This paragraph describes stable product guidance, tracing semantics, "
            "token accounting, and prompt-cache behavior for repeat requests."
            for i in range(1, 90)
        ],
        f"Stable cache key: {os.environ.get('CACHE_DEMO_KEY', 'langchain-anthropic-cache-demo')}",
    ]
)


def main() -> None:
    logger = braintrust.init_logger(project=PROJECT_NAME)
    handler = BraintrustCallbackHandler(logger=logger)
    model = ChatAnthropic(model=MODEL, max_tokens=64)

    messages: list[BaseMessage] = [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": CACHEABLE_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        HumanMessage(content=f"What is this document for? Run id: {uuid.uuid4().hex}"),
    ]

    for label in ("cache write", "cache read"):
        result = model.invoke(messages, config={"callbacks": [handler]})
        print(f"{label}: {result.content}")

    braintrust.flush()
    print(f"Logged demo spans to Braintrust project: {PROJECT_NAME}")


if __name__ == "__main__":
    main()
