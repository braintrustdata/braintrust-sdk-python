"""Test auto_instrument for Pydantic AI (no uninstrument available)."""

import asyncio

from braintrust.integrations.test_utils import run_auto_smoke


def _call(memory_logger):
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.settings import ModelSettings

    agent = Agent(
        OpenAIChatModel("gpt-4o-mini"),
        model_settings=ModelSettings(max_tokens=100),
    )

    result = asyncio.run(agent.run("Say hi"))
    assert result.output

    spans = memory_logger.pop()
    assert len(spans) >= 1, f"Expected at least 1 span, got {len(spans)}"
    agent_spans = [s for s in spans if "agent_run" in s["span_attributes"]["name"]]
    assert len(agent_spans) >= 1, f"Expected agent_run span, got {[s['span_attributes']['name'] for s in spans]}"


run_auto_smoke("pydantic_ai", integration="pydantic_ai", run=_call)
print("SUCCESS")
