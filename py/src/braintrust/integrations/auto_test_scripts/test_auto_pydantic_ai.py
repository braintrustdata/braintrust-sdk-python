"""Test auto_instrument for Pydantic AI (no uninstrument available)."""

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


# 1. Instrument
results = auto_instrument()
assert results.get("pydantic_ai") == True

# 2. Idempotent
results2 = auto_instrument()
assert results2.get("pydantic_ai") == True

# 3. Make API call and verify span
with autoinstrument_test_context("test_auto_pydantic_ai", integration="pydantic_ai") as memory_logger:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.settings import ModelSettings

    agent = Agent(
        OpenAIChatModel("gpt-6-luna"),
        model_settings=ModelSettings(max_tokens=100, openai_reasoning_effort="none"),
    )

    result = agent.run_sync("Say hi")
    assert result.output
    assert agent.name == "agent"

    span_names = {span["span_attributes"]["name"] for span in memory_logger.pop()}
    assert {"agent_run_sync [agent]", "agent_run [agent]"} <= span_names

    agents = [Agent(TestModel())]
    agents[0].run_sync("Say hi")
    assert agents[0].name is None

    opt_out_agent = Agent(TestModel())
    opt_out_agent.run_sync("Say hi", infer_name=False)
    assert opt_out_agent.name is None
    span_names = [span["span_attributes"]["name"] for span in memory_logger.pop()]
    assert sorted(name for name in span_names if name.startswith("agent_run")) == [
        "agent_run",
        "agent_run",
        "agent_run_sync",
        "agent_run_sync",
    ]

print("SUCCESS")
