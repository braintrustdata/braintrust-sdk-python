"""Test auto_instrument for the OpenAI Agents SDK."""

import asyncio

import agents
from braintrust.integrations.openai_agents import BraintrustTracingProcessor
from braintrust.integrations.test_utils import run_auto_smoke


TEST_MODEL = "gpt-4o-mini"
TEST_PROMPT = "What is 2+2? Just the number."
TEST_AGENT_INSTRUCTIONS = "You are a helpful assistant. Be very concise."


def _is_patched() -> bool:
    provider = agents.tracing.get_trace_provider()
    processors = getattr(getattr(provider, "_multi_processor", None), "_processors", ())
    return any(isinstance(processor, BraintrustTracingProcessor) for processor in processors)


def _call(memory_logger):
    from agents import Agent
    from agents.run import AgentRunner

    async def run_agent():
        agent = Agent(name="test-agent", model=TEST_MODEL, instructions=TEST_AGENT_INSTRUCTIONS)
        return await AgentRunner().run(agent, TEST_PROMPT)

    result = asyncio.run(run_agent())
    assert result is not None
    assert hasattr(result, "final_output") or hasattr(result, "output")

    spans = memory_logger.pop()
    assert len(spans) >= 2, f"Expected at least 2 spans, got {len(spans)}"


run_auto_smoke("openai_agents", is_patched=_is_patched, integration="openai_agents", run=_call)
print("SUCCESS")
