"""Test auto_instrument for OpenAI Agents SDK."""

from agents.tracing import get_trace_provider
from braintrust.auto import auto_instrument
from braintrust.wrappers.openai import BraintrustTracingProcessor


results = auto_instrument()
assert results.get("openai_agent_sdk") == True
bt_processor = get_trace_provider()._multi_processor._processors[0]
assert isinstance(bt_processor, BraintrustTracingProcessor)

results2 = auto_instrument()
assert results2.get("openai_agent_sdk") == True

# with autoinstrument_test_context("test_auto_openai_agents_sdk") as memory_logger:
#   agent = Agent(
#         name="Assistant",
#         instructions="You only respond in haikus.",
#     )

#   result = Runner.run_sync(agent, "Tell me about recursion in programming.")

#   span = memory_logger.pop()
#   assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
#   span = spans[0]
