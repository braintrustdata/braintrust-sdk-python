"""Test auto_instrument for Google ADK."""

from braintrust.auto import auto_instrument

# 1. Instrument
results = auto_instrument()
assert results.get("adk") == True, "auto_instrument should return True for adk"

# 2. Idempotent
results2 = auto_instrument()
assert results2.get("adk") == True, "auto_instrument should still return True on second call"

# 3. Verify classes are patched
from google.adk import agents, runners
from google.adk.flows.llm_flows import base_llm_flow

assert getattr(agents.BaseAgent, "_braintrust_patched", False), "BaseAgent should be patched"
assert getattr(runners.Runner, "_braintrust_patched", False), "Runner should be patched"
assert getattr(base_llm_flow.BaseLlmFlow, "_braintrust_patched", False), "BaseLlmFlow should be patched"

print("SUCCESS")
