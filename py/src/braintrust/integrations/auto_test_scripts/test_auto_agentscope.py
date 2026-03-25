"""Test auto_instrument for AgentScope."""

import os
from pathlib import Path


os.environ["BRAINTRUST_CASSETTES_DIR"] = str(Path(__file__).resolve().parent.parent / "agentscope" / "cassettes")

from braintrust.auto import auto_instrument
from braintrust.wrappers.test_utils import autoinstrument_test_context


results = auto_instrument()
assert results.get("agentscope") == True, "auto_instrument should return True for agentscope"

results2 = auto_instrument()
assert results2.get("agentscope") == True, "auto_instrument should still return True on second call"

from agentscope.agent import AgentBase, ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import sequential_pipeline
from agentscope.tool import Toolkit


try:
    from agentscope.pipeline import fanout_pipeline
except ImportError:
    fanout_pipeline = None


assert hasattr(AgentBase.__call__, "__wrapped__"), "AgentBase.__call__ should be wrapped"
assert hasattr(sequential_pipeline, "__wrapped__"), "sequential_pipeline should be wrapped"
if fanout_pipeline is not None:
    assert hasattr(fanout_pipeline, "__wrapped__"), "fanout_pipeline should be wrapped"
assert hasattr(Toolkit.call_tool_function, "__wrapped__"), "Toolkit.call_tool_function should be wrapped"
assert hasattr(OpenAIChatModel.__call__, "__wrapped__"), "OpenAIChatModel.__call__ should be wrapped"


with autoinstrument_test_context("test_auto_agentscope") as memory_logger:
    agent = ReActAgent(
        name="Test Agent",
        sys_prompt="You are a helpful assistant. Be brief.",
        model=OpenAIChatModel(
            model_name="gpt-4o-mini",
            generate_kwargs={"temperature": 0},
        ),
        formatter=OpenAIChatFormatter(),
        toolkit=Toolkit(),
        memory=InMemoryMemory(),
    )
    if hasattr(agent, "set_console_output_enabled"):
        agent.set_console_output_enabled(False)
    elif hasattr(agent, "disable_console_output"):
        agent.disable_console_output()

    response = agent(
        Msg(
            name="user",
            content="Say hi in two words.",
            role="user",
        )
    )

    import asyncio

    result = asyncio.run(response)
    assert result is not None

    spans = memory_logger.pop()
    assert len(spans) >= 2, f"Expected at least 2 spans (agent + model), got {len(spans)}"

    agent_span = next(span for span in spans if span["span_attributes"]["name"] == "Test Agent.reply")
    llm_spans = [span for span in spans if span["span_attributes"]["type"].value == "llm"]

    assert agent_span["span_attributes"]["type"].value == "task"
    assert llm_spans, "Should have at least one LLM span"
    assert llm_spans[0]["metadata"]["model"] == "gpt-4o-mini"
    assert agent_span["span_id"] in llm_spans[0]["span_parents"]

print("SUCCESS")
