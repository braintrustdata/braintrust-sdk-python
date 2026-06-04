"""Test auto_instrument for AgentScope."""
# pylint: disable=import-error,no-name-in-module,no-value-for-parameter,no-member

import asyncio
import importlib

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


results = auto_instrument(openai=False)
assert results.get("agentscope") == True, "auto_instrument should return True for agentscope"

results2 = auto_instrument(openai=False)
assert results2.get("agentscope") == True, "auto_instrument should still return True on second call"

agent_module = importlib.import_module("agentscope.agent")
HAS_AGENT_REPLY_API = hasattr(agent_module, "Agent")

if HAS_AGENT_REPLY_API:
    from agentscope.agent import Agent
    from agentscope.credential import OpenAICredential
    from agentscope.message import UserMsg
    from agentscope.model import OpenAIChatModel
    from agentscope.tool import Toolkit

    assert hasattr(Agent.reply, "__wrapped__"), "Agent.reply should be wrapped"
    assert hasattr(Toolkit.call_tool, "__wrapped__"), "Toolkit.call_tool should be wrapped"
    assert hasattr(OpenAIChatModel.__call__, "__wrapped__"), "OpenAIChatModel.__call__ should be wrapped"

    model = OpenAIChatModel(
        credential=OpenAICredential(api_key="test-api-key"),
        model="gpt-4o-mini",
        parameters=OpenAIChatModel.Parameters(temperature=0),
        stream=False,
        max_retries=0,
    )
    agent = Agent(
        name="Test Agent",
        system_prompt="You are a helpful assistant. Be brief.",
        model=model,
        toolkit=Toolkit(),
    )
    message = UserMsg("user", "Say hello in exactly two words.")
else:
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
    message = Msg(
        name="user",
        content="Say hello in exactly two words.",
        role="user",
    )

if hasattr(agent, "set_console_output_enabled"):
    agent.set_console_output_enabled(False)
elif hasattr(agent, "disable_console_output"):
    agent.disable_console_output()

with autoinstrument_test_context("test_auto_agentscope", integration="agentscope") as memory_logger:
    result = asyncio.run(agent.reply(message) if HAS_AGENT_REPLY_API else agent(message))
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
