# pylint: disable=import-error,no-name-in-module,no-value-for-parameter,unexpected-keyword-arg,no-member
import importlib
import importlib.metadata
import inspect
import os

import pytest
from braintrust import logger
from braintrust.integrations.agentscope import setup_agentscope
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from packaging.version import Version


PROJECT_NAME = "test_agentscope"

setup_agentscope(project_name=PROJECT_NAME)


AGENTSCOPE_VERSION = Version(importlib.metadata.version("agentscope"))
IS_AGENTSCOPE_V2 = AGENTSCOPE_VERSION >= Version("2")
agent_module = importlib.import_module("agentscope.agent")
message_module = importlib.import_module("agentscope.message")
HAS_AGENT_REPLY_API = hasattr(agent_module, "Agent")
HAS_USER_MSG = hasattr(message_module, "UserMsg")


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _span_type(span):
    span_type = span["span_attributes"]["type"]
    return span_type.value if hasattr(span_type, "value") else span_type


def _make_model(*, stream: bool = False, api_key: str | None = None):
    from agentscope.model import OpenAIChatModel

    resolved_api_key = api_key if api_key is not None else os.environ["OPENAI_API_KEY"]
    if hasattr(OpenAIChatModel, "Parameters"):
        from agentscope.credential import OpenAICredential

        return OpenAIChatModel(
            credential=OpenAICredential(api_key=resolved_api_key),
            model="gpt-4o-mini",
            parameters=OpenAIChatModel.Parameters(temperature=0),
            stream=stream,
            max_retries=0,
        )

    return OpenAIChatModel(
        model_name="gpt-4o-mini",
        api_key=resolved_api_key,
        stream=stream,
        generate_kwargs={"temperature": 0},
    )


def _make_agent(name: str, sys_prompt: str, *, toolkit=None, multi_agent: bool = False, model=None):
    from agentscope.tool import Toolkit

    if HAS_AGENT_REPLY_API:
        from agentscope.agent import Agent

        agent = Agent(
            name=name,
            system_prompt=sys_prompt,
            model=model or _make_model(),
            toolkit=toolkit or Toolkit(),
        )
    else:
        from agentscope.agent import ReActAgent
        from agentscope.formatter import OpenAIChatFormatter, OpenAIMultiAgentFormatter
        from agentscope.memory import InMemoryMemory

        agent = ReActAgent(
            name=name,
            sys_prompt=sys_prompt,
            model=model or _make_model(),
            formatter=OpenAIMultiAgentFormatter() if multi_agent else OpenAIChatFormatter(),
            toolkit=toolkit or Toolkit(),
            memory=InMemoryMemory(),
        )
    if hasattr(agent, "set_console_output_enabled"):
        agent.set_console_output_enabled(False)
    elif hasattr(agent, "disable_console_output"):
        agent.disable_console_output()
    return agent


def _make_user_msg(content):
    from agentscope.message import Msg

    if HAS_USER_MSG:
        from agentscope.message import UserMsg

        return UserMsg("user", content)
    return Msg(name="user", content=content, role="user")


async def _run_agent(agent, content):
    msg = _make_user_msg(content)
    return await (agent.reply(msg) if HAS_AGENT_REPLY_API else agent(msg))


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_simple_agent_run(memory_logger):
    assert not memory_logger.pop()

    agent = _make_agent(
        "Friday",
        "You are a concise assistant. Answer in one sentence.",
    )

    response = await _run_agent(agent, "Say hello in exactly two words.")

    assert response is not None

    spans = memory_logger.pop()
    agent_span = next(span for span in spans if span["span_attributes"]["name"] == "Friday.reply")
    llm_spans = [span for span in spans if _span_type(span) == SpanTypeAttribute.LLM]

    assert agent_span["context"]["span_origin"]["instrumentation"]["name"] == "agentscope-auto"
    assert _span_type(agent_span) == "task"
    assert llm_spans
    llm_span = llm_spans[0]
    assert llm_span["metadata"]["model"] == "gpt-4o-mini"
    assert llm_span["metadata"]["provider"] == "openai"
    assert "args" not in llm_span["input"]
    assert llm_span["input"]["messages"][0]["role"] == "system"
    assert llm_span["input"]["messages"][1]["role"] == "user"
    assert llm_span["input"]["messages"][1]["content"][0]["text"] == "Say hello in exactly two words."
    assert llm_span["output"]["role"] == "assistant"
    assert llm_span["output"]["content"][0]["text"]  # non-empty LLM response
    assert "usage" not in llm_span["output"]
    assert llm_span["metrics"]["prompt_tokens"] > 0
    assert llm_span["metrics"]["completion_tokens"] > 0
    assert llm_span["metrics"]["tokens"] > 0
    assert agent_span["span_id"] in llm_span["span_parents"]


@pytest.mark.skipif(IS_AGENTSCOPE_V2, reason="AgentScope 2.x removed the pipeline module")
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_sequential_pipeline_creates_parent_span(memory_logger):
    from agentscope.message import Msg
    from agentscope.pipeline import sequential_pipeline

    assert not memory_logger.pop()

    agents = [
        _make_agent("Alice", "You rewrite the input as a short title.", multi_agent=True),
        _make_agent("Bob", "You answer the previous message in one sentence.", multi_agent=True),
    ]

    result = await sequential_pipeline(
        agents=agents,
        msg=Msg(
            name="user",
            content="Summarize why tests should use real recorded traffic.",
            role="user",
        ),
    )

    assert result is not None

    spans = memory_logger.pop()
    pipeline_span = next(span for span in spans if span["span_attributes"]["name"] == "sequential_pipeline.run")
    alice_span = next(span for span in spans if span["span_attributes"]["name"] == "Alice.reply")
    bob_span = next(span for span in spans if span["span_attributes"]["name"] == "Bob.reply")

    assert _span_type(pipeline_span) == "task"
    assert pipeline_span["span_id"] in alice_span["span_parents"]
    assert pipeline_span["span_id"] in bob_span["span_parents"]


@pytest.mark.skipif(IS_AGENTSCOPE_V2, reason="AgentScope 2.x removed execute_python_code")
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_tool_use_creates_tool_span(memory_logger):
    from agentscope.message import Msg
    from agentscope.tool import Toolkit, execute_python_code

    assert not memory_logger.pop()

    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    agent = _make_agent(
        "Jarvis",
        "You are a helpful assistant. Use tools when required and keep answers brief.",
        toolkit=toolkit,
    )

    response = await agent(
        Msg(
            name="user",
            content="Use Python to compute 6 * 7 and return just the result.",
            role="user",
        )
    )

    assert response is not None

    spans = memory_logger.pop()
    tool_spans = [span for span in spans if _span_type(span) == "tool"]

    assert tool_spans
    assert tool_spans[0]["span_attributes"]["name"] == "execute_python_code.execute"
    assert tool_spans[0]["input"]["tool_name"] == "execute_python_code"
    assert tool_spans[0]["output"]["content"]

    llm_spans = [span for span in spans if _span_type(span) == SpanTypeAttribute.LLM]
    assert llm_spans
    llm_span = llm_spans[0]
    assert llm_span["output"]["role"] == "assistant"
    assert llm_span["output"]["content"][0]["type"] == "tool_use"
    assert "usage" not in llm_span["output"]
    # Tool definitions belong in metadata.tools, NOT input.
    assert "tools" not in llm_span["input"]
    assert llm_span["metadata"].get("tools")
    tool_names = {tool.get("name") or tool.get("function", {}).get("name") for tool in llm_span["metadata"]["tools"]}
    assert "execute_python_code" in tool_names


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_streaming_model_call(memory_logger):
    """A streaming LLM call must produce one span with accumulated output + TTFT."""
    assert not memory_logger.pop()

    model = _make_model(stream=True)
    agent = _make_agent("Streamer", "You are concise. Answer in one sentence.", model=model)

    response = await _run_agent(agent, "Say hi in five words.")
    assert response is not None

    spans = memory_logger.pop()
    llm_spans = [span for span in spans if _span_type(span) == SpanTypeAttribute.LLM]

    # One span per API call, not per chunk.
    assert len(llm_spans) >= 1
    llm_span = llm_spans[0]

    assert llm_span["metadata"]["provider"] == "openai"
    assert llm_span["metadata"]["model"] == "gpt-4o-mini"
    assert llm_span["output"]["role"] == "assistant"
    assert llm_span["output"]["content"]
    assert llm_span["metrics"]["time_to_first_token"] > 0
    assert llm_span["metrics"]["prompt_tokens"] > 0
    assert llm_span["metrics"]["completion_tokens"] > 0
    assert llm_span["metrics"]["tokens"] > 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_model_call_error_propagates(memory_logger):
    """Provider errors must propagate and the span must log the Exception instance."""
    assert not memory_logger.pop()

    model = _make_model(api_key="sk-invalid-braintrust-test-key")

    messages = [_make_user_msg("hello")] if IS_AGENTSCOPE_V2 else [{"role": "user", "content": "hello"}]
    with pytest.raises(Exception) as exc_info:
        await model(messages)

    assert type(exc_info.value).__module__.startswith("openai")

    spans = memory_logger.pop()
    llm_spans = [span for span in spans if _span_type(span) == SpanTypeAttribute.LLM]
    assert llm_spans
    llm_span = llm_spans[0]
    assert llm_span["metadata"]["provider"] == "openai"
    # error is logged (the exact serialized shape is Braintrust's concern; we
    # just verify it was populated as a truthy value, i.e. the wrapper called
    # span.log(error=exc) rather than swallowing).
    assert llm_span.get("error")


@pytest.mark.skipif(not IS_AGENTSCOPE_V2, reason="AgentScope 2.x Toolkit.call_tool API")
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_v2_toolkit_call_tool_creates_tool_span(memory_logger):
    from agentscope.message import TextBlock, ToolCallBlock
    from agentscope.state import AgentState
    from agentscope.tool import FunctionTool, ToolChunk, Toolkit

    assert not memory_logger.pop()

    def answer():
        return ToolChunk(content=[TextBlock(text="42")])

    toolkit = Toolkit(tools=[FunctionTool(answer, name="answer")])

    stream = await toolkit.call_tool(
        ToolCallBlock(id="test-call", name="answer", input="{}"),
        AgentState(),
    )
    chunks = [chunk async for chunk in stream]

    assert chunks

    spans = memory_logger.pop()
    tool_span = next(span for span in spans if _span_type(span) == "tool")
    assert tool_span["span_attributes"]["name"] == "answer.execute"
    assert tool_span["input"]["tool_name"] == "answer"


def test_setup_agentscope_is_idempotent():
    """Repeat setup calls must not double-wrap patched targets."""
    from agentscope.model import OpenAIChatModel

    wrapped = inspect.getattr_static(OpenAIChatModel, "__call__")
    assert hasattr(wrapped, "__wrapped__")

    setup_agentscope(project_name=PROJECT_NAME)
    setup_agentscope(project_name=PROJECT_NAME)

    assert inspect.getattr_static(OpenAIChatModel, "__call__") is wrapped


class TestAutoInstrumentAgentScope:
    def test_auto_instrument_agentscope(self):
        verify_autoinstrument_script("test_auto_agentscope.py")
