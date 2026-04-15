from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.agentscope import setup_agentscope
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger


PROJECT_NAME = "test_agentscope"

setup_agentscope(project_name=PROJECT_NAME)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "cassette_library_dir": str(Path(__file__).parent / "cassettes"),
    }


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _span_type(span):
    span_type = span["span_attributes"]["type"]
    return span_type.value if hasattr(span_type, "value") else span_type


def _make_model(*, stream: bool = False):
    from agentscope.model import OpenAIChatModel

    return OpenAIChatModel(
        model_name="gpt-4o-mini",
        stream=stream,
        generate_kwargs={"temperature": 0},
    )


def _make_agent(name: str, sys_prompt: str, *, toolkit=None, multi_agent: bool = False):
    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIChatFormatter, OpenAIMultiAgentFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.tool import Toolkit

    agent = ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=_make_model(),
        formatter=OpenAIMultiAgentFormatter() if multi_agent else OpenAIChatFormatter(),
        toolkit=toolkit or Toolkit(),
        memory=InMemoryMemory(),
    )
    if hasattr(agent, "set_console_output_enabled"):
        agent.set_console_output_enabled(False)
    elif hasattr(agent, "disable_console_output"):
        agent.disable_console_output()
    return agent


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_simple_agent_run(memory_logger):
    from agentscope.message import Msg

    assert not memory_logger.pop()

    agent = _make_agent(
        "Friday",
        "You are a concise assistant. Answer in one sentence.",
    )

    response = await agent(
        Msg(
            name="user",
            content="Say hello in exactly two words.",
            role="user",
        )
    )

    assert response is not None

    spans = memory_logger.pop()
    agent_span = next(span for span in spans if span["span_attributes"]["name"] == "Friday.reply")
    llm_spans = [span for span in spans if _span_type(span) == SpanTypeAttribute.LLM]

    assert _span_type(agent_span) == "task"
    assert llm_spans
    assert llm_spans[0]["metadata"]["model"] == "gpt-4o-mini"
    assert "args" not in llm_spans[0]["input"]
    assert llm_spans[0]["input"]["messages"][0]["role"] == "system"
    assert llm_spans[0]["input"]["messages"][1]["role"] == "user"
    assert llm_spans[0]["input"]["messages"][1]["content"][0]["text"] == "Say hello in exactly two words."
    assert llm_spans[0]["output"]["role"] == "assistant"
    assert llm_spans[0]["output"]["content"][0]["text"] == "Hello there."
    assert "usage" not in llm_spans[0]["output"]
    assert agent_span["span_id"] in llm_spans[0]["span_parents"]


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
    assert llm_spans[0]["output"]["role"] == "assistant"
    assert llm_spans[0]["output"]["content"][0]["type"] == "tool_use"
    assert "usage" not in llm_spans[0]["output"]


@pytest.mark.asyncio
async def test_model_call_wrapper_stream_logs_final_output_and_metrics(memory_logger):
    from braintrust.integrations.agentscope.tracing import _model_call_wrapper

    assert not memory_logger.pop()

    class FakeOpenAIChatModel:
        model_name = "gpt-4o-mini"

    async def wrapped(*_args, **_kwargs):
        async def _stream():
            yield {"content": [{"type": "text", "text": "Hello"}]}
            yield {
                "content": [{"type": "text", "text": "Hello there!"}],
                "usage": {"prompt_tokens": 29, "completion_tokens": 3, "total_tokens": 32},
            }

        return _stream()

    stream = await _model_call_wrapper(
        wrapped,
        FakeOpenAIChatModel(),
        args=([{"role": "user", "content": "Say hi in two words."}],),
        kwargs={},
    )

    chunks = [chunk async for chunk in stream]

    assert chunks[-1]["content"][0]["text"] == "Hello there!"

    spans = memory_logger.pop()
    assert len(spans) == 1
    llm_span = spans[0]

    assert _span_type(llm_span) == SpanTypeAttribute.LLM
    assert llm_span["output"]["role"] == "assistant"
    assert llm_span["output"]["content"][0]["text"] == "Hello there!"
    assert llm_span["metrics"]["prompt_tokens"] == 29
    assert llm_span["metrics"]["completion_tokens"] == 3
    assert llm_span["metrics"]["tokens"] == 32


@pytest.mark.asyncio
async def test_model_call_wrapper_stream_span_covers_full_stream_duration(memory_logger):
    """Span end timestamp must be recorded after the stream is fully consumed, not before."""
    import asyncio

    from braintrust.integrations.agentscope.tracing import _model_call_wrapper

    assert not memory_logger.pop()

    class FakeModel:
        model_name = "gpt-4o-mini"

    async def wrapped(*_args, **_kwargs):
        async def _stream():
            for i in range(3):
                await asyncio.sleep(0.1)
                yield {"content": [{"type": "text", "text": f"chunk-{i}"}]}

        return _stream()

    stream = await _model_call_wrapper(
        wrapped,
        FakeModel(),
        args=([{"role": "user", "content": "hi"}],),
        kwargs={},
    )
    async for _ in stream:
        pass

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    m = span.get("metrics", {})
    duration_ms = (m["end"] - m["start"]) * 1000
    # Stream takes ~300ms (3 chunks × 100ms). The span duration must reflect that.
    assert duration_ms >= 200, f"Span duration {duration_ms:.0f}ms is too short; span ended before stream was consumed"


@pytest.mark.asyncio
async def test_toolkit_call_tool_function_wrapper_stream_span_covers_full_stream_duration(memory_logger):
    """Tool span end timestamp must be recorded after the stream is fully consumed, not before."""
    import asyncio

    from braintrust.integrations.agentscope.tracing import _toolkit_call_tool_function_wrapper

    assert not memory_logger.pop()

    class FakeToolkit:
        pass

    class FakeToolCall:
        name = "my_tool"

    async def wrapped(*_args, **_kwargs):
        async def _stream():
            for i in range(3):
                await asyncio.sleep(0.1)
                yield f"chunk-{i}"

        return _stream()

    stream = await _toolkit_call_tool_function_wrapper(
        wrapped,
        FakeToolkit(),
        args=(FakeToolCall(),),
        kwargs={},
    )
    async for _ in stream:
        pass

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    m = span.get("metrics", {})
    duration_ms = (m["end"] - m["start"]) * 1000
    # Stream takes ~300ms (3 chunks × 100ms). The span duration must reflect that.
    assert duration_ms >= 200, f"Span duration {duration_ms:.0f}ms is too short; span ended before stream was consumed"


class TestAutoInstrumentAgentScope:
    def test_auto_instrument_agentscope(self):
        verify_autoinstrument_script("test_auto_agentscope.py")
