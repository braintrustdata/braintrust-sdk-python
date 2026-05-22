import pytest
from braintrust import logger
from braintrust.integrations.autogen import setup_autogen
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger


PROJECT_NAME = "test_autogen"
setup_autogen(project_name=PROJECT_NAME)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _span_type(span):
    value = span["span_attributes"]["type"]
    return value.value if hasattr(value, "value") else value


def _make_agent(*, tools=None, stream=False):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0)
    return AssistantAgent(
        "assistant",
        model_client=model_client,
        tools=tools,
        model_client_stream=stream,
        system_message="You are concise. Answer directly.",
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_autogen_agent_run_creates_braintrust_spans(memory_logger):
    assert not memory_logger.pop()
    agent = _make_agent()

    result = await agent.run(task="Say hello in exactly two words.")

    assert result.messages[-1].content
    spans = memory_logger.pop()
    agent_span = next(span for span in spans if span["span_attributes"]["name"] == "assistant.run")
    assert _span_type(agent_span) == "task"
    assert agent_span["input"]["task"] == "Say hello in exactly two words."
    assert agent_span["metadata"]["component"] == "agent"
    assert agent_span["metadata"]["agent_name"] == "assistant"
    assert agent_span["output"]["messages"][-1]["source"] == "assistant"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_autogen_tool_run_is_child_of_agent_span(memory_logger):
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert not memory_logger.pop()
    agent = _make_agent(tools=[add])

    result = await agent.run(task="Use the add tool to compute 2 + 3. Return only the number.")

    assert result.messages[-1].content
    spans = memory_logger.pop()
    agent_span = next(span for span in spans if span["span_attributes"]["name"] == "assistant.run")
    tool_span = next(span for span in spans if _span_type(span) == SpanTypeAttribute.TOOL)
    assert tool_span["span_attributes"]["name"] == "add.run"
    assert tool_span["metadata"]["component"] == "tool"
    assert tool_span["metadata"]["tool_name"] == "add"
    assert tool_span["input"] == {"a": 2, "b": 3}
    assert tool_span["output"] == 5
    assert agent_span["span_id"] in tool_span["span_parents"]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_autogen_run_stream_aggregates_events(memory_logger):
    assert not memory_logger.pop()
    agent = _make_agent(stream=True)

    events = [event async for event in agent.run_stream(task="Say hello in exactly two words.")]

    assert events
    spans = memory_logger.pop()
    stream_span = next(span for span in spans if span["span_attributes"]["name"] == "assistant.run_stream")
    assert _span_type(stream_span) == "task"
    assert stream_span["output"]["events"][-1]["messages"][-1]["source"] == "assistant"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_autogen_team_run_creates_team_span(memory_logger):
    from autogen_agentchat.teams import RoundRobinGroupChat

    assert not memory_logger.pop()
    agent = _make_agent()
    team = RoundRobinGroupChat([agent], name="writing_team", max_turns=1)

    result = await team.run(task="Say hello in exactly two words.")

    assert result.messages[-1].content
    spans = memory_logger.pop()
    team_span = next(span for span in spans if span["span_attributes"]["name"] == "writing_team.run")
    assert _span_type(team_span) == "task"
    assert team_span["input"]["task"] == "Say hello in exactly two words."
    assert team_span["metadata"]["component"] == "team"
    assert team_span["metadata"]["team_name"] == "writing_team"
    assert team_span["metadata"]["participant_names"] == ["assistant"]


def test_autogen_auto_instrument_subprocess():
    verify_autoinstrument_script("test_auto_autogen.py")
