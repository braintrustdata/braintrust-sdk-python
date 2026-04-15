# pyright: reportPrivateUsage=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.agno import setup_agno
from braintrust.integrations.agno import tracing as agno_tracing_module
from braintrust.integrations.agno.patchers import wrap_agent, wrap_team
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.logger import start_span
from braintrust.test_helpers import init_test_logger

from ._test_agno_helpers import (
    PROJECT_NAME,
    StrictSpan,
    isawaitable,
    make_fake_async_dispatch_component,
    make_fake_component,
    make_fake_error_component,
    make_fake_private_public_component,
)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "cassette_library_dir": str(Path(__file__).parent / "cassettes"),
    }


@pytest.fixture(scope="module", autouse=True)
def setup_wrapper():
    setup_agno(project_name=PROJECT_NAME)
    yield


@pytest.mark.vcr
def test_agno_simple_agent_execution(memory_logger):
    agent_module = pytest.importorskip("agno.agent")
    openai_module = pytest.importorskip("agno.models.openai")
    Agent = agent_module.Agent
    OpenAIChat = openai_module.OpenAIChat

    assert not memory_logger.pop()

    agent = Agent(
        name="Author Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions="You are librarian. Answer the questions by only replying with the author that wrote the book.",
    )

    response = agent.run("Charlotte's Web")

    assert response
    assert response.content
    assert len(response.content) > 0

    spans = memory_logger.pop()
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"

    root_span = spans[0]
    assert root_span["span_attributes"]["name"] == "Author Agent.run"
    assert root_span["span_attributes"]["type"].value == "task"
    root_input = root_span["input"]
    if "input" in root_input:
        assert root_input["input"] == "Charlotte's Web"
    else:
        assert root_input["run_response"]["input"]["input_content"] == "Charlotte's Web"
    assert root_span["output"]["content"] == "E.B. White"
    assert root_span["output"]["status"] == "COMPLETED"
    assert root_span["output"]["model"] == "gpt-4o-mini"
    assert root_span["output"]["model_provider"] == "OpenAI"
    assert root_span["metrics"]["prompt_tokens"] > 0
    assert root_span["metrics"]["completion_tokens"] > 0
    assert (
        root_span["metrics"]["tokens"]
        == root_span["metrics"]["prompt_tokens"] + root_span["metrics"]["completion_tokens"]
    )
    assert root_span["metrics"]["duration"] > 0

    llm_span = spans[1]
    llm_span_name = llm_span["span_attributes"]["name"]
    assert "OpenAI" in llm_span_name
    assert llm_span_name.endswith(".response")
    assert llm_span["span_attributes"]["type"].value == "llm"
    assert llm_span["span_parents"] == [root_span["span_id"]]
    assert llm_span["metadata"]["model"] == "gpt-4o-mini"
    assert llm_span["metadata"]["provider"] == "OpenAI"

    messages = llm_span["input"]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "librarian" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Charlotte's Web"
    assert llm_span["output"]["content"] == "E.B. White"
    assert llm_span["metrics"]["prompt_tokens"] == 38
    assert llm_span["metrics"]["completion_tokens"] == 4
    assert llm_span["metrics"]["tokens"] == 42


def test_get_model_name_prefers_stable_provider_attribute():
    class FakeModel:
        provider = "OpenAI"

        def get_provider(self):
            return "OpenAI Chat"

    assert agno_tracing_module._get_model_name(FakeModel()) == "OpenAI"


class TestAutoInstrumentAgno:
    def test_auto_instrument_agno(self):
        verify_autoinstrument_script("test_auto_agno.py")


@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgent"),
        (wrap_team, "CompatTeam"),
    ],
)
def test_agno_public_run_stream_dispatcher_compat(memory_logger, wrapper, name):
    Component = wrapper(make_fake_component(name))
    instance = Component()

    chunks = list(instance.run("hello", stream=True))
    assert len(chunks) == 3

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == f"{name}.run"
    assert span["output"]["content"] == "hello-sync"
    assert span["metrics"]["prompt_tokens"] == 1
    assert span["metrics"]["completion_tokens"] == 2
    assert span["metrics"]["duration"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentAsync"),
        (wrap_team, "CompatTeamAsync"),
    ],
)
async def test_agno_public_arun_stream_dispatcher_compat(memory_logger, wrapper, name):
    Component = wrapper(make_fake_component(name))
    instance = Component()

    stream = instance.arun("hello", stream=True)
    if isawaitable(stream):
        stream = await stream
    chunks = [chunk async for chunk in stream]
    assert len(chunks) == 3

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == f"{name}.arun"
    assert span["output"]["content"] == "hello-async"
    assert span["metrics"]["prompt_tokens"] == 1
    assert span["metrics"]["completion_tokens"] == 2
    assert span["metrics"]["duration"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentAwaitedAsync"),
        (wrap_team, "CompatTeamAwaitedAsync"),
    ],
)
async def test_agno_public_arun_awaited_async_iterator_compat(memory_logger, wrapper, name):
    Component = wrapper(make_fake_async_dispatch_component(name))
    instance = Component()

    stream = await instance.arun("hello", stream=True)
    chunks = [chunk async for chunk in stream]
    assert len(chunks) == 3

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == f"{name}.arun"
    assert span["output"]["content"] == "hello-awaited-async"
    assert span["metrics"]["prompt_tokens"] == 1
    assert span["metrics"]["completion_tokens"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "StrictAgentAwaitedAsync"),
        (wrap_team, "StrictTeamAwaitedAsync"),
    ],
)
async def test_agno_public_arun_awaited_async_iterator_span_lifecycle(monkeypatch, wrapper, name):
    strict_span = StrictSpan()
    monkeypatch.setattr(agno_tracing_module, "start_span", lambda **kwargs: strict_span)

    Component = wrapper(make_fake_async_dispatch_component(name))
    instance = Component()

    stream = await instance.arun("hello", stream=True)
    assert strict_span.ended is False

    chunks = [chunk async for chunk in stream]
    assert len(chunks) == 3
    assert strict_span.ended is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentAsyncNonStream"),
        (wrap_team, "CompatTeamAsyncNonStream"),
    ],
)
async def test_agno_public_arun_non_stream_awaitable_compat(memory_logger, wrapper, name):
    Component = wrapper(make_fake_component(name))
    instance = Component()

    result = instance.arun("hello", stream=False)
    if isawaitable(result):
        result = await result

    assert result.content == "hello-async"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == f"{name}.arun"
    assert span["output"]


@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentSyncError"),
        (wrap_team, "CompatTeamSyncError"),
    ],
)
def test_agno_public_run_stream_error_path(memory_logger, wrapper, name):
    Component = wrapper(make_fake_error_component(name))
    instance = Component()

    with pytest.raises(RuntimeError, match="sync-stream-error"):
        list(instance.run("boom", stream=True))

    spans = memory_logger.pop()
    assert len(spans) == 1
    assert "sync-stream-error" in spans[0]["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentAsyncError"),
        (wrap_team, "CompatTeamAsyncError"),
    ],
)
async def test_agno_public_arun_stream_error_path(memory_logger, wrapper, name):
    Component = wrapper(make_fake_error_component(name))
    instance = Component()

    stream = instance.arun("boom", stream=True)
    if isawaitable(stream):
        stream = await stream

    with pytest.raises(RuntimeError, match="async-stream-error"):
        async for _ in stream:
            pass

    spans = memory_logger.pop()
    assert len(spans) == 1
    assert "async-stream-error" in spans[0]["error"]


@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentSyncEarlyBreak"),
        (wrap_team, "CompatTeamSyncEarlyBreak"),
    ],
)
def test_agno_public_run_stream_early_break(memory_logger, wrapper, name):
    Component = wrapper(make_fake_component(name))
    instance = Component()

    for _ in instance.run("hello", stream=True):
        break

    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["span_attributes"]["name"] == f"{name}.run"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentAsyncEarlyBreak"),
        (wrap_team, "CompatTeamAsyncEarlyBreak"),
    ],
)
async def test_agno_public_arun_stream_early_break(memory_logger, wrapper, name):
    Component = wrapper(make_fake_component(name))
    instance = Component()

    stream = instance.arun("hello", stream=True)
    if isawaitable(stream):
        stream = await stream

    async for _ in stream:
        break

    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["span_attributes"]["name"] == f"{name}.arun"


@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentParentSync"),
        (wrap_team, "CompatTeamParentSync"),
    ],
)
def test_agno_public_run_parent_span_nesting(memory_logger, wrapper, name):
    Component = wrapper(make_fake_component(name))
    instance = Component()

    with start_span(name="outer_sync_parent", type="task"):
        instance.run("hello")

    spans = memory_logger.pop()
    by_name = {s["span_attributes"]["name"]: s for s in spans}
    outer = by_name["outer_sync_parent"]
    child = by_name[f"{name}.run"]
    assert child["span_parents"] == [outer["span_id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentParentAsync"),
        (wrap_team, "CompatTeamParentAsync"),
    ],
)
async def test_agno_public_arun_parent_span_nesting(memory_logger, wrapper, name):
    Component = wrapper(make_fake_component(name))
    instance = Component()

    with start_span(name="outer_async_parent", type="task"):
        stream = instance.arun("hello", stream=True)
        if isawaitable(stream):
            stream = await stream
        async for _ in stream:
            pass

    spans = memory_logger.pop()
    by_name = {s["span_attributes"]["name"]: s for s in spans}
    outer = by_name["outer_async_parent"]
    child = by_name[f"{name}.arun"]
    assert child["span_parents"] == [outer["span_id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgentPrivatePrecedence"),
        (wrap_team, "CompatTeamPrivatePrecedence"),
    ],
)
async def test_agno_private_method_precedence_over_public(memory_logger, wrapper, name):
    Component = wrapper(make_fake_private_public_component(name))
    instance = Component()

    _ = instance.run("hello")
    _ = await instance.arun("hello")
    _ = instance._run("rr", "rm")
    _ = await instance._arun("rr", "hello")

    spans = memory_logger.pop()
    span_names = {s["span_attributes"]["name"] for s in spans}

    assert instance.calls == ["run", "arun", "_run", "_arun"]
    assert f"{name}.run" in span_names
    assert f"{name}.arun" in span_names
    assert len(spans) == 2
