# pyright: reportPrivateUsage=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
from inspect import isawaitable

import pytest
from braintrust import logger
from braintrust.logger import start_span
from braintrust.test_helpers import init_test_logger
from braintrust.wrappers.agno import agent as agno_agent_module
from braintrust.wrappers.agno import run_helpers as agno_run_helpers_module
from braintrust.wrappers.agno import setup_agno
from braintrust.wrappers.agno import team as agno_team_module
from braintrust.wrappers.agno.agent import wrap_agent
from braintrust.wrappers.agno.team import wrap_team
from braintrust.wrappers.test_utils import verify_autoinstrument_script

TEST_ORG_ID = "test-org-123"
PROJECT_NAME = "test-agno-app"


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.mark.vcr
def test_agno_simple_agent_execution(memory_logger):
    Agent = pytest.importorskip("agno.agent.Agent")
    OpenAIChat = pytest.importorskip("agno.models.openai.OpenAIChat")

    setup_agno(project_name=PROJECT_NAME)

    assert not memory_logger.pop()

    # Create and configure the agent
    agent = Agent(
        name="Author Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions="You are librarian. Answer the questions by only replying with the author that wrote the book.",
    )

    response = agent.run("Charlotte's Web")

    # Basic assertion that the agent produced a response
    assert response
    assert response.content
    assert len(response.content) > 0

    # Check the spans generated
    spans = memory_logger.pop()
    assert len(spans) > 0

    # More detailed assertions based on expected span structure
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"

    # Check the root span (Agent.run)
    root_span = spans[0]
    assert root_span["span_attributes"]["name"] == "Author Agent.run"
    assert root_span["span_attributes"]["type"].value == "task"
    assert root_span["input"]["run_response"]["input"]["input_content"] == "Charlotte's Web"
    assert root_span["output"]["content"] == "E.B. White"
    assert root_span["output"]["status"] == "COMPLETED"
    assert root_span["output"]["model"] == "gpt-4o-mini"
    assert root_span["output"]["model_provider"] == "OpenAI"

    # Check metrics in root span
    assert "metrics" in root_span
    assert root_span["metrics"]["prompt_tokens"] > 0
    assert root_span["metrics"]["completion_tokens"] > 0
    assert (
        root_span["metrics"]["total_tokens"]
        == root_span["metrics"]["prompt_tokens"] + root_span["metrics"]["completion_tokens"]
    )
    assert root_span["metrics"]["duration"] > 0

    # Check the LLM span (OpenAI.response)
    llm_span = spans[1]
    assert llm_span["span_attributes"]["name"] == "OpenAI.response"
    assert llm_span["span_attributes"]["type"].value == "llm"
    assert llm_span["span_parents"] == [root_span["span_id"]]
    assert llm_span["metadata"]["model"] == "gpt-4o-mini"
    assert llm_span["metadata"]["provider"] == "OpenAI"

    # Check messages in LLM span input
    assert "messages" in llm_span["input"]
    messages = llm_span["input"]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "librarian" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Charlotte's Web"

    # Check LLM span output
    assert llm_span["output"]["content"] == "E.B. White"

    # Check LLM span metrics
    assert llm_span["metrics"]["prompt_tokens"] == 38
    assert llm_span["metrics"]["completion_tokens"] == 4
    assert llm_span["metrics"]["tokens"] == 42


class TestAutoInstrumentAgno:
    """Tests for auto_instrument() with Agno."""

    def test_auto_instrument_agno(self):
        """Test auto_instrument patches Agno and creates spans."""
        verify_autoinstrument_script("test_auto_agno.py")


class _FakeMetrics:
    def __init__(self):
        self.input_tokens = 1
        self.output_tokens = 2
        self.total_tokens = 3
        self.duration = 0.1
        self.time_to_first_token = 0.01


class _FakeRunOutput:
    def __init__(self, content: str):
        self.content = content
        self.status = "COMPLETED"
        self.model = "fake-model"
        self.model_provider = "FakeProvider"
        self.metrics = _FakeMetrics()


class _FakeEvent:
    def __init__(self, event: str, **kwargs):
        self.event = event
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_fake_component(name: str):
    class FakeComponent:
        def __init__(self):
            self.name = name

        def run(self, input, stream=False, **kwargs):
            if stream:
                def _stream():
                    yield _FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    yield _FakeEvent("RunContent", content=f"{input}-sync")
                    yield _FakeEvent("RunCompleted", metrics=_FakeMetrics())

                return _stream()
            return _FakeRunOutput(f"{input}-sync")

        def arun(self, input, stream=False, **kwargs):
            if stream:
                async def _astream():
                    yield _FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    yield _FakeEvent("RunContent", content=f"{input}-async")
                    yield _FakeEvent("RunCompleted", metrics=_FakeMetrics())

                return _astream()

            async def _result():
                return _FakeRunOutput(f"{input}-async")

            return _result()

    return FakeComponent


def _make_fake_async_dispatch_component(name: str):
    class FakeComponent:
        def __init__(self):
            self.name = name

        async def arun(self, input, stream=False, **kwargs):
            if stream:
                async def _astream():
                    yield _FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    yield _FakeEvent("RunContent", content=f"{input}-awaited-async")
                    yield _FakeEvent("RunCompleted", metrics=_FakeMetrics())

                return _astream()
            return {"content": f"{input}-awaited-async"}

    return FakeComponent


def _make_fake_error_component(name: str):
    class FakeComponent:
        def __init__(self):
            self.name = name

        def run(self, input, stream=False, **kwargs):
            if stream:
                def _stream():
                    yield _FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    raise RuntimeError("sync-stream-error")

                return _stream()
            return _FakeRunOutput(f"{input}-sync")

        def arun(self, input, stream=False, **kwargs):
            if stream:
                async def _astream():
                    yield _FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    raise RuntimeError("async-stream-error")

                return _astream()

            async def _result():
                return _FakeRunOutput(f"{input}-async")

            return _result()

    return FakeComponent


def _make_fake_private_public_component(name: str):
    class FakeComponent:
        def __init__(self):
            self.name = name
            self.calls = []

        def _run(self, run_response=None, run_messages=None, **kwargs):
            self.calls.append("_run")
            return _FakeRunOutput("private-run")

        def run(self, input, **kwargs):
            self.calls.append("run")
            return _FakeRunOutput("public-run")

        async def _arun(self, run_response=None, input=None, **kwargs):
            self.calls.append("_arun")
            return _FakeRunOutput("private-arun")

        def arun(self, input, **kwargs):
            self.calls.append("arun")

            async def _result():
                return _FakeRunOutput("public-arun")

            return _result()

    return FakeComponent


@pytest.mark.parametrize(
    "wrapper,name",
    [
        (wrap_agent, "CompatAgent"),
        (wrap_team, "CompatTeam"),
    ],
)
def test_agno_public_run_stream_dispatcher_compat(memory_logger, wrapper, name):
    """Ensures public run(stream=True) dispatchers are traced as a single streamed task span."""
    Component = wrapper(_make_fake_component(name))
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
    """Covers async streaming when arun returns an async iterator directly."""
    Component = wrapper(_make_fake_component(name))
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
    """Covers async streaming when arun must be awaited before yielding an async iterator."""
    Component = wrapper(_make_fake_async_dispatch_component(name))
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


class _StrictSpan:
    def __init__(self):
        self.ended = False

    def set_current(self):
        return None

    def unset_current(self):
        return None

    def log(self, **kwargs):
        if self.ended:
            raise AssertionError("log called after span.end()")

    def end(self):
        self.ended = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module,wrapper,name",
    [
        (agno_agent_module, wrap_agent, "StrictAgentAwaitedAsync"),
        (agno_team_module, wrap_team, "StrictTeamAwaitedAsync"),
    ],
)
async def test_agno_public_arun_awaited_async_iterator_span_lifecycle(monkeypatch, module, wrapper, name):
    """Guards against ending the span before awaited async-stream consumption completes."""
    strict_span = _StrictSpan()
    monkeypatch.setattr(module, "start_span", lambda **kwargs: strict_span)
    monkeypatch.setattr(agno_run_helpers_module, "start_span", lambda **kwargs: strict_span)

    Component = wrapper(_make_fake_async_dispatch_component(name))
    instance = Component()

    stream = await instance.arun("hello", stream=True)
    # Span must remain open until the async stream is consumed.
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
    """Validates non-streaming async dispatcher path logs output without stream-specific handling."""
    Component = wrapper(_make_fake_component(name))
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
    """Ensures sync stream exceptions are surfaced and recorded on the task span."""
    Component = wrapper(_make_fake_error_component(name))
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
    """Ensures async stream exceptions are surfaced and recorded on the task span."""
    Component = wrapper(_make_fake_error_component(name))
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
    """Covers early consumer break from sync stream without span lifecycle regressions."""
    Component = wrapper(_make_fake_component(name))
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
    """Covers early consumer break from async stream without span lifecycle regressions."""
    Component = wrapper(_make_fake_component(name))
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
    """Confirms public run spans nest under an already-active parent span."""
    Component = wrapper(_make_fake_component(name))
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
    """Confirms public arun streaming spans nest under an already-active parent span."""
    Component = wrapper(_make_fake_component(name))
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
    """Ensures classes from older Agno versions that expose private run methods still trace those paths."""
    Component = wrapper(_make_fake_private_public_component(name))
    instance = Component()

    _ = instance.run("hello")
    _ = await instance.arun("hello")
    _ = instance._run("rr", "rm")
    _ = await instance._arun("rr", "hello")

    spans = memory_logger.pop()
    span_names = {s["span_attributes"]["name"] for s in spans}

    # Calling public methods should not trigger tracing when private wrappers are present.
    assert instance.calls == ["run", "arun", "_run", "_arun"]
    # Private methods are traced, and they use the same span names as public run/arun.
    assert f"{name}.run" in span_names
    assert f"{name}.arun" in span_names
    assert len(spans) == 2
