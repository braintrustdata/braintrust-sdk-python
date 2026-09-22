# pyright: reportPrivateUsage=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
import asyncio
import gc
import time

import pytest
from braintrust import logger
from braintrust.integrations.agno import setup_agno
from braintrust.integrations.agno import tracing as agno_tracing_module
from braintrust.integrations.agno.patchers import wrap_agent, wrap_team
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.logger import Attachment, start_span
from braintrust.test_helpers import init_test_logger

from ._test_agno_helpers import (
    PROJECT_NAME,
    FakeEvent,
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


@pytest.fixture(scope="module", autouse=True)
def setup_wrapper():
    setup_agno(project_name=PROJECT_NAME)
    yield


_TOOL_METADATA_ONLY_KEYS = ("tools", "tool_choice", "functions", "tool_call_limit")


def _assert_tool_fields_not_in_input(llm_span) -> None:
    """Guardrail against regressing the SKILL rule that puts tool definitions in metadata."""
    for forbidden in _TOOL_METADATA_ONLY_KEYS:
        assert forbidden not in llm_span["input"], f"{forbidden!r} must live under metadata, not input"


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
    assert response.session_id
    assert root_span["metadata"]["session_id"] == response.session_id
    assert root_span["context"]["span_origin"]["instrumentation"]["name"] == "agno-auto"
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
    # Tool-related fields must not leak into `input` even when they are absent
    # from this particular call — see integrations SKILL "Span Design / Fields".
    _assert_tool_fields_not_in_input(llm_span)
    assert llm_span["output"]["content"] == "E.B. White"
    assert llm_span["metrics"]["prompt_tokens"] > 0
    assert llm_span["metrics"]["completion_tokens"] > 0
    assert (
        llm_span["metrics"]["tokens"]
        == llm_span["metrics"]["prompt_tokens"] + llm_span["metrics"]["completion_tokens"]
    )


async def _pause_agno_run(memory_logger, component, async_mode):
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from agno.team import Team
    from agno.tools import tool

    cls = Agent if component == "agent" else Team
    if not hasattr(cls, "continue_run"):
        pytest.skip("This Agno version does not support team continuation")

    @tool(external_execution=True)
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        raise AssertionError("The client must execute this tool")

    instance = cls(
        name="Weather Assistant",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[get_weather],
        instructions="Use get_weather to find the weather. Report the result briefly.",
        session_id="configured-session",
        **({"members": []} if component == "team" else {}),
    )
    if async_mode:
        paused = await instance.arun("What's the weather in Paris?", session_id="run-session")
    else:
        paused = instance.run("What's the weather in Paris?")
    assert paused.is_paused
    original = memory_logger.pop()[0]
    assert paused.session_id == ("run-session" if async_mode else "configured-session")
    assert original["metadata"]["session_id"] == paused.session_id

    # Older Agno uses updated_tools; newer versions use requirements.
    if getattr(paused, "requirements", None):
        for requirement in paused.requirements:
            requirement.set_external_execution_result("The weather in Paris is 72F and sunny.")
        continuation = {"requirements": paused.requirements}
    else:
        for execution in paused.tools:
            if execution.external_execution_required:
                execution.result = "The weather in Paris is 72F and sunny."
        continuation = {"updated_tools": paused.tools}

    return instance, paused, continuation, original


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "component,async_mode,stream",
    [
        pytest.param("agent", False, False, id="response-sync-agent"),
        pytest.param("agent", True, False, id="response-async-agent"),
        pytest.param("agent", False, True, id="stream-sync-agent"),
        pytest.param("agent", True, True, id="stream-async-agent"),
        # Team uses the same dispatch wrappers; cover both patch targets without
        # repeating every return contract already exercised by Agent.
        pytest.param("team", False, False, id="response-sync-team"),
        pytest.param("team", True, True, id="stream-async-team"),
    ],
)
async def test_agno_resume_session_id(memory_logger, component, async_mode, stream):
    instance, paused, continuation, original = await _pause_agno_run(memory_logger, component, async_mode)

    # Resume without an ambient parent, as in a separate AG-UI request.
    assert logger.current_span() == logger.NOOP_SPAN
    method = instance.acontinue_run if async_mode else instance.continue_run
    result = method(paused, stream=stream, **continuation)
    if isawaitable(result):
        result = await result
    if stream:
        if async_mode:
            chunks = [chunk async for chunk in result]
        else:
            chunks = list(result)
        assert chunks
    else:
        assert result.content
        assert not result.is_paused

    resumed = memory_logger.pop()
    root = resumed[0]
    assert root["span_attributes"]["type"].value == "task"
    assert root["metadata"]["session_id"] == paused.session_id
    assert root["root_span_id"] != original["root_span_id"]
    assert root["span_attributes"]["name"] == f"Weather Assistant.{'a' if async_mode else ''}continue_run"
    assert root["context"]["span_origin"]["instrumentation"]["name"] == "agno-auto"
    llm_spans = [s for s in resumed if s["span_attributes"]["type"].value == "llm"]
    assert llm_spans
    assert all(s["span_parents"] == [root["span_id"]] for s in llm_spans)
    assert logger.current_span() == logger.NOOP_SPAN


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "async_mode,vcr_cassette_name",
    [
        (False, "test_agno_resume_session_id[stream-sync-agent]"),
        (True, "test_agno_resume_session_id[stream-async-agent]"),
    ],
    ids=["sync", "async"],
)
@pytest.mark.parametrize("consume", [False, True], ids=["unstarted", "partial"])
@pytest.mark.parametrize("cleanup", ["close", "gc"])
async def test_agno_resume_stream_cleanup(memory_logger, async_mode, vcr_cassette_name, consume, cleanup):
    instance, paused, continuation, _ = await _pause_agno_run(memory_logger, "agent", async_mode)
    with start_span(name="caller") as parent:
        method = instance.acontinue_run if async_mode else instance.continue_run
        stream = method(paused, stream=True, **continuation)
        if isawaitable(stream):
            stream = await stream
        assert logger.current_span() is parent
        if consume:
            # Stop after actual content, with the nested model stream suspended.
            if async_mode:
                async for chunk in stream:
                    if getattr(chunk, "content", None):
                        break
            else:
                for chunk in stream:
                    if getattr(chunk, "content", None):
                        break
            assert logger.current_span() is parent
        if cleanup == "close":
            if async_mode:
                await stream.aclose()
                await stream.aclose()
            else:
                stream.close()
                stream.close()
        del stream
        gc.collect()
        assert logger.current_span() is parent
        with start_span(name="after"):
            pass

    spans = memory_logger.pop()
    root = next(s for s in spans if s["span_attributes"]["name"].endswith("continue_run"))
    assert root["span_parents"] == [parent.span_id]
    assert root["metadata"]["session_id"] == paused.session_id
    assert "end" in root["metrics"]
    assert bool(root["output"].get("content")) == consume
    after = next(s for s in spans if s["span_attributes"]["name"] == "after")
    assert after["span_parents"] == [parent.span_id]
    gc.collect()
    assert not memory_logger.pop()


@pytest.mark.asyncio
async def test_agno_async_stream_cancellation_restores_context(memory_logger):
    # Local coverage for cancellation at an actual await: cassette playback
    # cannot reliably suspend the provider at a specific point.
    waiting = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def source():
        try:
            with start_span(name="nested"):
                yield FakeEvent("RunContent", content="partial")
                waiting.set()
                await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    with start_span(name="caller") as parent:
        span = start_span(name="continuation")
        span.set_current()
        stream = agno_tracing_module._trace_async_stream(source(), span, time.time())
        assert (await stream.__anext__()).content == "partial"
        assert logger.current_span() is parent
        task = asyncio.create_task(stream.__anext__())
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned_up.is_set()
        assert logger.current_span() is parent
        await stream.aclose()

    spans = memory_logger.pop()
    root = next(s for s in spans if s["span_attributes"]["name"] == "continuation")
    nested = next(s for s in spans if s["span_attributes"]["name"] == "nested")
    assert root["span_parents"] == [parent.span_id]
    assert nested["span_parents"] == [root["span_id"]]
    assert root["output"]["content"] == "partial"
    assert "end" in root["metrics"]
    assert "end" in nested["metrics"]
    del stream
    gc.collect()
    assert not memory_logger.pop()


@pytest.mark.vcr
def test_agno_agent_tools_metadata_placement(memory_logger):
    """Tool definitions must live in `metadata.tools`, not in the span input.

    Cassette: recorded against the real OpenAI API with a small ``get_weather``
    tool. Re-record with ``nox -s "test_agno(latest)" -- --vcr-record=all -k
    "test_agno_agent_tools_metadata_placement"``.
    """
    agent_module = pytest.importorskip("agno.agent")
    openai_module = pytest.importorskip("agno.models.openai")
    Agent = agent_module.Agent
    OpenAIChat = openai_module.OpenAIChat

    def get_weather(agent: Agent, city: str) -> str:
        """Return the current weather for *city*."""
        assert agent.name == "Weather Agent"
        return f"The weather in {city} is 72F and sunny."

    assert not memory_logger.pop()

    agent = Agent(
        name="Weather Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[get_weather],
        instructions="Use the get_weather tool to answer questions.",
    )

    response = agent.run("What's the weather in Paris?")
    assert response and response.content

    spans = memory_logger.pop()
    tool_spans = [s for s in spans if s["span_attributes"]["type"].value == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0]["input"] == {"city": "Paris"}
    assert tool_spans[0]["metadata"] == {"name": "get_weather", "entrypoint": "get_weather"}
    llm_spans = [s for s in spans if s["span_attributes"]["type"].value == "llm"]
    assert llm_spans, "expected at least one llm span"

    for llm_span in llm_spans:
        _assert_tool_fields_not_in_input(llm_span)
        tools_meta = llm_span["metadata"].get("tools")
        assert tools_meta, "expected metadata.tools to be populated on llm spans"
        # Agno passes its own tool schema (name / description / parameters) to
        # Model.response — spec placement rule is satisfied as long as the tool
        # definitions live in metadata, not input. Normalization to the fully
        # OpenAI-wrapped `{"type": "function", "function": {...}}` shape is
        # tracked separately.
        names = {t.get("name") or t.get("function", {}).get("name") for t in tools_meta}
        assert "get_weather" in names


@pytest.mark.vcr
def test_agno_agent_image_input_materializes_attachment(memory_logger):
    """Inline image bytes must be replaced by a Braintrust ``Attachment``.

    Cassette: recorded against the real OpenAI vision API with a 1x1 PNG.
    Re-record with ``nox -s "test_agno(latest)" -- --vcr-record=all -k
    "test_agno_agent_image_input_materializes_attachment"``.
    """
    agent_module = pytest.importorskip("agno.agent")
    openai_module = pytest.importorskip("agno.models.openai")
    media_module = pytest.importorskip("agno.media")
    Agent = agent_module.Agent
    OpenAIChat = openai_module.OpenAIChat
    Image = media_module.Image

    # 1x1 transparent PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
        b"\x89\x00\x00\x00\rIDATx\xdac\xfc\xcf\xc0\xf0\x1f\x00\x05\x05\x02"
        b"\x00_\xc8\xf1\xd2\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    assert not memory_logger.pop()

    agent = Agent(
        name="Vision Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions="Describe the image in one word.",
    )

    response = agent.run("What's in this image?", images=[Image(content=png_bytes, format="png")])
    assert response and response.content

    spans = memory_logger.pop()
    llm_spans = [s for s in spans if s["span_attributes"]["type"].value == "llm"]
    assert llm_spans

    def _find_attachments(obj):
        found = []
        if isinstance(obj, Attachment):
            found.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                found.extend(_find_attachments(v))
        elif isinstance(obj, list):
            for v in obj:
                found.extend(_find_attachments(v))
        return found

    attachments = _find_attachments(llm_spans[0]["input"])
    assert attachments, "expected image bytes to be materialized as an Attachment"
    att = attachments[0]
    assert att.reference["content_type"].startswith("image/")


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
