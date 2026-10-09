import copy
import os
from collections.abc import AsyncGenerator
from importlib.metadata import version as pkg_version
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.adk import setup_adk
from braintrust.integrations.adk.tracing import _create_thread_wrapper
from braintrust.logger import Attachment
from braintrust.test_helpers import init_test_logger
from google.adk import Agent


ADK_VERSION = tuple(int(x) for x in pkg_version("google-adk").split(".")[:3])
from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field


PROJECT_NAME = "test_adk"
ADK_MODEL = (
    "gemini-2.5-flash-lite"
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") in {"latest", "2.6.3"}
    else "gemini-2.0-flash"
)
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"

setup_adk(project_name=PROJECT_NAME)


@pytest.fixture(scope="module")
def vcr_config():
    """Google ADK VCR config - needs to uppercase HTTP methods (same as google_genai)."""
    record_mode = "none" if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) else "once"

    def before_record_request(request):
        # Normalize HTTP method to uppercase for consistency (Google API quirk)
        request.method = request.method.upper()
        return request

    return {
        "record_mode": record_mode,
        "filter_headers": [
            "authorization",
            "Authorization",
            "x-goog-api-key",
        ],
        "before_record_request": before_record_request,
        "decode_compressed_response": True,
    }


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


async def _create_runner(agent: Agent, *, app_name: str, user_id: str, session_id: str) -> Runner:
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    return Runner(agent=agent, app_name=app_name, session_service=session_service)


async def _run_final_responses(runner: Runner, *, user_id: str, session_id: str, new_message: types.Content):
    return [
        event
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=new_message)
        if event.is_final_response()
    ]


def _assert_llm_output_shape(output):
    # Extra keys like model_version may appear in newer ADK versions, so only check the expected ones.
    assert output["content"]["role"] == "model"
    assert "parts" in output["content"]
    assert "finish_reason" in output
    assert "usage_metadata" in output
    if ADK_VERSION >= (1, 15, 0) and "avg_logprobs" in output:
        assert output["avg_logprobs"] is not None


def _adk_model_call_spans(spans):
    return [row for row in spans if row["span_attributes"]["name"].startswith("llm_call")]


def _assert_provider_owns_llm_spans(spans):
    """Each ADK model call has one Google GenAI ``llm`` child, which alone carries usage."""
    model_call_ids = [row["span_id"] for row in _adk_model_call_spans(spans)]
    llm_spans = [row for row in spans if row["span_attributes"]["type"] == "llm"]
    assert model_call_ids
    assert sorted(row["span_parents"][0] for row in llm_spans) == sorted(model_call_ids)
    for row in llm_spans:
        assert row["span_attributes"]["name"] == "generate_content"
        assert row["context"]["span_origin"]["instrumentation"]["name"] == "google-genai-auto"
        assert row["metrics"]["prompt_tokens"] > 0
    assert [row for row in spans if "prompt_tokens" in row.get("metrics", {})] == llm_spans
    return llm_spans


def get_weather(location: str):
    """Get the weather for a location."""
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "sunny",
        "humidity": "45%",
        "wind": "5 mph NW",
    }


def _extract_text_parts(contents):
    texts = []
    for content in contents or []:
        for part in content.get("parts", []):
            text = part.get("text")
            if text is not None:
                texts.append(text)
    return texts


def test_create_thread_wrapper_exception_does_not_double_invoke_target():
    """Regression test: target exceptions must not cause a second invocation."""
    call_count = 0

    def create_thread(target, *args, **kwargs):
        return target(*args, **kwargs)

    def target():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _create_thread_wrapper(create_thread, None, (target,), {})

    assert call_count == 1


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_multi_turn_history_is_logged(memory_logger):
    """Multi-turn session history should be visible in traced LLM requests."""
    assert not memory_logger.pop()

    app_name = "conversation_app"
    user_id = "test-user"
    session_id = "test-session-conversation"
    agent = Agent(
        name="conversation_agent",
        model=ADK_MODEL,
        instruction=(
            "You are a concise assistant. "
            "When the user says their name, acknowledge it briefly. "
            "When later asked to recall it, answer with just the name."
        ),
    )
    runner = await _create_runner(agent, app_name=app_name, user_id=user_id, session_id=session_id)

    async def run_message(text: str) -> str:
        user_msg = types.Content(role="user", parts=[types.Part(text=text)])
        responses = await _run_final_responses(runner, user_id=user_id, session_id=session_id, new_message=user_msg)
        assert responses
        return responses[0].content.parts[0].text

    first_response_text = await run_message("Hi, my name is Alice.")
    second_response_text = await run_message("What name did I tell you?")

    memory_logger.flush()
    spans = memory_logger.pop()

    invocation_spans = [row for row in spans if row["span_attributes"]["name"] == f"invocation [{app_name}]"]
    assert len(invocation_spans) == 2
    for span in invocation_spans:
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "adk-auto"
    assert {span["metadata"]["session_id"] for span in invocation_spans} == {session_id}
    assert {span["input"]["new_message"]["parts"][0]["text"] for span in invocation_spans} == {
        "Hi, my name is Alice.",
        "What name did I tell you?",
    }

    assert len(_assert_provider_owns_llm_spans(spans)) == 2
    model_call_spans = _adk_model_call_spans(spans)

    follow_up_span = next(
        span
        for span in model_call_spans
        if "What name did I tell you?" in _extract_text_parts(span["input"]["contents"])
    )
    follow_up_texts = _extract_text_parts(follow_up_span["input"]["contents"])

    assert "Hi, my name is Alice." in follow_up_texts
    assert "What name did I tell you?" in follow_up_texts
    assert first_response_text in follow_up_texts
    assert "alice" in second_response_text.lower()


@pytest.mark.vcr
def test_adk_sync_runner_run_does_not_duplicate_invocation_spans(memory_logger):
    """Runner.run() emits one invocation span AND preserves Braintrust context through
    ADK's thread bridge (Runner.run dispatches to a background thread)."""
    import asyncio

    from braintrust import start_span
    from braintrust.util import LazyValue

    assert not memory_logger.pop()

    agent = Agent(
        name="weather_agent",
        model=ADK_MODEL,
        instruction="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather.",
        tools=[get_weather],
    )

    app_name = "weather_app"
    user_id = "test-user"
    session_id = "test-session"

    runner = asyncio.run(_create_runner(agent, app_name=app_name, user_id=user_id, session_id=session_id))
    user_msg = types.Content(role="user", parts=[types.Part(text="What's the weather in San Francisco?")])

    # The memory_logger fixture overrides via thread-local (_override_bg_logger),
    # but Runner.run() dispatches to a background thread where that's invisible.
    # We must also set _global_bg_logger so spans emitted on the worker thread
    # are captured.
    original_global_bg_logger = logger._state._global_bg_logger
    logger._state._global_bg_logger = LazyValue(lambda: memory_logger, use_mutex=False)
    try:
        with start_span(name="adk_thread_parent") as parent_span:
            responses = [
                event
                for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_msg)
                if event.is_final_response()
            ]
    finally:
        logger._state._global_bg_logger = original_global_bg_logger

    assert responses
    spans = memory_logger.pop()

    invocation_spans = [row for row in spans if row["span_attributes"]["name"] == f"invocation [{app_name}]"]
    assert len(invocation_spans) == 1, (
        f"expected exactly one invocation span for Runner.run(), got {len(invocation_spans)}: "
        f"{[span['span_id'] for span in invocation_spans]}"
    )

    invocation_span = invocation_spans[0]
    agent_spans = [row for row in spans if row["span_attributes"]["name"] == "agent_run [weather_agent]"]
    assert len(agent_spans) == 1
    assert invocation_span["span_id"] in agent_spans[0].get("span_parents", []), (
        f"agent span should be parented to the single sync invocation span {invocation_span['span_id']}, "
        f"got parents {agent_spans[0].get('span_parents')}"
    )

    # Thread-bridge context propagation: every ADK span emitted on the worker
    # thread should share the outer parent's root_span_id.
    adk_spans = [row for row in spans if row["context"]["span_origin"]["instrumentation"]["name"] == "adk-auto"]
    assert adk_spans
    for row in adk_spans:
        assert row["root_span_id"] == parent_span.root_span_id, (
            f"{row['span_attributes']['name']} lost thread context: "
            f"{row['root_span_id']} != {parent_span.root_span_id}"
        )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_custom_base_agent_root_preserves_agent_spans(memory_logger):
    """A plain BaseAgent root must preserve agent attribution for its whole subtree."""
    assert not memory_logger.pop()

    class CustomAgent(BaseAgent):
        async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(role="model", parts=[types.Part(text="step")]),
            )
            for sub_agent in self.sub_agents:
                async for event in sub_agent.run_async(ctx):
                    yield event

    app_name = "custom_root_app"
    user_id = "test-user"
    session_id = "test-session-custom-root"
    agent = CustomAgent(
        name="custom_root",
        sub_agents=[
            LlmAgent(
                name="child",
                model="gemini-2.5-flash-lite",
                instruction="Reply with only the word hello.",
            )
        ],
    )
    runner = await _create_runner(agent, app_name=app_name, user_id=user_id, session_id=session_id)

    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="Say hello.")]),
    ):
        pass

    spans = memory_logger.pop()
    expected_names = {
        f"invocation [{app_name}]",
        "agent_run [custom_root]",
        "agent_run [child]",
        "call_llm",
        "llm_call [direct_response]",
        "generate_content",
    }
    spans_by_name = {span["span_attributes"]["name"]: span for span in spans}
    assert len(spans) == len(expected_names)
    assert spans_by_name.keys() == expected_names

    parent_name_by_id = {span["span_id"]: name for name, span in spans_by_name.items()}
    assert parent_name_by_id[spans_by_name["agent_run [custom_root]"]["span_parents"][0]] == f"invocation [{app_name}]"
    assert parent_name_by_id[spans_by_name["agent_run [child]"]["span_parents"][0]] == "agent_run [custom_root]"
    assert parent_name_by_id[spans_by_name["call_llm"]["span_parents"][0]] == "agent_run [child]"
    assert parent_name_by_id[spans_by_name["llm_call [direct_response]"]["span_parents"][0]] == "call_llm"
    assert parent_name_by_id[spans_by_name["generate_content"]["span_parents"][0]] == "llm_call [direct_response]"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_braintrust_integration(memory_logger):
    assert not memory_logger.pop()

    agent = Agent(
        name="weather_agent",
        model=ADK_MODEL,
        instruction="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather.",
        tools=[get_weather],
    )

    # Set up session
    APP_NAME = "weather_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session"

    runner = await _create_runner(agent, app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    user_msg = types.Content(role="user", parts=[types.Part(text="What's the weather in San Francisco?")])

    responses = await _run_final_responses(runner, user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg)

    assert len(responses) > 0
    assert responses[0].content
    assert responses[0].content.parts

    response_text = responses[0].content.parts[0].text
    assert any(word in response_text.lower() for word in ["weather", "san francisco", "72", "sunny"]), (
        f"Response doesn't mention weather: {response_text}"
    )

    spans = memory_logger.pop()

    # Check that we have the expected span types
    span_types = {row["span_attributes"]["type"] for row in spans}
    assert "task" in span_types, "Missing 'task' spans"
    assert "llm" in span_types, "Missing 'llm' spans"

    # Verify the invocation span
    invocation_spans = [row for row in spans if row["span_attributes"]["name"] == "invocation [weather_app]"]
    assert len(invocation_spans) > 0, "Missing invocation span"
    invocation_span = invocation_spans[0]

    # Check invocation input
    assert "input" in invocation_span, "Missing input in invocation span"
    assert "new_message" in invocation_span["input"], "Missing new_message in input"
    assert invocation_span["input"]["new_message"]["parts"][0]["text"] == "What's the weather in San Francisco?"

    # Check metadata
    assert "metadata" in invocation_span, "Missing metadata in invocation span"
    assert invocation_span["metadata"]["user_id"] == "test-user"
    assert invocation_span["metadata"]["session_id"] == "test-session"

    # Verify LLM call spans
    _assert_provider_owns_llm_spans(spans)
    model_call_spans = _adk_model_call_spans(spans)
    assert len(model_call_spans) >= 2, "Should have at least 2 LLM calls (tool selection and response generation)"

    # Check tool selection LLM call
    tool_selection_spans = [span for span in model_call_spans if "tool_selection" in span["span_attributes"]["name"]]
    assert len(tool_selection_spans) > 0, "Missing tool selection LLM call"

    tool_selection_span = tool_selection_spans[0]
    assert "output" in tool_selection_span, "Missing output in tool selection span"
    assert "content" in tool_selection_span["output"], "Missing content in tool selection output"
    # Verify it called the get_weather function
    function_call = tool_selection_span["output"]["content"]["parts"][0]["function_call"]
    assert function_call["name"] == "get_weather"
    assert function_call["args"]["location"] == "San Francisco"

    adk_spans = [row for row in spans if row["context"]["span_origin"]["instrumentation"]["name"] == "adk-auto"]
    span_types_by_origin = {row["span_attributes"]["type"] for row in adk_spans}
    assert span_types_by_origin == {"task", "tool"}, (
        f"adk-auto origin should be on task/tool spans only: {span_types_by_origin}"
    )

    for span in model_call_spans:
        meta = span["metadata"]
        assert meta.get("provider") == "google", (
            f"Missing metadata.provider=google on {span['span_attributes']['name']}"
        )
        assert meta.get("model"), "Missing metadata.model on llm_call span"
        assert meta.get("tools"), "metadata.tools should be non-empty for a tool-using agent"
        tool_names = [
            fn.get("name") for tool_entry in meta["tools"] for fn in (tool_entry.get("function_declarations") or [])
        ]
        assert "get_weather" in tool_names, f"get_weather missing from metadata.tools: {tool_names}"
        assert "tools" not in span["input"].get("config", {}), "tools should not be in input.config"

    # Check response generation LLM call
    response_gen_spans = [
        span for span in model_call_spans if "response_generation" in span["span_attributes"]["name"]
    ]
    assert len(response_gen_spans) > 0, "Missing response generation LLM call"

    response_span = response_gen_spans[0]
    assert "output" in response_span, "Missing output in response generation span"
    response_output = response_span["output"]["content"]["parts"][0]["text"]
    assert "san francisco" in response_output.lower(), "Response doesn't mention San Francisco"
    assert "72" in response_output, "Response doesn't mention temperature"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_subagent_transfer_does_not_log_generator_exit(memory_logger):
    """Successful LlmAgent delegation must not mark spans as failed during generator cleanup."""
    assert not memory_logger.pop()

    delegation_model = "gemini-2.5-flash"
    specialist = Agent(
        name="capital_specialist",
        model=delegation_model,
        description="The only agent allowed to answer geography questions.",
        instruction="Answer geography questions accurately and in one short sentence.",
    )
    coordinator = Agent(
        name="coordinator",
        model=delegation_model,
        description="Routes geography questions to the capital specialist without answering them.",
        instruction=(
            "You cannot answer questions yourself. For every request, immediately call "
            "transfer_to_agent with agent_name='capital_specialist'."
        ),
        sub_agents=[specialist],
    )

    app_name = "delegation_app"
    user_id = "test-user"
    session_id = "test-session-delegation"
    runner = await _create_runner(
        coordinator,
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    user_msg = types.Content(
        role="user",
        parts=[types.Part(text="What is the capital of France? Delegate this to the specialist.")],
    )

    events = [
        event
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_msg,
        )
    ]

    assert any(event.actions.transfer_to_agent == specialist.name for event in events)
    final_responses = [event for event in events if event.is_final_response()]
    assert final_responses
    assert "paris" in final_responses[-1].content.parts[0].text.lower()

    spans = memory_logger.pop()
    assert spans
    assert all("error" not in span for span in spans), [
        (span["span_attributes"]["name"], span.get("error")) for span in spans if "error" in span
    ]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_nested_subagent_tool_calls_are_traced(memory_logger):
    assert not memory_logger.pop()

    def get_weather(location: str):
        """Get the weather for a location."""
        return {
            "location": location,
            "temperature": "72°F",
            "condition": "sunny",
        }

    leaf_agent = Agent(
        name="weather_agent",
        model=ADK_MODEL,
        instruction="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather.",
        tools=[get_weather],
    )
    agent = SequentialAgent(
        name="root_agent",
        sub_agents=[
            ParallelAgent(
                name="parallel_weather_agent",
                sub_agents=[leaf_agent],
            )
        ],
    )

    app_name = "nested_weather_app"
    user_id = "test-user"
    session_id = "test-session-nested"

    runner = await _create_runner(agent, app_name=app_name, user_id=user_id, session_id=session_id)
    user_msg = types.Content(role="user", parts=[types.Part(text="What's the weather in San Francisco?")])

    responses = await _run_final_responses(runner, user_id=user_id, session_id=session_id, new_message=user_msg)

    assert responses
    assert responses[0].content
    response_text = responses[0].content.parts[0].text
    assert "san francisco" in response_text.lower()

    spans = memory_logger.pop()

    tool_spans = [row for row in spans if row["span_attributes"]["type"] == "tool"]
    assert len(tool_spans) == 1, (
        f"Expected one tool span, got {[row['span_attributes']['name'] for row in tool_spans]}"
    )

    tool_span = tool_spans[0]
    assert tool_span["span_attributes"]["name"] == "tool [get_weather]"
    assert tool_span["input"]["arguments"] == {"location": "San Francisco"}
    assert tool_span["output"]["location"] == "San Francisco"
    assert tool_span["output"]["temperature"] == "72°F"


@pytest.mark.asyncio
@pytest.mark.skipif(ADK_VERSION < (2, 10, 0), reason="Workflow nodes require ADK 2.10+")
async def test_adk_workflow_tool_node_is_traced(memory_logger):
    """A graph ToolNode should emit tool and workflow node spans."""
    from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
    from google.adk.workflow import START, Workflow

    assert not memory_logger.pop()

    tool = LoadArtifactsTool()
    workflow = Workflow(
        name="weather_workflow",
        edges=[(START, tool)],
    )
    app_name = "workflow_weather_app"
    user_id = "test-user"
    session_id = "test-session-workflow"
    runner = await _create_runner(workflow, app_name=app_name, user_id=user_id, session_id=session_id)

    events = [
        event
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text='{"artifact_names": ["report.txt"]}')]),
        )
    ]

    assert events
    spans = memory_logger.pop()
    invocation = next(row for row in spans if row["span_attributes"]["name"] == f"invocation [{app_name}]")
    workflow_span = next(row for row in spans if row["span_attributes"]["name"] == "workflow [weather_workflow]")
    tool_span = next(row for row in spans if row["span_attributes"]["type"] == "tool")

    assert workflow_span["span_attributes"]["type"] == "task"
    assert workflow_span["span_parents"] == [invocation["span_id"]]
    assert tool_span["span_attributes"]["name"] == "tool [load_artifacts]"
    assert tool_span["input"]["arguments"] == {"artifact_names": ["report.txt"]}
    assert tool_span["output"]["artifact_names"] == ["report.txt"]
    assert "temporarily inserted and removed" in tool_span["output"]["status"]
    tool_node_span = next(row for row in spans if row["span_id"] == tool_span["span_parents"][0])
    assert tool_node_span["span_attributes"]["name"] == "workflow_node [load_artifacts]"
    assert tool_node_span["output"]["artifact_names"] == ["report.txt"]


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.skipif(ADK_VERSION < (2, 10, 0), reason="Workflow nodes require ADK 2.10+")
async def test_adk_workflow_root_traces_recorded_agent_tool_calls(memory_logger):
    """A real ADK LLM/tool turn stays nested under the Workflow task span."""
    from google.adk.workflow import START, Workflow

    assert not memory_logger.pop()

    weather_agent = Agent(
        name="weather_agent",
        model=ADK_MODEL,
        instruction="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather.",
        tools=[get_weather],
    )
    workflow = Workflow(name="weather_workflow", edges=[(START, weather_agent)])
    app_name = "workflow_weather_app"
    user_id = "test-user"
    session_id = "test-session-workflow-recorded"
    runner = await _create_runner(workflow, app_name=app_name, user_id=user_id, session_id=session_id)
    response = await _run_final_responses(
        runner,
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="What's the weather in San Francisco?")]),
    )

    assert response
    spans = memory_logger.pop()
    invocation = next(row for row in spans if row["span_attributes"]["name"] == f"invocation [{app_name}]")
    workflow_span = next(row for row in spans if row["span_attributes"]["name"] == "workflow [weather_workflow]")
    agent_span = next(row for row in spans if row["span_attributes"]["name"] == "agent_run [weather_agent]")
    tool_span = next(row for row in spans if row["span_attributes"]["name"] == "tool [get_weather]")
    assert workflow_span["span_parents"] == [invocation["span_id"]]
    assert agent_span["span_parents"] == [workflow_span["span_id"]]
    assert tool_span["span_attributes"]["type"] == "tool"
    assert tool_span["input"]["arguments"] == {"location": "San Francisco"}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_max_tokens_captures_content(memory_logger):
    """Test that content is captured even when MAX_TOKENS finish reason occurs."""
    assert not memory_logger.pop()

    agent = Agent(
        name="creative_agent",
        model=ADK_MODEL,
        instruction="You are a creative storyteller.",
        generate_content_config=types.GenerateContentConfig(
            max_output_tokens=50,  # Set low to trigger MAX_TOKENS
            temperature=0.7,
        ),
    )

    APP_NAME = "creative_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-max-tokens"

    runner = await _create_runner(agent, app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    user_msg = types.Content(role="user", parts=[types.Part(text="Tell me a long story about a lighthouse.")])

    responses = await _run_final_responses(runner, user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg)

    assert len(responses) > 0
    spans = memory_logger.pop()

    # Find the LLM call span
    model_call_spans = _adk_model_call_spans(spans)
    assert len(model_call_spans) > 0, "Missing LLM call span"

    model_call_span = model_call_spans[0]
    assert "output" in model_call_span, "Missing output in LLM span"

    # Sampling config from generate_content_config is captured in input.config
    config = model_call_span["input"]["config"]
    assert config["max_output_tokens"] == 50
    assert config["temperature"] == 0.7

    output = model_call_span["output"]

    # When MAX_TOKENS is hit, we should still have content captured
    # The integration should merge content from earlier events if the final event lacks it
    # Every recorded cassette ends with MAX_TOKENS.
    assert output["finish_reason"] == "MAX_TOKENS"
    assert "content" in output, "Content should be captured even with MAX_TOKENS"
    assert output["content"] is not None, "Content should not be None"
    assert "parts" in output["content"], "Content should have parts"
    assert len(output["content"]["parts"]) > 0, "Content parts should not be empty"

    # Verify the text was actually captured
    text_content = output["content"]["parts"][0].get("text", "")
    assert len(text_content) > 0, "Should have captured some text content before MAX_TOKENS"

    # Verify usage metadata is present
    assert "usage_metadata" in output, "Should have usage metadata"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_binary_data_attachment_conversion(memory_logger):
    """Test that binary data in messages is converted to Attachment references."""
    assert not memory_logger.pop()

    agent = Agent(
        name="vision_agent",
        model=ADK_MODEL,
        instruction="You are a helpful assistant that can analyze images.",
        generate_content_config=types.GenerateContentConfig(
            max_output_tokens=150,
        ),
    )

    APP_NAME = "vision_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-image"

    runner = await _create_runner(agent, app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    # Load test image from shared SDK fixtures
    image_data = (FIXTURES_DIR / "test-image.png").read_bytes()

    # Create message with inline binary data
    user_msg = types.Content(
        role="user",
        parts=[
            types.Part(inline_data=types.Blob(mime_type="image/png", data=image_data)),
            types.Part(text="What color is this image?"),
        ],
    )

    responses = await _run_final_responses(runner, user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg)

    assert len(responses) > 0

    spans = memory_logger.pop()

    # Find the invocation span
    invocation_spans = [row for row in spans if row["span_attributes"]["name"] == "invocation [vision_app]"]
    assert len(invocation_spans) > 0, "Missing invocation span"
    invocation_span = invocation_spans[0]

    # Verify the input contains properly serialized content
    assert "input" in invocation_span, "Missing input in invocation span"
    assert "new_message" in invocation_span["input"], "Missing new_message in input"

    new_message = invocation_span["input"]["new_message"]
    assert "parts" in new_message, "Missing parts in new_message"
    assert len(new_message["parts"]) == 2, "Should have 2 parts (image and text)"

    # First part should be the image as an Attachment reference
    image_part = new_message["parts"][0]
    assert "image_url" in image_part, "Image part should have image_url field"
    assert "url" in image_part["image_url"], "image_url should have url field"

    attachment_ref = image_part["image_url"]["url"]
    # Verify it's an Attachment object, not raw binary data
    assert isinstance(attachment_ref, Attachment), "Attachment should be an Attachment object"
    ref = attachment_ref.reference
    assert "key" in ref, "Attachment reference should have a key"
    assert "filename" in ref, "Attachment reference should have a filename"
    assert "content_type" in ref, "Attachment reference should have a content_type"
    assert ref["content_type"] == "image/png", "Content type should be image/png"
    assert ref["filename"] == "image.png", "Filename should be image.png"

    # Second part should be the text
    text_part = new_message["parts"][1]
    assert "text" in text_part, "Second part should have text"
    assert text_part["text"] == "What color is this image?", "Text content should match"

    # Verify no raw binary data is present in the logged span
    span_str = str(invocation_span)
    # Check that the binary PNG signature is NOT in the logged data
    assert b"\x89PNG".hex() not in span_str, "Raw binary data should not be in logged span"
    assert "89504e47" not in span_str.lower(), "Raw binary data (hex) should not be in logged span"

    # Find model-call spans and verify they also don't contain raw binary
    model_call_spans = _adk_model_call_spans(spans)
    assert len(model_call_spans) > 0, "Should have model-call spans"

    for model_call_span in model_call_spans:
        if "input" in model_call_span and "contents" in model_call_span["input"]:
            llm_str = str(model_call_span["input"])
            assert b"\x89PNG".hex() not in llm_str, "Raw binary data should not be in LLM span input"
            assert "89504e47" not in llm_str.lower(), "Raw binary data (hex) should not be in LLM span input"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_captures_metrics(memory_logger):
    """setup_adk() alone yields one usage-bearing Google GenAI ``llm`` span under a task ``llm_call``."""
    assert not memory_logger.pop()

    agent = Agent(
        name="metrics_agent",
        model=ADK_MODEL,
        instruction="You are a helpful assistant.",
    )

    APP_NAME = "metrics_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-metrics"

    runner = await _create_runner(agent, app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    user_msg = types.Content(role="user", parts=[types.Part(text="Say hello in 3 words")])

    responses = await _run_final_responses(runner, user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg)

    assert len(responses) > 0

    spans = memory_logger.pop()

    # setup_adk() enables Google GenAI, which owns the one usage-bearing ``llm``
    # span per Gemini request, so trace totals count its usage once.
    _assert_provider_owns_llm_spans(spans)

    # ADK's model-call span keeps its request/response, call type, and time to
    # first token, as a task parent of the provider span.
    (adk_model_span,) = _adk_model_call_spans(spans)
    assert adk_model_span["span_attributes"]["type"] == "task"
    assert adk_model_span["span_attributes"]["name"] == "llm_call [direct_response]"
    assert "usage_metadata" in adk_model_span["output"]
    assert 0 < adk_model_span["metrics"]["time_to_first_token"] < 10


# _determine_llm_call_type paths are exercised through the VCR-backed
# integration tests: `test_adk_braintrust_integration` asserts the
# tool_selection / response_generation span names, and the direct_response
# path is asserted in `test_adk_captures_metrics`.


class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age", ge=0, le=150)
    address: Address = Field(description="Person's address")


CITY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "Name of the city",
        },
        "population": {
            "type": "integer",
            "description": "Population of the city",
            "minimum": 0,
        },
        "country": {
            "type": "string",
            "description": "Country where the city is located",
        },
    },
    "required": ["city", "country"],
}


# _capture_config's allowlisted fields are exercised through the VCR-backed
# integration tests: response_schema and response_json_schema
# (`test_adk_structured_output_schema`), and the sampling params
# (`test_adk_max_tokens_captures_content`).


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_kwargs", "prompt", "schema_field", "expected_schema"),
    [
        pytest.param(
            # Nested Pydantic output_schema is sent as response_schema
            {
                "name": "nested_agent",
                "instruction": "Return a person with their address.",
                "output_schema": Person,
                "output_key": "person_data",
            },
            "Give me info about Alice who lives in Paris, France.",
            "response_schema",
            {
                "properties": {
                    "name": {
                        "description": "Person's name",
                        "title": "Name",
                        "type": "string",
                    },
                    "age": {
                        "description": "Person's age",
                        "maximum": 150,
                        "minimum": 0,
                        "title": "Age",
                        "type": "integer",
                    },
                    "address": {
                        "$ref": "#/$defs/Address",
                        "description": "Person's address",
                    },
                },
                "$defs": {
                    "Address": {
                        "properties": {
                            "street": {
                                "description": "Street address",
                                "title": "Street",
                                "type": "string",
                            },
                            "city": {
                                "description": "City name",
                                "title": "City",
                                "type": "string",
                            },
                            "country": {
                                "description": "Country name",
                                "title": "Country",
                                "type": "string",
                            },
                        },
                        "required": ["street", "city", "country"],
                        "title": "Address",
                        "type": "object",
                    },
                },
                "required": ["name", "age", "address"],
                "title": "Person",
                "type": "object",
            },
            id="pydantic_output_schema",
        ),
        pytest.param(
            # Plain JSON schema dict passed via generate_content_config should be preserved
            {
                "name": "city_agent",
                "instruction": "You are a City Information Agent. Provide city information.",
                "generate_content_config": types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=CITY_JSON_SCHEMA,
                ),
            },
            "Tell me about Tokyo",
            "response_json_schema",
            copy.deepcopy(CITY_JSON_SCHEMA),
            id="response_json_schema_dict",
        ),
    ],
)
async def test_adk_structured_output_schema(memory_logger, agent_kwargs, prompt, schema_field, expected_schema):
    """Test that structured output schemas are properly serialized into the LLM span input."""
    from unittest.mock import ANY

    assert not memory_logger.pop()

    agent = LlmAgent(model=ADK_MODEL, **agent_kwargs)

    APP_NAME = "schema_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-schema"

    runner = await _create_runner(agent, app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    user_msg = types.Content(role="user", parts=[types.Part(text=prompt)])

    responses = await _run_final_responses(runner, user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg)

    assert len(responses) > 0

    spans = memory_logger.pop()

    # Find LLM span with the schema
    model_call_spans_with_schema = [
        span
        for span in _adk_model_call_spans(spans)
        if "input" in span and "config" in span["input"] and span["input"]["config"].get(schema_field) is not None
    ]

    assert len(model_call_spans_with_schema) > 0, f"Should have at least one LLM call with {schema_field}"

    model_call_span = model_call_spans_with_schema[0]

    # Assert complete input structure
    assert model_call_span["input"] == {
        "model": ADK_MODEL,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "config": {
            "system_instruction": ANY,  # Contains agent name
            "response_mime_type": "application/json",
            schema_field: expected_schema,
        },
        "live_connect_config": ANY,
    }

    _assert_llm_output_shape(model_call_span["output"])


class TestAutoInstrumentADK:
    """Tests for auto_instrument() with Google ADK."""

    def test_auto_instrument_adk(self):
        """Test auto_instrument patches ADK classes and is idempotent."""
        from braintrust.integrations.test_utils import verify_autoinstrument_script

        verify_autoinstrument_script("test_auto_adk.py")


class TestManualWrapADK:
    """Tests the public manual wrapping helpers in an isolated process."""

    @pytest.mark.skipif(ADK_VERSION < (2, 10, 0), reason="Workflow nodes require ADK 2.10+")
    def test_manual_wrap_adk_workflow(self):
        from braintrust.integrations.test_utils import verify_autoinstrument_script

        verify_autoinstrument_script("test_manual_wrap_adk_workflow.py")
