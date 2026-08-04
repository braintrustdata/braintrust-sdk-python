import os
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
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from pydantic import BaseModel, Field


PROJECT_NAME = "test_adk"
ADK_MODEL = (
    "gemini-2.5-flash-lite" if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") == "latest" else "gemini-2.0-flash"
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
        responses = []
        user_msg = types.Content(role="user", parts=[types.Part(text=text)])
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_msg):
            if event.is_final_response():
                responses.append(event)
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

    llm_spans = [row for row in spans if row["span_attributes"]["type"] == "llm"]
    assert len(llm_spans) == 2

    follow_up_span = next(
        span for span in llm_spans if "What name did I tell you?" in _extract_text_parts(span["input"]["contents"])
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

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    user_msg = types.Content(role="user", parts=[types.Part(text="What's the weather in San Francisco?")])

    responses = []
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

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
    llm_spans = [row for row in spans if row["span_attributes"]["type"] == "llm"]
    assert len(llm_spans) >= 2, "Should have at least 2 LLM calls (tool selection and response generation)"

    # Check tool selection LLM call
    tool_selection_spans = [span for span in llm_spans if "tool_selection" in span["span_attributes"]["name"]]
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
    assert {"task", "llm", "tool"} <= span_types_by_origin, (
        f"adk-auto origin missing on task/llm/tool spans: {span_types_by_origin}"
    )

    for span in llm_spans:
        meta = span["metadata"]
        assert meta.get("provider") == "google", (
            f"Missing metadata.provider=google on {span['span_attributes']['name']}"
        )
        assert meta.get("model"), "Missing metadata.model on llm span"
        assert meta.get("tools"), "metadata.tools should be non-empty for a tool-using agent"
        tool_names = [
            fn.get("name") for tool_entry in meta["tools"] for fn in (tool_entry.get("function_declarations") or [])
        ]
        assert "get_weather" in tool_names, f"get_weather missing from metadata.tools: {tool_names}"
        assert "tools" not in span["input"].get("config", {}), "tools should not be in input.config"

    # Check response generation LLM call
    response_gen_spans = [span for span in llm_spans if "response_generation" in span["span_attributes"]["name"]]
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

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    user_msg = types.Content(role="user", parts=[types.Part(text="What's the weather in San Francisco?")])

    responses = []
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

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

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    user_msg = types.Content(role="user", parts=[types.Part(text="Tell me a long story about a lighthouse.")])

    responses = []
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

    assert len(responses) > 0
    spans = memory_logger.pop()

    # Find the LLM call span
    llm_spans = [row for row in spans if row["span_attributes"]["type"] == "llm"]
    assert len(llm_spans) > 0, "Missing LLM call span"

    llm_span = llm_spans[0]
    assert "output" in llm_span, "Missing output in LLM span"

    # Sampling config from generate_content_config is captured in input.config
    config = llm_span["input"]["config"]
    assert config["max_output_tokens"] == 50
    assert config["temperature"] == 0.7

    output = llm_span["output"]

    # When MAX_TOKENS is hit, we should still have content captured
    # The integration should merge content from earlier events if the final event lacks it
    if "finish_reason" in output and output["finish_reason"] == "MAX_TOKENS":
        # This is the MAX_TOKENS case - verify we still captured content
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

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

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

    responses = []
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

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

    # Find LLM spans and verify they also don't contain raw binary
    llm_spans = [row for row in spans if row["span_attributes"]["type"] == "llm"]
    assert len(llm_spans) > 0, "Should have LLM spans"

    for llm_span in llm_spans:
        if "input" in llm_span and "contents" in llm_span["input"]:
            llm_str = str(llm_span["input"])
            assert b"\x89PNG".hex() not in llm_str, "Raw binary data should not be in LLM span input"
            assert "89504e47" not in llm_str.lower(), "Raw binary data (hex) should not be in LLM span input"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_usage_metadata_metrics(memory_logger):
    """Google Search usage includes tool prompts and reasoning in normalized totals."""
    assert not memory_logger.pop()

    agent = Agent(
        name="usage_metadata_agent",
        model="gemini-2.5-flash",
        instruction="Use Google Search to answer the user's question accurately and concisely.",
        tools=[google_search],
    )
    app_name = "usage_metadata_app"
    user_id = "test-user"
    session_id = "test-session-usage-metadata"
    runner = await _create_runner(agent, app_name=app_name, user_id=user_id, session_id=session_id)
    user_msg = types.Content(
        role="user",
        parts=[types.Part(text="What is the current population of Tokyo, Japan? Answer in one sentence.")],
    )

    events = [event async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_msg)]
    usage_metadata = next(
        (event.usage_metadata for event in reversed(events) if getattr(event, "usage_metadata", None) is not None),
        None,
    )
    assert usage_metadata is not None
    assert usage_metadata.tool_use_prompt_token_count > 0
    assert usage_metadata.thoughts_token_count > 0

    spans = memory_logger.pop()
    llm_spans = [row for row in spans if row["span_attributes"].get("type") == "llm"]
    assert len(llm_spans) == 1
    llm_span = llm_spans[0]
    metrics = llm_span["metrics"]

    assert metrics["prompt_tokens"] == (usage_metadata.prompt_token_count + usage_metadata.tool_use_prompt_token_count)
    assert metrics["completion_tokens"] == (
        usage_metadata.candidates_token_count + usage_metadata.thoughts_token_count
    )
    assert metrics["completion_reasoning_tokens"] == usage_metadata.thoughts_token_count
    assert metrics["tokens"] == usage_metadata.total_token_count
    assert metrics["tokens"] == metrics["prompt_tokens"] + metrics["completion_tokens"]
    assert llm_span["metadata"]["usage_by_modality"]["tool_use_prompt_tokens_details"] == [
        detail.model_dump(exclude_none=True) for detail in usage_metadata.tool_use_prompt_tokens_details
    ]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_captures_metrics(memory_logger):
    """Test that token usage metrics are captured from LLM responses."""
    assert not memory_logger.pop()

    agent = Agent(
        name="metrics_agent",
        model=ADK_MODEL,
        instruction="You are a helpful assistant.",
    )

    APP_NAME = "metrics_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-metrics"

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    user_msg = types.Content(role="user", parts=[types.Part(text="Say hello in 3 words")])

    responses = []
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

    assert len(responses) > 0

    spans = memory_logger.pop()

    # Find LLM spans
    llm_spans = [row for row in spans if row["span_attributes"].get("type") == "llm"]
    assert len(llm_spans) > 0, "Should have LLM spans"

    # Verify metrics are present in at least one LLM span
    llm_span_with_metrics = None
    for llm_span in llm_spans:
        if "metrics" in llm_span and llm_span["metrics"]:
            llm_span_with_metrics = llm_span
            break

    assert llm_span_with_metrics is not None, "At least one LLM span should have metrics"

    metrics = llm_span_with_metrics["metrics"]

    # Verify core token metrics are present
    assert "prompt_tokens" in metrics, "Metrics should include prompt_tokens"
    assert "completion_tokens" in metrics, "Metrics should include completion_tokens"
    assert "tokens" in metrics, "Metrics should include total tokens"

    # Verify token counts are reasonable
    assert metrics["prompt_tokens"] > 0, "prompt_tokens should be greater than 0"
    assert metrics["completion_tokens"] > 0, "completion_tokens should be greater than 0"
    assert metrics["tokens"] > 0, "total tokens should be greater than 0"
    assert metrics["tokens"] == metrics["prompt_tokens"] + metrics["completion_tokens"], (
        "total tokens should equal prompt + completion tokens"
    )

    # Verify time to first token is captured for streaming responses
    assert "time_to_first_token" in metrics, "Metrics should include time_to_first_token"
    assert metrics["time_to_first_token"] > 0, "time_to_first_token should be greater than 0"
    assert metrics["time_to_first_token"] < 10, "time_to_first_token should be reasonable (< 10 seconds)"

    # Verify model name is captured in metadata
    metadata = llm_span_with_metrics.get("metadata", {})
    assert "model" in metadata, "Metadata should include model name"
    assert metadata["model"] == ADK_MODEL, "Model name should match the agent's model"
    assert metadata.get("provider") == "google", "Metadata should include provider=google"

    # No tools configured — _determine_llm_call_type should mark this as direct_response
    assert "direct_response" in llm_span_with_metrics["span_attributes"]["name"], (
        f"Expected direct_response call type, got {llm_span_with_metrics['span_attributes']['name']}"
    )


# _determine_llm_call_type paths are exercised through the VCR-backed
# integration tests: `test_adk_braintrust_integration` asserts the
# tool_selection / response_generation span names, and the direct_response
# path is asserted in `test_adk_captures_metrics`.


class CapitalOutput(BaseModel):
    capital: str = Field(description="The capital of the country.")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_structured_output_pydantic(memory_logger):
    """Test that structured output with Pydantic models is properly captured."""
    from unittest.mock import ANY

    assert not memory_logger.pop()

    structured_capital_agent = LlmAgent(
        name="capital_agent",
        model=ADK_MODEL,
        instruction="""You are a Capital Information Agent. Given a country, respond ONLY with a JSON object containing the capital. Format: {"capital": "capital_name"}""",
        output_schema=CapitalOutput,
        output_key="found_capital",
    )

    APP_NAME = "capital_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-structured"

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=structured_capital_agent, app_name=APP_NAME, session_service=session_service)

    user_msg = types.Content(role="user", parts=[types.Part(text="What is the capital of France?")])

    responses = []
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

    assert len(responses) > 0

    spans = memory_logger.pop()

    # Find the LLM span that has response_schema in the config
    llm_spans_with_schema = [
        span
        for span in spans
        if span["span_attributes"]["type"] == "llm"
        and "input" in span
        and "config" in span["input"]
        and span["input"]["config"].get("response_schema") is not None
    ]

    assert len(llm_spans_with_schema) > 0, "Should have at least one LLM call with response_schema"

    llm_span = llm_spans_with_schema[0]

    # Assert the complete input structure - use ANY for values we don't care about
    assert llm_span["input"] == {
        "model": ANY,
        "contents": ANY,
        "config": {
            "system_instruction": ANY,
            "response_mime_type": ANY,
            "response_schema": {
                "properties": {
                    "capital": {"description": "The capital of the country.", "title": "Capital", "type": "string"}
                },
                "required": ["capital"],
                "title": "CapitalOutput",
                "type": "object",
            },
        },
        "live_connect_config": ANY,
    }


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_input_schema_serialization(memory_logger):
    """Test that input_schema with Pydantic models is properly serialized."""
    from unittest.mock import ANY

    class UserInput(BaseModel):
        name: str = Field(description="User's name")
        age: int = Field(description="User's age", ge=0)

    assert not memory_logger.pop()

    agent = LlmAgent(
        name="input_schema_agent",
        model=ADK_MODEL,
        instruction="You are a test agent with input schema.",
        input_schema=UserInput,
    )

    APP_NAME = "input_schema_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-input-schema"

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    user_msg = types.Content(role="user", parts=[types.Part(text='{"name":"Alice","age":30}')])

    responses = []
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

    assert len(responses) > 0

    spans = memory_logger.pop()

    # Find LLM span - input_schema is on the agent, but we verify serialization doesn't break
    llm_spans = [span for span in spans if span["span_attributes"]["type"] == "llm"]

    assert len(llm_spans) > 0, "Should have at least one LLM call"

    llm_span = llm_spans[0]

    # Assert complete input structure
    assert llm_span["input"] == {
        "model": ADK_MODEL,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": '{"name":"Alice","age":30}'}],
            }
        ],
        "config": {
            "system_instruction": ANY,  # Contains agent name
        },
        "live_connect_config": ANY,
    }

    # Assert output contains expected keys (extra keys like model_version may appear in newer ADK versions)
    output = llm_span["output"]
    assert output["content"]["role"] == "model"
    assert "parts" in output["content"]
    assert "finish_reason" in output
    assert "usage_metadata" in output
    if ADK_VERSION >= (1, 15, 0) and "avg_logprobs" in output:
        assert output["avg_logprobs"] is not None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_complex_nested_schema(memory_logger):
    """Test that complex nested Pydantic schemas are properly serialized."""
    from unittest.mock import ANY

    class Address(BaseModel):
        street: str = Field(description="Street address")
        city: str = Field(description="City name")
        country: str = Field(description="Country name")

    class Person(BaseModel):
        name: str = Field(description="Person's name")
        age: int = Field(description="Person's age", ge=0, le=150)
        address: Address = Field(description="Person's address")

    assert not memory_logger.pop()

    nested_agent = LlmAgent(
        name="nested_agent",
        model=ADK_MODEL,
        instruction="Return a person with their address.",
        output_schema=Person,
        output_key="person_data",
    )

    APP_NAME = "nested_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-nested"

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=nested_agent, app_name=APP_NAME, session_service=session_service)

    user_msg = types.Content(
        role="user", parts=[types.Part(text="Give me info about Alice who lives in Paris, France.")]
    )

    responses = []
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

    assert len(responses) > 0

    spans = memory_logger.pop()

    # Find LLM span with response_schema
    llm_spans_with_schema = [
        span
        for span in spans
        if span["span_attributes"]["type"] == "llm"
        and "input" in span
        and "config" in span["input"]
        and span["input"]["config"].get("response_schema") is not None
    ]

    assert len(llm_spans_with_schema) > 0, "Should have at least one LLM call with response_schema"

    llm_span = llm_spans_with_schema[0]

    # Assert complete input structure with nested schema
    assert llm_span["input"] == {
        "model": ADK_MODEL,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Give me info about Alice who lives in Paris, France."}],
            }
        ],
        "config": {
            "system_instruction": ANY,  # Contains agent name
            "response_mime_type": "application/json",
            "response_schema": {
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
        },
        "live_connect_config": ANY,
    }

    # Assert output contains expected keys (extra keys like model_version may appear in newer ADK versions)
    output = llm_span["output"]
    assert output["content"]["role"] == "model"
    assert "parts" in output["content"]
    assert "finish_reason" in output
    assert "usage_metadata" in output
    if ADK_VERSION >= (1, 15, 0) and "avg_logprobs" in output:
        assert output["avg_logprobs"] is not None


# _capture_config's allowlisted fields are exercised through the VCR-backed
# integration tests: response_schema (`test_adk_structured_output_pydantic`,
# `test_adk_complex_nested_schema`), input_schema (`test_adk_input_schema_serialization`),
# response_json_schema (`test_adk_response_json_schema_dict`), and the sampling
# params (`test_adk_generation_config_is_logged`).


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_adk_response_json_schema_dict(memory_logger):
    """Test that Google ADK with response_json_schema (plain dict) is properly captured."""
    from unittest.mock import ANY

    # Use a plain JSON schema dict (not Pydantic)
    json_schema_dict = {
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

    assert not memory_logger.pop()

    # Pass JSON schema via generate_content_config
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=json_schema_dict,
    )

    json_schema_agent = LlmAgent(
        name="city_agent",
        model=ADK_MODEL,
        instruction="You are a City Information Agent. Provide city information.",
        generate_content_config=config,
    )

    APP_NAME = "city_app"
    USER_ID = "test-user"
    SESSION_ID = "test-session-json-dict"

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    runner = Runner(agent=json_schema_agent, app_name=APP_NAME, session_service=session_service)

    user_msg = types.Content(role="user", parts=[types.Part(text="Tell me about Tokyo")])

    responses = []
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=user_msg):
        if event.is_final_response():
            responses.append(event)

    assert len(responses) > 0

    spans = memory_logger.pop()

    # Find LLM span with response_json_schema
    llm_spans_with_schema = [
        span
        for span in spans
        if span["span_attributes"]["type"] == "llm"
        and "input" in span
        and "config" in span["input"]
        and span["input"]["config"].get("response_json_schema") is not None
    ]

    assert len(llm_spans_with_schema) > 0, "Should have at least one LLM call with response_json_schema"

    llm_span = llm_spans_with_schema[0]

    # Assert complete input structure - plain JSON schema dict should be preserved
    assert llm_span["input"] == {
        "model": ADK_MODEL,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Tell me about Tokyo"}],
            }
        ],
        "config": {
            "system_instruction": ANY,  # Contains agent name
            "response_mime_type": "application/json",
            "response_json_schema": {
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
            },
        },
        "live_connect_config": ANY,
    }

    # Assert output contains expected keys (extra keys like model_version may appear in newer ADK versions)
    output = llm_span["output"]
    assert output["content"]["role"] == "model"
    assert "parts" in output["content"]
    assert "finish_reason" in output
    assert "usage_metadata" in output
    if ADK_VERSION >= (1, 15, 0) and "avg_logprobs" in output:
        assert output["avg_logprobs"] is not None


class TestAutoInstrumentADK:
    """Tests for auto_instrument() with Google ADK."""

    def test_auto_instrument_adk(self):
        """Test auto_instrument patches ADK classes and is idempotent."""
        from braintrust.integrations.test_utils import verify_autoinstrument_script

        verify_autoinstrument_script("test_auto_adk.py")
