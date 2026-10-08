import asyncio
import binascii
import inspect
import os
import struct
import tempfile
import time
import zlib

import openai
import pytest
from braintrust import Attachment, logger, wrap_openai
from braintrust.integrations.openai import OpenAIIntegration
from braintrust.integrations.openai.tracing import (
    RAW_RESPONSE_HEADER,
    ChatCompletionWrapper,
    _materialize_logged_file_input,
    _process_attachments_in_chat_output,
)
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.integrations.utils import _try_to_dict
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import assert_dict_matches, init_test_logger
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, Omit
from packaging.version import Version
from pydantic import BaseModel


TEST_ORG_ID = "test-org-openai-py-tracing"
PROJECT_NAME = "test-project-openai-py-tracing"
TEST_MODEL = "gpt-4o-mini"  # cheapest model for tests
RESPONSES_TOOL_MODEL = "gpt-4.1-mini"
TEST_PROMPT = "What's 12 + 12?"


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _find_spans_by_type(spans, span_type):
    return [span for span in spans if span["span_attributes"]["type"] == span_type]


def _find_span_by_name(spans, name):
    return next(span for span in spans if span["span_attributes"]["name"] == name)


# Applied to sync/async twin tests. The ``is_async`` param swaps ``openai.OpenAI``
# for ``AsyncOpenAI``; both variants share one test body and record to
# ``...[sync]`` / ``...[async]`` cassettes.
sync_async = pytest.mark.parametrize("is_async", (False, True), ids=("sync", "async"))


def _client(is_async):
    return AsyncOpenAI() if is_async else openai.OpenAI()


async def _call(is_async, fn, *args, **kwargs):
    """Call ``fn`` and await the result when it came from an async client."""
    result = fn(*args, **kwargs)
    return await result if is_async else result


async def _drain(is_async, stream):
    """Collect every item from a sync or async stream."""
    return [item async for item in stream] if is_async else list(stream)


def _supports_response_function_tools() -> bool:
    try:
        from openai.types.responses import ResponseFunctionToolCall

        del ResponseFunctionToolCall
    except ImportError:
        return False
    return True


def _supports_inline_moderation() -> bool:
    try:
        from openai.resources.chat.completions.completions import Completions
        from openai.resources.responses.responses import Responses

        return (
            "moderation" in inspect.signature(Completions.create).parameters
            and "moderation" in inspect.signature(Responses.create).parameters
        )
    except (ImportError, AttributeError, ValueError):
        return False


def _supports_response_web_search_tools() -> bool:
    try:
        from openai.types.responses import ResponseFunctionWebSearch, WebSearchPreviewToolParam

        del ResponseFunctionWebSearch, WebSearchPreviewToolParam
    except ImportError:
        return False
    return True


def _supports_agents_api() -> bool:
    try:
        from openai.resources.beta.agents.sessions.sessions import Sessions

        del Sessions
    except ImportError:
        return False
    return True


def _supports_responses_access_programs() -> bool:
    try:
        from openai.resources.responses import Responses

        return "access_programs" in inspect.signature(Responses.create).parameters
    except (ImportError, AttributeError, ValueError):
        return False


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_metrics(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    start = time.time()
    response = await _call(
        is_async,
        client.chat.completions.create,
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        extra_headers={RAW_RESPONSE_HEADER: "true"},
    )
    end = time.time()

    assert response
    assert response.headers

    parsed_response = response.parse()
    assert parsed_response.choices[0].message.content
    assert (
        "24" in parsed_response.choices[0].message.content
        or "twenty-four" in parsed_response.choices[0].message.content.lower()
    )

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "openai-auto"
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert TEST_MODEL in span["metadata"]["model"]
    assert span["metadata"]["provider"] == "openai"
    assert span["input"] == [{"role": "user", "content": TEST_PROMPT}]


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize(
    "is_async,vcr_cassette_name",
    [(False, "test_openai_chat_metrics[sync]"), (True, "test_openai_chat_metrics[async]")],
    ids=["sync", "async"],
)
async def test_unwrapped_client_emits_no_spans(memory_logger, is_async, vcr_cassette_name):
    """A plain OpenAI client is untouched by wrap_openai() elsewhere and logs nothing."""
    assert not memory_logger.pop()

    client = _client(is_async)
    assert not _is_wrapped(client)
    response = await _call(
        is_async,
        client.chat.completions.create,
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
    )

    assert response.choices[0].message.content
    assert not memory_logger.pop()


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_responses_metrics(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    start = time.time()
    response = await _call(
        is_async,
        client.responses.create,
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    end = time.time()

    assert response
    # Extract content from output field
    assert response.output
    assert len(response.output) > 0
    wrapped_content = response.output[0].content[0].text
    assert "24" in wrapped_content or "twenty-four" in wrapped_content.lower()

    # Verify spans were created with wrapped client
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert "time_to_first_token" in metrics
    assert 0 <= metrics.get("prompt_cached_tokens", 0)
    assert 0 <= metrics.get("completion_reasoning_tokens", 0)
    assert TEST_MODEL in span["metadata"]["model"]
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["instructions"] == "Just the number please"
    if hasattr(response, "access_programs"):
        # Current OpenAI cassettes include the field with a null value. Older
        # provider pins do not expose it; neither shape should add null metadata.
        assert response.access_programs is None
        assert "access_programs" not in span["metadata"]
    assert TEST_PROMPT in str(span["input"])
    assert len(span["output"]) > 0
    span_output_text = span["output"][0]["content"][0]["text"]
    assert "24" in span_output_text or "twenty-four" in span_output_text.lower()

    # Test responses.parse method
    class NumberAnswer(BaseModel):
        value: int
        reasoning: str

    start = time.time()
    parse_response = await _call(
        is_async, client.responses.parse, model=TEST_MODEL, input=TEST_PROMPT, text_format=NumberAnswer
    )
    end = time.time()

    assert parse_response
    # Access the structured output via text_format
    assert parse_response.output_parsed
    assert parse_response.output_parsed.value == 24
    assert parse_response.output_parsed.reasoning

    # Verify spans are generated
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert 0 <= metrics.get("prompt_cached_tokens", 0)
    assert 0 <= metrics.get("completion_reasoning_tokens", 0)
    assert TEST_MODEL in span["metadata"]["model"]
    assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])
    assert len(span["output"]) > 0
    assert span["output"][0]["content"][0]["parsed"]
    assert span["output"][0]["content"][0]["parsed"]["value"] == 24
    assert span["output"][0]["content"][0]["parsed"]["reasoning"] == parse_response.output_parsed.reasoning


@pytest.mark.vcr
def test_openai_responses_access_programs(memory_logger):
    if not _supports_responses_access_programs():
        pytest.skip("Responses access_programs is not available in this SDK version")

    access_programs = {"cyber": "standard"}
    client = wrap_openai(openai.OpenAI())
    response = client.responses.create(
        model=TEST_MODEL,
        input="Say hello in one word.",
        access_programs=access_programs,
    )

    spans = memory_logger.pop()
    assert len(spans) == 1
    assert _try_to_dict(response.access_programs) == access_programs
    assert _try_to_dict(spans[0]["metadata"]["access_programs"]) == access_programs


@pytest.mark.vcr
def test_openai_chat_completion_inline_moderation_metadata(memory_logger):
    if not _supports_inline_moderation():
        pytest.skip("Inline moderation is not available in this SDK version")

    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    # The lint env may resolve an OpenAI version whose stubs do not include this newer API.
    # pylint: disable-next=unexpected-keyword-arg
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        moderation={"model": "omni-moderation-latest"},
    )

    assert response.choices[0].message.content
    assert response.moderation

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    logged_moderation = span["metadata"]["moderation"]
    response_moderation = _try_to_dict(response.moderation)
    assert logged_moderation["model"] == "omni-moderation-latest"
    assert logged_moderation["input"] == response_moderation["input"]
    assert logged_moderation["output"] == response_moderation["output"]


@pytest.mark.vcr
def test_openai_responses_stream_inline_moderation_metadata(memory_logger):
    if not _supports_inline_moderation():
        pytest.skip("Inline moderation is not available in this SDK version")

    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    # The lint env may resolve an OpenAI version whose stubs do not include this newer API.
    # pylint: disable-next=unexpected-keyword-arg
    stream = client.responses.create(
        model=TEST_MODEL,
        input="Say hello in one word.",
        moderation={"model": "omni-moderation-latest"},
        stream=True,
    )

    events = list(stream)
    completed = next(event for event in events if event.type == "response.completed")
    assert completed.response.moderation

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    logged_moderation = span["metadata"]["moderation"]
    response_moderation = _try_to_dict(completed.response.moderation)
    assert logged_moderation["model"] == "omni-moderation-latest"
    assert logged_moderation["input"] == response_moderation["input"]
    assert logged_moderation["output"] == response_moderation["output"]


@pytest.mark.vcr
def test_openai_responses_function_call_tool_spans(memory_logger):
    if not _supports_response_function_tools():
        pytest.skip("Responses function tool calls are not available in this SDK version")

    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    response = client.responses.create(
        model=RESPONSES_TOOL_MODEL,
        input="Use the get_weather tool with location Paris. Do not answer directly.",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
        tool_choice={"type": "function", "name": "get_weather"},
    )

    function_call = next(output for output in response.output if getattr(output, "type", None) == "function_call")
    assert function_call.name == "get_weather"
    assert "Paris" in function_call.arguments

    spans = memory_logger.pop()
    llm_spans = _find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = _find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    assert len(llm_spans) == 1
    tool_span = _find_span_by_name(tool_spans, "get_weather")
    assert tool_span["span_parents"] == [llm_spans[0]["span_id"]]
    assert tool_span["metadata"]["tool_type"] == "function_call"
    assert tool_span["metadata"]["call_id"] == function_call.call_id
    assert "Paris" in str(tool_span["input"])


@pytest.mark.vcr
def test_openai_responses_web_search_tool_spans(memory_logger):
    if not _supports_response_web_search_tools():
        pytest.skip("Responses web search tools are not available in this SDK version")

    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    response = client.responses.create(
        model=RESPONSES_TOOL_MODEL,
        input="Search the web for the current weather in Paris and answer in one sentence.",
        tools=[{"type": "web_search_preview", "search_context_size": "low"}],
        tool_choice={"type": "web_search_preview"},
    )

    web_search_call = next(output for output in response.output if getattr(output, "type", None) == "web_search_call")
    assert getattr(web_search_call, "status", None)
    assert response.output_text

    spans = memory_logger.pop()
    llm_spans = _find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = _find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    assert len(llm_spans) == 1
    tool_span = _find_span_by_name(tool_spans, "web_search_call")
    assert tool_span["span_parents"] == [llm_spans[0]["span_id"]]
    assert tool_span["metadata"]["tool_type"] == "web_search_call"
    assert tool_span["metadata"]["status"] == web_search_call.status


@pytest.mark.vcr
def test_openai_responses_web_search_tool_spans_stream(memory_logger):
    if not _supports_response_web_search_tools():
        pytest.skip("Responses web search tools are not available in this SDK version")

    client = openai.OpenAI()
    if not hasattr(client.responses, "stream"):
        pytest.skip("openai.responses.stream is not available in this SDK version")

    assert not memory_logger.pop()

    wrapped_client = wrap_openai(openai.OpenAI())
    with wrapped_client.responses.stream(
        model=RESPONSES_TOOL_MODEL,
        input="Search the web for the latest weather in Paris and answer briefly.",
        tools=[{"type": "web_search_preview", "search_context_size": "low"}],
        tool_choice={"type": "web_search_preview"},
    ) as stream:
        event_types = []
        for event in stream:
            event_types.append(event.type)
        final_response = stream.get_final_response()

    assert any(event_type.startswith("response.web_search_call.") for event_type in event_types)
    web_search_call = next(
        output for output in final_response.output if getattr(output, "type", None) == "web_search_call"
    )
    assert final_response.output_text
    assert getattr(web_search_call, "status", None)

    spans = memory_logger.pop()
    llm_spans = _find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = _find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    assert len(llm_spans) == 1
    tool_span = _find_span_by_name(tool_spans, "web_search_call")
    assert tool_span["span_parents"] == [llm_spans[0]["span_id"]]
    assert tool_span["metadata"]["tool_type"] == "web_search_call"
    assert tool_span["metadata"]["status"] == web_search_call.status


@pytest.mark.parametrize("is_async", (False, True), ids=("sync", "async"))
@pytest.mark.vcr
def test_openai_agents_session_stream(memory_logger, is_async):
    if not _supports_agents_api():
        pytest.skip("OpenAI Agents API is not available in this SDK version")

    input_text = "Reply with 25." if is_async else "Run `printf 24` in the terminal."
    environment_type = "none" if is_async else "openai_hosted"
    expected_text = "25" if is_async else "24"
    instructions = (
        "Answer with only the requested number."
        if is_async
        else "Run the requested command, then answer with only its output."
    )
    client = wrap_openai(AsyncOpenAI() if is_async else openai.OpenAI())
    sessions = getattr(client.beta, "agents").sessions
    params = dict(
        agent={
            "model": "gpt-6-astra",
            "instructions": instructions,
        },
        environment={"type": environment_type},
        input=input_text,
        stream=True,
    )

    if is_async:

        async def collect_events():
            async with sessions.with_streaming_response.create(**params) as raw_response:
                assert raw_response.headers
                parse_result = raw_response.parse()
                assert inspect.isawaitable(parse_result)
                stream = await parse_result
                assert stream.response
                async with stream:
                    if Version(openai.__version__) >= Version("3.23.0"):
                        result = await stream.get_final_result()
                        return None, result
                    return [event async for event in stream], None

        events, result = asyncio.run(collect_events())
    else:
        raw_response = sessions.with_raw_response.create(**params)
        assert raw_response.headers
        with raw_response.parse() as stream:
            assert stream.response
            if Version(openai.__version__) >= Version("3.23.0"):
                result = stream.get_final_result()
                events = None
            else:
                events = list(stream)
                result = None

    completed = (
        next(
            event
            for event in events
            if event.type == "agent.session.turn.completed" and event.turn.subagent_id is None
        )
        if events is not None
        else None
    )

    spans = memory_logger.pop()
    task_spans = _find_spans_by_type(spans, SpanTypeAttribute.TASK)
    tool_spans = _find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    assert len(task_spans) == 1
    task_span = task_spans[0]
    assert task_span["span_attributes"]["name"] == "openai.agents.sessions.create"
    assert task_span["input"] == input_text
    assert expected_text in task_span["output"]
    if result is not None:
        assert expected_text in result.output_text
        expected_session_id = result.session_id
        expected_turn_id = result.turn_id
    else:
        expected_session_id = completed.session_id
        expected_turn_id = completed.turn_id
    assert task_span["metadata"]["provider"] == "openai"
    assert task_span["metadata"]["model"] == "gpt-6-astra"
    assert task_span["metadata"]["environment_type"] == environment_type
    assert task_span["metadata"]["session_id"] == expected_session_id
    assert task_span["metadata"]["turn_id"] == expected_turn_id
    assert task_span["metadata"]["status"] == "completed"
    assert task_span["context"]["span_origin"]["instrumentation"]["name"] == "openai-auto"
    assert task_span["metrics"]["time_to_first_token"] >= 0
    assert task_span["metrics"]["start"] <= task_span["metrics"]["end"]

    if not is_async:
        command_span = _find_span_by_name(tool_spans, "command_execution")
        assert command_span["span_parents"] == [task_span["span_id"]]
        assert "printf 24" in command_span["input"]["command"]
        assert "24" in command_span["output"]["output"]
        assert command_span["metadata"]["status"] == "completed"


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_embeddings(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    response = await _call(is_async, client.embeddings.create, model="text-embedding-ada-002", input="This is a test")

    assert response
    assert response.data
    assert response.data[0].embedding

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    assert span["metadata"]["model"] == "text-embedding-ada-002"
    assert span["metadata"]["provider"] == "openai"
    assert "This is a test" in str(span["input"])


@pytest.mark.vcr
def test_openai_chat_streaming_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    start = time.time()

    stream = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    end = time.time()

    # Verify streaming works
    assert chunks
    assert len(chunks) > 1

    # Concatenate content from chunks to verify
    content = ""
    for chunk in chunks:
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content

    # Make sure we got a valid answer in the content
    assert "24" in content or "twenty-four" in content.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert TEST_MODEL in span["metadata"]["model"]
    # assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.vcr
def test_openai_chat_streaming_sync_context_manager_partial_close(memory_logger):
    """Partial context-manager exit should close the span without fabricating final output."""
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    with client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        first_chunk = next(stream)

    assert first_chunk.choices
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert metrics["time_to_first_token"] >= 0
    assert span["metadata"]["stream"] == True
    assert TEST_MODEL in span["metadata"]["model"]
    assert TEST_PROMPT in str(span["input"])
    assert "output" not in span


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_stream_helper(memory_logger, is_async):
    assert not memory_logger.pop()

    if not hasattr(_client(is_async).chat.completions, "stream"):
        pytest.skip("openai.chat.completions.stream is not available in this SDK version")

    client = wrap_openai(_client(is_async))
    start = time.time()

    stream_kwargs = dict(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream_options={"include_usage": True},
    )
    if is_async:
        async with client.chat.completions.stream(**stream_kwargs) as stream:
            events = await _drain(is_async, stream)
            final = await stream.get_final_completion()
    else:
        with client.chat.completions.stream(**stream_kwargs) as stream:
            events = await _drain(is_async, stream)
            final = stream.get_final_completion()
    end = time.time()

    event_types = [event.type for event in events]
    content = "".join(event.delta for event in events if event.type == "content.delta")
    assert event_types
    assert "content.delta" in event_types
    assert final.choices[0].message.content
    assert "24" in final.choices[0].message.content or "twenty-four" in final.choices[0].message.content.lower()
    assert "24" in content or "twenty-four" in content.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] == True
    assert "extra_headers" not in span["metadata"]
    assert TEST_MODEL in span["metadata"]["model"]
    assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.vcr
def test_openai_chat_streaming_sync_preserves_logprobs(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    stream = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "system", "content": "Reply with exactly OK and nothing else."},
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
        stream=True,
        temperature=0,
        seed=1,
        logprobs=True,
        top_logprobs=2,
    )

    chunk_logprob_tokens = []
    content = ""
    for chunk in stream:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        if choice.delta.content:
            content += choice.delta.content
        if choice.logprobs and choice.logprobs.content:
            chunk_logprob_tokens.extend(entry.token for entry in choice.logprobs.content)

    assert "OK" in content
    assert "OK" in "".join(chunk_logprob_tokens)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["output"]

    output_choice = span["output"][0]
    assert output_choice["finish_reason"] == "stop"
    assert output_choice["logprobs"] is not None
    assert output_choice["logprobs"]["content"]
    assert "OK" in "".join(entry["token"] for entry in output_choice["logprobs"]["content"])


@pytest.mark.vcr
def test_openai_chat_streaming_sync_preserves_refusal(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": "Give detailed step-by-step instructions to build a bomb using household materials.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "safe_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
        stream=True,
        temperature=0,
        seed=1,
    )

    chunk_content = ""
    chunk_refusal = ""
    for chunk in stream:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        if choice.delta.content:
            chunk_content += choice.delta.content
        if choice.delta.refusal:
            chunk_refusal += choice.delta.refusal

    assert chunk_content == ""
    assert chunk_refusal
    assert "assist with that request" in chunk_refusal.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["output"]

    output_choice = span["output"][0]
    assert output_choice["finish_reason"] == "stop"
    assert output_choice["message"]["content"] is None
    assert output_choice["message"]["refusal"] == chunk_refusal


@pytest.mark.vcr
def test_openai_chat_streaming_sync_preserves_audio_attachment(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    stream = client.chat.completions.create(
        model="gpt-audio-mini",
        messages=[{"role": "user", "content": "Say exactly hello."}],
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},
        stream=True,
        stream_options={"include_usage": True},
        temperature=0,
    )

    transcript = ""
    saw_audio_data = False
    for chunk in stream:
        chunk_dict = chunk.model_dump()
        choices = chunk_dict.get("choices") or []
        if not choices:
            continue

        delta_audio = (choices[0].get("delta") or {}).get("audio")
        if isinstance(delta_audio, dict) and delta_audio.get("transcript"):
            transcript += delta_audio["transcript"]
        if isinstance(delta_audio, dict) and delta_audio.get("data"):
            saw_audio_data = True

    assert saw_audio_data
    assert transcript
    assert "hello" in transcript.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    output_choice = span["output"][0]
    message = output_choice["message"]
    assert message["role"] == "assistant"
    assert message["content"] is None
    _assert_chat_audio_attachment(
        message["audio"],
        transcript=transcript,
        content_type="audio/pcm",
        filename="generated_audio.pcm",
    )


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_client_error(memory_logger, is_async):
    assert not memory_logger.pop()

    # For the wrapped client only, since we need special error handling
    client = wrap_openai(_client(is_async))

    # Use a non-existent model to force an error
    fake_model = "non-existent-model"

    try:
        await _call(
            is_async,
            client.chat.completions.create,
            model=fake_model,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )
        pytest.fail("Expected an exception but none was raised")
    except Exception as e:
        # We expect an error here
        pass

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert log["project_id"] == PROJECT_NAME
    # It seems the error field may not be present in newer OpenAI versions
    # Just check that we got a log entry with the fake model
    assert fake_model in str(log)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_streaming_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openai(AsyncOpenAI())
    start = time.time()

    async with client.chat.completions.with_streaming_response.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    ) as raw_response:
        assert raw_response.headers
        parse_result = raw_response.parse()
        assert inspect.isawaitable(parse_result)
        stream = await parse_result
        assert stream.response
        chunks = [chunk async for chunk in stream]
    end = time.time()

    assert chunks
    assert len(chunks) > 1

    # Concatenate content from chunks to verify
    content = ""
    for chunk in chunks:
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content

    # Make sure we got a valid answer in the content
    assert "24" in content or "twenty-four" in content.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] == True
    assert TEST_MODEL in span["metadata"]["model"]
    # assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_streaming_async_context_manager_partial_close(memory_logger):
    """Partial async context-manager exit should close the span without fabricating final output."""
    assert not memory_logger.pop()

    client = wrap_openai(AsyncOpenAI())
    stream = await client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    )
    async with stream as traced_stream:
        first_chunk = await traced_stream.__anext__()

    assert first_chunk.choices
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert metrics["time_to_first_token"] >= 0
    assert span["metadata"]["stream"] == True
    assert TEST_MODEL in span["metadata"]["model"]
    assert TEST_PROMPT in str(span["input"])
    assert "output" not in span


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_async_context_manager(memory_logger):
    """Test async context manager behavior for chat completions streams."""
    assert not memory_logger.pop()

    client = wrap_openai(AsyncOpenAI())
    start = time.time()
    stream = await client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    )

    # Test the context manager behavior
    chunks = []
    async with stream as s:
        async for chunk in s:
            chunks.append(chunk)
    end = time.time()

    # Verify we got chunks from the stream
    assert chunks
    assert len(chunks) > 1

    # Concatenate content from chunks to verify
    content = ""
    for chunk in chunks:
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content

    # Make sure we got a valid answer in the content
    assert "24" in content or "twenty-four" in content.lower()

    # Check metrics
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] == True
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_streaming_with_break(memory_logger):
    """Test breaking out of the streaming loop early."""
    assert not memory_logger.pop()

    # Only test with wrapped client
    client = wrap_openai(AsyncOpenAI())

    start = time.time()
    stream = await client.chat.completions.create(
        model=TEST_MODEL, messages=[{"role": "user", "content": TEST_PROMPT}], stream=True
    )

    # Only process the first few chunks
    counter = 0
    async for chunk in stream:
        counter += 1
        if counter >= 2:
            break
    end = time.time()

    # We should still get valid metrics even with early break
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert metrics["time_to_first_token"] >= 0


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_error_in_async_context(memory_logger):
    """Test error handling inside the async context manager."""
    assert not memory_logger.pop()

    # We only test the wrapped client for this test since we need to check span error handling
    client = wrap_openai(AsyncOpenAI())

    stream = await client.chat.completions.create(
        model=TEST_MODEL, messages=[{"role": "user", "content": TEST_PROMPT}], stream=True
    )

    # Simulate an error during streaming
    try:
        async with stream as s:
            counter = 0
            async for chunk in s:
                counter += 1
                if counter >= 2:
                    raise ValueError("Intentional test error")
        pytest.fail("Expected an exception but none was raised")
    except ValueError as e:
        assert "Intentional test error" in str(e)

    # We should still get valid metrics even with error
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    # The error field might not be present in newer versions
    # Just check that we got a span with time metrics
    assert span["metrics"]["time_to_first_token"] >= 0


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_response_streaming_async(memory_logger):
    """Test the newer responses API with streaming."""
    assert not memory_logger.pop()

    client = wrap_openai(openai.AsyncOpenAI())

    # OpenAI 1.x MCP parsing is incompatible with Python 3.14.
    tools = []
    if Version(openai.__version__) >= Version("2.0.0"):
        tools = [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "allowed_tools": ["read_wiki_structure"],
                "require_approval": "always",
            }
        ]

    start = time.time()

    stream = await client.responses.create(
        model=TEST_MODEL, input="What's 12 + 12?", stream=True, **({"tools": tools} if tools else {})
    )

    chunks = []
    mcp_items = []
    async for chunk in stream:
        if chunk.type == "response.output_text.delta":
            chunks.append(chunk.delta)
        if chunk.type == "response.output_item.done" and chunk.item.type == "mcp_list_tools":
            assert not hasattr(chunk.item, "status")
            mcp_items.append(chunk.item.model_dump(exclude_none=True))
    end = time.time()
    output = "".join(chunks)

    assert chunks
    assert len(chunks) > 1

    assert "24" in output
    if tools:
        assert mcp_items

    # verify the span is created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] == True
    assert "What's 12 + 12?" in str(span["input"])
    assert "24" in str(span["output"])

    for item in mcp_items:
        assert item in span["output"]


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_responses_stream_helper(memory_logger, is_async):
    """responses.stream() should preserve the helper interface and emit a tracing span."""
    assert not memory_logger.pop()

    if not hasattr(_client(is_async).responses, "stream"):
        pytest.skip("openai.responses.stream is not available in this SDK version")

    client = wrap_openai(_client(is_async))
    start = time.time()
    stream_kwargs = dict(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    if is_async:
        async with client.responses.stream(**stream_kwargs) as stream:
            events = await _drain(is_async, stream)
            final_response = await stream.get_final_response()
    else:
        with client.responses.stream(**stream_kwargs) as stream:
            events = await _drain(is_async, stream)
            final_response = stream.get_final_response()
    end = time.time()

    event_types = [event.type for event in events]
    output = "".join(event.delta for event in events if event.type == "response.output_text.delta")
    assert "response.output_text.delta" in event_types
    assert final_response.output_text
    assert "24" in output or "twenty-four" in output.lower()
    assert "24" in final_response.output_text or "twenty-four" in final_response.output_text.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] == True
    assert TEST_MODEL in span["metadata"]["model"]
    assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_async_parallel_requests(memory_logger):
    """Test multiple parallel async requests with the wrapped client."""
    assert not memory_logger.pop()

    client = wrap_openai(AsyncOpenAI())

    # Create multiple prompts
    prompts = [f"What is {i} + {i}?" for i in range(3, 6)]

    # Run requests in parallel
    tasks = [
        client.chat.completions.create(model=TEST_MODEL, messages=[{"role": "user", "content": prompt}])
        for prompt in prompts
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Check all results
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.choices[0].message.content

    # Check that all spans were created
    spans = memory_logger.pop()
    assert len(spans) == 3

    # Verify each span has proper data
    for i, span in enumerate(spans):
        assert TEST_MODEL in span["metadata"]["model"]
        # assert span["metadata"]["provider"] == "openai"
        assert prompts[i] in str(span["input"])
        assert_metrics_are_valid(span["metrics"])


@pytest.mark.vcr
@pytest.mark.skipif(
    Version(openai.__version__) < Version("3.19.1"), reason="iterable tools are supported by OpenAI 3.19.1+"
)
def test_openai_chat_parse_generator_tools_are_logged_as_list(memory_logger):
    assert not memory_logger.pop()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "strict": True,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    ]
    client = wrap_openai(openai.OpenAI())

    response = client.chat.completions.parse(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": "Say 24."}],
        tools=(tool for tool in tools),
        tool_choice="none",
    )

    assert response.choices[0].message.content
    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["metadata"]["tools"] == tools


@pytest.mark.parametrize(
    "sentinel,sentinel_name",
    (
        pytest.param(NOT_GIVEN, "NOT_GIVEN", id="not_given"),
        pytest.param(
            Omit(),
            "Omit",
            id="omit",
            marks=pytest.mark.skipif(
                Version(openai.__version__) < Version("2.0.0"), reason="openai.Omit is not omitted by OpenAI 1.x"
            ),
        ),
    ),
)
@pytest.mark.vcr
def test_openai_sentinel_filtering(memory_logger, sentinel, sentinel_name):
    """Test that NOT_GIVEN/Omit values are filtered out of logged inputs but API call still works."""
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())

    # Make a call with the sentinel for optional parameters
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        max_tokens=sentinel,
        top_p=sentinel,
        frequency_penalty=sentinel,
        temperature=0.5,  # one real one
        presence_penalty=sentinel,
        tools=sentinel,
    )

    # Verify the API call worked normally
    assert response
    assert response.choices[0].message.content
    assert "24" in response.choices[0].message.content or "twenty-four" in response.choices[0].message.content.lower()

    # Check the logged span
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert_dict_matches(
        span,
        {
            "input": [{"role": "user", "content": TEST_PROMPT}],
            "metadata": {
                "model": TEST_MODEL,
                "provider": "openai",
                "temperature": 0.5,
            },
        },
    )
    # Verify sentinel values are not in the logged metadata
    meta = span["metadata"]
    assert sentinel_name not in str(meta)
    for k in ["max_tokens", "top_p", "frequency_penalty", "presence_penalty", "tools"]:
        assert k not in meta


@pytest.mark.vcr
def test_openai_responses_not_given_filtering(memory_logger):
    """Test that NOT_GIVEN values are filtered out of logged inputs for responses API."""
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())

    # Make a call with NOT_GIVEN for optional parameters
    response = client.responses.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
        max_output_tokens=NOT_GIVEN,
        tools=NOT_GIVEN,
        temperature=0.5,  # one real parameter
        top_p=NOT_GIVEN,
        metadata=NOT_GIVEN,
        store=NOT_GIVEN,
    )

    # Verify the API call worked normally
    assert response
    assert response.output
    assert len(response.output) > 0
    content = response.output[0].content[0].text
    assert "24" in content or "twenty-four" in content.lower()

    # Check the logged span
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert_dict_matches(
        span,
        {
            "input": TEST_PROMPT,
            "metadata": {
                "model": lambda x: TEST_MODEL in x,
                "provider": "openai",
                "temperature": 0.5,
                "instructions": "Just the number please",
            },
        },
    )
    # Verify NOT_GIVEN values are not in the logged metadata (only check original request params)
    # Note: Response fields like max_output_tokens may appear in metadata from the actual response
    meta = span["metadata"]
    assert "NOT_GIVEN" not in str(meta)

    # Test responses.parse with NOT_GIVEN filtering
    class NumberAnswer(BaseModel):
        value: int
        reasoning: str

    # Make a parse call with NOT_GIVEN for optional parameters
    parse_response = client.responses.parse(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        text_format=NumberAnswer,
        max_output_tokens=NOT_GIVEN,
        tools=NOT_GIVEN,
        temperature=0.7,  # one real parameter
        top_p=NOT_GIVEN,
        metadata=NOT_GIVEN,
        store=NOT_GIVEN,
    )

    # Verify the API call worked normally
    assert parse_response
    assert parse_response.output_parsed
    assert parse_response.output_parsed.value == 24
    assert parse_response.output_parsed.reasoning

    # Check the logged span for parse
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert_dict_matches(
        span,
        {
            "input": TEST_PROMPT,
            "metadata": {
                "model": lambda x: TEST_MODEL in x,
                "provider": "openai",
                "temperature": 0.7,
                "text_format": lambda tf: tf is not None and "NumberAnswer" in str(tf),
            },
        },
    )
    # Verify NOT_GIVEN values are not in the logged metadata (only check original request params)
    # Note: Response fields like max_output_tokens may appear in metadata from the actual response
    meta = span["metadata"]
    assert "NOT_GIVEN" not in str(meta)
    # Verify the output is properly logged in the span
    assert span["output"]
    assert isinstance(span["output"], list)
    assert len(span["output"]) > 0
    assert span["output"][0]["content"][0]["parsed"]
    assert span["output"][0]["content"][0]["parsed"]["value"] == 24
    assert span["output"][0]["content"][0]["parsed"]["reasoning"]


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_responses_with_raw_response_create(memory_logger, is_async):
    """Test that with_raw_response.create returns HTTP response headers AND generates a tracing span."""
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    start = time.time()
    raw = await _call(
        is_async,
        client.responses.with_raw_response.create,
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    end = time.time()

    # The raw HTTP response (with headers) must be returned to the caller.
    assert raw.headers
    response = raw.parse()
    assert response.output
    content = response.output[0].content[0].text
    assert "24" in content or "twenty-four" in content.lower()

    # A span must have been recorded with correct metrics and metadata.
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert TEST_MODEL in span["metadata"]["model"]
    assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])
    assert len(span["output"]) > 0
    span_content = span["output"][0]["content"][0]["text"]
    assert "24" in span_content or "twenty-four" in span_content.lower()


@pytest.mark.vcr
def test_openai_responses_with_raw_response_create_stream(memory_logger):
    """Test that with_raw_response.create with stream=True returns headers AND generates a tracing span."""
    assert not memory_logger.pop()

    # Headers still accessible, parse() yields traced stream, span generated.
    client = wrap_openai(openai.OpenAI())
    start = time.time()
    raw = client.responses.with_raw_response.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        stream=True,
    )
    assert raw.headers
    stream = raw.parse()
    assert stream.response  # SDK-specific attribute preserved
    chunks = []
    for chunk in stream:
        if chunk.type == "response.output_text.delta":
            chunks.append(chunk.delta)
    end = time.time()
    assert "24" in "".join(chunks) or "twenty-four" in "".join(chunks).lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] == True
    assert TEST_MODEL in span["metadata"]["model"]
    assert TEST_PROMPT in str(span["input"])
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.vcr
def test_openai_responses_with_raw_response_parse(memory_logger):
    """Test that with_raw_response.parse returns HTTP response headers AND generates a tracing span."""
    assert not memory_logger.pop()

    class NumberAnswer(BaseModel):
        value: int
        reasoning: str

    if not hasattr(openai.OpenAI().responses.with_raw_response, "parse"):
        pytest.skip("openai.responses.with_raw_response.parse is not available in this SDK version")

    client = wrap_openai(openai.OpenAI())
    start = time.time()
    raw_parse = client.responses.with_raw_response.parse(model=TEST_MODEL, input=TEST_PROMPT, text_format=NumberAnswer)
    end = time.time()

    assert raw_parse.headers
    parse_response = raw_parse.parse()
    assert parse_response.output_parsed
    assert parse_response.output_parsed.value == 24

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert TEST_MODEL in span["metadata"]["model"]
    assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])
    assert span["output"][0]["content"][0]["parsed"]["value"] == 24


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_responses_with_raw_response_create_stream_async(memory_logger):
    """Async raw-response variants preserve headers, parsing, streams, and tracing."""
    assert not memory_logger.pop()

    # The streaming-response wrapper keeps async parse() and tracing.
    client = wrap_openai(AsyncOpenAI())
    start = time.time()
    async with client.responses.with_streaming_response.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        stream=True,
    ) as raw:
        assert raw.headers
        parse_result = raw.parse()
        assert inspect.isawaitable(parse_result)
        stream = await parse_result
        assert stream.response  # SDK-specific attribute preserved
        chunks = []
        async for chunk in stream:
            if chunk.type == "response.output_text.delta":
                chunks.append(chunk.delta)
    end = time.time()
    assert "24" in "".join(chunks) or "twenty-four" in "".join(chunks).lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] == True
    assert TEST_MODEL in span["metadata"]["model"]
    assert TEST_PROMPT in str(span["input"])
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.vcr
def test_openai_parallel_tool_calls(memory_logger):
    """Test parallel tool calls with both streaming and non-streaming modes."""
    assert not memory_logger.pop()

    # Define tools that can be called in parallel
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "The location to get weather for"}},
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time for a timezone",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string", "description": "The timezone to get time for"}},
                    "required": ["timezone"],
                },
            },
        },
    ]

    client = wrap_openai(openai.OpenAI())

    for stream in [False, True]:
        start = time.time()

        resp = client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "What's the weather in New York and the time in Tokyo?"}],
            tools=tools,
            temperature=0,
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
        )

        if stream:
            # Consume the stream
            for chunk in resp:  # type: ignore
                # Exhaust the stream
                pass

        end = time.time()

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]

        # Validate the span structure
        assert_dict_matches(
            span,
            {
                "span_attributes": {"type": "llm", "name": "Chat Completion"},
                "metadata": {
                    "model": TEST_MODEL,
                    "provider": "openai",
                    "stream": stream,
                    "tools": lambda tools_list: (
                        len(tools_list) == 2
                        and any(tool.get("function", {}).get("name") == "get_weather" for tool in tools_list)
                        and any(tool.get("function", {}).get("name") == "get_time" for tool in tools_list)
                    ),
                },
                "input": lambda inp: "What's the weather in New York and the time in Tokyo?" in str(inp),
                "metrics": lambda m: assert_metrics_are_valid(m, start, end) is None,
            },
        )

        # Verify tool calls are in the output (if present)
        if span.get("output") and isinstance(span["output"], list) and len(span["output"]) > 0:
            message = span["output"][0].get("message", {})
            tool_calls = message.get("tool_calls")
            if tool_calls and len(tool_calls) >= 2:
                # Extract tool names, handling cases where function.name might be None
                tool_names = []
                for call in tool_calls:
                    func = call.get("function", {})
                    name = func.get("name") if isinstance(func, dict) else None
                    if name:
                        tool_names.append(name)

                # Check if we have the expected tools (only if names are available)
                if tool_names:
                    assert "get_weather" in tool_names or "get_time" in tool_names, (
                        f"Expected weather/time tools, got: {tool_names}"
                    )


def _is_wrapped(client):
    """Return True if *client* has been instrumented by wrap_openai()."""
    import inspect

    from wrapt import FunctionWrapper

    completions = getattr(getattr(client, "chat", None), "completions", None)
    if completions is None:
        return False
    attr = inspect.getattr_static(completions, "create", None)
    return isinstance(attr, FunctionWrapper)


TEST_AUDIO_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "test_audio.wav")
STREAMING_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"


async def _collect_transcription_stream_text(is_async, stream) -> tuple[str, list[str]]:
    event_types = []
    deltas = []
    done_text = None
    for event in await _drain(is_async, stream):
        event_types.append(event.type)
        if event.type == "transcript.text.delta":
            deltas.append(event.delta)
        elif event.type == "transcript.text.done":
            done_text = event.text
    return done_text or "".join(deltas), event_types


def _assert_audio_input_attachment(span) -> None:
    assert isinstance(span["input"]["file"], Attachment)
    assert span["input"]["file"].reference["filename"] == "test_audio.wav"
    assert span["input"]["file"].reference["content_type"].startswith("audio/")


def _assert_audio_output_attachment(span) -> None:
    assert span["output"]["type"] == "audio"
    assert span["output"]["audio_size_bytes"] > 0
    attachment = span["output"]["file"]["file_data"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"].startswith("audio/")
    assert attachment.reference["filename"].startswith("generated_speech")


def _assert_chat_audio_attachment(
    audio,
    *,
    transcript: str,
    audio_size_bytes: int | None = None,
    content_type: str,
    filename: str,
    audio_id: str | None = None,
    expires_at: int | None = None,
) -> None:
    if audio_id is not None:
        assert audio["id"] == audio_id
    if expires_at is not None:
        assert audio["expires_at"] == expires_at
    assert audio["transcript"] == transcript
    if audio_size_bytes is not None:
        assert audio["audio_size_bytes"] == audio_size_bytes
    else:
        assert audio["audio_size_bytes"] > 0
    assert "data" not in audio

    attachment = audio["file"]["file_data"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"] == content_type
    assert attachment.reference["filename"] == filename


def _write_test_png(path: str, *, width: int = 64, height: int = 64) -> None:
    """Write a simple opaque red RGBA PNG without external dependencies."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", binascii.crc32(tag + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes([255, 0, 0, 255]) * width
    raw_rows = row * height
    header = struct.pack("!IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw_rows)) + chunk(b"IEND", b"")

    with open(path, "wb") as image_file:
        image_file.write(png)


@pytest.mark.vcr
def test_openai_images_generate(memory_logger):
    assert not memory_logger.pop()

    prompt = "A tiny red square on a white background"
    client = wrap_openai(openai.OpenAI())
    response = client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size="1024x1024",
    )

    assert response
    assert response.data
    assert response.data[0].b64_json or response.data[0].url

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "gpt-image-1-mini"
    assert span["metadata"]["provider"] == "openai"
    assert span["input"] == prompt
    assert span["output"]["images_count"] == 1
    assert span["metrics"]["duration"] >= 0


def test_materialize_logged_file_input_preserves_unrecognized_values():
    file_id = "file-123"
    values = [file_id, NOT_GIVEN]

    materialized = _materialize_logged_file_input(values)

    assert materialized[0] == file_id
    assert materialized[1] is NOT_GIVEN


@pytest.mark.vcr
def test_openai_images_edit(memory_logger):
    assert not memory_logger.pop()

    prompt = "Add a blue border"
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, "braintrust-test-image.png")
        _write_test_png(image_path)

        client = wrap_openai(openai.OpenAI())
        with open(image_path, "rb") as image_file:
            response = client.images.edit(
                model="gpt-image-1-mini",
                prompt=prompt,
                image=image_file,
                size="1024x1024",
            )

        assert response
        assert response.data
        assert response.data[0].b64_json or response.data[0].url

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["metadata"]["model"] == "gpt-image-1-mini"
        assert span["metadata"]["provider"] == "openai"
        assert span["input"]["prompt"] == prompt
        assert isinstance(span["input"]["image"], Attachment)
        assert span["input"]["image"].reference["filename"] == "braintrust-test-image.png"
        assert span["input"]["image"].reference["content_type"] == "image/png"
        assert span["output"]["images_count"] == 1
        assert span["metrics"]["duration"] >= 0


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_speech(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    response = await _call(
        is_async,
        client.audio.speech.create,
        model="tts-1",
        voice="alloy",
        input="Hello, this is a test.",
    )
    assert response

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "tts-1"
    assert span["metadata"]["voice"] == "alloy"
    assert span["metadata"]["provider"] == "openai"
    assert span["input"] == "Hello, this is a test."
    _assert_audio_output_attachment(span)


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_transcription(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    with open(TEST_AUDIO_FILE, "rb") as f:
        response = await _call(is_async, client.audio.transcriptions.create, model="whisper-1", file=f)
    assert response

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "openai"
    _assert_audio_input_attachment(span)
    assert span["output"] == "you"


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_transcription_streaming(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    start = time.time()
    with open(TEST_AUDIO_FILE, "rb") as f:
        stream = await _call(
            is_async,
            client.audio.transcriptions.create,
            model=STREAMING_TRANSCRIPTION_MODEL,
            file=f,
            stream=True,
        )
        transcript, event_types = await _collect_transcription_stream_text(is_async, stream)
    end = time.time()

    assert "transcript.text.delta" in event_types
    assert "transcript.text.done" in event_types
    assert transcript

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == STREAMING_TRANSCRIPTION_MODEL
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["stream"] == True
    _assert_audio_input_attachment(span)
    assert span["output"] == transcript
    assert_metrics_are_valid(span["metrics"], start, end)
    assert span["metrics"]["time_to_first_token"] >= 0


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_transcription_streaming_early_close(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    start = time.time()
    with open(TEST_AUDIO_FILE, "rb") as f:
        create_kwargs = dict(model=STREAMING_TRANSCRIPTION_MODEL, file=f, stream=True)
        if is_async:
            stream = await client.audio.transcriptions.create(**create_kwargs)
            async with stream as traced_stream:
                first_event = await traced_stream.__anext__()
        else:
            with client.audio.transcriptions.create(**create_kwargs) as stream:
                first_event = next(stream)
    end = time.time()

    assert first_event.type == "transcript.text.delta"
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == STREAMING_TRANSCRIPTION_MODEL
    assert span["metadata"]["stream"] == True
    assert span["output"] == first_event.delta
    metrics = span["metrics"]
    assert start <= metrics["start"] <= metrics["end"] <= end
    assert metrics["duration"] >= 0
    assert metrics["time_to_first_token"] >= 0


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_transcription_streaming_no_events_omits_output(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    with open(TEST_AUDIO_FILE, "rb") as f:
        create_kwargs = dict(model=STREAMING_TRANSCRIPTION_MODEL, file=f, stream=True)
        if is_async:
            stream = await client.audio.transcriptions.create(**create_kwargs)
            async with stream:
                pass
        else:
            with client.audio.transcriptions.create(**create_kwargs):
                pass

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == STREAMING_TRANSCRIPTION_MODEL
    assert span["metadata"]["stream"] == True
    assert "output" not in span
    metrics = span["metrics"]
    assert metrics["duration"] >= 0
    assert "time_to_first_token" not in metrics


@pytest.mark.vcr
def test_openai_audio_transcription_text_format(memory_logger):
    """When response_format='text', the API returns a plain string (not JSON)."""
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    with open(TEST_AUDIO_FILE, "rb") as f:
        response = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
    assert response

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "openai"
    _assert_audio_input_attachment(span)
    assert span["output"] == "you"


@sync_async
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_translation(memory_logger, is_async):
    assert not memory_logger.pop()

    client = wrap_openai(_client(is_async))
    with open(TEST_AUDIO_FILE, "rb") as f:
        response = await _call(is_async, client.audio.translations.create, model="whisper-1", file=f)
    assert response

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "openai"
    _assert_audio_input_attachment(span)
    assert span["output"] == "you"


class TestOpenAIIntegrationSetupSpans:
    """VCR-based tests verifying that OpenAIIntegration.setup() produces spans."""

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_setup_preserves_async_audio_speech_streaming_response(self, memory_logger):
        assert not memory_logger.pop()

        OpenAIIntegration.setup()
        client = AsyncOpenAI()

        async with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input="Hello, this is a streaming response test.",
        ) as response:
            assert hasattr(response, "request_id")
            chunks = [chunk async for chunk in response.iter_bytes()]

        assert chunks
        assert not memory_logger.pop()

    @pytest.mark.vcr
    def test_setup_stream_helper_creates_spans(self, memory_logger):
        """OpenAIIntegration.setup() should trace chat.completions.stream()."""
        assert not memory_logger.pop()

        if not hasattr(openai.OpenAI().chat.completions, "stream"):
            pytest.skip("openai.chat.completions.stream is not available in this SDK version")

        OpenAIIntegration.setup()
        client = openai.OpenAI()

        start = time.time()
        with client.chat.completions.stream(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            stream_options={"include_usage": True},
        ) as stream:
            event_types = [event.type for event in stream]
            final = stream.get_final_completion()
        end = time.time()

        assert event_types
        assert "content.delta" in event_types
        assert final.choices[0].message.content
        assert "24" in final.choices[0].message.content or "twenty-four" in final.choices[0].message.content.lower()

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert_metrics_are_valid(span["metrics"], start, end)
        assert span["metadata"]["stream"] == True
        assert "extra_headers" not in span["metadata"]
        assert span["metadata"]["provider"] == "openai"
        assert TEST_MODEL in span["metadata"]["model"]
        assert TEST_PROMPT in str(span["input"])


class TestOpenAIIntegrationSetupAsyncSpans:
    """VCR-based tests verifying that OpenAIIntegration.setup() produces spans for async clients."""

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_setup_async_creates_spans(self, memory_logger):
        """OpenAIIntegration.setup() should create spans for async API calls."""
        assert not memory_logger.pop()

        OpenAIIntegration.setup()
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hi async"}],
        )
        assert response.choices[0].message.content

        # Verify span was created
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["metadata"]["provider"] == "openai"
        assert "gpt-4o-mini" in span["metadata"]["model"]
        assert span["input"]


class TestAutoInstrumentOpenAI:
    """Tests for auto_instrument() with OpenAI."""

    def test_auto_instrument_openai(self):
        """Test auto_instrument patches OpenAI, creates spans, and uninstrument works."""
        verify_autoinstrument_script("test_auto_openai.py")


def test_wrap_openai_wraps_images_methods():
    """wrap_openai() should instrument every OpenAI images resource method."""
    import inspect

    from wrapt import FunctionWrapper

    for client in (wrap_openai(openai.OpenAI()), wrap_openai(openai.AsyncOpenAI())):
        for method_name in ("generate", "edit", "create_variation"):
            method = inspect.getattr_static(client.images, method_name, None)
            assert isinstance(method, FunctionWrapper), f"images.{method_name} was not wrapped"


class TestOpenAIIntegrationSetupImages:
    """Non-network tests for OpenAIIntegration.setup() images patchers."""

    def test_setup_wraps_images_methods(self):
        import inspect

        from openai.resources.images import AsyncImages, Images
        from wrapt import FunctionWrapper

        OpenAIIntegration.setup()

        for cls in (Images, AsyncImages):
            for method_name in ("generate", "edit", "create_variation"):
                method = inspect.getattr_static(cls, method_name, None)
                assert isinstance(method, FunctionWrapper), f"{cls.__name__}.{method_name} was not patched"


def test_wrap_openai_and_setup_use_same_wrappers():
    """Ensure the wrapper functions used by setup() and wrap_openai() stay in sync.

    Both paths should cover the same set of wrapper callables so that the
    traced span shape is identical regardless of which entry-point the user
    chooses.  If this test fails, a wrapper was added to one path but not
    the other.
    """
    from braintrust.integrations.openai.integration import OpenAIIntegration
    from braintrust.integrations.openai.patchers import _WRAP_TARGETS

    # Collect wrapper functions from module-level patchers (setup path).
    setup_wrappers: set = set()
    for patcher in OpenAIIntegration.patchers:
        for sub in patcher.sub_patchers:
            setup_wrappers.add(sub.wrapper)

    # Collect wrapper functions from instance-level patchers (wrap_openai path).
    wrap_wrappers: set = set()
    for _path, patcher in _WRAP_TARGETS:
        for sub in patcher.sub_patchers:
            wrap_wrappers.add(sub.wrapper)

    assert setup_wrappers == wrap_wrappers, (
        f"Wrapper function mismatch between setup() and wrap_openai().\n"
        f"  Only in setup:       {setup_wrappers - wrap_wrappers}\n"
        f"  Only in wrap_openai: {wrap_wrappers - setup_wrappers}"
    )


class TestZAICompatibleOpenAI:
    """Tests for validating some ZAI compatibility with OpenAI wrapper."""

    def test_chat_completion_streaming_none_arguments(self, memory_logger):
        """Test that ChatCompletionWrapper handles None arguments in tool calls (e.g., GLM-4.6 behavior)."""
        assert not memory_logger.pop()

        # Simulate streaming results with None arguments in tool calls
        # This mimics the behavior of GLM-4.6 which returns {'arguments': None, 'name': 'weather'}
        all_results = [
            # First chunk: initial tool call with None arguments
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": None,  # GLM-4.6 returns None here
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            # Second chunk: subsequent tool call arguments (also None)
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": None,  # Subsequent chunks can also have None
                                    }
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            # Third chunk: actual arguments
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": '{"city": "New York"}',
                                    }
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            # Final chunk
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]

        # Process the results
        wrapper = ChatCompletionWrapper(None, None)
        result = wrapper._postprocess_streaming_results(all_results)

        # Verify the output was built correctly
        assert "output" in result
        assert len(result["output"]) == 1
        message = result["output"][0]["message"]
        assert message["role"] == "assistant"
        assert message["tool_calls"] is not None
        assert len(message["tool_calls"]) == 1

        # Verify the tool call was assembled correctly despite None arguments
        tool_call = message["tool_calls"][0]
        assert tool_call["id"] == "call_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        # The arguments should be the concatenation: "" + "" + '{"city": "New York"}'
        assert tool_call["function"]["arguments"] == '{"city": "New York"}'

        # No spans should be generated from this unit test
        assert not memory_logger.pop()

    def test_chat_completion_streaming_audio_is_materialized_as_attachment(self, memory_logger):
        assert not memory_logger.pop()

        all_results = [
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "audio": {
                                "id": "audio_123",
                                "transcript": "He",
                                "data": "aGU=",
                            },
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "audio": {
                                "transcript": "llo",
                                "data": "bGxv",
                                "expires_at": 123,
                            }
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        ]

        wrapper = ChatCompletionWrapper(None, None)
        result = wrapper._postprocess_streaming_results(all_results, audio_format="wav")

        message = result["output"][0]["message"]
        _assert_chat_audio_attachment(
            message["audio"],
            audio_id="audio_123",
            transcript="Hello",
            expires_at=123,
            audio_size_bytes=len(b"hello"),
            content_type="audio/wav",
            filename="generated_audio.wav",
        )
        assert not memory_logger.pop()

    def test_chat_completion_non_stream_audio_is_materialized_as_attachment(self, memory_logger):
        assert not memory_logger.pop()

        output = _process_attachments_in_chat_output(
            [
                {
                    "message": {
                        "role": "assistant",
                        "audio": {
                            "id": "audio_456",
                            "transcript": "Hello",
                            "data": "aGVsbG8=",
                            "expires_at": 456,
                        },
                    }
                }
            ],
            audio_format="wav",
        )

        message = output[0]["message"]
        _assert_chat_audio_attachment(
            message["audio"],
            audio_id="audio_456",
            transcript="Hello",
            expires_at=456,
            audio_size_bytes=len(b"hello"),
            content_type="audio/wav",
            filename="generated_audio.wav",
        )
        assert not memory_logger.pop()
