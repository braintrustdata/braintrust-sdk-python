import asyncio
import binascii
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
)
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import assert_dict_matches, init_test_logger
from braintrust.wrappers.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
from pydantic import BaseModel


TEST_ORG_ID = "test-org-openai-py-tracing"
PROJECT_NAME = "test-project-openai-py-tracing"
TEST_MODEL = "gpt-4o-mini"  # cheapest model for tests
RESPONSES_TOOL_MODEL = "gpt-4.1-mini"
TEST_PROMPT = "What's 12 + 12?"
TEST_SYSTEM_PROMPT = "You are a helpful assistant that only responds with numbers."


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _find_spans_by_type(spans, span_type):
    return [span for span in spans if span["span_attributes"]["type"] == span_type]


def _find_span_by_name(spans, name):
    return next(span for span in spans if span["span_attributes"]["name"] == name)


def _supports_response_function_tools() -> bool:
    try:
        from openai.types.responses import ResponseFunctionToolCall

        del ResponseFunctionToolCall
    except ImportError:
        return False
    return True


def _supports_response_web_search_tools() -> bool:
    try:
        from openai.types.responses import ResponseFunctionWebSearch, WebSearchPreviewToolParam

        del ResponseFunctionWebSearch, WebSearchPreviewToolParam
    except ImportError:
        return False
    return True


@pytest.mark.vcr
def test_openai_chat_metrics(memory_logger):
    assert not memory_logger.pop()

    unwrapped_client = openai.OpenAI()
    wrapped_client = wrap_openai(openai.OpenAI())
    clients = [unwrapped_client, wrapped_client]

    for client in clients:
        start = time.time()
        response = client.chat.completions.create(
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

        if not _is_wrapped(client):
            assert not memory_logger.pop()
            continue

        # Verify spans were created with wrapped client
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span
        metrics = span["metrics"]
        assert_metrics_are_valid(metrics, start, end)
        assert TEST_MODEL in span["metadata"]["model"]
        assert span["metadata"]["provider"] == "openai"
        assert TEST_PROMPT in str(span["input"])


@pytest.mark.vcr
def test_openai_responses_metrics(memory_logger):
    assert not memory_logger.pop()

    # First test with an unwrapped client
    unwrapped_client = openai.OpenAI()
    unwrapped_response = unwrapped_client.responses.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    assert unwrapped_response
    assert unwrapped_response.output
    assert len(unwrapped_response.output) > 0
    unwrapped_content = unwrapped_response.output[0].content[0].text

    # No spans should be generated with unwrapped client
    assert not memory_logger.pop()

    # Now test with wrapped client
    client = wrap_openai(openai.OpenAI())
    start = time.time()
    response = client.responses.create(
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

    # Both should contain a numeric response for the math question
    assert "24" in unwrapped_content or "twenty-four" in unwrapped_content.lower()
    assert "24" in wrapped_content or "twenty-four" in wrapped_content.lower()

    # Verify spans were created with wrapped client
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
    span_output_text = span["output"][0]["content"][0]["text"]
    assert "24" in span_output_text or "twenty-four" in span_output_text.lower()

    # Test responses.parse method
    class NumberAnswer(BaseModel):
        value: int
        reasoning: str

    # First test with unwrapped client - should work but no spans
    parse_response = unwrapped_client.responses.parse(model=TEST_MODEL, input=TEST_PROMPT, text_format=NumberAnswer)
    assert parse_response
    # Access the structured output via text_format
    assert parse_response.output_parsed
    assert parse_response.output_parsed.value == 24
    assert parse_response.output_parsed.reasoning

    # No spans should be generated with unwrapped client
    assert not memory_logger.pop()

    # Now test with wrapped client - should generate spans
    start = time.time()
    parse_response = client.responses.parse(model=TEST_MODEL, input=TEST_PROMPT, text_format=NumberAnswer)
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
def test_openai_responses_metadata_preservation(memory_logger):
    """Test that additional metadata fields in responses are preserved."""
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())

    # Test with responses.create - the response object has various metadata fields
    start = time.time()
    response = client.responses.create(
        model=TEST_MODEL,
        input="What is 10 + 10?",
        instructions="Respond with just the number",
    )
    end = time.time()

    assert response
    assert response.output

    # Check that the response has metadata fields like id, created_at, object, etc.
    assert hasattr(response, "id")
    assert hasattr(response, "created_at")
    assert hasattr(response, "object")
    assert hasattr(response, "model")

    # Verify spans capture metadata
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    # Check that span metadata includes the parameters
    assert TEST_MODEL in span["metadata"]["model"]  # Model name may include version date
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["instructions"] == "Respond with just the number"

    # Check that response metadata is preserved (non-output, non-usage fields)
    # The metadata should be in span["metadata"] after our changes
    assert "metadata" in span
    if "id" in span.get("metadata", {}):
        # Response metadata like id, created, object should be preserved
        assert span["metadata"]["id"] == response.id

    # Verify metrics are properly extracted
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert "time_to_first_token" in metrics

    # Test with responses.parse to ensure metadata is preserved there too
    class SimpleAnswer(BaseModel):
        value: int

    start = time.time()
    parse_response = client.responses.parse(
        model=TEST_MODEL,
        input="What is 15 + 15?",
        text_format=SimpleAnswer,
    )
    end = time.time()

    assert parse_response
    assert parse_response.output_parsed
    assert parse_response.output_parsed.value == 30

    # Verify metadata preservation in parse response
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    # Check parameters are in metadata
    assert TEST_MODEL in span["metadata"]["model"]  # Model name may include version date
    assert span["metadata"]["provider"] == "openai"

    # Verify the structured output is captured
    assert span["output"][0]["content"][0]["parsed"]["value"] == 30

    # Check metrics
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)


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


@pytest.mark.vcr
def test_openai_embeddings(memory_logger):
    assert not memory_logger.pop()

    client = openai.OpenAI()
    response = client.embeddings.create(model="text-embedding-ada-002", input="This is a test")

    assert response
    assert response.data
    assert response.data[0].embedding

    assert not memory_logger.pop()

    client2 = wrap_openai(openai.OpenAI())

    start = time.time()
    response2 = client2.embeddings.create(model="text-embedding-ada-002", input="This is a test")
    end = time.time()

    assert response2
    assert response2.data
    assert response2.data[0].embedding

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

    clients = [(openai.OpenAI(), False), (wrap_openai(openai.OpenAI()), True)]

    for client, is_wrapped in clients:
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

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        # Verify spans were created with wrapped client
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
def test_openai_chat_stream_helper_sync(memory_logger):
    assert not memory_logger.pop()

    if not hasattr(openai.OpenAI().chat.completions, "stream"):
        pytest.skip("openai.chat.completions.stream is not available in this SDK version")

    clients = [(openai.OpenAI(), False), (wrap_openai(openai.OpenAI()), True)]

    for client, is_wrapped in clients:
        start = time.time()

        with client.chat.completions.stream(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            stream_options={"include_usage": True},
        ) as stream:
            event_types = []
            content_parts = []
            for event in stream:
                event_types.append(event.type)
                if event.type == "content.delta":
                    content_parts.append(event.delta)
            final = stream.get_final_completion()
        end = time.time()

        content = "".join(content_parts)
        assert event_types
        assert "content.delta" in event_types
        assert final.choices[0].message.content
        assert "24" in final.choices[0].message.content or "twenty-four" in final.choices[0].message.content.lower()
        assert "24" in content or "twenty-four" in content.lower()

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        metrics = span["metrics"]
        assert_metrics_are_valid(metrics, start, end)
        assert span["metadata"]["stream"] == True
        assert span["metadata"]["extra_headers"]["X-Stainless-Helper-Method"] == "chat.completions.stream"
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
def test_openai_chat_with_system_prompt(memory_logger):
    assert not memory_logger.pop()

    clients = [(openai.OpenAI(), False), (wrap_openai(openai.OpenAI()), True)]

    for client, is_wrapped in clients:
        response = client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "system", "content": TEST_SYSTEM_PROMPT}, {"role": "user", "content": TEST_PROMPT}],
        )

        assert response
        assert response.choices
        assert "24" in response.choices[0].message.content

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        inputs = span["input"]
        assert len(inputs) == 2
        assert inputs[0]["role"] == "system"
        assert inputs[0]["content"] == TEST_SYSTEM_PROMPT
        assert inputs[1]["role"] == "user"
        assert inputs[1]["content"] == TEST_PROMPT


@pytest.mark.vcr
def test_openai_client_comparison(memory_logger):
    """Test that wrapped and unwrapped clients produce the same output."""
    assert not memory_logger.pop()

    # Get regular and wrapped clients
    clients = [(openai.OpenAI(), False), (wrap_openai(openai.OpenAI()), True)]

    for client, is_wrapped in clients:
        response = client.chat.completions.create(
            model=TEST_MODEL, messages=[{"role": "user", "content": TEST_PROMPT}], temperature=0, seed=42
        )

        # Both should have data
        assert response.choices[0].message.content

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        # Verify spans were created with wrapped client
        spans = memory_logger.pop()
        assert len(spans) == 1


@pytest.mark.vcr
def test_openai_client_error(memory_logger):
    assert not memory_logger.pop()

    # For the wrapped client only, since we need special error handling
    client = wrap_openai(openai.OpenAI())

    # Use a non-existent model to force an error
    fake_model = "non-existent-model"

    try:
        client.chat.completions.create(model=fake_model, messages=[{"role": "user", "content": TEST_PROMPT}])
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


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_chat_async(memory_logger):
    assert not memory_logger.pop()

    # First test with an unwrapped async client
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        extra_headers={RAW_RESPONSE_HEADER: "true"},
    )

    assert resp
    assert resp.headers
    parsed_response = resp.parse()
    assert parsed_response.choices
    assert parsed_response.choices[0].message.content
    content = parsed_response.choices[0].message.content

    # Verify it contains a correct response
    assert "24" in content or "twenty-four" in content.lower()

    # No spans should be generated with unwrapped client
    assert not memory_logger.pop()

    # Now test with wrapped client
    client2 = wrap_openai(AsyncOpenAI())

    start = time.time()
    resp2 = await client2.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        extra_headers={RAW_RESPONSE_HEADER: "true"},
    )
    end = time.time()

    assert resp2
    assert resp2.headers
    parsed_response2 = resp2.parse()
    assert parsed_response2.choices
    assert parsed_response2.choices[0].message.content
    content2 = parsed_response2.choices[0].message.content

    # Verify the wrapped client also gives correct responses
    assert "24" in content2 or "twenty-four" in content2.lower()

    # Verify spans were created with wrapped client
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert TEST_MODEL in span["metadata"]["model"]
    # assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_responses_async(memory_logger):
    assert not memory_logger.pop()

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
        start = time.time()

        resp = await client.responses.create(
            model=TEST_MODEL,
            input=TEST_PROMPT,
            instructions="Just the number please",
        )
        end = time.time()

        assert resp
        assert resp.output
        assert len(resp.output) > 0

        # Extract the text from the output
        content = resp.output[0].content[0].text

        # Verify response contains correct answer
        assert "24" in content or "twenty-four" in content.lower()

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        # Verify spans were created with wrapped client
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        metrics = span["metrics"]
        assert_metrics_are_valid(metrics, start, end)
        assert 0 <= metrics.get("prompt_cached_tokens", 0)
        assert 0 <= metrics.get("completion_reasoning_tokens", 0)
        assert TEST_MODEL in span["metadata"]["model"]
        # assert span["metadata"]["provider"] == "openai"
        assert TEST_PROMPT in str(span["input"])

    # Test responses.parse method
    class NumberAnswer(BaseModel):
        value: int
        reasoning: str

    for client, is_wrapped in clients:
        if not is_wrapped:
            # Test unwrapped client first
            parse_response = await client.responses.parse(
                model=TEST_MODEL, input=TEST_PROMPT, text_format=NumberAnswer
            )
            assert parse_response
            # Access the structured output via text_format
            assert parse_response.output_parsed
            assert parse_response.output_parsed.value == 24
            assert parse_response.output_parsed.reasoning

            # No spans should be generated with unwrapped client
            assert not memory_logger.pop()
        else:
            # Test wrapped client
            start = time.time()
            parse_response = await client.responses.parse(
                model=TEST_MODEL, input=TEST_PROMPT, text_format=NumberAnswer
            )
            end = time.time()

            assert parse_response
            # Access the structured output via text_format
            assert parse_response.output_parsed
            assert parse_response.output_parsed.value == 24
            assert parse_response.output_parsed.reasoning

            # Verify spans were created
            spans = memory_logger.pop()
            assert len(spans) == 1
            span = spans[0]
            assert span
            metrics = span["metrics"]
            assert_metrics_are_valid(metrics, start, end)
            assert 0 <= metrics.get("prompt_cached_tokens", 0)
            assert 0 <= metrics.get("completion_reasoning_tokens", 0)
            assert TEST_MODEL in span["metadata"]["model"]
            # assert span["metadata"]["provider"] == "openai"
            assert TEST_PROMPT in str(span["input"])
            assert len(span["output"]) > 0
            assert span["output"][0]["content"][0]["parsed"]
            assert span["output"][0]["content"][0]["parsed"]["value"] == 24
            assert span["output"][0]["content"][0]["parsed"]["reasoning"] == parse_response.output_parsed.reasoning


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_embeddings_async(memory_logger):
    assert not memory_logger.pop()

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
        start = time.time()

        resp = await client.embeddings.create(model="text-embedding-ada-002", input="This is a test")
        end = time.time()

        assert resp
        assert resp.data
        assert resp.data[0].embedding

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        # Verify spans were created with wrapped client
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span
        assert span["metadata"]["model"] == "text-embedding-ada-002"
        assert span["metadata"]["provider"] == "openai"
        assert "This is a test" in str(span["input"])


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_streaming_async(memory_logger):
    assert not memory_logger.pop()

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
        start = time.time()

        stream = await client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
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

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        # Verify spans were created with wrapped client
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
async def test_openai_chat_stream_helper_async(memory_logger):
    assert not memory_logger.pop()

    if not hasattr(AsyncOpenAI().chat.completions, "stream"):
        pytest.skip("openai.chat.completions.stream is not available in this SDK version")

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
        start = time.time()

        async with client.chat.completions.stream(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            stream_options={"include_usage": True},
        ) as stream:
            event_types = []
            content_parts = []
            async for event in stream:
                event_types.append(event.type)
                if event.type == "content.delta":
                    content_parts.append(event.delta)
            final = await stream.get_final_completion()
        end = time.time()

        content = "".join(content_parts)
        assert event_types
        assert "content.delta" in event_types
        assert final.choices[0].message.content
        assert "24" in final.choices[0].message.content or "twenty-four" in final.choices[0].message.content.lower()
        assert "24" in content or "twenty-four" in content.lower()

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        metrics = span["metrics"]
        assert_metrics_are_valid(metrics, start, end)
        assert span["metadata"]["stream"] == True
        assert span["metadata"]["extra_headers"]["X-Stainless-Helper-Method"] == "chat.completions.stream"
        assert TEST_MODEL in span["metadata"]["model"]
        assert span["metadata"]["provider"] == "openai"
        assert TEST_PROMPT in str(span["input"])
        assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_async_with_system_prompt(memory_logger):
    assert not memory_logger.pop()

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
        response = await client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "system", "content": TEST_SYSTEM_PROMPT}, {"role": "user", "content": TEST_PROMPT}],
        )

        assert response
        assert response.choices
        assert "24" in response.choices[0].message.content

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        inputs = span["input"]
        assert len(inputs) == 2
        assert inputs[0]["role"] == "system"
        assert inputs[0]["content"] == TEST_SYSTEM_PROMPT
        assert inputs[1]["role"] == "user"
        assert inputs[1]["content"] == TEST_PROMPT


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_client_async_comparison(memory_logger):
    """Test that wrapped and unwrapped async clients produce the same output."""
    assert not memory_logger.pop()

    # Get regular and wrapped clients
    regular_client = AsyncOpenAI()
    wrapped_client = wrap_openai(AsyncOpenAI())

    # Test with regular client
    normal_response = await regular_client.chat.completions.create(
        model=TEST_MODEL, messages=[{"role": "user", "content": TEST_PROMPT}], temperature=0, seed=42
    )

    # No spans should be created for unwrapped client
    assert not memory_logger.pop()

    # Test with wrapped client
    wrapped_response = await wrapped_client.chat.completions.create(
        model=TEST_MODEL, messages=[{"role": "user", "content": TEST_PROMPT}], temperature=0, seed=42
    )

    # Both should have data
    assert normal_response.choices[0].message.content
    assert wrapped_response.choices[0].message.content

    # Verify spans were created with wrapped client
    spans = memory_logger.pop()
    assert len(spans) == 1


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_client_async_error(memory_logger):
    assert not memory_logger.pop()

    # For the wrapped client only, since we need special error handling
    client = wrap_openai(AsyncOpenAI())

    # Use a non-existent model to force an error
    fake_model = "non-existent-model"

    try:
        await client.chat.completions.create(model=fake_model, messages=[{"role": "user", "content": TEST_PROMPT}])
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
async def test_openai_chat_async_context_manager(memory_logger):
    """Test async context manager behavior for chat completions streams."""
    assert not memory_logger.pop()

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
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

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

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

    unwrapped_client = openai.AsyncOpenAI()
    wrapped_client = wrap_openai(openai.AsyncOpenAI())
    clients = [unwrapped_client, wrapped_client]

    for client in clients:
        start = time.time()

        stream = await client.responses.create(model=TEST_MODEL, input="What's 12 + 12?", stream=True)

        chunks = []
        async for chunk in stream:
            if chunk.type == "response.output_text.delta":
                chunks.append(chunk.delta)
        end = time.time()
        output = "".join(chunks)

        assert chunks
        assert len(chunks) > 1

        assert "24" in output

        if not _is_wrapped(client):
            assert not memory_logger.pop()
            continue
        # verify the span is created
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        metrics = span["metrics"]
        assert_metrics_are_valid(metrics, start, end)
        assert span["metadata"]["stream"] == True
        assert "What's 12 + 12?" in str(span["input"])
        assert "24" in str(span["output"])


@pytest.mark.vcr
def test_openai_responses_stream_helper(memory_logger):
    """responses.stream() should preserve the helper interface and emit a tracing span."""
    assert not memory_logger.pop()

    unwrapped_client = openai.OpenAI()
    if not hasattr(unwrapped_client.responses, "stream"):
        pytest.skip("openai.responses.stream is not available in this SDK version")

    with unwrapped_client.responses.stream(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    ) as stream:
        event_types = []
        chunks = []
        for event in stream:
            event_types.append(event.type)
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
        final_response = stream.get_final_response()

    output = "".join(chunks)
    assert "response.output_text.delta" in event_types
    assert final_response.output_text
    assert "24" in output or "twenty-four" in output.lower()
    assert "24" in final_response.output_text or "twenty-four" in final_response.output_text.lower()
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())
    start = time.time()
    with client.responses.stream(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    ) as stream:
        event_types = []
        chunks = []
        for event in stream:
            event_types.append(event.type)
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
        final_response = stream.get_final_response()
    end = time.time()

    output = "".join(chunks)
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
async def test_openai_responses_stream_helper_async(memory_logger):
    """Async responses.stream() should preserve the helper interface and emit a tracing span."""
    assert not memory_logger.pop()

    unwrapped_client = AsyncOpenAI()
    if not hasattr(unwrapped_client.responses, "stream"):
        pytest.skip("openai.responses.stream is not available in this SDK version")

    async with unwrapped_client.responses.stream(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    ) as stream:
        event_types = []
        chunks = []
        async for event in stream:
            event_types.append(event.type)
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
        final_response = await stream.get_final_response()

    output = "".join(chunks)
    assert "response.output_text.delta" in event_types
    assert final_response.output_text
    assert "24" in output or "twenty-four" in output.lower()
    assert "24" in final_response.output_text or "twenty-four" in final_response.output_text.lower()
    assert not memory_logger.pop()

    client = wrap_openai(AsyncOpenAI())
    start = time.time()
    async with client.responses.stream(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    ) as stream:
        event_types = []
        chunks = []
        async for event in stream:
            event_types.append(event.type)
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
        final_response = await stream.get_final_response()
    end = time.time()

    output = "".join(chunks)
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
def test_openai_not_given_filtering(memory_logger):
    """Test that NOT_GIVEN values are filtered out of logged inputs but API call still works."""
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI())

    # Make a call with NOT_GIVEN for optional parameters
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        max_tokens=NOT_GIVEN,
        top_p=NOT_GIVEN,
        frequency_penalty=NOT_GIVEN,
        temperature=0.5,  # one real one
        presence_penalty=NOT_GIVEN,
        tools=NOT_GIVEN,
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
    # Verify NOT_GIVEN values are not in the logged metadata
    meta = span["metadata"]
    assert "NOT_GIVEN" not in str(meta)
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


@pytest.mark.vcr
def test_openai_responses_with_raw_response_create(memory_logger):
    """Test that with_raw_response.create returns HTTP response headers AND generates a tracing span."""
    assert not memory_logger.pop()

    # Unwrapped client: with_raw_response should work but produce no spans.
    unwrapped_client = openai.OpenAI()
    raw = unwrapped_client.responses.with_raw_response.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    assert raw.headers  # HTTP response headers are accessible
    response = raw.parse()
    assert response.output
    content = response.output[0].content[0].text
    assert "24" in content or "twenty-four" in content.lower()
    assert not memory_logger.pop()

    # Wrapped client: with_raw_response should ALSO generate a span.
    client = wrap_openai(openai.OpenAI())
    start = time.time()
    raw = client.responses.with_raw_response.create(
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

    # Unwrapped client: headers accessible, stream iterable via parse(), no spans.
    unwrapped_client = openai.OpenAI()
    raw = unwrapped_client.responses.with_raw_response.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        stream=True,
    )
    assert raw.headers
    chunks = []
    for chunk in raw.parse():
        if chunk.type == "response.output_text.delta":
            chunks.append(chunk.delta)
    assert "24" in "".join(chunks) or "twenty-four" in "".join(chunks).lower()
    assert not memory_logger.pop()

    # Wrapped client: headers still accessible, parse() yields traced stream, span generated.
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

    unwrapped_client = openai.OpenAI()
    if not hasattr(unwrapped_client.responses.with_raw_response, "parse"):
        pytest.skip("openai.responses.with_raw_response.parse is not available in this SDK version")

    raw_parse = unwrapped_client.responses.with_raw_response.parse(
        model=TEST_MODEL, input=TEST_PROMPT, text_format=NumberAnswer
    )
    assert raw_parse.headers
    parse_response = raw_parse.parse()
    assert parse_response.output_parsed
    assert parse_response.output_parsed.value == 24
    assert not memory_logger.pop()

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
async def test_openai_responses_with_raw_response_async(memory_logger):
    """Async version of test_openai_responses_with_raw_response."""
    assert not memory_logger.pop()

    unwrapped_client = AsyncOpenAI()
    raw = await unwrapped_client.responses.with_raw_response.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    assert raw.headers
    response = raw.parse()
    assert response.output
    content = response.output[0].content[0].text
    assert "24" in content or "twenty-four" in content.lower()
    assert not memory_logger.pop()

    client = wrap_openai(AsyncOpenAI())
    start = time.time()
    raw = await client.responses.with_raw_response.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    end = time.time()

    assert raw.headers
    response = raw.parse()
    assert response.output
    content = response.output[0].content[0].text
    assert "24" in content or "twenty-four" in content.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert TEST_MODEL in span["metadata"]["model"]
    assert TEST_PROMPT in str(span["input"])
    assert len(span["output"]) > 0
    span_content = span["output"][0]["content"][0]["text"]
    assert "24" in span_content or "twenty-four" in span_content.lower()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_responses_with_raw_response_create_stream_async(memory_logger):
    """Async version of test_openai_responses_with_raw_response_create_stream."""
    assert not memory_logger.pop()

    # Unwrapped client: headers accessible, stream iterable via parse(), no spans.
    unwrapped_client = AsyncOpenAI()
    raw = await unwrapped_client.responses.with_raw_response.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        stream=True,
    )
    assert raw.headers
    chunks = []
    async for chunk in raw.parse():
        if chunk.type == "response.output_text.delta":
            chunks.append(chunk.delta)
    assert "24" in "".join(chunks) or "twenty-four" in "".join(chunks).lower()
    assert not memory_logger.pop()

    # Wrapped client: headers still accessible, parse() yields traced stream, span generated.
    client = wrap_openai(AsyncOpenAI())
    start = time.time()
    raw = await client.responses.with_raw_response.create(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        stream=True,
    )
    assert raw.headers
    stream = raw.parse()
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

    unwrapped_client = openai.OpenAI()
    wrapped_client = wrap_openai(openai.OpenAI())
    clients = [unwrapped_client, wrapped_client]

    for stream in [False, True]:
        for client in clients:
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

            if not _is_wrapped(client):
                assert not memory_logger.pop()
                continue

            # Verify spans were created with wrapped client
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
                        "tools": lambda tools_list: len(tools_list) == 2
                        and any(tool.get("function", {}).get("name") == "get_weather" for tool in tools_list)
                        and any(tool.get("function", {}).get("name") == "get_time" for tool in tools_list),
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
    clients = [(openai.OpenAI(), False), (wrap_openai(openai.OpenAI()), True)]

    for client, is_wrapped in clients:
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="256x256",
            response_format="url",
        )

        assert response
        assert response.data
        assert response.data[0].url

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["metadata"]["model"] == "dall-e-2"
        assert span["metadata"]["provider"] == "openai"
        assert span["metadata"]["response_format"] == "url"
        assert span["input"] == prompt
        assert span["output"]["images_count"] == 1
        assert span["output"]["images"][0]["image_url"]["url"].startswith("https://")
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

        clients = [(openai.OpenAI(), False), (wrap_openai(openai.OpenAI()), True)]

        for client, is_wrapped in clients:
            with open(image_path, "rb") as image_file:
                response = client.images.edit(
                    model="dall-e-2",
                    prompt=prompt,
                    image=image_file,
                    size="256x256",
                    response_format="url",
                )

            assert response
            assert response.data
            assert response.data[0].url

            if not is_wrapped:
                assert not memory_logger.pop()
                continue

            spans = memory_logger.pop()
            assert len(spans) == 1
            span = spans[0]
            assert span["metadata"]["model"] == "dall-e-2"
            assert span["metadata"]["provider"] == "openai"
            assert span["metadata"]["response_format"] == "url"
            assert span["input"]["prompt"] == prompt
            assert isinstance(span["input"]["image"], Attachment)
            assert span["input"]["image"].reference["filename"] == "braintrust-test-image.png"
            assert span["input"]["image"].reference["content_type"] == "image/png"
            assert span["output"]["images_count"] == 1
            assert span["output"]["images"][0]["image_url"]["url"].startswith("https://")
            assert span["metrics"]["duration"] >= 0


@pytest.mark.vcr
def test_openai_audio_speech(memory_logger):
    assert not memory_logger.pop()

    # Unwrapped client should produce no spans
    client = openai.OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="Hello, this is a test.",
    )
    assert response
    assert not memory_logger.pop()

    # Wrapped client should produce a span
    client2 = wrap_openai(openai.OpenAI())
    response2 = client2.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="Hello, this is a test.",
    )
    assert response2

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "tts-1"
    assert span["metadata"]["voice"] == "alloy"
    assert span["metadata"]["provider"] == "openai"
    assert span["input"] == "Hello, this is a test."
    assert span["output"] == {"type": "audio"}


@pytest.mark.vcr
def test_openai_audio_transcription(memory_logger):
    assert not memory_logger.pop()

    # Unwrapped client should produce no spans
    client = openai.OpenAI()
    with open(TEST_AUDIO_FILE, "rb") as f:
        response = client.audio.transcriptions.create(model="whisper-1", file=f)
    assert response
    assert not memory_logger.pop()

    # Wrapped client should produce a span
    client2 = wrap_openai(openai.OpenAI())
    with open(TEST_AUDIO_FILE, "rb") as f:
        response2 = client2.audio.transcriptions.create(model="whisper-1", file=f)
    assert response2

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "openai"
    assert span["output"] == "you"


@pytest.mark.vcr
def test_openai_audio_transcription_text_format(memory_logger):
    """When response_format='text', the API returns a plain string (not JSON)."""
    assert not memory_logger.pop()

    # Unwrapped client should produce no spans
    client = openai.OpenAI()
    with open(TEST_AUDIO_FILE, "rb") as f:
        response = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
    assert response
    assert not memory_logger.pop()

    # Wrapped client should produce a span with the plain-text output
    client2 = wrap_openai(openai.OpenAI())
    with open(TEST_AUDIO_FILE, "rb") as f:
        response2 = client2.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
    assert response2

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "openai"
    assert span["output"] == "you"


@pytest.mark.vcr
def test_openai_audio_translation(memory_logger):
    assert not memory_logger.pop()

    # Unwrapped client should produce no spans
    client = openai.OpenAI()
    with open(TEST_AUDIO_FILE, "rb") as f:
        response = client.audio.translations.create(model="whisper-1", file=f)
    assert response
    assert not memory_logger.pop()

    # Wrapped client should produce a span
    client2 = wrap_openai(openai.OpenAI())
    with open(TEST_AUDIO_FILE, "rb") as f:
        response2 = client2.audio.translations.create(model="whisper-1", file=f)
    assert response2

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "openai"
    assert span["output"] == "you"


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_speech_async(memory_logger):
    assert not memory_logger.pop()

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
        response = await client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="Hello, this is a test.",
        )
        assert response

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["metadata"]["model"] == "tts-1"
        assert span["metadata"]["voice"] == "alloy"
        assert span["metadata"]["provider"] == "openai"
        assert span["input"] == "Hello, this is a test."
        assert span["output"] == {"type": "audio"}


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_transcription_async(memory_logger):
    assert not memory_logger.pop()

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
        with open(TEST_AUDIO_FILE, "rb") as f:
            response = await client.audio.transcriptions.create(model="whisper-1", file=f)
        assert response

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["metadata"]["model"] == "whisper-1"
        assert span["metadata"]["provider"] == "openai"
        assert span["output"] == "you"


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_audio_translation_async(memory_logger):
    assert not memory_logger.pop()

    clients = [(AsyncOpenAI(), False), (wrap_openai(AsyncOpenAI()), True)]

    for client, is_wrapped in clients:
        with open(TEST_AUDIO_FILE, "rb") as f:
            response = await client.audio.translations.create(model="whisper-1", file=f)
        assert response

        if not is_wrapped:
            assert not memory_logger.pop()
            continue

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["metadata"]["model"] == "whisper-1"
        assert span["metadata"]["provider"] == "openai"
        assert span["output"] == "you"


class TestOpenAIIntegrationSetupSpans:
    """VCR-based tests verifying that OpenAIIntegration.setup() produces spans."""

    @pytest.mark.vcr
    def test_setup_creates_spans(self, memory_logger):
        """OpenAIIntegration.setup() should create spans when making API calls."""
        assert not memory_logger.pop()

        OpenAIIntegration.setup()
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hi"}],
        )
        assert response.choices[0].message.content

        # Verify span was created
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["metadata"]["provider"] == "openai"
        assert "gpt-4o-mini" in span["metadata"]["model"]
        assert span["input"]

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
        assert span["metadata"]["extra_headers"]["X-Stainless-Helper-Method"] == "chat.completions.stream"
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
