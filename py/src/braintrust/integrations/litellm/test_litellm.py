import asyncio
import os
import time

import litellm
import pytest
from braintrust import Attachment, logger
from braintrust.integrations.litellm import patch_litellm
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.test_helpers import assert_dict_matches, init_test_logger


TEST_ORG_ID = "test-org-litellm-py-tracing"
PROJECT_NAME = "test-project-litellm-py-tracing"
TEST_MODEL = "gpt-4o-mini"  # cheapest model for tests
TEST_PROMPT = "What's 12 + 12?"
TEST_SYSTEM_PROMPT = "You are a helpful assistant that only responds with numbers."
TEST_AUDIO_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "test_audio.wav")

RERANK_MODEL = "cohere/rerank-english-v3.0"
RERANK_QUERY = "What is the capital of France?"
RERANK_DOCUMENTS = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
]


def _assert_speech_output_attachment(span) -> None:
    assert span["output"]["type"] == "audio"
    assert span["output"]["audio_size_bytes"] > 0
    attachment = span["output"]["file"]["file_data"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"].startswith("audio/")
    assert attachment.reference["filename"].startswith("generated_speech")


@pytest.fixture(autouse=True)
def _patch():
    patch_litellm()


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.mark.vcr
def test_litellm_completion_metrics(memory_logger) -> None:
    assert not memory_logger.pop()

    start = time.time()
    response = litellm.completion(model=TEST_MODEL, messages=[{"role": "user", "content": TEST_PROMPT}])
    end = time.time()

    assert response
    assert response.choices[0].message.content
    assert "24" in response.choices[0].message.content or "twenty-four" in response.choices[0].message.content.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["metadata"]["provider"] == "litellm"
    assert TEST_PROMPT in str(span["input"])


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_acompletion_metrics(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    response = await litellm.acompletion(model=TEST_MODEL, messages=[{"role": "user", "content": TEST_PROMPT}])
    end = time.time()

    assert response
    assert response.choices[0].message.content
    assert "24" in response.choices[0].message.content or "twenty-four" in response.choices[0].message.content.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["metadata"]["provider"] == "litellm"
    assert TEST_PROMPT in str(span["input"])


@pytest.mark.vcr
def test_litellm_completion_streaming_sync(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    stream = litellm.completion(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=True,
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

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["metadata"]["provider"] == "litellm"
    assert TEST_PROMPT in str(span["input"])
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_acompletion_streaming_async(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    stream = await litellm.acompletion(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    end = time.time()

    # Verify streaming works
    assert chunks
    assert len(chunks) > 1

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["metadata"]["provider"] == "litellm"
    assert TEST_PROMPT in str(span["input"])


@pytest.mark.vcr
def test_litellm_responses_metrics(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    response = litellm.responses(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    end = time.time()

    assert response
    assert response.output
    assert len(response.output) > 0
    content = response.output[0].content[0].text
    assert "24" in content or "twenty-four" in content.lower()

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["metadata"]["provider"] == "litellm"
    assert TEST_PROMPT in str(span["input"])


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_aresponses_metrics(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    response = await litellm.aresponses(
        model=TEST_MODEL,
        input=TEST_PROMPT,
        instructions="Just the number please",
    )
    end = time.time()

    assert response
    assert response.output
    assert len(response.output) > 0
    content = response.output[0].content[0].text
    assert "24" in content or "twenty-four" in content.lower()

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["metadata"]["provider"] == "litellm"
    assert TEST_PROMPT in str(span["input"])


@pytest.mark.vcr
def test_litellm_embeddings(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    response = litellm.embedding(model="text-embedding-ada-002", input="This is a test")
    end = time.time()

    assert response
    assert response.data
    assert response.data[0]["embedding"]

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    assert span["metadata"]["model"] == "text-embedding-ada-002"
    assert span["metadata"]["provider"] == "litellm"
    assert "This is a test" in str(span["input"])


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_aembedding(memory_logger):
    assert not memory_logger.pop()

    response = await litellm.aembedding(model="text-embedding-ada-002", input="This is a test")

    assert response
    assert response.data
    assert response.data[0]["embedding"]

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    assert span["metadata"]["model"] == "text-embedding-ada-002"
    assert span["metadata"]["provider"] == "litellm"
    assert "This is a test" in str(span["input"])


@pytest.mark.vcr
def test_litellm_moderation(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    response = litellm.moderation(model="omni-moderation-latest", input="This is a test message")
    end = time.time()

    assert response
    assert response.results

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    assert span["metadata"]["model"] == "omni-moderation-latest"
    assert span["metadata"]["provider"] == "litellm"
    assert "This is a test message" in str(span["input"])


@pytest.mark.vcr
def test_litellm_image_generation(memory_logger):
    assert not memory_logger.pop()

    prompt = "A tiny red square on a white background"

    response = litellm.image_generation(
        model="dall-e-2",
        prompt=prompt,
        size="256x256",
        response_format="url",
    )

    assert response
    assert response.data
    assert response.data[0].url

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "dall-e-2"
    assert span["metadata"]["provider"] == "litellm"
    assert span["metadata"]["response_format"] == "url"
    assert span["input"] == prompt
    assert span["output"]["images_count"] == 1
    assert span["output"]["images"][0]["image_url"]["url"].startswith("https://")
    assert span["metrics"]["duration"] >= 0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_aimage_generation(memory_logger):
    assert not memory_logger.pop()

    prompt = "A tiny blue square on a white background"

    response = await litellm.aimage_generation(
        model="dall-e-2",
        prompt=prompt,
        size="256x256",
        response_format="url",
    )

    assert response
    assert response.data
    assert response.data[0].url

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "dall-e-2"
    assert span["metadata"]["provider"] == "litellm"
    assert span["metadata"]["response_format"] == "url"
    assert span["input"] == prompt
    assert span["output"]["images_count"] == 1
    assert span["output"]["images"][0]["image_url"]["url"].startswith("https://")
    assert span["metrics"]["duration"] >= 0


@pytest.mark.vcr
def test_litellm_completion_with_system_prompt(memory_logger):
    assert not memory_logger.pop()

    response = litellm.completion(
        model=TEST_MODEL,
        messages=[{"role": "system", "content": TEST_SYSTEM_PROMPT}, {"role": "user", "content": TEST_PROMPT}],
    )

    assert response
    assert response.choices
    assert "24" in response.choices[0].message.content

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
def test_litellm_transcription(memory_logger):
    assert not memory_logger.pop()

    with open(TEST_AUDIO_FILE, "rb") as f:
        response = litellm.transcription(model="whisper-1", file=f)

    assert response
    assert response.text == "you"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "litellm"
    assert isinstance(span["input"]["file"], Attachment)
    assert span["input"]["file"].reference["filename"] == "test_audio.wav"
    assert span["input"]["file"].reference["content_type"] in ("audio/x-wav", "audio/wav")  # OS-dependent
    assert span["output"] == "you"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_atranscription(memory_logger):
    assert not memory_logger.pop()

    with open(TEST_AUDIO_FILE, "rb") as f:
        response = await litellm.atranscription(model="whisper-1", file=f)

    assert response
    assert response.text == "you"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "litellm"
    assert isinstance(span["input"]["file"], Attachment)
    assert span["input"]["file"].reference["filename"] == "test_audio.wav"
    assert span["input"]["file"].reference["content_type"] in ("audio/x-wav", "audio/wav")  # OS-dependent
    assert span["output"] == "you"


@pytest.mark.vcr
def test_litellm_speech(memory_logger):
    assert not memory_logger.pop()

    response = litellm.speech(
        model="tts-1",
        voice="alloy",
        input="Hello, this is a test.",
        response_format="mp3",
    )

    assert response
    assert response.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "tts-1"
    assert span["metadata"]["voice"] == "alloy"
    assert span["metadata"]["response_format"] == "mp3"
    assert span["metadata"]["provider"] == "litellm"
    assert span["input"] == "Hello, this is a test."
    _assert_speech_output_attachment(span)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_aspeech(memory_logger):
    assert not memory_logger.pop()

    response = await litellm.aspeech(
        model="tts-1",
        voice="alloy",
        input="Hello, this is a test.",
        response_format="mp3",
    )

    assert response
    assert response.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "tts-1"
    assert span["metadata"]["voice"] == "alloy"
    assert span["metadata"]["response_format"] == "mp3"
    assert span["metadata"]["provider"] == "litellm"
    assert span["input"] == "Hello, this is a test."
    _assert_speech_output_attachment(span)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_acompletion_with_system_prompt(memory_logger):
    assert not memory_logger.pop()

    response = await litellm.acompletion(
        model=TEST_MODEL,
        messages=[{"role": "system", "content": TEST_SYSTEM_PROMPT}, {"role": "user", "content": TEST_PROMPT}],
    )

    assert response
    assert response.choices
    assert "24" in response.choices[0].message.content

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
def test_litellm_completion_error(memory_logger):
    assert not memory_logger.pop()

    # Use a non-existent model to force an error
    fake_model = "non-existent-model"

    try:
        litellm.completion(model=fake_model, messages=[{"role": "user", "content": TEST_PROMPT}])
        pytest.fail("Expected an exception but none was raised")
    except Exception:
        # We expect an error here
        pass

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert log["project_id"] == PROJECT_NAME
    # Check that we got a log entry with the fake model
    assert fake_model in str(log)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_acompletion_error(memory_logger):
    assert not memory_logger.pop()

    # Use a non-existent model to force an error
    fake_model = "non-existent-model"

    try:
        await litellm.acompletion(model=fake_model, messages=[{"role": "user", "content": TEST_PROMPT}])
        pytest.fail("Expected an exception but none was raised")
    except Exception:
        # We expect an error here
        pass

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert log["project_id"] == PROJECT_NAME
    # Check that we got a log entry with the fake model
    assert fake_model in str(log)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_async_parallel_requests(memory_logger):
    """Test multiple parallel async requests."""
    assert not memory_logger.pop()

    # Create multiple prompts
    prompts = [f"What is {i} + {i}?" for i in range(3, 6)]

    # Run requests in parallel
    tasks = [
        litellm.acompletion(model=TEST_MODEL, messages=[{"role": "user", "content": prompt}]) for prompt in prompts
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Check all results
    assert len(results) == 3
    for result in results:
        assert result.choices[0].message.content

    # Check that all spans were created
    spans = memory_logger.pop()
    assert len(spans) == 3

    # Verify each span has proper data
    for i, span in enumerate(spans):
        assert span["metadata"]["model"] == TEST_MODEL
        assert span["metadata"]["provider"] == "litellm"
        assert prompts[i] in str(span["input"])
        assert_metrics_are_valid(span["metrics"])


@pytest.mark.vcr
def test_litellm_tool_calls(memory_logger):
    """Test tool calls with LiteLLM."""
    assert not memory_logger.pop()

    # Define tools that can be called
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
    ]

    start = time.time()
    response = litellm.completion(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": "What's the weather in New York?"}],
        tools=tools,
        temperature=0,
    )
    end = time.time()

    print(response)
    assert response
    assert response.choices

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    # Validate the span structure
    assert_dict_matches(
        span,
        {
            "span_attributes": {"type": "llm", "name": "Completion"},
            "metadata": {
                "model": TEST_MODEL,
                "provider": "litellm",
                "tools": lambda tools_list: len(tools_list) == 1
                and any(tool.get("function", {}).get("name") == "get_weather" for tool in tools_list),
            },
            "input": lambda inp: "What's the weather in New York?" in str(inp),
            "metrics": lambda m: assert_metrics_are_valid(m, start, end) is None,
        },
    )


@pytest.mark.vcr
def test_litellm_responses_streaming_sync(memory_logger):
    """Test the responses API with streaming."""
    assert not memory_logger.pop()

    start = time.time()
    stream = litellm.responses(model=TEST_MODEL, input="What's 12 + 12?", stream=True)

    chunks = []
    for chunk in stream:
        if chunk.type == "response.output_text.delta":
            chunks.append(chunk.delta)
    end = time.time()

    output = "".join(chunks)
    assert chunks
    assert len(chunks) > 1
    assert "24" in output

    # Verify the span is created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] is True
    assert "What's 12 + 12?" in str(span["input"])
    assert "24" in str(span["output"])


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_aresponses_streaming_async(memory_logger):
    """Test the async responses API with streaming."""
    assert not memory_logger.pop()

    start = time.time()
    stream = await litellm.aresponses(model=TEST_MODEL, input="What's 12 + 12?", stream=True)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    end = time.time()

    assert chunks
    assert len(chunks) > 1

    # Verify the span is created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["stream"] is True
    assert "What's 12 + 12?" in str(span["input"])


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_async_streaming_with_break(memory_logger):
    """Test breaking out of the async streaming loop early."""
    assert not memory_logger.pop()

    start = time.time()
    stream = await litellm.acompletion(
        model=TEST_MODEL, messages=[{"role": "user", "content": TEST_PROMPT}], stream=True
    )

    time.sleep(0.1)  # time to first token sleep

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


def test_patch_litellm_responses():
    """Test that patch_litellm() patches responses (subprocess to avoid global state pollution)."""
    verify_autoinstrument_script("test_patch_litellm_responses.py")


def test_patch_litellm_aresponses():
    """Test that patch_litellm() patches aresponses (subprocess to avoid global state pollution)."""
    verify_autoinstrument_script("test_patch_litellm_aresponses.py")


def test_litellm_is_numeric_excludes_booleans():
    """Reproduce issue #1357: _is_numeric should exclude booleans.

    OpenRouter returns `is_byok: true` in usage data. Since Python's bool is
    a subclass of int, isinstance(True, int) is True. The _is_numeric function
    must explicitly exclude booleans so they don't end up in metrics, which
    causes a 400 from the API (expected number, received boolean).
    """
    from braintrust.util import is_numeric

    assert is_numeric(1)
    assert is_numeric(1.0)
    assert not is_numeric(True)
    assert not is_numeric(False)


def test_litellm_parse_metrics_excludes_booleans():
    """Reproduce issue #1357: _parse_metrics_from_usage should not include boolean fields.

    When OpenRouter returns usage data with `is_byok: true`, the metrics parser
    should filter it out rather than passing it through to the API.
    """
    from braintrust.integrations.litellm.tracing import _parse_metrics_from_usage

    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "is_byok": True,
    }
    metrics = _parse_metrics_from_usage(usage)

    assert "prompt_tokens" in metrics
    assert "completion_tokens" in metrics
    assert "tokens" in metrics
    assert "is_byok" not in metrics
    for key, value in metrics.items():
        assert not isinstance(value, bool)


@pytest.mark.vcr
def test_litellm_openrouter_no_booleans_in_metrics(memory_logger):
    """Reproduce issue #1357: OpenRouter returns is_byok boolean in usage.

    Makes a real litellm.completion call via OpenRouter. The response includes
    `is_byok: true` in usage, which must be filtered out of metrics to avoid
    a 400 from the Braintrust API.
    """
    import os

    assert not memory_logger.pop()

    start = time.time()
    response = litellm.completion(
        model="openrouter/openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
        api_key=os.environ.get("OPENROUTER_API_KEY", "fake-key"),
    )
    end = time.time()

    assert response
    assert response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    metrics = spans[0]["metrics"]

    # No boolean values should be in metrics
    for key, value in metrics.items():
        assert not isinstance(value, bool)
    assert "is_byok" not in metrics


@pytest.mark.vcr
def test_litellm_rerank(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    response = litellm.rerank(
        model=RERANK_MODEL,
        query=RERANK_QUERY,
        documents=RERANK_DOCUMENTS,
        top_n=2,
    )
    end = time.time()

    assert response
    assert response.results
    assert len(response.results) == 2
    # Paris should rank first for "capital of France".
    assert response.results[0]["index"] == 0

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "Rerank"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == "litellm"
    assert span["metadata"]["model"] == RERANK_MODEL
    assert span["metadata"]["top_n"] == 2
    assert span["metadata"]["document_count"] == 3
    assert span["input"] == {"query": RERANK_QUERY, "documents": RERANK_DOCUMENTS}

    assert isinstance(span["output"], list)
    assert len(span["output"]) == 2
    assert span["output"][0]["index"] == 0
    assert isinstance(span["output"][0]["relevance_score"], float)
    # Document payload must not leak into the span output.
    for entry in span["output"]:
        assert set(entry.keys()) == {"index", "relevance_score"}

    metrics = span["metrics"]
    assert metrics["start"] >= start
    assert metrics["end"] <= end
    assert metrics["end"] >= metrics["start"]
    assert metrics["duration"] >= 0
    # Cohere rerank bills in search_units.
    assert metrics.get("search_units") == 1


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_arerank(memory_logger):
    assert not memory_logger.pop()

    start = time.time()
    response = await litellm.arerank(
        model=RERANK_MODEL,
        query=RERANK_QUERY,
        documents=RERANK_DOCUMENTS,
        top_n=2,
    )
    end = time.time()

    assert response
    assert response.results
    assert len(response.results) == 2

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "Rerank"
    assert span["metadata"]["provider"] == "litellm"
    assert span["metadata"]["model"] == RERANK_MODEL
    assert span["metadata"]["top_n"] == 2
    assert span["metadata"]["document_count"] == 3
    assert span["input"] == {"query": RERANK_QUERY, "documents": RERANK_DOCUMENTS}
    assert isinstance(span["output"], list)
    assert len(span["output"]) == 2

    metrics = span["metrics"]
    assert metrics["start"] >= start
    assert metrics["end"] <= end
    assert metrics.get("search_units") == 1


def test_litellm_parse_rerank_metrics_from_meta():
    """Unit-level sanity check for ``_parse_rerank_metrics``.

    LiteLLM rerank responses follow Cohere's shape: usage lives under
    ``meta.billed_units`` and ``meta.tokens``.  ``billed_units`` wins when
    both are present.
    """
    from braintrust.integrations.litellm.tracing import _parse_rerank_metrics

    response = {
        "meta": {
            "billed_units": {
                "input_tokens": 7,
                "output_tokens": 3,
                "search_units": 1,
            },
            "tokens": {
                "input_tokens": 999,
                "output_tokens": 999,
            },
        }
    }
    metrics = _parse_rerank_metrics(response)
    # billed_units overrides the matching fields in tokens.
    assert metrics["prompt_tokens"] == 7
    assert metrics["completion_tokens"] == 3
    assert metrics["search_units"] == 1
    # derived total when no total_tokens is reported.
    assert metrics["tokens"] == 10

    # meta.billed_units alone also yields a derived total.
    metrics2 = _parse_rerank_metrics({"meta": {"billed_units": {"search_units": 1}}})
    assert metrics2 == {"search_units": 1}


def test_litellm_extract_rerank_output_drops_document():
    from braintrust.integrations.litellm.tracing import _extract_rerank_output

    response = {
        "results": [
            {"index": 0, "relevance_score": 0.9, "document": {"text": "private"}},
            {"index": 1, "relevance_score": 0.1, "document": {"text": "also private"}},
        ]
    }
    out = _extract_rerank_output(response)
    assert out == [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.1},
    ]


class TestAutoInstrumentLiteLLM:
    """Tests for auto_instrument() with LiteLLM."""

    def test_auto_instrument_litellm(self):
        """Test auto_instrument patches LiteLLM, creates spans, and uninstrument works."""
        verify_autoinstrument_script("test_auto_litellm.py")
