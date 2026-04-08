import gzip
import json
import os
import time
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.google_genai import setup_genai
from braintrust.logger import Attachment
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from braintrust.wrappers.test_utils import verify_autoinstrument_script
from google.genai import types


try:
    from google.genai import interactions
except ImportError:
    interactions = None

_needs_interactions = pytest.mark.skipif(interactions is None, reason="google-genai too old for interactions API")
from google.genai.client import Client


PROJECT_NAME = "test-genai-app"
MODEL = "gemini-2.0-flash-001"
EMBEDDING_MODEL = "gemini-embedding-001"
IMAGE_MODEL = "imagen-4.0-fast-generate-001"
REASONING_MODEL = "gemini-2.5-flash"
INTERACTIONS_MODEL = "gemini-2.5-flash"
FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "internal/golden/fixtures"
TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


def _sanitize_generate_images_body(value):
    if isinstance(value, dict):
        return {
            key: (
                TINY_PNG_BASE64
                if key == "bytesBase64Encoded" and isinstance(val, str)
                else _sanitize_generate_images_body(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_generate_images_body(item) for item in value]
    return value


def _sanitize_generate_images_response(response):
    body = response.get("body", {})
    payload = body.get("string")
    if not payload:
        return response

    is_bytes = isinstance(payload, bytes)
    is_gzipped = False

    if is_bytes:
        raw_payload = payload
        if raw_payload[:2] == b"\x1f\x8b":
            raw_payload = gzip.decompress(raw_payload)
            is_gzipped = True
        payload = raw_payload.decode("utf-8")

    try:
        parsed = json.loads(payload)
    except Exception:
        return response

    sanitized = _sanitize_generate_images_body(parsed)
    if sanitized == parsed:
        return response

    sanitized_payload = json.dumps(sanitized)
    if is_bytes:
        body["string"] = (
            gzip.compress(sanitized_payload.encode("utf-8")) if is_gzipped else sanitized_payload.encode("utf-8")
        )
    else:
        body["string"] = sanitized_payload
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """Google-specific VCR config - needs to uppercase HTTP methods."""
    record_mode = "none" if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) else "once"

    def before_record_request(request):
        # Normalize HTTP method to uppercase for consistency (Google API quirk)
        request.method = request.method.upper()
        return request

    def before_record_response(response):
        return _sanitize_generate_images_response(response)

    return {
        "record_mode": record_mode,
        "decode_compressed_response": True,
        "filter_headers": [
            "authorization",
            "Authorization",
            "x-api-key",
            "x-goog-api-key",
        ],
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
    }


@pytest.fixture(scope="module", autouse=True)
def setup_wrapper():
    """Setup genai wrapper once for all tests."""
    setup_genai(project_name=PROJECT_NAME)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


# Helper to assert metrics are valid
def _assert_metrics_are_valid(metrics, start=None, end=None):
    assert metrics["tokens"] > 0
    assert metrics["prompt_tokens"] > 0
    assert metrics["completion_tokens"] > 0
    if start and end:
        assert start <= metrics["start"] <= metrics["end"] <= end
    else:
        assert metrics["start"] <= metrics["end"]


def _assert_timing_metrics_are_valid(metrics, start=None, end=None):
    assert metrics["duration"] >= 0
    if start and end:
        assert start <= metrics["start"] <= metrics["end"] <= end
    else:
        assert metrics["start"] <= metrics["end"]


def _assert_attachment_part(part, *, content_type, filename):
    assert "image_url" in part
    assert "url" in part["image_url"]

    attachment = part["image_url"]["url"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["type"] == "braintrust_attachment"
    assert attachment.reference["content_type"] == content_type
    assert attachment.reference["filename"] == filename
    assert attachment.reference["key"]
    return attachment


def _assert_binary_not_logged(span, binary_data):
    span_str = str(span).lower()
    assert binary_data[:8].hex() not in span_str


# Test 1: Basic Completion (Sync)
@pytest.mark.vcr
@pytest.mark.parametrize(
    "mode",
    ["sync", "stream"],
)
def test_basic_completion(memory_logger, mode):
    """Test basic text completion in sync modes."""
    assert not memory_logger.pop()

    client = Client()
    start = time.time()

    if mode == "sync":
        response = client.models.generate_content(
            model=MODEL,
            contents="What is the capital of France?",
            config=types.GenerateContentConfig(
                max_output_tokens=100,
            ),
        )
        text = response.text
    elif mode == "stream":
        stream = client.models.generate_content_stream(
            model=MODEL,
            contents="What is the capital of France?",
            config=types.GenerateContentConfig(
                max_output_tokens=100,
            ),
        )
        text = ""
        for chunk in stream:
            if chunk.text:
                text += chunk.text

    end = time.time()

    # Verify response contains expected content
    assert "Paris" in text

    # Verify logging
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert "What is the capital of France?" in str(span["input"])
    assert span["output"]
    assert "Paris" in str(span["output"])
    _assert_metrics_are_valid(span["metrics"], start, end)


# Test 1b: Basic Completion (Async)
@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["async", "async_stream"],
)
async def test_basic_completion_async(memory_logger, mode):
    """Test basic text completion in async modes."""
    assert not memory_logger.pop()

    client = Client()
    start = time.time()

    if mode == "async":
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents="What is the capital of France?",
            config=types.GenerateContentConfig(
                max_output_tokens=100,
            ),
        )
        text = response.text
    elif mode == "async_stream":
        stream = await client.aio.models.generate_content_stream(
            model=MODEL,
            contents="What is the capital of France?",
            config=types.GenerateContentConfig(
                max_output_tokens=100,
            ),
        )
        text = ""
        async for chunk in stream:
            if chunk.text:
                text += chunk.text

    end = time.time()

    # Verify response contains expected content
    assert "Paris" in text

    # Verify logging
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert "What is the capital of France?" in str(span["input"])
    assert span["output"]
    assert "Paris" in str(span["output"])
    _assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_embed_content(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    start = time.time()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=["This is a test", "This is another test"],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=32,
        ),
    )
    end = time.time()

    assert response.embeddings
    assert len(response.embeddings) == 2
    assert response.embeddings[0].values
    assert len(response.embeddings[0].values) == 32

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == EMBEDDING_MODEL
    assert "RETRIEVAL_DOCUMENT" in str(span["input"])
    assert "This is a test" in str(span["input"])
    assert span["output"]["embedding_length"] == 32
    assert span["output"]["embeddings_count"] == 2
    _assert_timing_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_embed_content_async(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    start = time.time()
    response = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=["This is a test", "This is another test"],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=32,
        ),
    )
    end = time.time()

    assert response.embeddings
    assert len(response.embeddings) == 2
    assert response.embeddings[0].values
    assert len(response.embeddings[0].values) == 32

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == EMBEDDING_MODEL
    assert "RETRIEVAL_DOCUMENT" in str(span["input"])
    assert "This is a test" in str(span["input"])
    assert span["output"]["embedding_length"] == 32
    assert span["output"]["embeddings_count"] == 2
    _assert_timing_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_image_input(memory_logger):
    """Verify image inputs are traced as attachments instead of raw bytes."""
    assert not memory_logger.pop()

    image_data = (FIXTURES_DIR / "test-image.png").read_bytes()

    client = Client()
    start = time.time()
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_data, mime_type="image/png"),
            types.Part.from_text(text="What color is this image?"),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=150,
        ),
    )
    end = time.time()

    assert response.text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    contents = span["input"]["contents"]
    assert len(contents) == 2
    _assert_attachment_part(contents[0], content_type="image/png", filename="file.png")
    assert contents[1] == {"text": "What color is this image?"}
    _assert_binary_not_logged(span, image_data)
    assert span["output"]
    _assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_document_input(memory_logger):
    """Verify document inputs are traced as attachments instead of raw bytes."""
    assert not memory_logger.pop()

    pdf_data = (FIXTURES_DIR / "test-document.pdf").read_bytes()

    client = Client()
    start = time.time()
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
            types.Part.from_text(text="What is in this document?"),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=150,
        ),
    )
    end = time.time()

    assert response.text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    contents = span["input"]["contents"]
    assert len(contents) == 2
    _assert_attachment_part(contents[0], content_type="application/pdf", filename="file.pdf")
    assert contents[1] == {"text": "What is in this document?"}
    _assert_binary_not_logged(span, pdf_data)
    assert span["output"]
    _assert_metrics_are_valid(span["metrics"], start, end)


# Test 3: Tool Use (Sync)
@pytest.mark.vcr
@pytest.mark.parametrize(
    "mode",
    ["sync", "stream"],
)
def test_tool_use(memory_logger, mode):
    """Test function calling / tool use in sync modes."""
    assert not memory_logger.pop()

    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location.

        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: The unit of temperature (celsius or fahrenheit)
        """
        return f"22 degrees {unit} and sunny in {location}"

    client = Client()
    start = time.time()
    has_function_call = False

    if mode == "sync":
        response = client.models.generate_content(
            model=MODEL,
            contents="What is the weather like in Paris, France?",
            config=types.GenerateContentConfig(
                tools=[get_weather],
                max_output_tokens=500,
            ),
        )
        # Check if function was called (either in function_calls or automatic_function_calling_history)
        has_function_call = (hasattr(response, "function_calls") and response.function_calls) or (
            hasattr(response, "automatic_function_calling_history") and response.automatic_function_calling_history
        )
    elif mode == "stream":
        stream = client.models.generate_content_stream(
            model=MODEL,
            contents="What is the weather like in Paris, France?",
            config=types.GenerateContentConfig(
                tools=[get_weather],
                max_output_tokens=500,
            ),
        )
        chunks = list(stream)
        # Check if function was called in any chunk (either in function_calls or automatic_function_calling_history)
        has_function_call = any(
            (hasattr(chunk, "function_calls") and chunk.function_calls)
            or (hasattr(chunk, "automatic_function_calling_history") and chunk.automatic_function_calling_history)
            for chunk in chunks
        )

    end = time.time()

    # Verify function call was made
    assert has_function_call, f"Expected function call in {mode} mode but got has_function_call={has_function_call}"

    # Verify logging (automatic function calling may create multiple spans)
    spans = memory_logger.pop()
    assert len(spans) >= 1
    # Check the first span (initial request with tool call)
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert "Paris" in str(span["input"]) or "weather" in str(span["input"])
    assert span["output"]
    _assert_metrics_are_valid(span["metrics"], start, end)


# Test 3b: Tool Use (Async)
@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["async", "async_stream"],
)
async def test_tool_use_async(memory_logger, mode):
    """Test function calling / tool use in async modes."""
    assert not memory_logger.pop()

    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location.

        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: The unit of temperature (celsius or fahrenheit)
        """
        return f"22 degrees {unit} and sunny in {location}"

    client = Client()
    start = time.time()
    has_function_call = False

    if mode == "async":
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents="What is the weather like in Paris, France?",
            config=types.GenerateContentConfig(
                tools=[get_weather],
                max_output_tokens=500,
            ),
        )
        # Check if function was called (either in function_calls or automatic_function_calling_history)
        has_function_call = (hasattr(response, "function_calls") and response.function_calls) or (
            hasattr(response, "automatic_function_calling_history") and response.automatic_function_calling_history
        )
    elif mode == "async_stream":
        stream = await client.aio.models.generate_content_stream(
            model=MODEL,
            contents="What is the weather like in Paris, France?",
            config=types.GenerateContentConfig(
                tools=[get_weather],
                max_output_tokens=500,
            ),
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        # Check if function was called in any chunk (either in function_calls or automatic_function_calling_history)
        has_function_call = any(
            (hasattr(chunk, "function_calls") and chunk.function_calls)
            or (hasattr(chunk, "automatic_function_calling_history") and chunk.automatic_function_calling_history)
            for chunk in chunks
        )

    end = time.time()

    # Verify function call was made
    assert has_function_call, f"Expected function call in {mode} mode but got has_function_call={has_function_call}"

    # Verify logging (automatic function calling may create multiple spans)
    spans = memory_logger.pop()
    assert len(spans) >= 1
    # Check the first span (initial request with tool call)
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert "Paris" in str(span["input"]) or "weather" in str(span["input"])
    assert span["output"]
    _assert_metrics_are_valid(span["metrics"], start, end)


# Test 4: System Prompt
@pytest.mark.vcr
def test_system_prompt(memory_logger):
    """Test system instruction handling."""
    assert not memory_logger.pop()

    client = Client()
    response = client.models.generate_content(
        model=MODEL,
        contents="Tell me about the weather.",
        config=types.GenerateContentConfig(
            system_instruction="You are a pirate. Always respond in pirate speak.",
            max_output_tokens=150,
        ),
    )

    text = response.text
    assert text

    # Verify logging
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["input"]
    assert span["output"]
    # Check that system instruction is captured
    assert "pirate" in str(span["input"]).lower() or "system_instruction" in str(span)


# Test 5: Multi-turn Conversation
@pytest.mark.vcr
def test_multi_turn(memory_logger):
    """Test multi-turn conversation."""
    assert not memory_logger.pop()

    client = Client()
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="Hi, my name is Alice.")]),
            types.Content(role="model", parts=[types.Part.from_text(text="Hello Alice! Nice to meet you.")]),
            types.Content(role="user", parts=[types.Part.from_text(text="What did I just tell you my name was?")]),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=200,
        ),
    )

    text = response.text
    assert "Alice" in text

    # Verify logging
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["input"]
    assert span["output"]
    assert "Alice" in str(span["input"])


# Test 6: Temperature and Top P
@pytest.mark.vcr
def test_temperature_and_top_p(memory_logger):
    """Test temperature and top_p parameters."""
    assert not memory_logger.pop()

    client = Client()
    response = client.models.generate_content(
        model=MODEL,
        contents="Say something creative.",
        config=types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=50,
        ),
    )

    text = response.text
    assert text

    # Verify logging includes temperature and top_p
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL


# Test 7: Error Handling
@pytest.mark.vcr
def test_error_handling(memory_logger):
    """Test that errors are properly logged."""
    assert not memory_logger.pop()

    client = Client()
    fake_model = "there-is-no-such-model"

    try:
        client.models.generate_content(
            model=fake_model,
            contents="Hello",
            config=types.GenerateContentConfig(
                max_output_tokens=100,
            ),
        )
    except Exception:
        pass
    else:
        raise Exception("should have raised an exception")

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert log["project_id"] == PROJECT_NAME
    assert log["error"]


@pytest.mark.vcr
def test_stop_sequences(memory_logger):
    """Test stop sequences parameter."""
    assert not memory_logger.pop()

    client = Client()
    response = client.models.generate_content(
        model=MODEL,
        contents="Write a short story about a robot.",
        config=types.GenerateContentConfig(
            max_output_tokens=500,
            stop_sequences=["END", "\n\n"],
        ),
    )

    text = response.text
    assert text

    # Verify logging
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL


@pytest.mark.vcr
def test_prefill(memory_logger):
    """Verify prefilled model context is preserved in traced input."""
    assert not memory_logger.pop()

    client = Client()
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="Write a haiku about coding.")]),
            types.Content(role="model", parts=[types.Part.from_text(text="Here is a haiku:")]),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=200,
        ),
    )

    assert response.text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["input"]["contents"] == [
        {"role": "user", "parts": [{"text": "Write a haiku about coding."}]},
        {"role": "model", "parts": [{"text": "Here is a haiku:"}]},
    ]
    assert span["output"]


@pytest.mark.vcr
def test_short_max_tokens(memory_logger):
    """Verify truncated responses still log useful output and request config."""
    assert not memory_logger.pop()

    client = Client()
    response = client.models.generate_content(
        model=MODEL,
        contents="What is AI?",
        config=types.GenerateContentConfig(
            max_output_tokens=5,
        ),
    )

    assert response.text
    assert response.candidates
    assert response.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == MODEL
    assert span["input"]["config"]["max_output_tokens"] == 5
    assert span["output"]


@pytest.mark.vcr
def test_tool_use_with_result(memory_logger):
    """Verify function-response turns are captured in traced conversation history."""
    assert not memory_logger.pop()

    client = Client()

    function = types.FunctionDeclaration(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The mathematical operation",
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "a", "b"],
        },
    )
    tool = types.Tool(function_declarations=[function])

    first_response = client.models.generate_content(
        model=MODEL,
        contents="What is 127 multiplied by 49?",
        config=types.GenerateContentConfig(
            tools=[tool],
            max_output_tokens=500,
        ),
    )

    assert first_response.candidates
    assert first_response.function_calls
    tool_call = first_response.function_calls[0]
    assert tool_call.name == "calculate"

    second_response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="What is 127 multiplied by 49?")]),
            first_response.candidates[0].content,
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=tool_call.name,
                        response={"result": 127 * 49},
                    )
                ],
            ),
        ],
        config=types.GenerateContentConfig(
            tools=[tool],
            max_output_tokens=500,
        ),
    )

    assert second_response.text
    assert "6223" in second_response.text

    spans = memory_logger.pop()
    assert len(spans) == 2

    first_span, second_span = spans
    assert first_span["metadata"]["model"] == MODEL
    assert "calculate" in str(first_span["input"])
    assert first_span["output"]

    assert second_span["metadata"]["model"] == MODEL
    assert "function_response" in str(second_span["input"])
    assert "6223" in str(second_span["input"])
    assert "6223" in str(second_span["output"])


@pytest.mark.vcr
def test_reasoning(memory_logger):
    """Verify reasoning-enabled responses log reasoning metrics across follow-up calls."""
    assert not memory_logger.pop()

    client = Client()

    first_prompt = (
        "Look at this sequence: 2, 6, 12, 20, 30. What is the pattern and what would be the formula for the nth term?"
    )
    follow_up_prompt = "Using the pattern you discovered, what would be the 10th term? And can you find the sum of the first 10 terms?"
    reasoning_config = types.GenerateContentConfig(
        max_output_tokens=512,
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=128),
    )

    first_response = client.models.generate_content(
        model=REASONING_MODEL,
        contents=first_prompt,
        config=reasoning_config,
    )
    assert first_response.candidates

    follow_up_response = client.models.generate_content(
        model=REASONING_MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=first_prompt)]),
            first_response.candidates[0].content,
            types.Content(role="user", parts=[types.Part.from_text(text=follow_up_prompt)]),
        ],
        config=reasoning_config,
    )

    assert follow_up_response.text
    assert "110" in follow_up_response.text
    assert "sum" in follow_up_response.text.lower()

    spans = memory_logger.pop()
    assert len(spans) == 2

    first_span, second_span = spans
    for span in spans:
        assert span["metadata"]["model"] == REASONING_MODEL
        assert span["input"]["config"]["thinking_config"]["include_thoughts"] is True
        assert span["metrics"]["completion_reasoning_tokens"] > 0
        assert span["output"]

    assert first_prompt in str(first_span["input"])
    assert follow_up_prompt in str(second_span["input"])


def test_attachment_in_config(memory_logger):
    """Test that attachments in config are preserved through serialization."""
    from braintrust.bt_json import bt_safe_deep_copy
    from braintrust.logger import Attachment

    attachment = Attachment(data=b"config data", filename="config.txt", content_type="text/plain")

    # Simulate config with attachment
    config = {"temperature": 0.5, "context_file": attachment, "max_output_tokens": 100}

    # Test bt_safe_deep_copy preserves attachment
    copied = bt_safe_deep_copy(config)
    assert copied["context_file"] is attachment
    assert copied["temperature"] == 0.5


@pytest.mark.vcr
def test_generate_images(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    start = time.time()

    response = client.models.generate_images(
        model=IMAGE_MODEL,
        prompt="A watercolor fox in a forest",
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="BLOCK_LOW_AND_ABOVE",
            include_rai_reason=True,
        ),
    )
    end = time.time()

    assert len(response.generated_images) == 1
    assert response.generated_images[0].image
    assert response.generated_images[0].image.image_bytes

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == IMAGE_MODEL
    assert span["input"]["prompt"] == "A watercolor fox in a forest"
    assert span["input"]["config"]["number_of_images"] == 1
    assert span["input"]["config"]["aspect_ratio"] == "1:1"
    assert span["input"]["config"]["safety_filter_level"] == "BLOCK_LOW_AND_ABOVE"
    assert span["input"]["config"]["include_rai_reason"] is True
    assert span["output"]["generated_images_count"] == 1
    generated_image = span["output"]["generated_images"][0]
    assert generated_image["image_size_bytes"] > 0
    assert generated_image["mime_type"] in {"image/png", "image/jpeg", "image/webp"}

    # Verify the image bytes are stored as an Attachment for upload to object storage
    assert "image_url" in generated_image
    attachment = generated_image["image_url"]["url"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["type"] == "braintrust_attachment"
    assert attachment.reference["content_type"] == generated_image["mime_type"]
    assert attachment.reference["filename"].startswith("generated_image_")
    assert attachment.reference["key"]

    _assert_timing_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_generate_images_async(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    start = time.time()

    response = await client.aio.models.generate_images(
        model=IMAGE_MODEL,
        prompt="A watercolor fox in a forest",
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="BLOCK_LOW_AND_ABOVE",
            include_rai_reason=True,
        ),
    )
    end = time.time()

    assert len(response.generated_images) == 1
    assert response.generated_images[0].image
    assert response.generated_images[0].image.image_bytes

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == IMAGE_MODEL
    assert span["input"]["prompt"] == "A watercolor fox in a forest"
    assert span["input"]["config"]["number_of_images"] == 1
    assert span["input"]["config"]["aspect_ratio"] == "1:1"
    assert span["input"]["config"]["safety_filter_level"] == "BLOCK_LOW_AND_ABOVE"
    assert span["input"]["config"]["include_rai_reason"] is True
    assert span["output"]["generated_images_count"] == 1
    generated_image = span["output"]["generated_images"][0]
    assert generated_image["image_size_bytes"] > 0
    assert generated_image["mime_type"] in {"image/png", "image/jpeg", "image/webp"}

    # Verify the image bytes are stored as an Attachment for upload to object storage
    assert "image_url" in generated_image
    attachment = generated_image["image_url"]["url"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["type"] == "braintrust_attachment"
    assert attachment.reference["content_type"] == generated_image["mime_type"]
    assert attachment.reference["filename"].startswith("generated_image_")
    assert attachment.reference["key"]

    _assert_timing_metrics_are_valid(span["metrics"], start, end)


def test_nested_attachments_in_contents(memory_logger):
    """Test that nested attachments in contents are preserved."""
    from braintrust.bt_json import bt_safe_deep_copy
    from braintrust.logger import Attachment, ExternalAttachment

    attachment1 = Attachment(data=b"file1", filename="file1.txt", content_type="text/plain")
    attachment2 = ExternalAttachment(url="s3://bucket/file2.pdf", filename="file2.pdf", content_type="application/pdf")

    # Simulate contents with nested attachments
    contents = [
        {"role": "user", "parts": [{"text": "Check these files"}, {"file": attachment1}]},
        {"role": "model", "parts": [{"text": "Analyzed"}, {"result_file": attachment2}]},
    ]

    copied = bt_safe_deep_copy(contents)

    # Verify attachments preserved
    assert copied[0]["parts"][1]["file"] is attachment1
    assert copied[1]["parts"][1]["result_file"] is attachment2


def test_attachment_with_pydantic_model(memory_logger):
    """Test that attachments work alongside Pydantic model serialization."""
    from braintrust.bt_json import bt_safe_deep_copy
    from braintrust.logger import Attachment
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str
        value: int

    attachment = Attachment(data=b"model data", filename="model.txt", content_type="text/plain")

    # Structure with both Pydantic model and attachment
    data = {"model_config": TestModel(name="test", value=42), "context_file": attachment}

    copied = bt_safe_deep_copy(data)

    # Pydantic model should be converted to dict
    assert isinstance(copied["model_config"], dict)
    assert copied["model_config"]["name"] == "test"

    # Attachment should be preserved
    assert copied["context_file"] is attachment


GROUNDING_MODEL = "gemini-2.0-flash-001"


def _assert_grounding_metadata(span_output):
    """Assert that grounding metadata is present and well-structured in span output."""
    # The grounding_metadata should be present on the first candidate
    candidates = span_output.get("candidates", [])
    assert candidates, "Expected candidates in span output"

    first_candidate = candidates[0]
    grounding = first_candidate.get("grounding_metadata")
    assert grounding is not None, (
        f"Expected grounding_metadata on first candidate, got keys: {list(first_candidate.keys())}"
    )

    # web_search_queries should be a non-empty list of strings
    web_search_queries = grounding.get("web_search_queries")
    assert web_search_queries, "Expected web_search_queries in grounding_metadata"
    assert isinstance(web_search_queries, list)
    assert all(isinstance(q, str) for q in web_search_queries)

    # grounding_chunks should contain search result snippets
    grounding_chunks = grounding.get("grounding_chunks")
    assert grounding_chunks, "Expected grounding_chunks in grounding_metadata"
    assert isinstance(grounding_chunks, list)

    # grounding_supports should link response segments to chunks
    grounding_supports = grounding.get("grounding_supports")
    assert grounding_supports, "Expected grounding_supports in grounding_metadata"
    assert isinstance(grounding_supports, list)


# Test: Google Search Grounding (Sync)
@pytest.mark.vcr
@pytest.mark.parametrize(
    "mode",
    ["sync", "stream"],
)
def test_google_search_grounding(memory_logger, mode):
    """Test that Google Search grounding metadata is captured in span output."""
    assert not memory_logger.pop()

    client = Client()
    start = time.time()

    if mode == "sync":
        response = client.models.generate_content(
            model=GROUNDING_MODEL,
            contents="What is the current population of Tokyo, Japan?",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=300,
            ),
        )
        text = response.text
    elif mode == "stream":
        stream = client.models.generate_content_stream(
            model=GROUNDING_MODEL,
            contents="What is the current population of Tokyo, Japan?",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=300,
            ),
        )
        text = ""
        for chunk in stream:
            if chunk.text:
                text += chunk.text

    end = time.time()

    # Verify response contains expected content
    assert text
    assert len(text) > 0

    # Verify logging
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == GROUNDING_MODEL
    assert "population" in str(span["input"]).lower() or "Tokyo" in str(span["input"])
    assert span["output"]
    _assert_metrics_are_valid(span["metrics"], start, end)

    # Verify grounding metadata is captured
    _assert_grounding_metadata(span["output"])


# Test: Google Search Grounding (Async)
@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["async", "async_stream"],
)
async def test_google_search_grounding_async(memory_logger, mode):
    """Test that Google Search grounding metadata is captured in async span output."""
    assert not memory_logger.pop()

    client = Client()
    start = time.time()

    if mode == "async":
        response = await client.aio.models.generate_content(
            model=GROUNDING_MODEL,
            contents="What is the current population of Tokyo, Japan?",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=300,
            ),
        )
        text = response.text
    elif mode == "async_stream":
        stream = await client.aio.models.generate_content_stream(
            model=GROUNDING_MODEL,
            contents="What is the current population of Tokyo, Japan?",
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=300,
            ),
        )
        text = ""
        async for chunk in stream:
            if chunk.text:
                text += chunk.text

    end = time.time()

    # Verify response contains expected content
    assert text
    assert len(text) > 0

    # Verify logging
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == GROUNDING_MODEL
    assert "population" in str(span["input"]).lower() or "Tokyo" in str(span["input"])
    assert span["output"]
    _assert_metrics_are_valid(span["metrics"], start, end)

    # Verify grounding metadata is captured
    _assert_grounding_metadata(span["output"])


def _find_spans_by_type(spans, span_type):
    return [span for span in spans if span["span_attributes"]["type"] == span_type]


def _find_span_by_name(spans, name):
    return next(span for span in spans if span["span_attributes"]["name"] == name)


def _interaction_function_tool():
    return interactions.Function(
        type="function",
        name="get_weather",
        description="Get the current weather for a location.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )


@_needs_interactions
@pytest.mark.vcr
def test_interactions_create_and_get(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    response = client.interactions.create(
        model=INTERACTIONS_MODEL,
        input="What is the capital of France?",
    )
    fetched = client.interactions.get(response.id, include_input=True)

    assert response.status == "completed"
    assert fetched.id == response.id
    assert fetched.status == "completed"

    spans = memory_logger.pop()
    create_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.LLM), "interactions.create")
    get_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.TASK), "interactions.get")

    assert create_span["metadata"]["model"] == INTERACTIONS_MODEL
    assert create_span["metadata"]["interaction_id"] == response.id
    assert create_span["output"]["status"] == "completed"
    assert "Paris" in create_span["output"]["text"]
    assert create_span["metrics"]["prompt_tokens"] > 0
    assert create_span["metrics"]["completion_tokens"] > 0

    assert get_span["input"]["id"] == response.id
    assert get_span["metadata"]["interaction_id"] == response.id
    assert get_span["output"]["status"] == "completed"
    assert "Paris" in get_span["output"]["text"]
    assert "France" in str(get_span["output"]["outputs"])


@_needs_interactions
@pytest.mark.vcr
def test_interactions_create_stream(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    events = list(
        client.interactions.create(
            model=INTERACTIONS_MODEL,
            input="Say hi in five words or less.",
            stream=True,
        )
    )

    assert events

    spans = memory_logger.pop()
    create_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.LLM), "interactions.create")

    assert create_span["metadata"]["model"] == INTERACTIONS_MODEL
    assert create_span["output"]["status"] == "completed"
    assert create_span["output"]["text"]
    assert "hi" in create_span["output"]["text"].lower()
    assert create_span["metrics"]["time_to_first_token"] >= 0
    assert "content.start" in create_span["metadata"]["stream_event_types"]
    assert "interaction.complete" in create_span["metadata"]["stream_event_types"]


@_needs_interactions
@pytest.mark.vcr
def test_interactions_tool_call_and_follow_up(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    tool = _interaction_function_tool()

    first_response = client.interactions.create(
        model=INTERACTIONS_MODEL,
        input="What is the weather like in Paris? Use the tool.",
        tools=[tool],
    )
    tool_call = next(output for output in first_response.outputs if output.type == "function_call")

    second_response = client.interactions.create(
        model=INTERACTIONS_MODEL,
        previous_interaction_id=first_response.id,
        input=interactions.FunctionResultContent(
            type="function_result",
            call_id=tool_call.id,
            name=tool_call.name,
            result={"forecast": "sunny"},
        ),
        tools=[tool],
    )

    assert first_response.status == "requires_action"
    assert second_response.status == "completed"
    assert "sunny" in second_response.outputs[-1].text.lower()

    spans = memory_logger.pop()
    llm_spans = _find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = _find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    first_span = next(span for span in llm_spans if span["metadata"]["interaction_id"] == first_response.id)
    second_span = next(span for span in llm_spans if span["metadata"]["interaction_id"] == second_response.id)
    tool_span = _find_span_by_name(tool_spans, "get_weather")

    assert first_span["output"]["status"] == "requires_action"
    assert second_span["metadata"]["previous_interaction_id"] == first_response.id
    assert second_span["output"]["status"] == "completed"
    assert "sunny" in second_span["output"]["text"].lower()

    assert tool_span["input"] == {"location": "Paris"}
    assert tool_span["span_parents"] == [first_span["span_id"]]


@_needs_interactions
@pytest.mark.vcr
def test_interactions_tool_span_stays_active_during_local_tool_work(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    tool = _interaction_function_tool()

    first_response = client.interactions.create(
        model=INTERACTIONS_MODEL,
        input="What is the weather like in Paris? Use the tool.",
        tools=[tool],
    )
    tool_call = next(output for output in first_response.outputs if output.type == "function_call")

    with logger.start_span(name="nested_tool_work", type=SpanTypeAttribute.TASK) as nested_tool_work:
        nested_tool_work.log(output={"forecast": "sunny"})

    second_response = client.interactions.create(
        model=INTERACTIONS_MODEL,
        previous_interaction_id=first_response.id,
        input=interactions.FunctionResultContent(
            type="function_result",
            call_id=tool_call.id,
            name=tool_call.name,
            result={"forecast": "sunny"},
        ),
        tools=[tool],
    )

    assert first_response.status == "requires_action"
    assert second_response.status == "completed"

    spans = memory_logger.pop()
    llm_spans = _find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = _find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    first_span = next(span for span in llm_spans if span["metadata"]["interaction_id"] == first_response.id)
    second_span = next(span for span in llm_spans if span["metadata"]["interaction_id"] == second_response.id)
    tool_span = _find_span_by_name(tool_spans, "get_weather")
    nested_span = _find_span_by_name(spans, "nested_tool_work")

    assert tool_span["span_parents"] == [first_span["span_id"]]
    assert nested_span["span_parents"] == [tool_span["span_id"]]
    assert tool_span["metrics"]["start"] <= nested_span["metrics"]["start"]
    assert tool_span["metrics"]["end"] >= nested_span["metrics"]["end"]
    assert second_span.get("span_parents") in (None, [])


@_needs_interactions
@pytest.mark.vcr
def test_interactions_delete(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    response = client.interactions.create(
        model=INTERACTIONS_MODEL,
        input="Reply with exactly ok.",
    )
    assert response.id

    create_spans = memory_logger.pop()
    assert create_spans

    delete_response = client.interactions.delete(response.id)
    assert delete_response == {}

    spans = memory_logger.pop()
    delete_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.TASK), "interactions.delete")

    assert delete_span["input"]["id"] == response.id
    assert delete_span["output"] == {}
    assert delete_span["metrics"]["duration"] >= 0


@_needs_interactions
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_interactions_async_round_trip(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    response = await client.aio.interactions.create(
        model=INTERACTIONS_MODEL,
        input="What is the capital of Italy?",
    )
    fetched = await client.aio.interactions.get(response.id, include_input=True)
    deleted = await client.aio.interactions.delete(response.id)

    assert response.status == "completed"
    assert fetched.id == response.id
    assert deleted == {}

    spans = memory_logger.pop()
    create_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.LLM), "interactions.create")
    get_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.TASK), "interactions.get")
    delete_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.TASK), "interactions.delete")

    assert create_span["metadata"]["model"] == INTERACTIONS_MODEL
    assert create_span["output"]["status"] == "completed"
    assert "Rome" in create_span["output"]["text"]

    assert get_span["input"]["id"] == response.id
    assert "Italy" in str(get_span["output"]["outputs"])

    assert delete_span["input"]["id"] == response.id
    assert delete_span["output"] == {}


@_needs_interactions
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_interactions_async_stream(memory_logger):
    assert not memory_logger.pop()

    client = Client()
    stream = await client.aio.interactions.create(
        model=INTERACTIONS_MODEL,
        input="Say hi shortly.",
        stream=True,
    )

    events = []
    async for event in stream:
        events.append(event)

    assert events

    spans = memory_logger.pop()
    create_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.LLM), "interactions.create")

    assert create_span["output"]["status"] == "completed"
    assert create_span["output"]["text"]
    assert "hi" in create_span["output"]["text"].lower()
    assert create_span["metrics"]["time_to_first_token"] >= 0
    assert "content.delta" in create_span["metadata"]["stream_event_types"]


class TestAutoInstrumentGoogleGenAI:
    """Tests for auto_instrument() with Google GenAI."""

    def test_auto_instrument_google_genai(self):
        """Test auto_instrument patches Google GenAI and creates spans."""
        verify_autoinstrument_script("test_auto_google_genai.py")
