import base64
import importlib
import inspect
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from braintrust import Attachment, logger
from braintrust.integrations.mistral import MistralIntegration, wrap_mistral
from braintrust.integrations.mistral.tracing import (
    _aggregate_completion_events,
    _chat_complete_async_wrapper,
    _chat_complete_wrapper,
    _conversations_start_wrapper,
    _normalize_mistral_multimodal_value,
    _ocr_process_wrapper,
)
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_spans_by_type, init_test_logger


pytest.importorskip("mistralai")

try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral

try:
    Chat = importlib.import_module("mistralai.client.chat").Chat
    Embeddings = importlib.import_module("mistralai.client.embeddings").Embeddings
    Fim = importlib.import_module("mistralai.client.fim").Fim
    Agents = importlib.import_module("mistralai.client.agents").Agents
    Conversations = importlib.import_module("mistralai.client.conversations").Conversations
    Ocr = importlib.import_module("mistralai.client.ocr").Ocr
    Transcriptions = importlib.import_module("mistralai.client.transcriptions").Transcriptions
    models = importlib.import_module("mistralai.client.models")
except ImportError:
    Chat = importlib.import_module("mistralai.chat").Chat
    Embeddings = importlib.import_module("mistralai.embeddings").Embeddings
    Fim = importlib.import_module("mistralai.fim").Fim
    Agents = importlib.import_module("mistralai.agents").Agents
    Conversations = importlib.import_module("mistralai.conversations").Conversations
    Ocr = importlib.import_module("mistralai.ocr").Ocr
    Transcriptions = importlib.import_module("mistralai.transcriptions").Transcriptions
    models = importlib.import_module("mistralai.models")

try:
    Speech = importlib.import_module("mistralai.client.speech").Speech
except ImportError:
    Speech = None


PROJECT_NAME = "test-mistral-sdk"
CHAT_MODEL = "mistral-small-latest"
AGENT_MODEL = CHAT_MODEL
EMBEDDING_MODEL = "mistral-embed"
FIM_MODEL = "codestral-latest"
OCR_MODEL = "mistral-ocr-latest"
TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
TEST_PDF_DATA_URL = (
    "data:application/pdf;base64,"
    "JVBERi0xLjAKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PmVuZG9iagoyIDAgb2JqCjw8L1R5cGUvUGFnZXMvS2lkc1szIDAgUl0vQ291bnQgMT4+ZW5kb2JqCjMgMCBvYmoKPDwvVHlwZS9QYWdlL01lZGlhQm94WzAgMCA2MTIgNzkyXT4+ZW5kb2JqCnhyZWYKMCA0CjAwMDAwMDAwMDAgNjU1MzUgZg0KMDAwMDAwMDAxMCAwMDAwMCBuDQowMDAwMDAwMDUzIDAwMDAwIG4NCjAwMDAwMDAxMDIgMDAwMDAgbg0KdHJhaWxlcgo8PC9TaXplIDQvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgoxNDkKJUVPRg=="
)
AUDIO_TRANSCRIPTION_MODEL = "voxtral-mini-2507"
SPEECH_MODEL = "voxtral-mini-tts-latest"
SPEECH_VOICE_ID = "en_paul_neutral"
TEST_AUDIO_FILE = Path(__file__).resolve().parents[2] / "fixtures" / "test_audio.wav"


def _transcription_file_payload(file_obj):
    return {
        "file_name": TEST_AUDIO_FILE.name,
        "content": file_obj,
        "content_type": "audio/wav",
    }


def _assert_speech_output_attachment(span):
    assert span["output"]["type"] == "audio"
    assert span["output"]["audio_size_bytes"] > 0
    assert span["output"]["mime_type"] == "audio/mpeg"
    attachment = span["output"]["file"]["file_data"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["filename"].startswith("generated_speech")
    assert attachment.reference["content_type"] == "audio/mpeg"


def _assert_transcription_complete_span(span, response_text, start, end, *, check_content_type=False):
    assert isinstance(span["input"]["file"], Attachment)
    assert span["input"]["file"].reference["filename"] == TEST_AUDIO_FILE.name
    if check_content_type:
        assert span["input"]["file"].reference["content_type"] == "audio/wav"
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == AUDIO_TRANSCRIPTION_MODEL
    assert span["output"] == response_text
    assert span["metrics"]["prompt_audio_seconds"] > 0
    assert span["metrics"]["tokens"] > 0
    assert_metrics_are_valid(span["metrics"], start, end)


def _assert_speech_complete_span(span, start, end):
    assert span["input"] == "Hello from Braintrust."
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == SPEECH_MODEL
    assert span["metadata"]["voice_id"] == SPEECH_VOICE_ID
    assert span["metadata"]["response_format"] == "mp3"
    _assert_speech_output_attachment(span)
    assert start <= span["metrics"]["start"] <= span["metrics"]["end"] <= end
    assert span["metrics"]["duration"] >= 0


def _assert_conversation_span(span, expected_input, start, end, *, expected_content, stream=False):
    assert span["input"] == expected_input
    assert span["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["metadata"]["conversation_id"]
    assert span["output"][0]["type"] == "message.output"
    assert expected_content in span["output"][0]["content"]
    if stream:
        assert span["metadata"]["stream"] == True
        assert span["metrics"]["time_to_first_token"] >= 0
    assert_metrics_are_valid(span["metrics"], start, end)


def _audio_method_refs():
    refs = {}
    for cls, methods in (
        (Transcriptions, ("complete", "complete_async", "stream", "stream_async")),
        (Speech, ("complete", "complete_async") if Speech is not None else ()),
    ):
        if cls is None:
            continue
        for method in methods:
            refs[(cls, method)] = inspect.getattr_static(cls, method)
    return refs


def _conversation_method_refs():
    refs = {}
    for method in (
        "start",
        "start_async",
        "start_stream",
        "start_stream_async",
        "append",
        "append_async",
        "append_stream",
        "append_stream_async",
        "restart",
        "restart_async",
        "restart_stream",
        "restart_stream_async",
    ):
        refs[(Conversations, method)] = inspect.getattr_static(Conversations, method)
    return refs


def _restore_method_refs(monkeypatch, refs):
    for (cls, method), original in refs.items():
        monkeypatch.setattr(cls, method, original)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _get_client():
    return Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))


def _weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _assert_ocr_metrics_are_valid(metrics, start, end):
    assert metrics["duration"] >= 0
    assert metrics["pages_processed"] >= 1
    assert metrics["doc_size_bytes"] > 0
    assert start <= metrics["start"] <= metrics["end"] <= end


@contextmanager
def _temporary_agent(client):
    manager = getattr(getattr(client, "beta", None), "agents", None)
    assert manager is not None, "Mistral beta.agents is required for agent tests"

    agent = manager.create(
        model=AGENT_MODEL,
        name=f"braintrust-test-agent-{int(time.time() * 1000)}",
        instructions="You are concise. Keep responses under five words.",
    )
    agent_id = getattr(agent, "id", None) or getattr(agent, "agent_id", None)
    assert agent_id, "Expected created agent to include an id"

    try:
        yield agent_id
    finally:
        manager.delete(agent_id=agent_id)


@pytest.mark.vcr
def test_wrap_mistral_chat_complete_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    assert "4" in str(response.choices[0].message.content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"] == [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert "4" in str(span["output"])
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_mistral_chat_complete_tool_spans(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "Use get_weather for Paris. Do not answer directly."}],
        tools=[_weather_tool()],
        tool_choice="any",
        max_tokens=100,
    )

    assert response.choices[0].message.tool_calls
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"

    spans = memory_logger.pop()
    assert len(find_spans_by_type(spans, SpanTypeAttribute.LLM)) == 1
    assert len(find_spans_by_type(spans, SpanTypeAttribute.TOOL)) == 1


@pytest.mark.vcr
def test_wrap_mistral_chat_stream_tool_spans(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with client.chat.stream(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "Use get_weather for Paris. Do not answer directly."}],
        tools=[_weather_tool()],
        tool_choice="any",
        max_tokens=100,
    ) as stream:
        chunks = list(stream)

    assert chunks
    spans = memory_logger.pop()
    assert len(find_spans_by_type(spans, SpanTypeAttribute.LLM)) == 1
    assert len(find_spans_by_type(spans, SpanTypeAttribute.TOOL)) == 1


@pytest.mark.vcr
def test_wrap_mistral_chat_stream_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    with client.chat.stream(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 5+5? Reply with just the number."}],
        max_tokens=10,
    ) as stream:
        chunks = list(stream)
    end = time.time()

    assert chunks
    streamed_text = "".join(
        choice.delta.content or ""
        for chunk in chunks
        for choice in (chunk.data.choices or [])
        if getattr(choice, "delta", None) is not None and isinstance(choice.delta.content, str)
    )
    assert "10" in streamed_text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["metadata"]["stream"] == True
    assert span["metrics"]["time_to_first_token"] >= 0
    assert "10" in str(span["output"])
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_chat_complete_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = await client.chat.complete_async(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 3+3? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    assert "6" in str(response.choices[0].message.content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert "6" in str(span["output"])
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_mistral_beta_conversations_start_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = client.beta.conversations.start(
        model=CHAT_MODEL,
        inputs="What is 2+2? Reply with just the number.",
    )
    end = time.time()

    assert response.outputs
    assert response.outputs[0].type == "message.output"
    assert "4" in str(response.outputs[0].content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_conversation_span(
        spans[0],
        "What is 2+2? Reply with just the number.",
        start,
        end,
        expected_content="4",
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_beta_conversations_append_async(memory_logger):
    assert not memory_logger.pop()

    client = _get_client()
    initial_response = client.beta.conversations.start(
        model=CHAT_MODEL,
        inputs="What is 1+1? Reply with just the number.",
    )

    wrapped_client = wrap_mistral(client)
    start = time.time()
    response = await wrapped_client.beta.conversations.append_async(
        conversation_id=initial_response.conversation_id,
        inputs="What is 2+3? Reply with just the number.",
    )
    end = time.time()

    assert response.outputs
    assert response.outputs[0].type == "message.output"
    assert "5" in str(response.outputs[0].content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_conversation_span(
        spans[0],
        "What is 2+3? Reply with just the number.",
        start,
        end,
        expected_content="5",
    )


@pytest.mark.vcr
def test_wrap_mistral_beta_conversations_start_stream_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    with client.beta.conversations.start_stream(
        model=CHAT_MODEL,
        inputs="What is 3+4? Reply with just the number.",
    ) as stream:
        events = list(stream)
    end = time.time()

    assert events

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_conversation_span(
        spans[0],
        "What is 3+4? Reply with just the number.",
        start,
        end,
        expected_content="7",
        stream=True,
    )


@pytest.mark.vcr
def test_wrap_mistral_beta_conversations_restart_sync(memory_logger):
    assert not memory_logger.pop()

    client = _get_client()
    initial_response = client.beta.conversations.start(
        model=CHAT_MODEL,
        inputs="What is 9+1? Reply with just the number.",
    )

    wrapped_client = wrap_mistral(client)
    start = time.time()
    response = wrapped_client.beta.conversations.restart(
        conversation_id=initial_response.conversation_id,
        from_entry_id=initial_response.outputs[0].id,
        inputs="What is 6+6? Reply with just the number.",
    )
    end = time.time()

    assert response.outputs
    assert response.outputs[0].type == "message.output"
    assert "12" in str(response.outputs[0].content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_conversation_span(
        spans[0],
        "What is 6+6? Reply with just the number.",
        start,
        end,
        expected_content="12",
    )


@pytest.mark.vcr
def test_wrap_mistral_agents_complete_tool_spans(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with _temporary_agent(client) as agent_id:
        response = client.agents.complete(
            agent_id=agent_id,
            messages=[{"role": "user", "content": "Use get_weather for Paris. Do not answer directly."}],
            tools=[_weather_tool()],
            tool_choice="any",
            max_tokens=100,
        )

    assert response.choices[0].message.tool_calls
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
    spans = memory_logger.pop()
    assert len(find_spans_by_type(spans, SpanTypeAttribute.LLM)) == 1
    assert len(find_spans_by_type(spans, SpanTypeAttribute.TOOL)) == 1


@pytest.mark.vcr
def test_wrap_mistral_agents_stream_tool_spans(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with _temporary_agent(client) as agent_id:
        with client.agents.stream(
            agent_id=agent_id,
            messages=[{"role": "user", "content": "Use get_weather for Paris. Do not answer directly."}],
            tools=[_weather_tool()],
            tool_choice="any",
            max_tokens=100,
        ) as stream:
            chunks = list(stream)

    assert chunks
    spans = memory_logger.pop()
    assert len(find_spans_by_type(spans, SpanTypeAttribute.LLM)) == 1
    assert len(find_spans_by_type(spans, SpanTypeAttribute.TOOL)) == 1


@pytest.mark.vcr
def test_wrap_mistral_agents_complete_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with _temporary_agent(client) as agent_id:
        start = time.time()
        response = client.agents.complete(
            agent_id=agent_id,
            messages=[{"role": "user", "content": "What is 8+1? Reply with just the number."}],
            max_tokens=10,
        )
        end = time.time()

    assert "9" in str(response.choices[0].message.content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"] == [{"role": "user", "content": "What is 8+1? Reply with just the number."}]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["agent_id"] == agent_id
    assert "9" in str(span["output"])
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_mistral_agents_stream_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with _temporary_agent(client) as agent_id:
        start = time.time()
        with client.agents.stream(
            agent_id=agent_id,
            messages=[{"role": "user", "content": "What is 6+5? Reply with just the number."}],
            max_tokens=10,
        ) as stream:
            chunks = list(stream)
        end = time.time()

    assert chunks
    streamed_text = "".join(
        choice.delta.content or ""
        for chunk in chunks
        for choice in (chunk.data.choices or [])
        if getattr(choice, "delta", None) is not None and isinstance(choice.delta.content, str)
    )
    assert "11" in streamed_text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["agent_id"] == agent_id
    assert span["metadata"]["stream"] == True
    assert span["metrics"]["time_to_first_token"] >= 0
    assert "11" in str(span["output"])
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_agents_complete_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with _temporary_agent(client) as agent_id:
        start = time.time()
        response = await client.agents.complete_async(
            agent_id=agent_id,
            messages=[{"role": "user", "content": "What is 7+2? Reply with just the number."}],
            max_tokens=10,
        )
        end = time.time()

    assert "9" in str(response.choices[0].message.content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["agent_id"] == agent_id
    assert "9" in str(span["output"])
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_agents_stream_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with _temporary_agent(client) as agent_id:
        start = time.time()
        stream = await client.agents.stream_async(
            agent_id=agent_id,
            messages=[{"role": "user", "content": "What is 4+8? Reply with just the number."}],
            max_tokens=10,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        end = time.time()

    assert chunks

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["agent_id"] == agent_id
    assert span["metadata"]["stream"] == True
    assert span["metrics"]["time_to_first_token"] >= 0
    assert span["output"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_mistral_ocr_process_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = client.ocr.process(
        model=OCR_MODEL,
        document={
            "type": "document_url",
            "document_url": TEST_PDF_DATA_URL,
            "document_name": "test.pdf",
        },
    )
    end = time.time()

    assert response.pages

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == OCR_MODEL
    assert span["metadata"]["document_type"] == "document_url"
    assert span["metadata"]["page_count"] == len(response.pages)
    assert span["output"]["pages"]
    file_data_value = span["input"]["document"]["file"]["file_data"]
    assert isinstance(file_data_value, Attachment)
    assert file_data_value.reference["type"] == "braintrust_attachment"
    assert file_data_value.reference["content_type"] == "application/pdf"
    assert file_data_value.reference["filename"] == "test.pdf"
    assert file_data_value.reference["key"]
    _assert_ocr_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_ocr_process_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = await client.ocr.process_async(
        model=OCR_MODEL,
        document={
            "type": "image_url",
            "image_url": TINY_PNG_DATA_URL,
        },
    )
    end = time.time()

    assert response.pages

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == OCR_MODEL
    assert span["metadata"]["document_type"] == "image_url"
    assert span["metadata"]["page_count"] == len(response.pages)
    assert span["output"]["pages"]
    image_url_value = span["input"]["document"]["image_url"]["url"]
    assert isinstance(image_url_value, Attachment)
    assert image_url_value.reference["type"] == "braintrust_attachment"
    assert image_url_value.reference["content_type"] == "image/png"
    assert image_url_value.reference["filename"] == "image.png"
    assert image_url_value.reference["key"]
    _assert_ocr_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_mistral_embeddings_create(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs="braintrust tracing",
    )
    end = time.time()

    assert response.data
    assert response.data[0].embedding

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"] == "braintrust tracing"
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == EMBEDDING_MODEL
    assert span["output"]["embeddings_count"] == 1
    assert span["output"]["embedding_length"] == len(response.data[0].embedding)
    assert span["metrics"]["prompt_tokens"] > 0
    assert span["metrics"]["tokens"] > 0
    assert start <= span["metrics"]["start"] <= span["metrics"]["end"] <= end


@pytest.mark.vcr
def test_wrap_mistral_fim_complete_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = client.fim.complete(
        model=FIM_MODEL,
        prompt="def add(a, b):\n    return ",
        suffix="\n\nprint(add(2, 3))",
        max_tokens=16,
    )
    end = time.time()

    assert response.choices
    assert response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"]["prompt"] == "def add(a, b):\n    return "
    assert span["input"]["suffix"] == "\n\nprint(add(2, 3))"
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == FIM_MODEL
    assert "return" in str(span["input"])
    assert span["output"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_mistral_fim_stream_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    with client.fim.stream(
        model=FIM_MODEL,
        prompt="def multiply(a, b):\n    return ",
        suffix="\n\nprint(multiply(3, 4))",
        max_tokens=16,
    ) as stream:
        chunks = list(stream)
    end = time.time()

    assert chunks
    streamed_text = "".join(
        choice.delta.content or ""
        for chunk in chunks
        for choice in (chunk.data.choices or [])
        if getattr(choice, "delta", None) is not None and isinstance(choice.delta.content, str)
    )
    assert streamed_text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == FIM_MODEL
    assert span["metadata"]["stream"] == True
    assert span["metrics"]["time_to_first_token"] >= 0
    assert span["output"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_fim_complete_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = await client.fim.complete_async(
        model=FIM_MODEL,
        prompt="def subtract(a, b):\n    return ",
        suffix="\n\nprint(subtract(5, 2))",
        max_tokens=16,
    )
    end = time.time()

    assert response.choices
    assert response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == FIM_MODEL
    assert span["output"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_fim_stream_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    stream = await client.fim.stream_async(
        model=FIM_MODEL,
        prompt="def divide(a, b):\n    return ",
        suffix="\n\nprint(divide(8, 2))",
        max_tokens=16,
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    end = time.time()

    assert chunks

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == FIM_MODEL
    assert span["metadata"]["stream"] == True
    assert span["metrics"]["time_to_first_token"] >= 0
    assert span["output"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_mistral_audio_transcriptions_complete_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with TEST_AUDIO_FILE.open("rb") as file_obj:
        start = time.time()
        response = client.audio.transcriptions.complete(
            model=AUDIO_TRANSCRIPTION_MODEL,
            file=_transcription_file_payload(file_obj),
        )
        end = time.time()

    assert response.text

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_transcription_complete_span(spans[0], response.text, start, end, check_content_type=True)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_audio_transcriptions_complete_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with TEST_AUDIO_FILE.open("rb") as file_obj:
        start = time.time()
        response = await client.audio.transcriptions.complete_async(
            model=AUDIO_TRANSCRIPTION_MODEL,
            file=_transcription_file_payload(file_obj),
        )
        end = time.time()

    assert response.text

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_transcription_complete_span(spans[0], response.text, start, end)


@pytest.mark.vcr
def test_wrap_mistral_audio_transcriptions_stream_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with TEST_AUDIO_FILE.open("rb") as file_obj:
        start = time.time()
        with client.audio.transcriptions.stream(
            model=AUDIO_TRANSCRIPTION_MODEL,
            file=_transcription_file_payload(file_obj),
        ) as stream:
            events = list(stream)
        end = time.time()

    assert events
    streamed_text = "".join(
        event.data.text
        for event in events
        if getattr(event, "event", None) == "transcription.text.delta"
        and isinstance(getattr(event.data, "text", None), str)
    )
    final_text = next(
        event.data.text
        for event in reversed(events)
        if getattr(event, "event", None) == "transcription.done" and isinstance(getattr(event.data, "text", None), str)
    )
    assert streamed_text == final_text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == AUDIO_TRANSCRIPTION_MODEL
    assert span["metadata"]["stream"] == True
    assert span["output"] == final_text
    assert span["metrics"]["prompt_audio_seconds"] > 0
    assert span["metrics"]["time_to_first_token"] >= 0
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_mistral_audio_speech_complete_sync(memory_logger):
    if Speech is None:
        pytest.skip("Mistral speech API is unavailable in this SDK version")

    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = client.audio.speech.complete(
        input="Hello from Braintrust.",
        model=SPEECH_MODEL,
        response_format="mp3",
        voice_id=SPEECH_VOICE_ID,
    )
    end = time.time()

    assert response.audio_data

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_speech_complete_span(spans[0], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_mistral_audio_speech_complete_async(memory_logger):
    if Speech is None:
        pytest.skip("Mistral speech API is unavailable in this SDK version")

    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = await client.audio.speech.complete_async(
        input="Hello from Braintrust.",
        model=SPEECH_MODEL,
        response_format="mp3",
        voice_id=SPEECH_VOICE_ID,
    )
    end = time.time()

    assert response.audio_data

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_speech_complete_span(spans[0], start, end)


@pytest.mark.vcr(cassette_name="test_wrap_mistral_audio_transcriptions_complete_sync")
def test_mistral_integration_setup_instruments_audio_transcriptions(memory_logger, monkeypatch):
    assert not memory_logger.pop()

    original_audio_methods = _audio_method_refs()

    assert MistralIntegration.setup()
    client = _get_client()
    try:
        with TEST_AUDIO_FILE.open("rb") as file_obj:
            start = time.time()
            response = client.audio.transcriptions.complete(
                model=AUDIO_TRANSCRIPTION_MODEL,
                file=_transcription_file_payload(file_obj),
            )
            end = time.time()
    finally:
        _restore_method_refs(monkeypatch, original_audio_methods)

    assert response.text

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_transcription_complete_span(spans[0], response.text, start, end)


@pytest.mark.vcr
def test_mistral_integration_setup_instruments_beta_conversations(memory_logger, monkeypatch):
    assert not memory_logger.pop()

    original_conversation_methods = _conversation_method_refs()

    assert MistralIntegration.setup()
    client = _get_client()
    try:
        start = time.time()
        response = client.beta.conversations.start(
            model=CHAT_MODEL,
            inputs="What is 4+4? Reply with just the number.",
        )
        end = time.time()
    finally:
        _restore_method_refs(monkeypatch, original_conversation_methods)

    assert response.outputs
    assert "8" in str(response.outputs[0].content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_conversation_span(
        spans[0],
        "What is 4+4? Reply with just the number.",
        start,
        end,
        expected_content="8",
    )


@pytest.mark.vcr
def test_mistral_integration_setup_creates_spans(memory_logger, monkeypatch):
    assert not memory_logger.pop()

    original_complete = inspect.getattr_static(Chat, "complete")
    original_complete_async = inspect.getattr_static(Chat, "complete_async")
    original_stream = inspect.getattr_static(Chat, "stream")
    original_stream_async = inspect.getattr_static(Chat, "stream_async")
    original_embeddings_create = inspect.getattr_static(Embeddings, "create")
    original_embeddings_create_async = inspect.getattr_static(Embeddings, "create_async")
    original_fim_complete = inspect.getattr_static(Fim, "complete")
    original_fim_complete_async = inspect.getattr_static(Fim, "complete_async")
    original_fim_stream = inspect.getattr_static(Fim, "stream")
    original_fim_stream_async = inspect.getattr_static(Fim, "stream_async")
    original_agents_complete = inspect.getattr_static(Agents, "complete")
    original_agents_complete_async = inspect.getattr_static(Agents, "complete_async")
    original_agents_stream = inspect.getattr_static(Agents, "stream")
    original_agents_stream_async = inspect.getattr_static(Agents, "stream_async")
    original_ocr_process = inspect.getattr_static(Ocr, "process")
    original_ocr_process_async = inspect.getattr_static(Ocr, "process_async")
    original_audio_methods = _audio_method_refs()

    assert MistralIntegration.setup()
    client = _get_client()
    start = time.time()
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    monkeypatch.setattr(Chat, "complete", original_complete)
    monkeypatch.setattr(Chat, "complete_async", original_complete_async)
    monkeypatch.setattr(Chat, "stream", original_stream)
    monkeypatch.setattr(Chat, "stream_async", original_stream_async)
    monkeypatch.setattr(Embeddings, "create", original_embeddings_create)
    monkeypatch.setattr(Embeddings, "create_async", original_embeddings_create_async)
    monkeypatch.setattr(Fim, "complete", original_fim_complete)
    monkeypatch.setattr(Fim, "complete_async", original_fim_complete_async)
    monkeypatch.setattr(Fim, "stream", original_fim_stream)
    monkeypatch.setattr(Fim, "stream_async", original_fim_stream_async)
    monkeypatch.setattr(Agents, "complete", original_agents_complete)
    monkeypatch.setattr(Agents, "complete_async", original_agents_complete_async)
    monkeypatch.setattr(Agents, "stream", original_agents_stream)
    monkeypatch.setattr(Agents, "stream_async", original_agents_stream_async)
    monkeypatch.setattr(Ocr, "process", original_ocr_process)
    monkeypatch.setattr(Ocr, "process_async", original_ocr_process_async)
    _restore_method_refs(monkeypatch, original_audio_methods)

    assert "4" in str(response.choices[0].message.content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert "4" in str(span["output"])
    assert_metrics_are_valid(span["metrics"], start, end)


def test_mistral_integration_setup_is_idempotent(monkeypatch):
    first_complete = inspect.getattr_static(Chat, "complete")
    first_complete_async = inspect.getattr_static(Chat, "complete_async")
    first_stream = inspect.getattr_static(Chat, "stream")
    first_stream_async = inspect.getattr_static(Chat, "stream_async")
    first_embeddings_create = inspect.getattr_static(Embeddings, "create")
    first_embeddings_create_async = inspect.getattr_static(Embeddings, "create_async")
    first_fim_complete = inspect.getattr_static(Fim, "complete")
    first_fim_complete_async = inspect.getattr_static(Fim, "complete_async")
    first_fim_stream = inspect.getattr_static(Fim, "stream")
    first_fim_stream_async = inspect.getattr_static(Fim, "stream_async")
    first_agents_complete = inspect.getattr_static(Agents, "complete")
    first_agents_complete_async = inspect.getattr_static(Agents, "complete_async")
    first_agents_stream = inspect.getattr_static(Agents, "stream")
    first_agents_stream_async = inspect.getattr_static(Agents, "stream_async")
    first_ocr_process = inspect.getattr_static(Ocr, "process")
    first_ocr_process_async = inspect.getattr_static(Ocr, "process_async")
    first_conversation_methods = _conversation_method_refs()
    first_audio_methods = _audio_method_refs()

    assert MistralIntegration.setup()
    patched_complete = inspect.getattr_static(Chat, "complete")
    patched_complete_async = inspect.getattr_static(Chat, "complete_async")
    patched_stream = inspect.getattr_static(Chat, "stream")
    patched_stream_async = inspect.getattr_static(Chat, "stream_async")
    patched_embeddings_create = inspect.getattr_static(Embeddings, "create")
    patched_embeddings_create_async = inspect.getattr_static(Embeddings, "create_async")
    patched_fim_complete = inspect.getattr_static(Fim, "complete")
    patched_fim_complete_async = inspect.getattr_static(Fim, "complete_async")
    patched_fim_stream = inspect.getattr_static(Fim, "stream")
    patched_fim_stream_async = inspect.getattr_static(Fim, "stream_async")
    patched_agents_complete = inspect.getattr_static(Agents, "complete")
    patched_agents_complete_async = inspect.getattr_static(Agents, "complete_async")
    patched_agents_stream = inspect.getattr_static(Agents, "stream")
    patched_agents_stream_async = inspect.getattr_static(Agents, "stream_async")
    patched_ocr_process = inspect.getattr_static(Ocr, "process")
    patched_ocr_process_async = inspect.getattr_static(Ocr, "process_async")
    patched_conversation_methods = _conversation_method_refs()
    patched_audio_methods = _audio_method_refs()

    assert MistralIntegration.setup()
    assert inspect.getattr_static(Chat, "complete") is patched_complete
    assert inspect.getattr_static(Chat, "complete_async") is patched_complete_async
    assert inspect.getattr_static(Chat, "stream") is patched_stream
    assert inspect.getattr_static(Chat, "stream_async") is patched_stream_async
    assert inspect.getattr_static(Embeddings, "create") is patched_embeddings_create
    assert inspect.getattr_static(Embeddings, "create_async") is patched_embeddings_create_async
    assert inspect.getattr_static(Fim, "complete") is patched_fim_complete
    assert inspect.getattr_static(Fim, "complete_async") is patched_fim_complete_async
    assert inspect.getattr_static(Fim, "stream") is patched_fim_stream
    assert inspect.getattr_static(Fim, "stream_async") is patched_fim_stream_async
    assert inspect.getattr_static(Agents, "complete") is patched_agents_complete
    assert inspect.getattr_static(Agents, "complete_async") is patched_agents_complete_async
    assert inspect.getattr_static(Agents, "stream") is patched_agents_stream
    assert inspect.getattr_static(Agents, "stream_async") is patched_agents_stream_async
    assert inspect.getattr_static(Ocr, "process") is patched_ocr_process
    assert inspect.getattr_static(Ocr, "process_async") is patched_ocr_process_async
    for key, method in patched_conversation_methods.items():
        assert inspect.getattr_static(*key) is method
    for key, method in patched_audio_methods.items():
        assert inspect.getattr_static(*key) is method

    monkeypatch.setattr(Chat, "complete", first_complete)
    monkeypatch.setattr(Chat, "complete_async", first_complete_async)
    monkeypatch.setattr(Chat, "stream", first_stream)
    monkeypatch.setattr(Chat, "stream_async", first_stream_async)
    monkeypatch.setattr(Embeddings, "create", first_embeddings_create)
    monkeypatch.setattr(Embeddings, "create_async", first_embeddings_create_async)
    monkeypatch.setattr(Fim, "complete", first_fim_complete)
    monkeypatch.setattr(Fim, "complete_async", first_fim_complete_async)
    monkeypatch.setattr(Fim, "stream", first_fim_stream)
    monkeypatch.setattr(Fim, "stream_async", first_fim_stream_async)
    monkeypatch.setattr(Agents, "complete", first_agents_complete)
    monkeypatch.setattr(Agents, "complete_async", first_agents_complete_async)
    monkeypatch.setattr(Agents, "stream", first_agents_stream)
    monkeypatch.setattr(Agents, "stream_async", first_agents_stream_async)
    monkeypatch.setattr(Ocr, "process", first_ocr_process)
    monkeypatch.setattr(Ocr, "process_async", first_ocr_process_async)
    _restore_method_refs(monkeypatch, first_conversation_methods)
    _restore_method_refs(monkeypatch, first_audio_methods)


def test_chat_complete_wrapper_logs_errors(memory_logger):
    assert not memory_logger.pop()

    def fail(*args, **kwargs):
        raise RuntimeError("sync boom")

    with pytest.raises(RuntimeError, match="sync boom"):
        _chat_complete_wrapper(
            fail,
            None,
            (),
            {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"] == [{"role": "user", "content": "hello"}]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert "sync boom" in span["error"]


@pytest.mark.asyncio
async def test_chat_complete_async_wrapper_logs_errors(memory_logger):
    assert not memory_logger.pop()

    async def fail(*args, **kwargs):
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        await _chat_complete_async_wrapper(
            fail,
            None,
            (),
            {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"] == [{"role": "user", "content": "hello"}]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert "async boom" in span["error"]


def test_normalize_mistral_multimodal_value_converts_image_url_data_uri_to_attachment():
    sanitized = _normalize_mistral_multimodal_value(
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        }
    )

    assert isinstance(sanitized["image_url"]["url"], Attachment)
    assert sanitized["image_url"]["url"].reference["content_type"] == "image/png"


def test_normalize_mistral_multimodal_value_converts_image_url_string_data_uri_to_attachment():
    sanitized = _normalize_mistral_multimodal_value(
        {
            "type": "image_url",
            "image_url": "data:image/png;base64,aGVsbG8=",
        }
    )

    assert isinstance(sanitized["image_url"]["url"], Attachment)
    assert sanitized["image_url"]["url"].reference["content_type"] == "image/png"


def test_normalize_mistral_multimodal_value_converts_document_url_data_uri_to_attachment():
    sanitized = _normalize_mistral_multimodal_value(
        {
            "type": "document_url",
            "document_url": TEST_PDF_DATA_URL,
            "document_name": "test.pdf",
        }
    )

    assert sanitized["type"] == "file"
    assert isinstance(sanitized["file"]["file_data"], Attachment)
    assert sanitized["file"]["file_data"].reference["content_type"] == "application/pdf"
    assert sanitized["file"]["filename"] == "test.pdf"


def test_normalize_mistral_multimodal_value_converts_large_base64_input_audio_to_attachment():
    sanitized = _normalize_mistral_multimodal_value(
        {
            "type": "input_audio",
            "input_audio": base64.b64encode(b"hello" * 16).decode("ascii"),
        }
    )

    assert isinstance(sanitized["input_audio"], Attachment)
    assert sanitized["input_audio"].reference["filename"] == "input_audio.bin"


def test_normalize_mistral_multimodal_value_leaves_non_base64_input_audio_unchanged():
    original = {"type": "input_audio", "input_audio": "not base64"}

    sanitized = _normalize_mistral_multimodal_value(original)

    assert sanitized == original


class _UnsupportedUsageResponse:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


@pytest.mark.parametrize(
    ("wrapper", "kwargs", "response", "missing_metric_keys"),
    [
        (
            _chat_complete_wrapper,
            {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": "hello"}],
            },
            _UnsupportedUsageResponse(usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
            {"tokens", "prompt_tokens", "completion_tokens"},
        ),
        (
            _conversations_start_wrapper,
            {
                "model": CHAT_MODEL,
                "inputs": [{"role": "user", "content": "hello"}],
            },
            _UnsupportedUsageResponse(usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}),
            {"tokens", "prompt_tokens", "completion_tokens"},
        ),
        (
            _ocr_process_wrapper,
            {
                "model": OCR_MODEL,
                "document": {"type": "document_url", "document_url": TEST_PDF_DATA_URL},
            },
            _UnsupportedUsageResponse(usage_info={"pages_processed": 1, "doc_size_bytes": 70}),
            {"pages_processed", "doc_size_bytes"},
        ),
    ],
)
def test_wrappers_ignore_usage_when_response_normalization_fails(
    memory_logger, wrapper, kwargs, response, missing_metric_keys
):
    assert not memory_logger.pop()

    result = wrapper(lambda *args, **kwargs: response, None, (), kwargs)

    assert result is response

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    for key in missing_metric_keys:
        assert key not in span["metrics"]
    assert span["metrics"]["duration"] >= 0


def test_aggregate_completion_events_merges_tool_calls_and_content():
    events = [
        models.CompletionEvent(
            data={
                "id": "cmpl_123",
                "model": CHAT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup_", "arguments": '{"city":"San'},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ),
        models.CompletionEvent(
            data={
                "id": "cmpl_123",
                "model": CHAT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": ' Francisco"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
        ),
    ]

    aggregated = _aggregate_completion_events(events)

    assert aggregated["id"] == "cmpl_123"
    assert aggregated["model"] == CHAT_MODEL
    assert aggregated["usage"]["total_tokens"] == 14
    assert aggregated["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = aggregated["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "lookup_weather"
    assert tool_call["function"]["arguments"] == '{"city":"San Francisco"}'


class TestAutoInstrumentMistral:
    def test_auto_instrument_mistral(self):
        verify_autoinstrument_script("test_auto_mistral.py")
