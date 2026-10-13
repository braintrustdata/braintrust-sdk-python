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
    models = importlib.import_module("mistralai.client.models")
except ImportError:
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
AUDIO_TRANSCRIPTION_RESPONSE_MODELS = {AUDIO_TRANSCRIPTION_MODEL, "voxtral-mini-latest"}
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


def _assert_transcription_complete_span(span, response_text, start, end):
    assert isinstance(span["input"]["file"], Attachment)
    assert span["input"]["file"].reference["filename"] == TEST_AUDIO_FILE.name
    assert span["input"]["file"].reference["content_type"] == "audio/wav"
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] in AUDIO_TRANSCRIPTION_RESPONSE_MODELS
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


def _assert_conversation_tool_span(span):
    assert span["span_attributes"]["type"] == SpanTypeAttribute.TOOL
    assert "21 * 2" in span["input"].get("code", "")
    assert span["output"].get("stdout") == "42\n" or span["output"].get("code_output") == "42\n"
    assert span["metadata"]["tool_type"] == "tool.execution"


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


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _get_client():
    return Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))


@pytest.fixture
def mistral_client(request, monkeypatch):
    if request.param == "setup":
        for composite in MistralIntegration.patchers:
            for patcher in composite.sub_patchers:
                if not patcher.applies(None, None):
                    continue
                module = patcher.resolve_root(None, None)
                class_name, method_name = patcher.target_path.split(".")
                cls = getattr(module, class_name)
                method = inspect.getattr_static(cls, method_name)
                monkeypatch.setattr(cls, method_name, method)
                # wrapt forwards patch markers to the original callable; restore
                # those too so another setup case can instrument it again.
                marker = patcher.patch_marker_attr()
                monkeypatch.setattr(method, marker, getattr(method, marker, False), raising=False)
        assert MistralIntegration.setup()
        return _get_client()
    return wrap_mistral(_get_client())


def _completion_params(method):
    if "logprobs" not in inspect.signature(method).parameters:
        return {}
    return {
        "logprobs": True,
        "top_logprobs": 2,
        "prompt_logprobs": True,
        "top_prompt_logprobs": 2,
        "min_tokens": 1,
        "repetition_penalty": 1.1,
        "top_k": 20,
    }


def _assert_completion_params(span, params):
    for key, value in params.items():
        assert span["metadata"][key] == value


def _assert_completion_logprobs(span, response):
    for key in ("logprobs", "prompt_logprobs"):
        expected = getattr(response.choices[0], key)
        assert hasattr(expected, "content") and expected.content, (
            f"Expected populated {key}; re-record this cassette with Mistral 3.2.0."
        )
        assert span["output"][0][key] == expected.model_dump(mode="python", by_alias=True)


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
    params = _completion_params(client.chat.complete)
    start = time.time()
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
        **params,
    )
    end = time.time()

    assert "4" in str(response.choices[0].message.content)
    assert response.usage.prompt_tokens_details["cached_tokens"] == 0

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "mistral-auto"
    assert span["input"] == [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == CHAT_MODEL
    _assert_completion_params(span, params)
    if params:
        _assert_completion_logprobs(span, response)
    assert "4" in str(span["output"])
    assert span["metrics"]["prompt_cached_tokens"] == 0
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
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    assert len(llm_spans) == 1
    span = llm_spans[0]
    service_tier = getattr(response.usage, "service_tier", None)
    if service_tier is not None:
        assert span["metadata"]["usage_service_tier"] == service_tier
    else:
        assert "usage_service_tier" not in span["metadata"]
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
    params = _completion_params(client.chat.stream)
    start = time.time()
    with client.chat.stream(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 5+5? Reply with just the number."}],
        max_tokens=10,
        **params,
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
    _assert_completion_params(span, params)
    if params:
        for key in ("logprobs", "prompt_logprobs"):
            expected_content = [
                token.model_dump(mode="python", by_alias=True)
                for chunk in chunks
                for choice in chunk.data.choices
                if hasattr(logprobs := getattr(choice, key), "content")
                for token in logprobs.content
            ]
            assert expected_content, f"Expected populated {key}; re-record this cassette with Mistral 3.2.0."
            assert span["output"][0][key]["content"] == expected_content
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


@pytest.mark.vcr(cassette_name="test_wrap_mistral_beta_conversations_start_sync")
@pytest.mark.parametrize("mistral_client", ["wrap", "setup"], indirect=True)
def test_mistral_beta_conversations_start_sync(memory_logger, mistral_client):
    assert not memory_logger.pop()

    start = time.time()
    response = mistral_client.beta.conversations.start(
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
def test_wrap_mistral_beta_conversations_start_tool_spans(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    response = client.beta.conversations.start(
        model=CHAT_MODEL,
        inputs="Use the code interpreter to calculate 21 * 2. Return only the result.",
        tools=[{"type": "code_interpreter"}],
    )

    assert response.outputs
    assert any(output.type == "tool.execution" for output in response.outputs)

    spans = memory_logger.pop()
    task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)
    assert len(task_spans) == 1
    assert len(tool_spans) == 1
    _assert_conversation_tool_span(tool_spans[0])


@pytest.mark.vcr
def test_wrap_mistral_beta_conversations_start_stream_tool_spans(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    with client.beta.conversations.start_stream(
        model=CHAT_MODEL,
        inputs="Use the code interpreter to calculate 21 * 2. Return only the result.",
        tools=[{"type": "code_interpreter"}],
    ) as stream:
        events = list(stream)

    assert events
    assert any(getattr(event, "event", None) == "tool.execution.done" for event in events)

    spans = memory_logger.pop()
    task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)
    assert len(task_spans) == 1
    assert len(tool_spans) == 1
    _assert_conversation_tool_span(tool_spans[0])


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
    usage = next(chunk.data.usage for chunk in reversed(chunks) if getattr(chunk.data, "usage", None) is not None)
    assert usage.prompt_tokens_details["cached_tokens"] == 80

    spans = memory_logger.pop()
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    assert len(llm_spans) == 1
    assert llm_spans[0]["metrics"]["prompt_cached_tokens"] == 80
    assert len(find_spans_by_type(spans, SpanTypeAttribute.TOOL)) == 1


@pytest.mark.vcr
def test_wrap_mistral_agents_complete_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    params = _completion_params(client.agents.complete)
    with _temporary_agent(client) as agent_id:
        start = time.time()
        response = client.agents.complete(
            agent_id=agent_id,
            messages=[{"role": "user", "content": "What is 8+1? Reply with just the number."}],
            max_tokens=10,
            **params,
        )
        end = time.time()

    assert "9" in str(response.choices[0].message.content)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"] == [{"role": "user", "content": "What is 8+1? Reply with just the number."}]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["agent_id"] == agent_id
    _assert_completion_params(span, params)
    if params:
        _assert_completion_logprobs(span, response)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mistral_client", "method_name"),
    [
        pytest.param(
            "wrap",
            "complete",
            id="sync-wrap",
            marks=pytest.mark.vcr(cassette_name="test_wrap_mistral_audio_transcriptions_complete_sync"),
        ),
        pytest.param(
            "setup",
            "complete",
            id="sync-setup",
            marks=pytest.mark.vcr(cassette_name="test_wrap_mistral_audio_transcriptions_complete_sync"),
        ),
        pytest.param(
            "wrap",
            "complete_async",
            id="async-wrap",
            marks=pytest.mark.vcr(cassette_name="test_wrap_mistral_audio_transcriptions_complete_async"),
        ),
    ],
    indirect=["mistral_client"],
)
async def test_mistral_audio_transcriptions_complete(memory_logger, mistral_client, method_name):
    assert not memory_logger.pop()

    with TEST_AUDIO_FILE.open("rb") as file_obj:
        start = time.time()
        response = getattr(mistral_client.audio.transcriptions, method_name)(
            model=AUDIO_TRANSCRIPTION_MODEL,
            file=_transcription_file_payload(file_obj),
        )
        if inspect.isawaitable(response):
            response = await response
        end = time.time()

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
    assert span["metadata"]["model"] in AUDIO_TRANSCRIPTION_RESPONSE_MODELS
    assert span["metadata"]["stream"] == True
    assert span["output"] == final_text
    assert span["metrics"]["prompt_audio_seconds"] > 0
    if final_text:
        assert span["metrics"]["time_to_first_token"] >= 0
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name",
    [
        pytest.param(
            "complete", id="sync", marks=pytest.mark.vcr(cassette_name="test_wrap_mistral_audio_speech_complete_sync")
        ),
        pytest.param(
            "complete_async",
            id="async",
            marks=pytest.mark.vcr(cassette_name="test_wrap_mistral_audio_speech_complete_async"),
        ),
    ],
)
async def test_wrap_mistral_audio_speech_complete(memory_logger, method_name):
    if Speech is None:
        pytest.skip("Mistral speech API is unavailable in this SDK version")

    assert not memory_logger.pop()

    client = wrap_mistral(_get_client())
    start = time.time()
    response = getattr(client.audio.speech, method_name)(
        input="Hello from Braintrust.",
        model=SPEECH_MODEL,
        response_format="mp3",
        voice_id=SPEECH_VOICE_ID,
    )
    if inspect.isawaitable(response):
        response = await response
    end = time.time()

    assert response.audio_data

    spans = memory_logger.pop()
    assert len(spans) == 1
    _assert_speech_complete_span(spans[0], start, end)


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


@pytest.mark.parametrize(
    ("value", "attachment_path", "expected_reference"),
    [
        pytest.param(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ("image_url", "url"),
            {"content_type": "image/png"},
            id="image_url_data_uri_to_attachment",
        ),
        pytest.param(
            {"type": "input_audio", "input_audio": base64.b64encode(b"hello" * 16).decode("ascii")},
            ("input_audio",),
            {"filename": "input_audio.bin"},
            id="large_base64_input_audio_to_attachment",
        ),
        # attachment_path=None means the value should come back unchanged.
        pytest.param(
            {"type": "input_audio", "input_audio": "not base64"},
            None,
            None,
            id="non_base64_input_audio_unchanged",
        ),
    ],
)
def test_normalize_mistral_multimodal_value(value, attachment_path, expected_reference):
    sanitized = _normalize_mistral_multimodal_value(value)

    if attachment_path is None:
        assert sanitized == value
        return

    attachment = sanitized
    for key in attachment_path:
        attachment = attachment[key]
    assert isinstance(attachment, Attachment)
    for key, expected in expected_reference.items():
        assert attachment.reference[key] == expected


class _PlainResponse:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


@pytest.mark.parametrize(
    ("wrapper", "kwargs", "response", "expected_metric_keys"),
    [
        (
            _chat_complete_wrapper,
            {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": "hello"}],
            },
            _PlainResponse(usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
            {"tokens", "prompt_tokens", "completion_tokens"},
        ),
        (
            _conversations_start_wrapper,
            {
                "model": CHAT_MODEL,
                "inputs": [{"role": "user", "content": "hello"}],
            },
            _PlainResponse(usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}),
            {"tokens", "prompt_tokens", "completion_tokens"},
        ),
        (
            _ocr_process_wrapper,
            {
                "model": OCR_MODEL,
                "document": {"type": "document_url", "document_url": TEST_PDF_DATA_URL},
            },
            _PlainResponse(usage_info={"pages_processed": 1, "doc_size_bytes": 70}),
            {"pages_processed", "doc_size_bytes"},
        ),
    ],
)
def test_wrappers_read_usage_from_plain_python_responses(
    memory_logger, wrapper, kwargs, response, expected_metric_keys
):
    assert not memory_logger.pop()

    result = wrapper(lambda *args, **kwargs: response, None, (), kwargs)

    assert result is response

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    for key in expected_metric_keys:
        assert key in span["metrics"]
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
    assert getattr(aggregated["usage"], "total_tokens", None) == 14
    assert aggregated["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = aggregated["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "lookup_weather"
    assert tool_call["function"]["arguments"] == '{"city":"San Francisco"}'


def test_auto_instrument_mistral():
    verify_autoinstrument_script("test_auto_mistral.py")
