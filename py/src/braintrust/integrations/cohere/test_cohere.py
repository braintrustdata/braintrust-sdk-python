"""Tests for the Cohere integration."""

import asyncio
import inspect
import os
import time
from pathlib import Path

import pytest
from braintrust import Attachment, logger
from braintrust.integrations.cohere import CohereIntegration, wrap_cohere
from braintrust.integrations.cohere.patchers import (
    AsyncChatPatcher,
    AsyncChatStreamPatcher,
    AsyncEmbedPatcher,
    AsyncRerankPatcher,
    AsyncTranscriptionsCreatePatcher,
    AsyncV2ChatPatcher,
    AsyncV2ChatStreamPatcher,
    AsyncV2EmbedPatcher,
    AsyncV2RerankPatcher,
    ChatPatcher,
    ChatStreamPatcher,
    EmbedPatcher,
    RerankPatcher,
    TranscriptionsCreatePatcher,
    V2ChatPatcher,
    V2ChatStreamPatcher,
    V2EmbedPatcher,
    V2RerankPatcher,
)
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_spans_by_type, init_test_logger


pytest.importorskip("cohere")

import cohere  # noqa: E402


PROJECT_NAME = "test-cohere-sdk"
CHAT_MODEL = "command-a-03-2025"
EMBED_MODEL = "embed-english-v3.0"
RERANK_MODEL = "rerank-english-v3.0"
TRANSCRIBE_MODEL = "cohere-transcribe-03-2026"
TEST_AUDIO_FILE = Path(__file__).resolve().parents[2] / "fixtures" / "test_audio.wav"
COHERE_API_KEY = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY") or "co-test-dummy-api-key-for-vcr-tests"


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _supports_client(name: str, *methods: str) -> bool:
    client = getattr(cohere, name, None)
    return client is not None and all(hasattr(client, method) for method in methods)


def _v1_client():
    return cohere.Client(api_key=COHERE_API_KEY)


def _v1_async_client():
    return cohere.AsyncClient(api_key=COHERE_API_KEY)


def _v2_client(*, require_methods: tuple[str, ...] = ("chat",)):
    if not _supports_client("ClientV2", *require_methods):
        pytest.skip(f"Cohere ClientV2 missing required methods: {', '.join(require_methods)}")
    return cohere.ClientV2(api_key=COHERE_API_KEY)


def _v2_async_client(*, require_methods: tuple[str, ...] = ("chat",)):
    if not _supports_client("AsyncClientV2", *require_methods):
        pytest.skip(f"Cohere AsyncClientV2 missing required methods: {', '.join(require_methods)}")
    return cohere.AsyncClientV2(api_key=COHERE_API_KEY)


# A restoration context that snapshots and restores the patched methods on the
# Cohere classes. Integration setup mutates global class state; tests that
# exercise ``setup()`` must not leak wrappers into other tests.
@pytest.fixture
def clean_cohere_methods():
    from cohere.base_client import AsyncBaseCohere, BaseCohere

    targets = [
        (BaseCohere, "chat"),
        (BaseCohere, "chat_stream"),
        (BaseCohere, "embed"),
        (BaseCohere, "rerank"),
        (AsyncBaseCohere, "chat"),
        (AsyncBaseCohere, "chat_stream"),
        (AsyncBaseCohere, "embed"),
        (AsyncBaseCohere, "rerank"),
    ]
    try:
        from cohere.v2.client import AsyncV2Client, V2Client
    except ImportError:
        pass
    else:
        for cls in (V2Client, AsyncV2Client):
            for attr in ("chat", "chat_stream", "embed", "rerank"):
                if hasattr(cls, attr):
                    targets.append((cls, attr))

    try:
        from cohere.audio.transcriptions.client import AsyncTranscriptionsClient, TranscriptionsClient
    except ImportError:
        pass
    else:
        for cls in (TranscriptionsClient, AsyncTranscriptionsClient):
            if hasattr(cls, "create"):
                targets.append((cls, "create"))

    originals = [(cls, attr, inspect.getattr_static(cls, attr)) for cls, attr in targets]
    # Also capture patch markers so we can clear them.
    marker_attrs = set()
    for patcher in (
        ChatPatcher,
        ChatStreamPatcher,
        EmbedPatcher,
        RerankPatcher,
        AsyncChatPatcher,
        AsyncChatStreamPatcher,
        AsyncEmbedPatcher,
        AsyncRerankPatcher,
        V2ChatPatcher,
        V2ChatStreamPatcher,
        V2EmbedPatcher,
        V2RerankPatcher,
        AsyncV2ChatPatcher,
        AsyncV2ChatStreamPatcher,
        AsyncV2EmbedPatcher,
        AsyncV2RerankPatcher,
        TranscriptionsCreatePatcher,
        AsyncTranscriptionsCreatePatcher,
    ):
        marker_attrs.add(patcher.patch_marker_attr())

    try:
        yield
    finally:
        for cls, attr, original in originals:
            setattr(cls, attr, original)
        # Clear any patch markers that setup() may have added. wrapt can forward
        # setattr from a FunctionWrapper onto the wrapped function, so the
        # restored original may still carry the marker; clear it from both
        # class and restored function.
        for cls, _, original in originals:
            for marker in marker_attrs:
                if hasattr(cls, marker):
                    try:
                        delattr(cls, marker)
                    except AttributeError:
                        pass
                if hasattr(original, marker):
                    try:
                        delattr(original, marker)
                    except AttributeError:
                        pass


# ---------------------------------------------------------------------------
# Unit / local tests (no network)
# ---------------------------------------------------------------------------


def test_wrap_cohere_returns_unsupported_unchanged(caplog):
    invalid = object()
    assert wrap_cohere(invalid) is invalid

    invalid_dict = {"foo": "bar"}
    assert wrap_cohere(invalid_dict) is invalid_dict


def test_wrap_cohere_is_idempotent():
    client = _v1_client()
    wrapped_once = wrap_cohere(client)
    wrapped_twice = wrap_cohere(client)
    assert wrapped_once is client
    assert wrapped_twice is client
    assert getattr(client, "__braintrust_cohere_traced__", False) is True


def test_cohere_integration_available_patchers_ids():
    ids = CohereIntegration.available_patchers()
    assert set(ids) == {
        "cohere.chat.all",
        "cohere.chat_stream.all",
        "cohere.embed.all",
        "cohere.rerank.all",
        "cohere.audio.transcriptions.all",
    }


def test_audio_transcriptions_patchers_target_sdk_surface():
    """The audio transcription patchers must point at the Cohere SDK classes.

    Regression guard for https://github.com/braintrustdata/braintrust-sdk-python/issues/327:
    we must instrument both ``TranscriptionsClient.create`` and
    ``AsyncTranscriptionsClient.create`` on the ``cohere.audio.transcriptions``
    surface introduced in cohere>=6.1.0.
    """
    try:
        import cohere.audio.transcriptions.client as transcriptions_module
    except ImportError:
        pytest.skip("cohere SDK does not expose audio.transcriptions")

    assert TranscriptionsCreatePatcher.target_module == "cohere.audio.transcriptions.client"
    assert TranscriptionsCreatePatcher.target_path == "TranscriptionsClient.create"
    assert AsyncTranscriptionsCreatePatcher.target_module == "cohere.audio.transcriptions.client"
    assert AsyncTranscriptionsCreatePatcher.target_path == "AsyncTranscriptionsClient.create"

    assert hasattr(transcriptions_module.TranscriptionsClient, "create")
    assert hasattr(transcriptions_module.AsyncTranscriptionsClient, "create")


# ---------------------------------------------------------------------------
# VCR-backed integration tests
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_wrap_cohere_chat_v2_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_cohere(_v2_client(require_methods=("chat",)))

    start = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_tokens=10,
    )
    end = time.time()

    # Provider behavior preserved.
    assert response.message.role == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.chat"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["metadata"]["max_tokens"] == 10
    assert span["input"] == [{"role": "user", "content": "Say hi in one word."}]

    # Output is the normalized v2 message object.
    assert isinstance(span["output"], dict)
    assert span["output"]["role"] == "assistant"

    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_cohere_chat_v2_tool_call_spans(memory_logger):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("v2 tool-call cassette is recorded for the latest Cohere SDK")

    assert not memory_logger.pop()
    client = wrap_cohere(_v2_client(require_methods=("chat",)))

    response = client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "Use the get_weather tool for Paris."}],
        tools=[
            {
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
        ],
        tool_choice="REQUIRED",
        max_tokens=64,
    )

    tool_calls = response.message.tool_calls
    assert tool_calls
    assert tool_calls[0].function.name == "get_weather"

    spans = memory_logger.pop()
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    assert len(llm_spans) == 1
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]
    assert tool_span["span_attributes"]["name"] == "tool: get_weather"
    assert tool_span["span_parents"] == [llm_spans[0]["span_id"]]
    assert tool_span["metadata"]["tool_call_id"] == tool_calls[0].id
    assert tool_span["metadata"]["tool_type"] == "function"
    assert "Paris" in str(tool_span["input"])


@pytest.mark.vcr
def test_wrap_cohere_chat_v1_tool_call_spans(memory_logger):
    if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") != "latest":
        pytest.skip("v1 tool-call cassette is recorded for the latest Cohere SDK")

    assert not memory_logger.pop()
    client = wrap_cohere(_v1_client())

    response = client.chat(
        model=CHAT_MODEL,
        message="Use the get_weather tool for Paris.",
        tools=[
            {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameter_definitions": {"city": {"description": "City name", "type": "str", "required": True}},
            }
        ],
        force_single_step=True,
        max_tokens=64,
    )

    tool_calls = response.tool_calls
    assert tool_calls
    assert tool_calls[0].name == "get_weather"

    spans = memory_logger.pop()
    llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
    tool_spans = find_spans_by_type(spans, SpanTypeAttribute.TOOL)

    assert len(llm_spans) == 1
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]
    assert tool_span["span_attributes"]["name"] == "tool: get_weather"
    assert tool_span["span_parents"] == [llm_spans[0]["span_id"]]
    assert "Paris" in str(tool_span["input"])


@pytest.mark.vcr
def test_wrap_cohere_chat_v1_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_cohere(_v1_client())

    start = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        message="Say hi in one word.",
        max_tokens=10,
    )
    end = time.time()

    assert isinstance(response.text, str)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.chat"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["input"] == "Say hi in one word."
    assert isinstance(span["output"], str)
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_cohere_embed_v1_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_cohere(_v1_client())

    start = time.time()
    response = client.embed(
        texts=["braintrust tracing"],
        model=EMBED_MODEL,
        input_type="search_document",
    )
    end = time.time()

    # v1 embed returns a plain list-of-lists.
    assert isinstance(response.embeddings, list)
    assert len(response.embeddings) == 1
    assert isinstance(response.embeddings[0], list)
    assert response.embeddings[0]

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.embed"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == EMBED_MODEL
    assert span["metadata"]["input_type"] == "search_document"
    assert span["input"] == ["braintrust tracing"]

    assert span["output"]["embedding_count"] == 1
    assert span["output"]["embedding_length"] > 0

    metrics = span["metrics"]
    assert metrics["duration"] >= 0
    assert metrics["start"] <= metrics["end"]
    assert start <= metrics["start"] <= metrics["end"] <= end


@pytest.mark.vcr
def test_wrap_cohere_embed_v2_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_cohere(_v2_client(require_methods=("embed",)))

    start = time.time()
    response = client.embed(
        texts=["braintrust tracing"],
        model=EMBED_MODEL,
        input_type="search_document",
        embedding_types=["float"],
    )
    end = time.time()

    # Provider behavior preserved.
    assert hasattr(response, "embeddings")

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.embed"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == EMBED_MODEL
    assert span["metadata"]["input_type"] == "search_document"
    assert span["input"] == ["braintrust tracing"]

    assert span["output"]["embedding_count"] == 1
    assert span["output"]["embedding_length"] > 0

    metrics = span["metrics"]
    assert metrics["duration"] >= 0
    assert metrics["start"] <= metrics["end"]
    assert start <= metrics["start"] <= metrics["end"] <= end


@pytest.mark.vcr
def test_wrap_cohere_rerank_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_cohere(_v1_client())

    response = client.rerank(
        model=RERANK_MODEL,
        query="capital of france",
        documents=["Paris is in France", "Vienna is in Austria"],
        top_n=2,
    )
    assert len(response.results) == 2

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.rerank"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == RERANK_MODEL
    assert span["metadata"]["top_n"] == 2
    assert span["metadata"]["document_count"] == 2

    assert span["input"] == {
        "query": "capital of france",
        "documents": ["Paris is in France", "Vienna is in Austria"],
    }

    assert isinstance(span["output"], list)
    assert len(span["output"]) == 2
    assert span["output"][0]["index"] in (0, 1)
    assert isinstance(span["output"][0]["relevance_score"], float)

    # Search units should show up as a dedicated metric.
    assert span["metrics"].get("search_units") == 1


@pytest.mark.vcr
def test_wrap_cohere_chat_stream_v1_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_cohere(_v1_client())

    start = time.time()
    events = []
    for event in client.chat_stream(
        model=CHAT_MODEL,
        message="Say hi in one word.",
        max_tokens=8,
    ):
        events.append(event)
    end = time.time()

    assert events
    event_types = [
        event.get("event_type") if isinstance(event, dict) else getattr(event, "event_type", None) for event in events
    ]
    assert "stream-start" in event_types
    assert "stream-end" in event_types
    assert "text-generation" in event_types

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.chat_stream"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["input"] == "Say hi in one word."
    assert span["metadata"].get("finish_reason") in {"COMPLETE", "MAX_TOKENS", "STOP_SEQUENCE"}
    assert isinstance(span["output"], str) and span["output"]

    metrics = span["metrics"]
    assert metrics["start"] <= metrics["end"]
    assert start <= metrics["start"] <= metrics["end"] <= end
    assert metrics.get("prompt_tokens", 0) > 0
    assert metrics.get("completion_tokens", 0) > 0


@pytest.mark.vcr
def test_wrap_cohere_chat_stream_v2_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_cohere(_v2_client(require_methods=("chat_stream",)))

    start = time.time()
    events = []
    for event in client.chat_stream(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_tokens=8,
    ):
        events.append(event)
    end = time.time()

    assert events  # provider still yielded events

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.chat_stream"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["metadata"].get("finish_reason") in {"COMPLETE", "MAX_TOKENS", "STOP_SEQUENCE"}

    output = span["output"]
    # Either a v2-shaped dict or a v1-shaped string — both are acceptable
    # aggregations as long as we captured *some* content.
    if isinstance(output, dict):
        content = output.get("content")
        assert isinstance(content, str) and content
    else:
        assert isinstance(output, str) and output

    metrics = span["metrics"]
    assert metrics["start"] <= metrics["end"]
    assert start <= metrics["start"] <= metrics["end"] <= end
    assert metrics.get("prompt_tokens", 0) > 0
    assert metrics.get("completion_tokens", 0) > 0


@pytest.mark.vcr
def test_wrap_cohere_chat_v1_async(memory_logger):
    assert not memory_logger.pop()

    async def _run():
        client = wrap_cohere(_v1_async_client())
        return await client.chat(
            model=CHAT_MODEL,
            message="Say hi in one word.",
            max_tokens=10,
        )

    response = asyncio.run(_run())
    assert isinstance(response.text, str)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.chat"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert isinstance(span["output"], str)
    assert span["metrics"].get("prompt_tokens", 0) > 0


@pytest.mark.vcr
def test_cohere_integration_setup_patches_v2_chat(memory_logger, clean_cohere_methods):
    """CohereIntegration.setup() patches the global Cohere classes.

    Verifies setup is idempotent and that un-wrapped client instances
    created after setup() emit spans automatically.
    """
    assert not memory_logger.pop()

    assert CohereIntegration.setup() is True
    # Second call is a no-op but still reports success.
    assert CohereIntegration.setup() is True

    use_v2 = _supports_client("ClientV2", "chat")
    client = _v2_client(require_methods=("chat",)) if use_v2 else _v1_client()  # NOT manually wrapped
    if use_v2:
        response = client.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
        )
        assert response.message.role == "assistant"
    else:
        response = client.chat(
            model=CHAT_MODEL,
            message="Say hi in one word.",
            max_tokens=10,
        )
        assert isinstance(response.text, str)

    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["span_attributes"]["name"] == "cohere.chat"
    assert spans[0]["metadata"]["provider"] == "cohere"
    assert spans[0]["metadata"]["model"] == CHAT_MODEL


@pytest.mark.vcr
def test_wrap_cohere_audio_transcription_sync(memory_logger):
    pytest.importorskip("cohere.audio.transcriptions.client")
    assert not memory_logger.pop()

    client = wrap_cohere(_v1_client())

    start = time.time()
    with open(TEST_AUDIO_FILE, "rb") as file_obj:
        response = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            language="en",
            file=(TEST_AUDIO_FILE.name, file_obj, "audio/wav"),
            temperature=0.0,
        )
    end = time.time()

    # Provider behavior preserved.
    assert isinstance(response.text, str)
    assert response.text  # non-empty

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.audio.transcriptions.create"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == TRANSCRIBE_MODEL
    assert span["metadata"]["language"] == "en"
    assert span["metadata"]["temperature"] == 0.0

    # Input carries the audio file as an Attachment.
    file_attachment = span["input"]["file"]
    assert isinstance(file_attachment, Attachment)
    assert file_attachment.reference["filename"] == TEST_AUDIO_FILE.name
    assert file_attachment.reference["content_type"] == "audio/wav"

    # Output is the transcribed text.
    assert span["output"] == response.text

    # Cohere's transcription response does not expose token counts, so we
    # only assert the timing metrics we always record.
    metrics = span["metrics"]
    assert start <= metrics["start"] <= metrics["end"] <= end
    assert metrics["duration"] >= 0


@pytest.mark.vcr
def test_wrap_cohere_audio_transcription_async(memory_logger):
    pytest.importorskip("cohere.audio.transcriptions.client")
    assert not memory_logger.pop()

    async def _run():
        client = wrap_cohere(_v1_async_client())
        with open(TEST_AUDIO_FILE, "rb") as file_obj:
            return await client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                language="en",
                file=(TEST_AUDIO_FILE.name, file_obj, "audio/wav"),
            )

    response = asyncio.run(_run())
    assert isinstance(response.text, str)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "cohere.audio.transcriptions.create"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == TRANSCRIBE_MODEL
    assert span["metadata"]["language"] == "en"

    file_attachment = span["input"]["file"]
    assert isinstance(file_attachment, Attachment)
    assert file_attachment.reference["filename"] == TEST_AUDIO_FILE.name

    assert span["output"] == response.text


@pytest.mark.vcr
def test_cohere_integration_setup_patches_audio_transcriptions(memory_logger, clean_cohere_methods):
    """``CohereIntegration.setup()`` must wire up audio transcription tracing."""
    pytest.importorskip("cohere.audio.transcriptions.client")
    assert not memory_logger.pop()

    assert CohereIntegration.setup() is True
    # Second call is a no-op but still reports success.
    assert CohereIntegration.setup() is True

    client = _v1_client()  # NOT manually wrapped
    with open(TEST_AUDIO_FILE, "rb") as file_obj:
        response = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            language="en",
            file=(TEST_AUDIO_FILE.name, file_obj, "audio/wav"),
            temperature=0.0,
        )
    assert isinstance(response.text, str)

    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["span_attributes"]["name"] == "cohere.audio.transcriptions.create"
    assert spans[0]["metadata"]["provider"] == "cohere"
    assert spans[0]["metadata"]["model"] == TRANSCRIBE_MODEL


class TestAutoInstrumentCohere:
    def test_auto_instrument_cohere(self):
        verify_autoinstrument_script("test_auto_cohere.py")
