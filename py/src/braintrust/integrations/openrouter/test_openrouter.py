import inspect
import os
import time
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.openrouter import OpenRouterIntegration, wrap_openrouter
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


openrouter = pytest.importorskip("openrouter")
from openrouter import OpenRouter
from openrouter.chat import Chat
from openrouter.embeddings import Embeddings
from openrouter.responses import Responses


PROJECT_NAME = "test-openrouter-sdk"
CHAT_MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).resolve().parent / "cassettes")


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _get_client():
    return OpenRouter(api_key=os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.vcr
def test_wrap_openrouter_chat_send_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openrouter(_get_client())
    start = time.time()
    response = client.chat.send(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    assert "4" in response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"] == [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["model"] == "gpt-4o-mini"
    assert "4" in span["output"][0]["message"]["content"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_openrouter_chat_send_stream_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openrouter(_get_client())
    start = time.time()
    result = client.chat.send(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 5+5? Reply with just the number."}],
        max_tokens=10,
        stream=True,
    )
    chunks = list(result)
    end = time.time()

    assert chunks
    content = "".join(
        choice.delta.content or ""
        for chunk in chunks
        for choice in (chunk.choices or [])
        if getattr(choice, "delta", None) is not None
    )
    assert "10" in content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["model"] == "gpt-4o-mini"
    assert span["metrics"]["time_to_first_token"] >= 0
    assert "10" in span["output"][0]["message"]["content"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_openrouter_chat_send_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openrouter(_get_client())
    start = time.time()
    response = await client.chat.send_async(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 3+3? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    assert "6" in response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["model"] == "gpt-4o-mini"
    assert "6" in span["output"][0]["message"]["content"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_openrouter_embeddings_generate(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openrouter(_get_client())
    start = time.time()
    response = client.embeddings.generate(
        model=EMBEDDING_MODEL,
        input="braintrust tracing",
        input_type="query",
    )
    end = time.time()

    assert response.data
    assert response.data[0].embedding
    embedding_length = len(response.data[0].embedding)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    # The response model field from the embeddings API is "text-embedding-3-small"
    # (without the "openai/" prefix), so the provider falls back to "openrouter".
    assert span["metadata"]["provider"] in ("openai", "openrouter")
    assert span["metadata"]["model"] == "text-embedding-3-small"
    assert span["metadata"]["embedding_model"] == "text-embedding-3-small"
    assert span["metadata"]["input_type"] == "query"
    assert span["output"]["embedding_length"] == embedding_length
    assert span["output"]["embeddings_count"] == 1
    assert span["metrics"]["prompt_tokens"] > 0
    assert span["metrics"]["tokens"] > 0
    # Embeddings don't have completion_tokens, so we check duration directly
    # instead of using assert_metrics_are_valid (which expects completion_tokens).
    assert start <= span["metrics"]["start"] <= span["metrics"]["end"] <= end + 1


@pytest.mark.vcr
def test_wrap_openrouter_responses_send(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openrouter(_get_client())
    start = time.time()
    response = client.beta.responses.send(
        model=CHAT_MODEL,
        input="Say one short sentence about observability.",
        max_output_tokens=16,
        temperature=0,
    )
    end = time.time()

    # output_text may be None in some SDK versions; fall back to output content
    output_text = response.output_text or response.output[0].content[0].text
    assert output_text

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"] == "Say one short sentence about observability."
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["model"] == "gpt-4o-mini"
    assert span["metadata"]["status"] == "completed"
    assert span["metadata"]["id"]
    assert span["metrics"]["prompt_tokens"] > 0
    assert span["metrics"]["completion_tokens"] > 0
    assert span["metrics"]["tokens"] > 0
    assert span["output"][0]["type"] == "message"
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_openrouter_responses_send_async_stream(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openrouter(_get_client())
    start = time.time()
    result = await client.beta.responses.send_async(
        model=CHAT_MODEL,
        input="What is 7+7? Reply with just the number.",
        stream=True,
    )

    items = []
    async for item in result:
        items.append(item)
    end = time.time()

    assert len(items) >= 1

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["model"] == "gpt-4o-mini"
    assert span["metadata"]["status"] == "completed"
    assert span["metrics"]["prompt_tokens"] > 0
    assert span["metrics"]["completion_tokens"] > 0
    assert span["metrics"]["tokens"] > 0
    assert span["metrics"]["time_to_first_token"] >= 0
    assert span["output"][0]["content"][0]["text"]
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_openrouter_integration_setup_creates_spans(memory_logger, monkeypatch):
    assert not memory_logger.pop()

    original_send = inspect.getattr_static(Chat, "send")
    original_generate = inspect.getattr_static(Embeddings, "generate")
    original_responses_send = inspect.getattr_static(Responses, "send")

    assert OpenRouterIntegration.setup()
    client = _get_client()
    start = time.time()
    response = client.chat.send(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    monkeypatch.setattr(Chat, "send", original_send)
    monkeypatch.setattr(Embeddings, "generate", original_generate)
    monkeypatch.setattr(Responses, "send", original_responses_send)

    assert "4" in response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["model"] == "gpt-4o-mini"
    assert "4" in span["output"][0]["message"]["content"]
    assert_metrics_are_valid(span["metrics"], start, end)


def test_openrouter_integration_setup_is_idempotent(monkeypatch):
    first_send = inspect.getattr_static(Chat, "send")
    first_generate = inspect.getattr_static(Embeddings, "generate")
    first_responses_send = inspect.getattr_static(Responses, "send")

    assert OpenRouterIntegration.setup()
    patched_send = inspect.getattr_static(Chat, "send")
    patched_generate = inspect.getattr_static(Embeddings, "generate")
    patched_responses_send = inspect.getattr_static(Responses, "send")

    assert OpenRouterIntegration.setup()
    assert inspect.getattr_static(Chat, "send") is patched_send
    assert inspect.getattr_static(Embeddings, "generate") is patched_generate
    assert inspect.getattr_static(Responses, "send") is patched_responses_send
    assert patched_send is not None
    assert patched_generate is not None
    assert patched_responses_send is not None

    monkeypatch.setattr(Chat, "send", first_send)
    monkeypatch.setattr(Embeddings, "generate", first_generate)
    monkeypatch.setattr(Responses, "send", first_responses_send)


class TestAutoInstrumentOpenRouter:
    def test_auto_instrument_openrouter(self):
        verify_autoinstrument_script("test_auto_openrouter.py")
