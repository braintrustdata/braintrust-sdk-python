import inspect
import os
import time
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.cohere import CohereIntegration, wrap_cohere
from braintrust.integrations.cohere.tracing import _v2_chat_async_wrapper, _v2_chat_wrapper
from braintrust.test_helpers import init_test_logger
from braintrust.wrappers.test_utils import assert_metrics_are_valid, verify_autoinstrument_script


pytest.importorskip("cohere")
from cohere import AsyncClientV2, Client, ClientV2
from cohere.v2.client import V2Client


PROJECT_NAME = "test-cohere-sdk"
CHAT_MODEL = "command-r-plus-08-2024"
EMBED_MODEL = "embed-english-v3.0"
RERANK_MODEL = "rerank-v3.5"


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).resolve().parent / "cassettes")


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _get_client():
    return Client(api_key=os.environ.get("CO_API_KEY"))


def _get_client_v2():
    return ClientV2(api_key=os.environ.get("CO_API_KEY"))


async def _get_async_client_v2():
    return AsyncClientV2(api_key=os.environ.get("CO_API_KEY"))


@pytest.mark.vcr
def test_wrap_cohere_nested_v2_chat_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_cohere(_get_client())
    start = time.time()
    response = client.v2.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    assert response.message.content[0].text == "4"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["input"]["messages"] == [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["output"]["content"][0]["text"] == "4"
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_cohere_client_v2_chat_sync(memory_logger):
    assert not memory_logger.pop()

    client = wrap_cohere(_get_client_v2())
    start = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 5+5? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    assert response.message.content[0].text == "10"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["output"]["content"][0]["text"] == "10"
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_wrap_cohere_v2_chat_stream_async(memory_logger):
    assert not memory_logger.pop()

    client = wrap_cohere(await _get_async_client_v2())
    start = time.time()
    stream = client.chat_stream(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 8+8? Reply with just the number."}],
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
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert span["metadata"]["model"] == CHAT_MODEL
    assert span["metadata"]["stream"] == True
    assert span["metrics"]["time_to_first_token"] >= 0
    assert span["output"]["content"][0]["text"] == "16"
    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_cohere_v2_embed(memory_logger):
    assert not memory_logger.pop()

    client = wrap_cohere(_get_client_v2())
    start = time.time()
    response = client.embed(
        model=EMBED_MODEL,
        texts=["braintrust tracing"],
        input_type="search_query",
        embedding_types=["float"],
    )
    end = time.time()

    assert response.embeddings.float_

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert span["metadata"]["model"] == EMBED_MODEL
    assert span["output"]["embeddings_count"] == 1
    assert span["output"]["embedding_length"] == len(response.embeddings.float_[0])
    assert span["output"]["embedding_types"] == ["float"]
    assert span["metrics"]["prompt_tokens"] > 0
    assert span["metrics"]["tokens"] > 0
    assert start <= span["metrics"]["start"] <= span["metrics"]["end"] <= end


@pytest.mark.vcr
def test_wrap_cohere_v2_rerank(memory_logger):
    assert not memory_logger.pop()

    client = wrap_cohere(_get_client_v2())
    start = time.time()
    response = client.rerank(
        model=RERANK_MODEL,
        query="What is the capital of the United States?",
        documents=[
            "Carson City is the capital city of Nevada.",
            "Washington, D.C. is the capital of the United States.",
        ],
        top_n=2,
    )
    end = time.time()

    assert response.results[0].index == 1

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert span["metadata"]["model"] == RERANK_MODEL
    assert span["output"][0]["index"] == 1
    assert span["metrics"]["billed_search_units"] > 0
    assert start <= span["metrics"]["start"] <= span["metrics"]["end"] <= end


@pytest.mark.vcr
def test_cohere_integration_setup_creates_spans(memory_logger, monkeypatch):
    assert not memory_logger.pop()

    original_chat = inspect.getattr_static(V2Client, "chat")
    original_embed = inspect.getattr_static(V2Client, "embed")
    original_rerank = inspect.getattr_static(V2Client, "rerank")

    assert CohereIntegration.setup()
    client = _get_client_v2()
    start = time.time()
    response = client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "What is 4+4? Reply with just the number."}],
        max_tokens=10,
    )
    end = time.time()

    monkeypatch.setattr(V2Client, "chat", original_chat)
    monkeypatch.setattr(V2Client, "embed", original_embed)
    monkeypatch.setattr(V2Client, "rerank", original_rerank)

    assert response.message.content[0].text == "8"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert span["output"]["content"][0]["text"] == "8"
    assert_metrics_are_valid(span["metrics"], start, end)


def test_cohere_integration_setup_is_idempotent(monkeypatch):
    first_chat = inspect.getattr_static(V2Client, "chat")
    first_embed = inspect.getattr_static(V2Client, "embed")
    first_rerank = inspect.getattr_static(V2Client, "rerank")

    assert CohereIntegration.setup()
    patched_chat = inspect.getattr_static(V2Client, "chat")
    patched_embed = inspect.getattr_static(V2Client, "embed")
    patched_rerank = inspect.getattr_static(V2Client, "rerank")

    assert CohereIntegration.setup()
    assert inspect.getattr_static(V2Client, "chat") is patched_chat
    assert inspect.getattr_static(V2Client, "embed") is patched_embed
    assert inspect.getattr_static(V2Client, "rerank") is patched_rerank

    monkeypatch.setattr(V2Client, "chat", first_chat)
    monkeypatch.setattr(V2Client, "embed", first_embed)
    monkeypatch.setattr(V2Client, "rerank", first_rerank)


def test_v2_chat_wrapper_logs_errors(memory_logger):
    assert not memory_logger.pop()

    def fail(*args, **kwargs):
        raise RuntimeError("sync boom")

    with pytest.raises(RuntimeError, match="sync boom"):
        _v2_chat_wrapper(
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
    assert span["input"]["messages"] == [{"role": "user", "content": "hello"}]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert "sync boom" in span["error"]


@pytest.mark.asyncio
async def test_v2_chat_async_wrapper_logs_errors(memory_logger):
    assert not memory_logger.pop()

    async def fail(*args, **kwargs):
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        await _v2_chat_async_wrapper(
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
    assert span["input"]["messages"] == [{"role": "user", "content": "hello"}]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert "async boom" in span["error"]


class TestAutoInstrumentCohere:
    def test_auto_instrument_cohere(self):
        verify_autoinstrument_script("test_auto_cohere.py")
