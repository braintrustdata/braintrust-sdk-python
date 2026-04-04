import time

import httpx
import openai
import pytest
from braintrust import logger, wrap_openai
from braintrust.test_helpers import init_test_logger
from braintrust.wrappers.test_utils import assert_metrics_are_valid
from openai import AsyncOpenAI


PROJECT_NAME = "test-project-openai-py-tracing"
TEST_MODEL = "gpt-4o-mini"
TEST_PROMPT = "What's 12 + 12?"


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.mark.vcr
def test_openai_chat_streaming_sync_http2_preserves_stream_interface(memory_logger):
    assert not memory_logger.pop()

    unwrapped_client = openai.OpenAI(http_client=httpx.Client(http2=True))
    wrapped_client = wrap_openai(openai.OpenAI(http_client=httpx.Client(http2=True)))
    clients = [(unwrapped_client, False), (wrapped_client, True)]

    try:
        for client, wrapped in clients:
            start = time.time()

            stream = client.chat.completions.create(
                model=TEST_MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                stream=True,
                stream_options={"include_usage": True},
            )

            assert hasattr(stream, "response")
            assert hasattr(stream, "_iterator")

            chunks = []
            for chunk in stream:
                chunks.append(chunk)
            end = time.time()

            assert chunks
            assert len(chunks) > 1

            content = ""
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content

            assert "24" in content or "twenty-four" in content.lower()

            if not wrapped:
                assert not memory_logger.pop()
                continue

            spans = memory_logger.pop()
            assert len(spans) == 1
            span = spans[0]
            metrics = span["metrics"]
            assert_metrics_are_valid(metrics, start, end)
            assert TEST_MODEL in span["metadata"]["model"]
            assert span["metadata"]["provider"] == "openai"
            assert TEST_PROMPT in str(span["input"])
            assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()
    finally:
        unwrapped_client.close()
        wrapped_client.close()


@pytest.mark.vcr
def test_openai_chat_streaming_sync_http2_context_manager_preserves_wrapper(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openai(openai.OpenAI(http_client=httpx.Client(http2=True)))

    try:
        start = time.time()
        stream = client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            stream=True,
            stream_options={"include_usage": True},
        )

        assert hasattr(stream, "response")
        assert hasattr(stream, "_iterator")

        chunks = []
        with stream as entered_stream:
            assert entered_stream is stream
            for chunk in entered_stream:
                chunks.append(chunk)
        end = time.time()

        assert chunks
        assert len(chunks) > 1

        content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        assert "24" in content or "twenty-four" in content.lower()

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        metrics = span["metrics"]
        assert_metrics_are_valid(metrics, start, end)
        assert TEST_MODEL in span["metadata"]["model"]
        assert span["metadata"]["provider"] == "openai"
        assert TEST_PROMPT in str(span["input"])
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_chat_streaming_async_http2_context_manager_preserves_wrapper(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openai(AsyncOpenAI(http_client=httpx.AsyncClient(http2=True)))

    try:
        start = time.time()
        stream = await client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            stream=True,
            stream_options={"include_usage": True},
        )

        assert hasattr(stream, "response")
        assert hasattr(stream, "_iterator")

        chunks = []
        async with stream as entered_stream:
            assert entered_stream is stream
            async for chunk in entered_stream:
                chunks.append(chunk)
        end = time.time()

        assert chunks
        assert len(chunks) > 1

        content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        assert "24" in content or "twenty-four" in content.lower()

        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        metrics = span["metrics"]
        assert_metrics_are_valid(metrics, start, end)
        assert TEST_MODEL in span["metadata"]["model"]
        assert span["metadata"]["provider"] == "openai"
        assert TEST_PROMPT in str(span["input"])
    finally:
        await client.close()
