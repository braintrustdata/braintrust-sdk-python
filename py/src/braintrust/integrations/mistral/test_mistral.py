import importlib
import inspect
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.mistral import MistralIntegration, wrap_mistral
from braintrust.integrations.mistral.tracing import (
    _aggregate_completion_events,
    _chat_complete_async_wrapper,
    _chat_complete_wrapper,
)
from braintrust.test_helpers import init_test_logger
from braintrust.wrappers.test_utils import assert_metrics_are_valid, verify_autoinstrument_script


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
    models = importlib.import_module("mistralai.client.models")
except ImportError:
    Chat = importlib.import_module("mistralai.chat").Chat
    Embeddings = importlib.import_module("mistralai.embeddings").Embeddings
    Fim = importlib.import_module("mistralai.fim").Fim
    Agents = importlib.import_module("mistralai.agents").Agents
    models = importlib.import_module("mistralai.models")


PROJECT_NAME = "test-mistral-sdk"
CHAT_MODEL = "mistral-small-latest"
AGENT_MODEL = CHAT_MODEL
EMBEDDING_MODEL = "mistral-embed"
FIM_MODEL = "codestral-latest"


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).resolve().parent / "cassettes")


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _get_client():
    return Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))


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
