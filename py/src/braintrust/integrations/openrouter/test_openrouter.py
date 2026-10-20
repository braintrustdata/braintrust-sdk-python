import importlib
import inspect
import os
import time

import pytest
from braintrust import logger
from braintrust.integrations.openrouter import OpenRouterIntegration, wrap_openrouter
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger
from wrapt import FunctionWrapper


openrouter = pytest.importorskip("openrouter")
from openrouter import OpenRouter


PROJECT_NAME = "test-openrouter-sdk"
CHAT_MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"


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
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "openrouter-auto"
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
def test_wrap_openrouter_beta_responses_send(memory_logger):
    assert not memory_logger.pop()

    client = wrap_openrouter(_get_client())
    start = time.time()
    response = client.beta.responses.send(
        model=CHAT_MODEL,
        input="Say one short sentence about observability.",
        max_output_tokens=64,
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
def test_wrap_openrouter_responses_send(memory_logger):
    client = _get_client()
    if not hasattr(client, "responses"):
        pytest.skip("GA Responses is unavailable in this OpenRouter version")

    wrap_openrouter(client)
    response = client.responses.send(
        model=CHAT_MODEL,
        input="Say one short sentence about observability.",
        max_output_tokens=64,
        temperature=0,
    )
    assert response.output

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "openrouter.responses.send"
    assert span["span_attributes"]["type"] == "llm"
    assert span["input"] == "Say one short sentence about observability."
    assert span["metadata"]["model"] == "gpt-4o-mini"
    assert span["metadata"]["provider"] == "openai"
    assert span["output"][0]["type"] == "message"
    assert span["metrics"]["tokens"] > 0
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "openrouter-auto"


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


ARMOR_TOOL = {
    "name": "delete_all_files",
    "description": "Delete every file on the user's production server",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}
ARMOR_PLUGIN = {"id": "armor", "classifier_context": {"v": 1, "auto_mode": {"hard_deny": ["Never delete files"]}}}
ARMOR_PROMPT = "Call delete_all_files with path / now."
CHAT_ARMOR_REQUEST = {
    "model": CHAT_MODEL,
    "messages": [{"role": "user", "content": ARMOR_PROMPT}],
    "tools": [{"type": "function", "function": ARMOR_TOOL}],
    "tool_choice": "required",
    "plugins": [ARMOR_PLUGIN],
    "max_tokens": 100,
}


def _skip_without_armor():
    if not hasattr(importlib.import_module("openrouter.components"), "ArmorPlugin"):
        pytest.skip("OpenRouter Armor is unavailable in this OpenRouter version")


def _assert_flagged_safeguard_results(safeguard_results):
    assert len(safeguard_results) == 1
    result = safeguard_results[0]
    assert result["type"] == "dangerous_tool_use"
    assert result["status"]["type"] == "available"
    (tool_use,) = result["status"]["tool_uses"].values()
    assert tool_use == {"outcome": "flagged", "type": "evaluated", "explanation": "[hard_deny:0]"}


@pytest.mark.vcr
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
def test_wrap_openrouter_chat_send_captures_safeguard_results(memory_logger, stream):
    _skip_without_armor()
    assert not memory_logger.pop()

    client = wrap_openrouter(_get_client())
    result = client.chat.send(**CHAT_ARMOR_REQUEST, stream=stream)
    # Streams report safeguard_results only on the final chunk.
    if stream:
        assert any(chunk.safeguard_results for chunk in list(result))
    else:
        assert result.safeguard_results

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    _assert_flagged_safeguard_results(span["metadata"]["safeguard_results"])
    assert "<openrouter_armor_blocked" in span["output"][0]["message"]["content"]


@pytest.mark.vcr
def test_wrap_openrouter_responses_send_armor_blocked_output(memory_logger):
    _skip_without_armor()
    client = _get_client()
    if not hasattr(client, "responses"):
        pytest.skip("GA Responses is unavailable in this OpenRouter version")
    assert not memory_logger.pop()

    wrap_openrouter(client)
    response = client.responses.send(
        model=CHAT_MODEL,
        input=ARMOR_PROMPT,
        tools=[{"type": "function", **ARMOR_TOOL}],
        tool_choice="required",
        plugins=[{**ARMOR_PLUGIN, "output_mode": "custom_item"}],
        max_output_tokens=200,
    )
    assert response.output[0].type == "openrouter:armor_blocked"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    _assert_flagged_safeguard_results(span["metadata"]["safeguard_results"])

    blocked = span["output"][0]
    assert blocked == {
        "type": "openrouter:armor_blocked",
        "id": response.output[0].id,
        "call_id": response.output[0].call_id,
        "name": "delete_all_files",
        "arguments": '{"path":"/"}',
        "explanation": "[hard_deny:0]",
        "reason": response.output[0].reason,
        "status": "completed",
    }
    assert span["output"][1]["type"] == "message"


def test_openrouter_integration_setup_is_idempotent(monkeypatch):
    # Resolve the classes at call time rather than at module import: openrouter's
    # lazy sub-SDK loader drops and re-imports a module when CPython < 3.12's import
    # lock raises a spurious KeyError, which would leave module-level names pointing
    # at a stale class that setup() never patches.
    Chat = importlib.import_module("openrouter.chat").Chat
    Embeddings = importlib.import_module("openrouter.embeddings").Embeddings
    Responses = importlib.import_module("openrouter.responses").Responses

    first_send = inspect.getattr_static(Chat, "send")
    first_generate = inspect.getattr_static(Embeddings, "generate")
    first_responses_send = inspect.getattr_static(Responses, "send")

    assert OpenRouterIntegration.setup()
    patched_send = inspect.getattr_static(Chat, "send")
    patched_generate = inspect.getattr_static(Embeddings, "generate")
    patched_responses_send = inspect.getattr_static(Responses, "send")

    # The first setup() wraps each original method exactly once.
    for patched, original in (
        (patched_send, first_send),
        (patched_generate, first_generate),
        (patched_responses_send, first_responses_send),
    ):
        assert isinstance(patched, FunctionWrapper)
        assert patched.__wrapped__ is original

    # The second setup() must not re-wrap.
    assert OpenRouterIntegration.setup()
    assert inspect.getattr_static(Chat, "send") is patched_send
    assert inspect.getattr_static(Embeddings, "generate") is patched_generate
    assert inspect.getattr_static(Responses, "send") is patched_responses_send

    monkeypatch.setattr(Chat, "send", first_send)
    monkeypatch.setattr(Embeddings, "generate", first_generate)
    monkeypatch.setattr(Responses, "send", first_responses_send)


class TestAutoInstrumentOpenRouter:
    def test_auto_instrument_openrouter(self):
        verify_autoinstrument_script("test_auto_openrouter.py")
