"""Tests for the HuggingFace Hub integration."""

import asyncio
import inspect
import os
import time

import pytest
from braintrust import logger, start_span
from braintrust.integrations.huggingface_hub import HuggingFaceHubIntegration, wrap_huggingface_hub
from braintrust.integrations.huggingface_hub.patchers import (
    AsyncChatCompletionPatcher,
    AsyncFeatureExtractionPatcher,
    AsyncSentenceSimilarityPatcher,
    AsyncTextGenerationPatcher,
    ChatCompletionPatcher,
    FeatureExtractionPatcher,
    SentenceSimilarityPatcher,
    TextGenerationPatcher,
)
from braintrust.integrations.test_utils import assert_metrics_are_valid, verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


pytest.importorskip("huggingface_hub")

from huggingface_hub import AsyncInferenceClient, InferenceClient  # noqa: E402
from huggingface_hub.inference._client import InferenceClient as _SyncInferenceClient  # noqa: E402
from huggingface_hub.inference._generated._async_client import (  # noqa: E402
    AsyncInferenceClient as _AsyncInferenceClient,
)


PROJECT_NAME = "test-huggingface-hub-sdk"

# Chat completion is tested against ``provider="cerebras"``: it hosts
# ``meta-llama/Llama-3.1-8B-Instruct`` and is one of the named providers
# available on the entire matrix range. Using a specific named provider
# instead of ``"auto"`` keeps the recorded request URLs stable, so cassette
# routing does not depend on per-version HuggingFace routing decisions.
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CHAT_PROVIDER = "cerebras"

# Text generation is exercised via ``provider="auto"`` so the HuggingFace
# router can pick a backend that actually hosts the requested model.
TEXT_GEN_MODEL = "meta-llama/Llama-3.2-3B"
TEXT_GEN_PROVIDER = "auto"

# Embeddings + sentence similarity are tested against ``hf-inference``: that's
# the provider that exposes the sentence-transformers feature-extraction and
# sentence-similarity endpoints used here.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_PROVIDER = "hf-inference"
# Dummy token must start with ``hf_`` so the HuggingFace SDK accepts it for
# ``provider="auto"`` routing (the SDK validates the prefix locally before
# making any HTTP request).
HF_TOKEN = os.getenv("HF_TOKEN") or "hf_test_dummy_api_key_for_vcr_tests"


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture
def clean_hf_methods():
    """Snapshot and restore patched methods on the HuggingFace inference clients.

    Integration setup mutates global class state; tests that exercise
    ``setup()`` must not leak wrappers into other tests.
    """
    targets = [
        (_SyncInferenceClient, "chat_completion"),
        (_SyncInferenceClient, "text_generation"),
        (_SyncInferenceClient, "feature_extraction"),
        (_SyncInferenceClient, "sentence_similarity"),
        (_AsyncInferenceClient, "chat_completion"),
        (_AsyncInferenceClient, "text_generation"),
        (_AsyncInferenceClient, "feature_extraction"),
        (_AsyncInferenceClient, "sentence_similarity"),
    ]
    originals = [(cls, attr, inspect.getattr_static(cls, attr)) for cls, attr in targets]
    marker_attrs = {
        patcher.patch_marker_attr()
        for patcher in (
            ChatCompletionPatcher,
            AsyncChatCompletionPatcher,
            TextGenerationPatcher,
            AsyncTextGenerationPatcher,
            FeatureExtractionPatcher,
            AsyncFeatureExtractionPatcher,
            SentenceSimilarityPatcher,
            AsyncSentenceSimilarityPatcher,
        )
    }

    try:
        yield
    finally:
        for cls, attr, original in originals:
            setattr(cls, attr, original)
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


def _sync_client(*, model: str = CHAT_MODEL, provider: str = CHAT_PROVIDER) -> InferenceClient:
    return InferenceClient(model=model, provider=provider, token=HF_TOKEN)


def _async_client(*, model: str = CHAT_MODEL, provider: str = CHAT_PROVIDER) -> AsyncInferenceClient:
    return AsyncInferenceClient(model=model, provider=provider, token=HF_TOKEN)


# ---------------------------------------------------------------------------
# Unit / local tests (no network)
# ---------------------------------------------------------------------------


def test_integration_available_patcher_ids():
    ids = HuggingFaceHubIntegration.available_patchers()
    assert set(ids) == {
        "huggingface_hub.chat_completion.all",
        "huggingface_hub.text_generation.all",
        "huggingface_hub.feature_extraction.all",
        "huggingface_hub.sentence_similarity.all",
    }


def test_integration_pins_min_version_floor():
    """The integration must declare a min_version so the matrix floor is enforced.

    The 0.32.0 floor is the earliest release that exposes both the stable
    multi-provider ``InferenceClient(provider=...)`` surface and the
    ``provider="auto"`` routing mode the integration relies on.
    """
    assert HuggingFaceHubIntegration.min_version == "0.32.0"


def test_patchers_target_real_sdk_surfaces():
    """Each patcher must point at an attribute that actually exists on the SDK.

    Regression guard: if HuggingFace ever rename or move one of these
    methods, the patcher should fail at this assertion before the broader
    integration test surface drifts silently.
    """
    sync_targets = (
        (ChatCompletionPatcher, "chat_completion"),
        (TextGenerationPatcher, "text_generation"),
        (FeatureExtractionPatcher, "feature_extraction"),
        (SentenceSimilarityPatcher, "sentence_similarity"),
    )
    for patcher, attr in sync_targets:
        assert patcher.target_module == "huggingface_hub.inference._client"
        assert patcher.target_path == f"InferenceClient.{attr}"
        assert hasattr(_SyncInferenceClient, attr)

    async_targets = (
        (AsyncChatCompletionPatcher, "chat_completion"),
        (AsyncTextGenerationPatcher, "text_generation"),
        (AsyncFeatureExtractionPatcher, "feature_extraction"),
        (AsyncSentenceSimilarityPatcher, "sentence_similarity"),
    )
    for patcher, attr in async_targets:
        assert patcher.target_module == "huggingface_hub.inference._generated._async_client"
        assert patcher.target_path == f"AsyncInferenceClient.{attr}"
        assert hasattr(_AsyncInferenceClient, attr)


def test_wrap_huggingface_hub_returns_unsupported_unchanged():
    invalid = object()
    assert wrap_huggingface_hub(invalid) is invalid

    invalid_dict = {"foo": "bar"}
    assert wrap_huggingface_hub(invalid_dict) is invalid_dict


def test_wrap_huggingface_hub_is_idempotent():
    client = _sync_client()
    wrapped_once = wrap_huggingface_hub(client)
    wrapped_twice = wrap_huggingface_hub(client)
    assert wrapped_once is client
    assert wrapped_twice is client
    assert getattr(client, "__braintrust_huggingface_hub_traced__", False) is True


def test_setup_is_idempotent(clean_hf_methods):
    """Repeated ``setup()`` calls must not pile wrappers on top of each other."""
    assert HuggingFaceHubIntegration.setup() is True
    assert HuggingFaceHubIntegration.setup() is True


def test_chat_dot_completions_dot_create_alias_dispatches_to_chat_completion():
    """The OpenAI-compatible ``client.chat.completions.create`` alias must
    resolve to ``InferenceClient.chat_completion`` so that patching the latter
    transparently covers the former.  This is a structural assertion that
    does not require network access.
    """
    client = _sync_client()
    # Bound method equality compares both ``__func__`` and ``__self__``, so
    # this single assertion is what guarantees a single patch covers both.
    assert client.chat.completions.create == client.chat_completion


# ---------------------------------------------------------------------------
# VCR-backed integration tests
#
# All tests below replay HuggingFace inference traffic through cassettes
# under ``cassettes/<version>/``.  When recording new cassettes, run with
# ``--vcr-record=all`` and a real ``HF_TOKEN`` exported in the environment.
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_wrap_huggingface_hub_chat_completion_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client())

    start = time.time()
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_tokens=10,
    )
    end = time.time()

    assert response.choices
    assert response.choices[0].message.role == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    assert span["span_attributes"]["name"] == "huggingface.chat_completion"
    assert span["span_attributes"]["type"] == "llm"
    # With no parent span on the stack, the LLM span is its own root and has
    # no ``span_parents``.
    assert not span.get("span_parents")
    # The user's ``provider=`` kwarg overrides the default "huggingface"
    # identity so the span reflects the actual routing target.
    assert span["metadata"]["provider"] == CHAT_PROVIDER
    # ``model`` reflects what the provider actually served, which may be a
    # provider-internal alias (e.g. Cerebras returns ``"llama3.1-8b"`` for
    # ``meta-llama/Llama-3.1-8B-Instruct``). The request-side value is the
    # ``CHAT_MODEL`` HuggingFace ID; the response-side value is what the
    # backend reports back.
    assert isinstance(span["metadata"]["model"], str) and span["metadata"]["model"]
    assert span["metadata"]["max_tokens"] == 10
    assert span["input"] == [{"role": "user", "content": "Say hi in one word."}]

    # ``output`` is the SDK ``choices`` list verbatim.
    output = span["output"]
    assert isinstance(output, list) and output
    first_choice = output[0]
    message = first_choice.message if hasattr(first_choice, "message") else first_choice["message"]
    assert message.role == "assistant" if hasattr(message, "role") else message["role"] == "assistant"

    assert_metrics_are_valid(span["metrics"], start, end)


@pytest.mark.vcr
def test_wrap_huggingface_hub_chat_completion_openai_alias(memory_logger):
    """``client.chat.completions.create`` must produce the same span as ``chat_completion``."""
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client())

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_tokens=10,
    )
    assert response.choices
    assert response.choices[0].message.role == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.chat_completion"
    assert span["metadata"]["provider"] == CHAT_PROVIDER


@pytest.mark.vcr
def test_wrap_huggingface_hub_chat_completion_streaming(memory_logger):
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client())

    start = time.time()
    chunks = list(
        client.chat_completion(
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
            stream=True,
        )
    )
    end = time.time()
    assert chunks  # provider yielded events

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    # Streaming gets its own span name.
    assert span["span_attributes"]["name"] == "huggingface.chat_completion_stream"
    # Even though the span is started inside the wrapper and finalized later
    # when the iterator is exhausted, with no parent on the stack the span is
    # still its own root.
    assert not span.get("span_parents")
    assert span["metadata"]["provider"] == CHAT_PROVIDER

    # Aggregated output is ``{"choices": [{"index", "message": {...}, "finish_reason"?}]}``.
    output = span["output"]
    assert isinstance(output, dict)
    assert isinstance(output["choices"], list) and output["choices"]
    first_choice = output["choices"][0]
    assert first_choice["index"] == 0
    assert first_choice["message"]["role"] == "assistant"
    assert isinstance(first_choice["message"]["content"], str) and first_choice["message"]["content"]

    metrics = span["metrics"]
    assert metrics["start"] <= metrics["end"]
    assert start <= metrics["start"] <= metrics["end"] <= end


@pytest.mark.vcr(cassette_name="test_wrap_huggingface_hub_chat_completion_streaming")
def test_wrap_huggingface_hub_chat_completion_streaming_finalizes_on_early_close(memory_logger):
    """Closing a real provider stream early must finalize the span."""
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client())

    stream = client.chat_completion(
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_tokens=10,
        stream=True,
    )
    first_chunk = next(stream)
    assert first_chunk is not None
    stream.close()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.chat_completion_stream"
    assert span["metadata"]["provider"] == CHAT_PROVIDER
    assert span["output"]["choices"]

    # Double-close must be a no-op (no extra span emitted).
    stream.close()
    assert memory_logger.pop() == []


@pytest.mark.vcr(cassette_name="test_wrap_huggingface_hub_chat_completion_streaming")
def test_wrap_huggingface_hub_chat_completion_streaming_context_manager_finalizes(memory_logger):
    """Using a real provider stream as a ``with`` block must finalize on exit."""
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client())

    with client.chat_completion(
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_tokens=10,
        stream=True,
    ) as stream:
        first_chunk = next(stream)
        assert first_chunk is not None

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.chat_completion_stream"
    assert span["metadata"]["provider"] == CHAT_PROVIDER
    assert span["output"]["choices"]


@pytest.mark.vcr
def test_wrap_huggingface_hub_chat_completion_tool_calls(memory_logger):
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client())

    tools = [
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
    ]

    # ``tool_choice="required"`` ensures the model emits a tool call rather
    # than choosing to respond in natural language.
    response = client.chat_completion(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        tool_choice="required",
        max_tokens=64,
    )

    # Provider behavior preserved.
    assert response.choices
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls
    assert tool_calls[0].function.name == "get_weather"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["tools"] == tools

    # ``output`` is the SDK ``choices`` list; tool calls live on the
    # first choice's ``message.tool_calls``.
    output = span["output"]
    assert isinstance(output, list) and output
    first_choice = output[0]
    message = first_choice.message if hasattr(first_choice, "message") else first_choice["message"]
    logged_tool_calls = message.tool_calls if hasattr(message, "tool_calls") else message["tool_calls"]
    assert logged_tool_calls
    first_tc = logged_tool_calls[0]
    fn = first_tc.function if hasattr(first_tc, "function") else first_tc["function"]
    name = fn.name if hasattr(fn, "name") else fn["name"]
    assert name == "get_weather"


@pytest.mark.vcr
def test_wrap_huggingface_hub_chat_completion_error_logs_span(memory_logger):
    """A provider failure must still finalize the span with the error."""
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client(model="this-model-id-does-not-exist-xyz", provider="hf-inference"))

    with pytest.raises(Exception):
        client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
        )

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.chat_completion"
    assert span.get("error")


@pytest.mark.vcr
def test_wrap_huggingface_hub_chat_completion_async(memory_logger):
    async def _run():
        client = wrap_huggingface_hub(_async_client())
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
        )
        assert response.choices
        assert response.choices[0].message.role == "assistant"
        await client.close()

    assert not memory_logger.pop()
    asyncio.run(_run())

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.chat_completion"
    assert not span.get("span_parents")
    assert span["metadata"]["provider"] == CHAT_PROVIDER


@pytest.mark.vcr
def test_wrap_huggingface_hub_chat_completion_async_streaming(memory_logger):
    async def _run():
        client = wrap_huggingface_hub(_async_client())
        chunks = []
        async for chunk in await client.chat_completion(
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
            stream=True,
        ):
            chunks.append(chunk)
        await client.close()
        return chunks

    assert not memory_logger.pop()
    chunks = asyncio.run(_run())
    assert chunks

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.chat_completion_stream"
    output = span["output"]
    assert isinstance(output, dict)
    first_choice = output["choices"][0]
    assert isinstance(first_choice["message"]["content"], str) and first_choice["message"]["content"]


def _skip_if_text_generation_unavailable() -> None:
    """Skip ``text_generation`` tests on SDK versions whose provider router
    cannot dispatch text-generation requests to a currently-hosted backend.

    HuggingFace's ``provider="auto"`` on the latest pin transparently routes
    text-generation to a backend such as ``featherless-ai``; older SDKs in
    the matrix do not know about that provider and there is no public provider
    that exposes the ``text-generation`` task for these models on those
    releases. Patcher coverage for ``text_generation`` is still asserted
    structurally in :func:`test_patchers_target_real_sdk_surfaces`.
    """
    version = os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION")
    if version and version != "latest":
        pytest.skip(
            "text_generation is only routable through provider='auto' on the latest huggingface-hub pin in our matrix"
        )


@pytest.mark.vcr
def test_wrap_huggingface_hub_text_generation_sync(memory_logger):
    _skip_if_text_generation_unavailable()
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client(model=TEXT_GEN_MODEL, provider=TEXT_GEN_PROVIDER))

    response = client.text_generation(
        "The Braintrust SDK is",
        max_new_tokens=8,
    )

    assert isinstance(response, str)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.text_generation"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == TEXT_GEN_PROVIDER
    # ``model`` is whichever value the integration resolved (request override,
    # instance default, or server-reported value depending on provider).
    assert isinstance(span["metadata"]["model"], str) and span["metadata"]["model"]
    assert span["metadata"]["max_new_tokens"] == 8
    assert span["input"] == "The Braintrust SDK is"
    # ``output`` is always the ``{"generated_text": ...}`` object shape.
    assert isinstance(span["output"], dict)
    assert isinstance(span["output"]["generated_text"], str) and span["output"]["generated_text"]


@pytest.mark.vcr
def test_wrap_huggingface_hub_text_generation_details(memory_logger):
    """With ``details=True`` the response is a ``TextGenerationOutput`` object.

    HuggingFace's ``provider="auto"`` may route to a non-TGI backend whose
    ``details`` payload omits ``generated_tokens``; the span shape must still
    surface the generated text and not crash on the missing field.
    """
    _skip_if_text_generation_unavailable()
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client(model=TEXT_GEN_MODEL, provider=TEXT_GEN_PROVIDER))

    response = client.text_generation(
        "The Braintrust SDK is",
        max_new_tokens=8,
        details=True,
    )

    assert hasattr(response, "generated_text")
    assert hasattr(response, "details")

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == TEXT_GEN_PROVIDER
    assert span["metadata"]["details"] is True
    assert isinstance(span["output"], dict)
    assert isinstance(span["output"]["generated_text"], str) and span["output"]["generated_text"]


@pytest.mark.vcr
def test_wrap_huggingface_hub_feature_extraction_sync(memory_logger):
    pytest.importorskip("numpy")

    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client(model=EMBED_MODEL, provider=EMBED_PROVIDER))

    response = client.feature_extraction("braintrust tracing")

    # Provider behavior preserved: numpy array.
    assert hasattr(response, "shape")

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.feature_extraction"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == EMBED_PROVIDER
    # ``feature_extraction`` has no response-side ``model`` field, so the
    # request/instance ``model`` is what surfaces in metadata.
    assert span["metadata"]["model"] == EMBED_MODEL
    assert span["input"] == "braintrust tracing"

    output = span["output"]
    assert isinstance(output, dict)
    assert output["embedding_length"] > 0
    # 1D vector responses no longer carry ``embedding_count``.
    assert "embedding_count" not in output


@pytest.mark.vcr
def test_wrap_huggingface_hub_sentence_similarity_sync(memory_logger):
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client(model=EMBED_MODEL, provider=EMBED_PROVIDER))

    response = client.sentence_similarity(
        "Machine learning is so easy.",
        [
            "Deep learning is so straightforward.",
            "This is so difficult, like rocket science.",
        ],
    )

    assert isinstance(response, list)
    assert len(response) == 2
    assert all(isinstance(x, float) for x in response)

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.sentence_similarity"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == EMBED_PROVIDER
    assert span["input"]["sentence"] == "Machine learning is so easy."
    assert len(span["input"]["other_sentences"]) == 2
    assert isinstance(span["output"], list)
    assert len(span["output"]) == 2


# ---------------------------------------------------------------------------
# Parent / child span relationships
#
# These tests assert that HF spans nest correctly under a user-opened parent
# span. The HTTP request body is identical to the non-nested tests above, so
# cassettes only differ by file name; the parent-span logic is purely local.
# Streaming + async paths matter most because the LLM span is started inside
# the wrapper but finalized later — the parent context must be captured at
# ``start_span`` time, not at iterator-exhaustion time.
# ---------------------------------------------------------------------------


def _assert_child_of(child: dict, parent_id: str, root_id: str) -> None:
    """Assert *child* span nests under *parent_id* and shares *root_id*."""
    assert child.get("span_parents") == [parent_id], (
        f"expected span_parents=[{parent_id!r}], got {child.get('span_parents')!r}"
    )
    assert child.get("root_span_id") == root_id, (
        f"expected root_span_id={root_id!r}, got {child.get('root_span_id')!r}"
    )


@pytest.mark.vcr
def test_chat_completion_nests_under_parent_span(memory_logger):
    """Non-streaming chat span must nest under an outer user span."""
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client())

    with start_span(name="user.outer") as outer:
        outer_id = outer.span_id
        outer_root = outer.root_span_id
        client.chat_completion(
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
        )

    spans = memory_logger.pop()
    by_name = {span["span_attributes"]["name"]: span for span in spans}
    assert "user.outer" in by_name
    assert "huggingface.chat_completion" in by_name

    llm_span = by_name["huggingface.chat_completion"]
    _assert_child_of(llm_span, parent_id=outer_id, root_id=outer_root)


@pytest.mark.vcr
def test_chat_completion_streaming_nests_under_parent_span(memory_logger):
    """Streaming span must capture the parent at start time, not at finalize.

    The wrapper opens the span inside the call to ``chat_completion`` while
    the parent ``with`` block is still active, then returns a traced iterator.
    The span is finalized on iterator exhaustion. The parent context must be
    baked into the span's identity at start time so it does not drift even if
    the iterator is consumed outside the parent block.
    """
    assert not memory_logger.pop()
    client = wrap_huggingface_hub(_sync_client())

    with start_span(name="user.outer") as outer:
        outer_id = outer.span_id
        outer_root = outer.root_span_id
        iterator = client.chat_completion(
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
            stream=True,
        )
        # Drain the iterator *inside* the parent block so the cassette
        # interaction matches the non-nested streaming test's request shape.
        chunks = list(iterator)
    assert chunks

    spans = memory_logger.pop()
    by_name = {span["span_attributes"]["name"]: span for span in spans}
    assert "user.outer" in by_name
    assert "huggingface.chat_completion_stream" in by_name

    llm_span = by_name["huggingface.chat_completion_stream"]
    _assert_child_of(llm_span, parent_id=outer_id, root_id=outer_root)


@pytest.mark.vcr
def test_chat_completion_async_streaming_nests_under_parent_span(memory_logger):
    """Async streaming span must also capture the parent at start time."""
    captured: dict[str, str] = {}

    async def _run():
        client = wrap_huggingface_hub(_async_client())
        with start_span(name="user.outer") as outer:
            captured["outer_id"] = outer.span_id
            captured["outer_root"] = outer.root_span_id
            async for chunk in await client.chat_completion(
                messages=[{"role": "user", "content": "Say hi in one word."}],
                max_tokens=10,
                stream=True,
            ):
                assert chunk is not None
        await client.close()

    assert not memory_logger.pop()
    asyncio.run(_run())

    spans = memory_logger.pop()
    by_name = {span["span_attributes"]["name"]: span for span in spans}
    assert "user.outer" in by_name
    assert "huggingface.chat_completion_stream" in by_name

    llm_span = by_name["huggingface.chat_completion_stream"]
    _assert_child_of(llm_span, parent_id=captured["outer_id"], root_id=captured["outer_root"])


# ---------------------------------------------------------------------------
# Auto-instrument
# ---------------------------------------------------------------------------


class TestAutoInstrumentHuggingFaceHub:
    def test_auto_instrument_huggingface_hub(self):
        verify_autoinstrument_script("test_auto_huggingface_hub.py")
