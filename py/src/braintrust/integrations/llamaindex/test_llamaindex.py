"""Tests for the LlamaIndex integration.

These tests verify that LlamaIndex operations produce correct Braintrust spans
with proper hierarchy, input/output data, and metrics extraction.
"""

import pytest
from unittest.mock import ANY

from braintrust import logger
from braintrust.integrations.llamaindex import LlamaIndexIntegration, BraintrustSpanHandler
from braintrust.test_helpers import init_test_logger

from .helpers import assert_matches_object, find_spans_by_attributes


PROJECT_NAME = "llamaindex-py"


@pytest.fixture
def logger_memory_logger():
    test_logger = init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield (test_logger, bgl)


@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """Ensure handlers are registered for each test and cleaned up after."""
    from llama_index.core.instrumentation import get_dispatcher

    LlamaIndexIntegration.setup()

    yield

    # Clean up handlers to avoid cross-test interference
    dispatcher = get_dispatcher()
    dispatcher.span_handlers = [
        h for h in dispatcher.span_handlers if not isinstance(h, BraintrustSpanHandler)
    ]
    from braintrust.integrations.llamaindex.event_handler import BraintrustEventHandler

    dispatcher.event_handlers = [
        h for h in dispatcher.event_handlers if not isinstance(h, BraintrustEventHandler)
    ]


def test_integration_setup():
    """Test that setup registers handlers on the dispatcher."""
    from llama_index.core.instrumentation import get_dispatcher

    dispatcher = get_dispatcher()
    handler_types = [type(h).__name__ for h in dispatcher.span_handlers]
    assert "BraintrustSpanHandler" in handler_types

    event_handler_types = [type(h).__name__ for h in dispatcher.event_handlers]
    assert "BraintrustEventHandler" in event_handler_types


def test_integration_idempotent():
    """Calling setup() twice doesn't register duplicate handlers."""
    from llama_index.core.instrumentation import get_dispatcher

    LlamaIndexIntegration.setup()
    LlamaIndexIntegration.setup()

    dispatcher = get_dispatcher()
    bt_handlers = [h for h in dispatcher.span_handlers if isinstance(h, BraintrustSpanHandler)]
    assert len(bt_handlers) == 1


def test_auto_instrument_includes_llamaindex():
    """auto_instrument() should include llamaindex."""
    from braintrust.auto import auto_instrument

    result = auto_instrument()
    assert "llamaindex" in result
    assert result["llamaindex"] is True


@pytest.mark.vcr
def test_llm_complete(logger_memory_logger):
    """Test that LLM.complete() creates a span with input/output."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    with test_logger.start_span(name="test-complete") as parent:
        response = llm.complete("What is 2+2? Answer with just the number.")

    spans = memory_logger.pop()
    assert len(spans) >= 2

    # Find the LLM span
    llm_spans = find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1

    llm_span = llm_spans[0]
    assert llm_span["input"] is not None
    assert llm_span["output"] is not None
    assert "metadata" in llm_span
    assert llm_span["metadata"]["class"] == "OpenAI"


@pytest.mark.vcr
def test_llm_chat(logger_memory_logger):
    """Test that LLM.chat() creates a span with messages."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
    ]

    with test_logger.start_span(name="test-chat") as parent:
        response = llm.chat(messages)

    spans = memory_logger.pop()
    assert len(spans) >= 2

    llm_spans = find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1

    llm_span = llm_spans[0]
    assert llm_span["input"] is not None
    assert llm_span["output"] is not None

    # Output should contain the response content
    output = llm_span["output"]
    assert isinstance(output, dict)
    assert "content" in output or "role" in output


@pytest.mark.vcr
def test_llm_chat_metrics(logger_memory_logger):
    """Test that token metrics are extracted from LLM responses."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    with test_logger.start_span(name="test-metrics") as parent:
        response = llm.complete("Say hello")

    spans = memory_logger.pop()
    llm_spans = find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1

    llm_span = llm_spans[0]
    metrics = llm_span.get("metrics", {})
    # Token metrics should be present (extracted from raw response)
    assert "prompt_tokens" in metrics or "total_tokens" in metrics or "completion_tokens" in metrics


@pytest.mark.vcr
def test_document_processing(logger_memory_logger):
    """Test that document splitting creates spans."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    docs = [
        Document(text="Paris is the capital of France. The Eiffel Tower is in Paris."),
        Document(text="Berlin is the capital of Germany. The Brandenburg Gate is in Berlin."),
    ]

    splitter = SentenceSplitter(chunk_size=64, chunk_overlap=10)

    with test_logger.start_span(name="test-docproc") as parent:
        nodes = splitter.get_nodes_from_documents(docs)

    spans = memory_logger.pop()
    # Should have parent span + at least one SentenceSplitter span
    assert len(spans) >= 2

    # Find function spans (SentenceSplitter spans)
    func_spans = find_spans_by_attributes(spans, type="function")
    assert len(func_spans) >= 1

    func_span = func_spans[0]
    assert "SentenceSplitter" in func_span["span_attributes"]["name"]


@pytest.mark.vcr
def test_embedding(logger_memory_logger):
    """Test that embedding calls create spans with metadata."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.embeddings.openai import OpenAIEmbedding

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    with test_logger.start_span(name="test-embedding") as parent:
        embedding = embed_model.get_text_embedding("Hello world")

    spans = memory_logger.pop()
    assert len(spans) >= 2

    # Should have an embedding span
    func_spans = find_spans_by_attributes(spans, type="function")
    assert len(func_spans) >= 1

    embed_span = func_spans[0]
    assert "OpenAIEmbedding" in embed_span["span_attributes"]["name"]


@pytest.mark.vcr
def test_query_engine(logger_memory_logger):
    """Test that a query engine creates a full span hierarchy."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core import Document, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    docs = [
        Document(text="The capital of France is Paris. Paris has a population of 2.1 million."),
        Document(text="The Eiffel Tower is located in Paris, France. It was built in 1889."),
    ]

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    with test_logger.start_span(name="test-query-engine") as parent:
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query("What is the capital of France?")

    spans = memory_logger.pop()
    # Should have many spans: parent, query engine, retriever, synthesizer, LLM, embeddings
    assert len(spans) >= 4

    # Verify span types present
    span_types = {s.get("span_attributes", {}).get("type") for s in spans}
    assert "task" in span_types  # query engine and/or parent
    assert "llm" in span_types or "function" in span_types  # LLM call or embedding


@pytest.mark.vcr
def test_span_hierarchy(logger_memory_logger):
    """Test that spans maintain proper parent-child relationships."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    docs = [Document(text="Hello world. This is a test document with some content.")]
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=10)

    with test_logger.start_span(name="test-hierarchy") as parent:
        nodes = splitter.get_nodes_from_documents(docs)

    spans = memory_logger.pop()
    assert len(spans) >= 2

    # The manually created parent span
    parent_span = spans[0]
    root_span_id = parent_span["root_span_id"]

    # All spans should share the same root
    for span in spans:
        assert span["root_span_id"] == root_span_id


def test_llm_error_handling(logger_memory_logger):
    """Test that errors in LLM calls are captured."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", api_key="sk-invalid-key")

    with test_logger.start_span(name="test-error") as parent:
        try:
            response = llm.complete("Hello")
        except Exception:
            pass

    spans = memory_logger.pop()
    assert len(spans) >= 2

    # Find LLM span - should have error
    llm_spans = find_spans_by_attributes(spans, type="llm")
    if llm_spans:
        llm_span = llm_spans[0]
        assert llm_span.get("error") is not None


@pytest.mark.vcr
async def test_async_llm_complete(logger_memory_logger):
    """Test that async LLM.acomplete() creates spans."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    with test_logger.start_span(name="test-async-complete") as parent:
        response = await llm.acomplete("What is 2+2? Answer with just the number.")

    spans = memory_logger.pop()
    assert len(spans) >= 2

    llm_spans = find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1


@pytest.mark.vcr
async def test_async_llm_chat(logger_memory_logger):
    """Test that async LLM.achat() creates spans."""
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        ChatMessage(role=MessageRole.USER, content="Say hello"),
    ]

    with test_logger.start_span(name="test-async-chat") as parent:
        response = await llm.achat(messages)

    spans = memory_logger.pop()
    assert len(spans) >= 2

    llm_spans = find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1
