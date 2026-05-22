"""Tests for the LlamaIndex integration."""

import pytest
from braintrust import logger
from braintrust.integrations.llamaindex import BraintrustSpanHandler, LlamaIndexIntegration
from braintrust.test_helpers import init_test_logger


PROJECT_NAME = "llamaindex-py"


def _find_spans_by_attributes(spans, **attributes):
    result = []
    for span in spans:
        span_attrs = span.get("span_attributes") or {}
        if all(span_attrs.get(k) == v for k, v in attributes.items()):
            result.append(span)
    return result


@pytest.fixture
def logger_memory_logger():
    test_logger = init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield (test_logger, bgl)


@pytest.fixture(autouse=True)
def setup_and_cleanup():
    from llama_index.core.instrumentation import get_dispatcher

    LlamaIndexIntegration.setup()
    yield

    dispatcher = get_dispatcher()
    dispatcher.span_handlers = [h for h in dispatcher.span_handlers if not isinstance(h, BraintrustSpanHandler)]


def test_integration_setup():
    from llama_index.core.instrumentation import get_dispatcher

    dispatcher = get_dispatcher()
    handler_types = [type(h).__name__ for h in dispatcher.span_handlers]
    assert "BraintrustSpanHandler" in handler_types


def test_integration_idempotent():
    from llama_index.core.instrumentation import get_dispatcher

    LlamaIndexIntegration.setup()
    LlamaIndexIntegration.setup()

    dispatcher = get_dispatcher()
    bt_handlers = [h for h in dispatcher.span_handlers if isinstance(h, BraintrustSpanHandler)]
    assert len(bt_handlers) == 1


def test_auto_instrument_includes_llamaindex():
    from braintrust.auto import auto_instrument

    result = auto_instrument()
    assert "llamaindex" in result
    assert result["llamaindex"] is True


@pytest.mark.asyncio
async def test_streaming_outputs_are_not_stringified():
    from braintrust.integrations.llamaindex.tracing import _extract_response_output

    def stream():
        yield "chunk"

    async def async_stream():
        yield "chunk"

    async_gen = async_stream()
    try:
        assert _extract_response_output(stream()) is None
        assert _extract_response_output(async_gen) is None
    finally:
        await async_gen.aclose()


@pytest.mark.asyncio
async def test_coroutine_outputs_are_not_stringified():
    from braintrust.integrations.llamaindex.tracing import _extract_response_output

    async def coroutine():
        return "result"

    coro = coroutine()
    try:
        assert _extract_response_output(coro) is None
    finally:
        getattr(coro, "close")()


@pytest.mark.vcr
def test_llm_complete(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    with test_logger.start_span(name="test-complete"):
        llm.complete("What is 2+2? Answer with just the number.")

    spans = memory_logger.pop()
    assert len(spans) >= 2

    llm_spans = _find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1

    llm_span = llm_spans[0]
    assert llm_span["span_attributes"]["name"] == "OpenAI"
    assert llm_span["input"] is not None
    assert llm_span["output"] is not None
    assert llm_span["metadata"]["class"] == "OpenAI"
    assert llm_span["metadata"]["model"] == "gpt-4o-mini"


@pytest.mark.vcr
def test_llm_chat(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
    ]

    with test_logger.start_span(name="test-chat"):
        llm.chat(messages)

    spans = memory_logger.pop()
    assert len(spans) >= 2

    llm_spans = _find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1

    llm_span = llm_spans[0]
    assert llm_span["input"] is not None
    assert llm_span["output"] is not None
    assert isinstance(llm_span["output"], dict)
    assert "content" in llm_span["output"] or "role" in llm_span["output"]


def test_document_processing(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    docs = [
        Document(text="Paris is the capital of France. The Eiffel Tower is in Paris."),
        Document(text="Berlin is the capital of Germany. The Brandenburg Gate is in Berlin."),
    ]

    splitter = SentenceSplitter(chunk_size=64, chunk_overlap=10)

    with test_logger.start_span(name="test-docproc"):
        splitter.get_nodes_from_documents(docs)

    spans = memory_logger.pop()
    assert len(spans) >= 2

    func_spans = _find_spans_by_attributes(spans, type="function")
    assert len(func_spans) >= 1
    assert "SentenceSplitter" in func_spans[0]["span_attributes"]["name"]


@pytest.mark.vcr
def test_embedding(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.embeddings.openai import OpenAIEmbedding

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    with test_logger.start_span(name="test-embedding"):
        embed_model.get_text_embedding("Hello world")

    spans = memory_logger.pop()
    assert len(spans) >= 2

    func_spans = _find_spans_by_attributes(spans, type="function")
    assert len(func_spans) >= 1
    assert "OpenAIEmbedding" in func_spans[0]["span_attributes"]["name"]


@pytest.mark.vcr
def test_query_engine(logger_memory_logger):
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

    with test_logger.start_span(name="test-query-engine"):
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        query_engine = index.as_query_engine(llm=llm)
        query_engine.query("What is the capital of France?")

    spans = memory_logger.pop()
    assert len(spans) >= 4

    span_types = {s.get("span_attributes", {}).get("type") for s in spans}
    assert "task" in span_types
    assert "llm" in span_types or "function" in span_types


def test_span_hierarchy(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    docs = [Document(text="Hello world. This is a test document with some content.")]
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=10)

    with test_logger.start_span(name="test-hierarchy"):
        splitter.get_nodes_from_documents(docs)

    spans = memory_logger.pop()
    assert len(spans) >= 2

    root_span_id = spans[0]["root_span_id"]
    for span in spans:
        assert span["root_span_id"] == root_span_id


@pytest.mark.vcr
def test_llm_error_handling(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", api_key="sk-invalid-key")

    with test_logger.start_span(name="test-error"):
        try:
            llm.complete("Hello")
        except Exception:
            pass

    spans = memory_logger.pop()
    assert len(spans) >= 2

    llm_spans = _find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1
    assert llm_spans[0].get("error") is not None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_llm_complete(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    with test_logger.start_span(name="test-async-complete"):
        await llm.acomplete("What is 2+2? Answer with just the number.")

    spans = memory_logger.pop()
    assert len(spans) >= 2

    llm_spans = _find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_llm_chat(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        ChatMessage(role=MessageRole.USER, content="Say hello"),
    ]

    with test_logger.start_span(name="test-async-chat"):
        await llm.achat(messages)

    spans = memory_logger.pop()
    assert len(spans) >= 2

    llm_spans = _find_spans_by_attributes(spans, type="llm")
    assert len(llm_spans) >= 1
