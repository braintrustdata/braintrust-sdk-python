# pyright: reportPrivateUsage=false

from pathlib import Path

import langsmith
import pytest
from braintrust import flush, logger
from braintrust.integrations.langchain import BraintrustCallbackHandler, set_global_handler
from braintrust.integrations.langsmith import setup_langsmith
from braintrust.integrations.langsmith.tracing import reset_langsmith_state
from braintrust.test_helpers import init_test_logger, simulate_login
from braintrust.wrappers.test_utils import verify_autoinstrument_script
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


PROJECT_NAME = "langsmith-py"
MODEL = "gpt-4o-mini"
EXPECTED_ANSWER = "1 + 2 equals 3."


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).resolve().parent / "cassettes")


@pytest.fixture
def logger_memory_logger():
    test_logger = init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield (test_logger, bgl)


@pytest.fixture(autouse=True)
def restore_langsmith_state():
    from braintrust.integrations.langchain.context import clear_global_handler

    sentinel = object()
    original_traceable = langsmith.__dict__.get("traceable", sentinel)
    original_evaluate = langsmith.__dict__.get("evaluate", sentinel)
    original_aevaluate = langsmith.__dict__.get("aevaluate", sentinel)
    original_client_evaluate = langsmith.Client.evaluate
    original_client_aevaluate = langsmith.Client.aevaluate

    clear_global_handler()
    yield
    if original_traceable is sentinel:
        langsmith.__dict__.pop("traceable", None)
    else:
        langsmith.traceable = original_traceable
    if original_evaluate is sentinel:
        langsmith.__dict__.pop("evaluate", None)
    else:
        langsmith.evaluate = original_evaluate
    if original_aevaluate is sentinel:
        langsmith.__dict__.pop("aevaluate", None)
    else:
        langsmith.aevaluate = original_aevaluate
    langsmith.Client.evaluate = original_client_evaluate
    langsmith.Client.aevaluate = original_client_aevaluate
    clear_global_handler()
    reset_langsmith_state()


def _make_chain():
    prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")
    model = ChatOpenAI(
        model=MODEL,
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
    )
    return prompt | model


@pytest.mark.vcr
def test_setup_langsmith_traceable_standalone_creates_span(logger_memory_logger):
    _, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    assert setup_langsmith(project_name=PROJECT_NAME, standalone=True)
    init_test_logger(PROJECT_NAME)
    chain = _make_chain()

    @langsmith.traceable(name="langsmith-standalone")
    def run_chain(inputs: dict[str, str]) -> dict[str, str]:
        return {"answer": chain.invoke(inputs).content}

    result = run_chain({"number": "2"})
    flush()

    assert result == {"answer": EXPECTED_ANSWER}

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "langsmith-standalone"
    assert span["output"] == {"answer": EXPECTED_ANSWER}


@pytest.mark.vcr
def test_setup_langsmith_uses_standalone_env_var(logger_memory_logger, monkeypatch):
    _, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    monkeypatch.setenv("BRAINTRUST_LANGSMITH_STANDALONE", "1")
    assert setup_langsmith(project_name=PROJECT_NAME)
    init_test_logger(PROJECT_NAME)
    chain = _make_chain()

    @langsmith.traceable(name="langsmith-standalone-env")
    def run_chain(inputs: dict[str, str]) -> dict[str, str]:
        return {"answer": chain.invoke(inputs).content}

    assert run_chain({"number": "2"}) == {"answer": EXPECTED_ANSWER}
    flush()

    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["span_attributes"]["name"] == "langsmith-standalone-env"


@pytest.mark.vcr
def test_setup_langsmith_module_evaluate_uses_braintrust_eval(logger_memory_logger):
    _, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    assert setup_langsmith(project_name=PROJECT_NAME, standalone=True)
    init_test_logger(PROJECT_NAME)
    chain = _make_chain()

    def task(inputs: dict[str, str]) -> dict[str, str]:
        return {"answer": chain.invoke(inputs).content}

    def evaluator(inputs, outputs, reference_outputs):
        return {
            "key": "match",
            "score": 1.0 if outputs["answer"] == reference_outputs["outputs"]["answer"] else 0.0,
        }

    result = langsmith.evaluate(
        task,
        data=[{"inputs": {"number": "2"}, "outputs": {"answer": EXPECTED_ANSWER}}],
        evaluators=[evaluator],
        experiment_prefix="langsmith-eval",
    )
    flush()

    assert result.results[0].output == {"answer": EXPECTED_ANSWER}
    assert result.results[0].scores == {"match": 1.0}

    spans = memory_logger.pop()
    assert [getattr(span["span_attributes"]["type"], "value", span["span_attributes"]["type"]) for span in spans] == [
        "eval",
        "task",
        "score",
    ]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_setup_langsmith_module_aevaluate_uses_braintrust_eval(logger_memory_logger):
    _, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    assert setup_langsmith(project_name=PROJECT_NAME, standalone=True)
    init_test_logger(PROJECT_NAME)
    chain = _make_chain()

    def task(inputs: dict[str, str]) -> dict[str, str]:
        return {"answer": chain.invoke(inputs).content}

    def evaluator(inputs, outputs, reference_outputs):
        return {
            "key": "match",
            "score": 1.0 if outputs["answer"] == reference_outputs["outputs"]["answer"] else 0.0,
        }

    result = await langsmith.aevaluate(
        task,
        data=[{"inputs": {"number": "2"}, "outputs": {"answer": EXPECTED_ANSWER}}],
        evaluators=[evaluator],
        experiment_prefix="langsmith-aeval",
    )
    flush()

    assert result.results[0].output == {"answer": EXPECTED_ANSWER}
    assert result.results[0].scores == {"match": 1.0}

    spans = memory_logger.pop()
    assert [getattr(span["span_attributes"]["type"], "value", span["span_attributes"]["type"]) for span in spans] == [
        "eval",
        "task",
        "score",
    ]


@pytest.mark.vcr
def test_langsmith_traceable_coexists_with_langchain(logger_memory_logger):
    _, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    assert setup_langsmith(project_name=PROJECT_NAME, standalone=True)

    simulate_login()
    test_logger = init_test_logger(PROJECT_NAME)
    handler = BraintrustCallbackHandler(logger=test_logger)
    set_global_handler(handler)

    from braintrust.integrations.langchain.context import get_global_handler

    assert get_global_handler() is handler
    assert setup_langsmith(project_name=PROJECT_NAME, standalone=True)
    assert get_global_handler() is handler

    chain = _make_chain()

    @langsmith.traceable(name="langsmith-chain")
    def run_chain(inputs: dict[str, str]):
        return chain.invoke(inputs)

    message = run_chain({"number": "2"})

    assert message.content == EXPECTED_ANSWER

    spans = memory_logger.pop()
    assert len(spans) >= 4

    traceable_span = next(span for span in spans if span["span_attributes"].get("name") == "langsmith-chain")
    llm_spans = [
        span
        for span in spans
        if getattr(span["span_attributes"].get("type"), "value", span["span_attributes"].get("type")) == "llm"
    ]

    assert len(llm_spans) == 1
    assert llm_spans[0]["metadata"]["model"] == "gpt-4o-mini-2024-07-18"
    assert llm_spans[0]["root_span_id"] == traceable_span["span_id"]


class TestAutoInstrumentLangSmith:
    def test_auto_instrument_langsmith(self):
        verify_autoinstrument_script("test_auto_langsmith.py")
