import importlib

import pytest
from braintrust import logger
from braintrust.integrations.langchain import BraintrustCallbackHandler, set_global_handler, setup_langchain
from braintrust.integrations.langchain.helpers import find_spans_by_attributes
from braintrust.test_helpers import init_test_logger


PROJECT_NAME = "deepagents-py"


@pytest.fixture
def logger_memory_logger():
    test_logger = init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield (test_logger, bgl)


@pytest.fixture(autouse=True)
def clear_langchain_handler():
    context = importlib.import_module("braintrust.integrations.langchain.context")
    context.clear_global_handler()
    yield
    context.clear_global_handler()


def _chat_model():
    langchain_openai = importlib.import_module("langchain_openai")
    return langchain_openai.ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_completion_tokens=20,
    )


def _create_deep_agent(*, name="bt-deep-agent"):
    deepagents = importlib.import_module("deepagents")
    return deepagents.create_deep_agent(
        model=_chat_model(),
        tools=[],
        name=name,
    )


def _invoke_agent(agent, content="Reply with exactly the word blue."):
    return agent.invoke({"messages": [{"role": "user", "content": content}]})


def _assert_deepagents_spans(spans, root_name):
    assert spans

    root_spans = find_spans_by_attributes(spans, name=root_name, type="task")
    llm_spans = find_spans_by_attributes(spans, name="ChatOpenAI", type="llm")
    assert len(root_spans) == 1
    assert len(llm_spans) >= 1

    root_span = root_spans[0]
    deepagents_metadata = root_span["metadata"]["metadata"]
    assert deepagents_metadata["ls_integration"] == "deepagents"
    assert deepagents_metadata["lc_agent_name"] == root_name
    if "lc_versions" in deepagents_metadata:
        assert "deepagents" in deepagents_metadata["lc_versions"]
    assert "messages" in root_span["output"]


@pytest.mark.vcr
def test_setup_langchain_traces_deepagents_run(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    assert setup_langchain(project_name=PROJECT_NAME)
    set_global_handler(BraintrustCallbackHandler(logger=test_logger))

    agent = _create_deep_agent(name="bt-deep-agent")
    _invoke_agent(agent)

    _assert_deepagents_spans(memory_logger.pop(), "bt-deep-agent")


@pytest.mark.vcr
def test_langchain_callback_traces_prebuilt_deep_agent(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    stdout = importlib.import_module("langchain_core.tracers.stdout")
    existing_callback = stdout.ConsoleCallbackHandler()
    handler = BraintrustCallbackHandler(logger=test_logger)

    agent = _create_deep_agent(name="manual-agent").with_config(
        {
            "callbacks": [existing_callback, handler],
            "metadata": {"user_metadata": "kept"},
        }
    )
    assert existing_callback in agent.config["callbacks"]
    assert handler in agent.config["callbacks"]

    _invoke_agent(agent, content="Reply with exactly the word green.")

    spans = memory_logger.pop()
    _assert_deepagents_spans(spans, "manual-agent")
    root_spans = find_spans_by_attributes(spans, name="manual-agent", type="task")
    assert root_spans[0]["metadata"]["metadata"]["user_metadata"] == "kept"
