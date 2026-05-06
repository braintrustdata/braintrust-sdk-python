# pyright: reportTypedDictNotRequiredAccess=none
from unittest.mock import ANY

import pytest
from braintrust import logger
from braintrust.integrations.langchain import BraintrustCallbackHandler, set_global_handler, setup_langchain
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from .helpers import assert_matches_object


PROJECT_NAME = "langchain-py"


@pytest.fixture
def logger_memory_logger():
    test_logger = init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield (test_logger, bgl)


@pytest.fixture(autouse=True)
def clear_handler():
    from braintrust.integrations.langchain.context import clear_global_handler

    clear_global_handler()
    yield
    clear_global_handler()


@pytest.mark.vcr
def test_global_handler(logger_memory_logger):
    test_logger, memory_logger = logger_memory_logger
    assert not memory_logger.pop()

    handler = BraintrustCallbackHandler(logger=test_logger, debug=True)
    set_global_handler(handler)

    # Make sure the handler is registered in the LangChain library
    manager = CallbackManager.configure()
    assert next((h for h in manager.handlers if isinstance(h, BraintrustCallbackHandler)), None) == handler

    # Here's what a typical user would do
    prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
    )
    chain: RunnableSerializable[dict[str, str], BaseMessage] = prompt.pipe(model)

    message = chain.invoke({"number": "2"})

    spans = memory_logger.pop()
    assert len(spans) > 0

    root_span_id = spans[0]["span_id"]

    # Spans would be empty if the handler was not registered, let's make sure it logged what we expect
    assert_matches_object(
        spans,
        [
            {
                "span_attributes": {
                    "name": "RunnableSequence",
                    "type": "task",
                },
                "input": {"number": "2"},
                "output": {
                    "content": ANY,  # LLM response text
                    "additional_kwargs": ANY,
                    "response_metadata": ANY,
                    "type": "ai",
                },
                "metadata": {"tags": []},
                "span_id": root_span_id,
                "root_span_id": root_span_id,
            },
            {
                "span_attributes": {"name": "ChatPromptTemplate"},
                "input": {"number": "2"},
                "output": {
                    "messages": [
                        {
                            "content": ANY,  # Formatted prompt text
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "type": "human",
                        }
                    ]
                },
                "metadata": {"tags": ["seq:step:1"]},
                "root_span_id": root_span_id,
                "span_parents": [root_span_id],
            },
            {
                "span_attributes": {"name": "ChatOpenAI", "type": "llm"},
                "input": [
                    [
                        {
                            "content": ANY,  # Prompt message content
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "type": "human",
                        }
                    ]
                ],
                "output": {
                    "generations": [
                        [
                            {
                                "text": ANY,  # Generated text
                                "generation_info": ANY,
                                "type": "ChatGeneration",
                                "message": {
                                    "content": ANY,  # Message content
                                    "additional_kwargs": ANY,
                                    "response_metadata": ANY,
                                    "type": "ai",
                                },
                            }
                        ]
                    ],
                    "llm_output": {
                        "model_name": "gpt-4o-mini-2024-07-18",
                    },
                    "type": "LLMResult",
                },
                "metrics": {
                    "start": ANY,
                    "total_tokens": ANY,
                    "prompt_tokens": ANY,
                    "completion_tokens": ANY,
                    "end": ANY,
                },
                "metadata": {
                    "tags": ["seq:step:2"],
                    "model": "gpt-4o-mini-2024-07-18",
                },
                "root_span_id": root_span_id,
                "span_parents": [root_span_id],
            },
        ],
    )

    assert message.content == "1 + 2 equals 3."


def test_setup_langchain_installs_default_handler():
    from braintrust.integrations.langchain.context import get_global_handler

    manager = CallbackManager.configure()
    assert next((h for h in manager.handlers if isinstance(h, BraintrustCallbackHandler)), None) is None
    assert get_global_handler() is None

    assert setup_langchain()
    handler = get_global_handler()
    assert isinstance(handler, BraintrustCallbackHandler)

    manager = CallbackManager.configure()
    assert next((h for h in manager.handlers if isinstance(h, BraintrustCallbackHandler)), None) is handler

    assert setup_langchain()
    assert get_global_handler() is handler


class TestAutoInstrumentLangChain:
    def test_auto_instrument_langchain(self):
        verify_autoinstrument_script("test_auto_langchain.py")
