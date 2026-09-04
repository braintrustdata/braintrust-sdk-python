"""Test auto_instrument for LangChain."""

from braintrust.integrations.langchain import BraintrustCallbackHandler
from braintrust.integrations.langchain.context import clear_global_handler, get_global_handler
from braintrust.integrations.test_utils import run_auto_smoke
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Ensure a clean starting state so the pre-check reflects a fresh process.
clear_global_handler()
manager = CallbackManager.configure()
assert next((h for h in manager.handlers if isinstance(h, BraintrustCallbackHandler)), None) is None


def _is_patched() -> bool:
    return isinstance(get_global_handler(), BraintrustCallbackHandler)


def _call(memory_logger):
    prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
    )
    chain = prompt.pipe(model)

    message = chain.invoke({"number": "2"})
    assert message.content == "1 + 2 equals 3."

    spans = memory_logger.pop()
    assert len(spans) > 0


run_auto_smoke(
    "langchain",
    is_patched=_is_patched,
    cassette="test_global_handler",
    integration="langchain",
    run=_call,
)

# The handler installed via auto_instrument must also flow through CallbackManager.configure().
handler = get_global_handler()
manager = CallbackManager.configure()
assert next((h for h in manager.handlers if isinstance(h, BraintrustCallbackHandler)), None) is handler

print("SUCCESS")
