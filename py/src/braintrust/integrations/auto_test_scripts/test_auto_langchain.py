"""Test auto_instrument for LangChain."""

from braintrust.auto import auto_instrument
from braintrust.integrations.langchain import BraintrustCallbackHandler
from braintrust.integrations.langchain.context import clear_global_handler, get_global_handler
from braintrust.integrations.test_utils import autoinstrument_test_context
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# 1. Verify not patched initially.
clear_global_handler()
assert get_global_handler() is None
manager = CallbackManager.configure()
assert next((h for h in manager.handlers if isinstance(h, BraintrustCallbackHandler)), None) is None

# 2. Instrument.
results = auto_instrument()
assert results.get("langchain") == True
handler = get_global_handler()
assert isinstance(handler, BraintrustCallbackHandler)

manager = CallbackManager.configure()
assert next((h for h in manager.handlers if isinstance(h, BraintrustCallbackHandler)), None) is handler

# 3. Idempotent.
results2 = auto_instrument()
assert results2.get("langchain") == True
assert get_global_handler() is handler

# 4. Make an API call and verify spans.
with autoinstrument_test_context("test_global_handler", integration="langchain") as memory_logger:
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

print("SUCCESS")
