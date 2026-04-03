"""Test auto_instrument for LangSmith."""

import os
from pathlib import Path

import langsmith.client
import langsmith.evaluation._arunner
import langsmith.evaluation._runner
import langsmith.run_helpers
from braintrust.auto import auto_instrument
from braintrust.wrappers.test_utils import autoinstrument_test_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


_CASSETTES_DIR = Path(__file__).resolve().parent.parent / "langsmith" / "cassettes"


# 1. Verify not patched initially.
assert not getattr(langsmith.run_helpers.traceable, "__braintrust_patched_langsmith_traceable__", False)
assert not getattr(langsmith.evaluation._runner.evaluate, "__braintrust_patched_langsmith_evaluate_sync__", False)
assert not getattr(langsmith.evaluation._arunner.aevaluate, "__braintrust_patched_langsmith_evaluate_async__", False)
assert not getattr(langsmith.client.Client.evaluate, "__braintrust_patched_langsmith_client_evaluate__", False)
assert not getattr(langsmith.client.Client.aevaluate, "__braintrust_patched_langsmith_client_aevaluate__", False)


# 2. Instrument with standalone mode so only Braintrust runs.
os.environ["BRAINTRUST_LANGSMITH_STANDALONE"] = "1"
results = auto_instrument(
    openai=False,
    anthropic=False,
    litellm=False,
    pydantic_ai=False,
    google_genai=False,
    openrouter=False,
    agno=False,
    agentscope=False,
    claude_agent_sdk=False,
    dspy=False,
    adk=False,
    langchain=False,
    langsmith=True,
)
assert results.get("langsmith") == True

assert getattr(langsmith.run_helpers.traceable, "__braintrust_patched_langsmith_traceable__", False)
assert getattr(langsmith.evaluation._runner.evaluate, "__braintrust_patched_langsmith_evaluate_sync__", False)
assert getattr(langsmith.evaluation._arunner.aevaluate, "__braintrust_patched_langsmith_evaluate_async__", False)
assert getattr(langsmith.client.Client.evaluate, "__braintrust_patched_langsmith_client_evaluate__", False)
assert getattr(langsmith.client.Client.aevaluate, "__braintrust_patched_langsmith_client_aevaluate__", False)


# 3. Idempotent.
results2 = auto_instrument(
    openai=False,
    anthropic=False,
    litellm=False,
    pydantic_ai=False,
    google_genai=False,
    openrouter=False,
    agno=False,
    agentscope=False,
    claude_agent_sdk=False,
    dspy=False,
    adk=False,
    langchain=False,
    langsmith=True,
)
assert results2.get("langsmith") == True


# 4. Make an API call and verify span.
with autoinstrument_test_context("test_auto_langsmith", cassettes_dir=_CASSETTES_DIR) as memory_logger:
    prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
    )
    chain = prompt | model

    @langsmith.traceable(name="auto-langsmith")
    def run_chain(inputs: dict[str, str]) -> dict[str, str]:
        return {"answer": chain.invoke(inputs).content}

    result = run_chain({"number": "2"})
    assert result == {"answer": "1 + 2 equals 3."}

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["span_attributes"]["name"] == "auto-langsmith"
    assert span["output"] == {"answer": "1 + 2 equals 3."}

print("SUCCESS")
