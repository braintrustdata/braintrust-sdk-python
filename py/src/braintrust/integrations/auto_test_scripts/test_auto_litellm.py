"""Test auto_instrument for LiteLLM."""

import litellm
from braintrust.integrations.litellm import LiteLLMIntegration
from braintrust.integrations.test_utils import run_auto_smoke


def _is_patched() -> bool:
    return LiteLLMIntegration.patchers[0].is_patched(litellm, None)


def _call(memory_logger):
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi"}],
    )
    assert response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "openai"


# Disable OpenAI auto-instrumentation here because LiteLLM's OpenAI-backed
# chat path can otherwise produce both a LiteLLM span and an OpenAI span.
# This test is meant to validate LiteLLM instrumentation in isolation.
run_auto_smoke(
    "litellm",
    auto_instrument_kwargs={"openai": False},
    is_patched=_is_patched,
    integration="litellm",
    run=_call,
)
print("SUCCESS")
