"""Test auto_instrument for LiteLLM."""

import asyncio

import litellm
from braintrust.auto import auto_instrument
from braintrust.integrations.litellm import LiteLLMIntegration
from braintrust.integrations.test_decisions import (
    DECISIONS_INPUT,
    DECISIONS_MODEL,
    DECISIONS_QUESTIONS,
    assert_decisions_span,
)
from braintrust.integrations.test_utils import autoinstrument_test_context
from braintrust.integrations.versioning import detect_module_version, version_satisfies


# 1. Verify not patched initially
assert not LiteLLMIntegration.patchers[0].is_patched(litellm, None)

# 2. Instrument
# Disable OpenAI auto-instrumentation here because LiteLLM's OpenAI-backed
# chat path can otherwise produce both a LiteLLM span and an OpenAI span.
# This test is meant to validate LiteLLM instrumentation in isolation.
results = auto_instrument(openai=False)
assert results.get("litellm") == True
assert LiteLLMIntegration.patchers[0].is_patched(litellm, None)

# 3. Idempotent
results2 = auto_instrument(openai=False)
assert results2.get("litellm") == True

# 4. Make API call and verify span
with autoinstrument_test_context("test_auto_litellm", integration="litellm") as memory_logger:
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi"}],
    )
    assert response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "openai"

if version_satisfies(detect_module_version(litellm, ("litellm",)), ">=1.105.0"):
    with autoinstrument_test_context("test_litellm_decisions_metrics[async]", integration="litellm") as memory_logger:
        response = asyncio.run(
            litellm.adecisions(
                model=f"openai/{DECISIONS_MODEL}",
                input=DECISIONS_INPUT,
                questions=DECISIONS_QUESTIONS,
                safety_identifier="do-not-log-this-identifier",
            )
        )
        assert_decisions_span(
            memory_logger.pop(),
            response,
            name="Decision",
            model=f"openai/{DECISIONS_MODEL}",
            instrumentation="litellm-auto",
        )

    with autoinstrument_test_context("test_litellm_systemone_metrics[async]", integration="litellm") as memory_logger:
        response = asyncio.run(
            litellm.asystemone(
                f"openai/{DECISIONS_MODEL}",
                DECISIONS_INPUT,
                {"positive": {"type": "noul", "instructions": "Is the sentiment positive?"}},
            )
        )
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["type"] == "decision"
        assert span["input"]["state"] == DECISIONS_INPUT
        assert span["output"]["answers"] == [
            {"id": "positive", "type": "noul", "noul": response.answers["positive"].noul}
        ]
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "litellm-auto"

print("SUCCESS")
