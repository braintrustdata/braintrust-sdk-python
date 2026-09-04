"""Test auto_instrument for Anthropic."""

import os

import anthropic
from braintrust.integrations.test_utils import run_auto_smoke


_TRACING_MODULE = "braintrust.integrations.anthropic.tracing"


def _is_patched() -> bool:
    return (
        type(anthropic.Anthropic(api_key="test-key").messages).__module__ == _TRACING_MODULE
        and type(anthropic.AsyncAnthropic(api_key="test-key").messages).__module__ == _TRACING_MODULE
    )


def _call(memory_logger):
    model = (
        "claude-haiku-4-5-20251001"
        if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") == "latest"
        else "claude-3-haiku-20240307"
    )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hi"}],
    )
    assert response.content[0].text

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "anthropic"
    assert "claude" in span["metadata"]["model"]


run_auto_smoke("anthropic", is_patched=_is_patched, integration="anthropic", run=_call)
print("SUCCESS")
