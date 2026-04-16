"""Test auto_instrument for Anthropic."""

import anthropic
from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


# 1. Verify not patched initially
original_sync_module = type(anthropic.Anthropic(api_key="test-key").messages).__module__
original_async_module = type(anthropic.AsyncAnthropic(api_key="test-key").messages).__module__

# 2. Instrument
results = auto_instrument()
assert results.get("anthropic") == True

patched_sync = anthropic.Anthropic(api_key="test-key")
patched_async = anthropic.AsyncAnthropic(api_key="test-key")
assert type(patched_sync.messages).__module__ == "braintrust.integrations.anthropic.tracing"
assert type(patched_async.messages).__module__ == "braintrust.integrations.anthropic.tracing"
assert type(patched_sync.messages).__module__ != original_sync_module
assert type(patched_async.messages).__module__ != original_async_module

# 3. Idempotent
results2 = auto_instrument()
assert results2.get("anthropic") == True

# 4. Make API call and verify span
with autoinstrument_test_context("test_auto_anthropic", integration="anthropic") as memory_logger:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hi"}],
    )
    assert response.content[0].text

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "anthropic"
    assert "claude" in span["metadata"]["model"]

print("SUCCESS")
