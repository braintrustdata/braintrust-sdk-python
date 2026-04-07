"""Test auto_instrument for OpenRouter."""

import os

import openrouter
from braintrust.auto import auto_instrument
from braintrust.wrappers.test_utils import autoinstrument_test_context


results = auto_instrument()
assert results.get("openrouter") == True

results2 = auto_instrument()
assert results2.get("openrouter") == True

with autoinstrument_test_context("test_auto_openrouter") as memory_logger:
    client = openrouter.OpenRouter(api_key=os.environ.get("OPENROUTER_API_KEY"))
    response = client.chat.send(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    assert "4" in response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["model"] == "gpt-4o-mini"
    assert "4" in span["output"][0]["message"]["content"]

print("SUCCESS")
