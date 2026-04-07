"""Test auto_instrument for Cohere."""

import os
from pathlib import Path

from braintrust.auto import auto_instrument
from braintrust.wrappers.test_utils import autoinstrument_test_context
from cohere import ClientV2


results = auto_instrument()
assert results.get("cohere") == True

results2 = auto_instrument()
assert results2.get("cohere") == True

COHERE_CASSETTES_DIR = Path(__file__).resolve().parent.parent / "cohere" / "cassettes"

with autoinstrument_test_context("test_auto_cohere", cassettes_dir=COHERE_CASSETTES_DIR) as memory_logger:
    client = ClientV2(api_key=os.environ.get("CO_API_KEY"))
    response = client.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    assert response.message.content[0].text == "4"

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["api_version"] == "2"
    assert span["metadata"]["model"] == "command-a-03-2025"
    assert span["output"]["content"][0]["text"] == "4"

print("SUCCESS")
