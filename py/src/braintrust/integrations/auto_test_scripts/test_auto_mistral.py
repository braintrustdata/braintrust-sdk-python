"""Test auto_instrument for Mistral."""

import os
from pathlib import Path

from braintrust.auto import auto_instrument
from braintrust.wrappers.test_utils import autoinstrument_test_context


try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral


results = auto_instrument()
assert results.get("mistral") == True

results2 = auto_instrument()
assert results2.get("mistral") == True

MISTRAL_CASSETTES_DIR = Path(__file__).resolve().parent.parent / "mistral" / "cassettes"

with autoinstrument_test_context("test_auto_mistral", cassettes_dir=MISTRAL_CASSETTES_DIR) as memory_logger:
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
    )
    assert "4" in str(response.choices[0].message.content)

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "mistral"
    assert span["metadata"]["model"] == "mistral-small-latest"
    assert "4" in str(span["output"])

print("SUCCESS")
