"""Test auto_instrument for Mistral."""

import os

from braintrust.integrations.test_utils import run_auto_smoke


try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral


def _call(memory_logger):
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


run_auto_smoke("mistral", integration="mistral", run=_call)
print("SUCCESS")
