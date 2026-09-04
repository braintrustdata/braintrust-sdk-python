"""Test auto_instrument for Cohere."""

import os


os.environ.setdefault("CO_API_KEY", "co-test-dummy-api-key-for-vcr-tests")
os.environ.setdefault("COHERE_API_KEY", os.environ["CO_API_KEY"])

import cohere
from braintrust.integrations.test_utils import run_auto_smoke


def _call(memory_logger):
    use_v2 = hasattr(cohere, "ClientV2") and hasattr(cohere.ClientV2, "chat")
    if use_v2:
        client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
        response = client.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
        )
        assert response.message.role == "assistant"
    else:
        client = cohere.Client(api_key=os.environ["CO_API_KEY"])
        response = client.chat(
            model="command-a-03-2025",
            message="Say hi in one word.",
            max_tokens=10,
        )
        assert isinstance(response.text, str)

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == "command-a-03-2025"
    assert span["span_attributes"]["name"] == "cohere.chat"


run_auto_smoke("cohere", integration="cohere", run=_call)
print("SUCCESS")
