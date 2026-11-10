"""Test auto_instrument for Cohere."""

import os


os.environ.setdefault("CO_API_KEY", "co-test-dummy-api-key-for-vcr-tests")
os.environ.setdefault("COHERE_API_KEY", os.environ["CO_API_KEY"])

import cohere
from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


results = auto_instrument()
assert results.get("cohere") is True

results2 = auto_instrument()
assert results2.get("cohere") is True


TEST_MODEL = "command-r7b-12-2024"


with autoinstrument_test_context("test_auto_cohere", integration="cohere") as memory_logger:
    use_v2 = hasattr(cohere, "ClientV2") and hasattr(cohere.ClientV2, "chat")
    if use_v2:
        client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
        response = client.chat(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
        )
        assert response.message.role == "assistant"
    else:
        client = cohere.Client(api_key=os.environ["CO_API_KEY"])
        response = client.chat(
            model=TEST_MODEL,
            message="Say hi in one word.",
            max_tokens=10,
        )
        assert isinstance(response.text, str)

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["span_attributes"]["name"] == "cohere.chat"

print("SUCCESS")
