"""Test auto_instrument for HuggingFace Hub."""

import os


# Dummy token must start with ``hf_`` so the HuggingFace SDK accepts it for
# ``provider="auto"`` routing (validated locally before any HTTP request).
os.environ.setdefault("HF_TOKEN", "hf_test_dummy_api_key_for_vcr_tests")

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context
from huggingface_hub import InferenceClient


results = auto_instrument()
assert results.get("huggingface_hub") is True

results2 = auto_instrument()
assert results2.get("huggingface_hub") is True


CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


with autoinstrument_test_context("test_auto_huggingface_hub", integration="huggingface_hub") as memory_logger:
    # ``provider="cerebras"`` hosts ``meta-llama/Llama-3.1-8B-Instruct`` across
    # the matrix; ``hf-inference`` no longer hosts most conversational checkpoints.
    client = InferenceClient(model=CHAT_MODEL, provider="cerebras", token=os.environ["HF_TOKEN"])
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_tokens=10,
    )
    assert response.choices
    assert response.choices[0].message.role == "assistant"

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    # User-supplied ``provider`` overrides the default "huggingface" identity.
    assert span["metadata"]["provider"] == "cerebras"
    assert span["span_attributes"]["name"] == "huggingface.chat_completion"

print("SUCCESS")
