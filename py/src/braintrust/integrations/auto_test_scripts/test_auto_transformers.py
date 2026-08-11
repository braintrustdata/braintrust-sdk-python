"""Subprocess coverage for Transformers auto-instrumentation."""

# Keep the large Transformers/PyTorch dependencies isolated to their nox job.
# pylint: disable=import-error

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


results = auto_instrument(transformers=True)
assert results.get("transformers") is True
assert auto_instrument(transformers=True).get("transformers") is True

from transformers import pipeline  # noqa: E402


with autoinstrument_test_context("test_auto_transformers", use_vcr=False) as memory_logger:
    generator = pipeline(
        "text-generation",
        model="hf-internal-testing/tiny-random-LlamaForCausalLM",
        device=-1,
    )
    response = generator("Hello", do_sample=False, max_new_tokens=1)
    assert response

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "huggingface.transformers.text_generation"
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "transformers-auto"

print("SUCCESS")
