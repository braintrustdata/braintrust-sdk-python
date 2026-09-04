"""Subprocess coverage for Transformers auto-instrumentation."""

# Keep the large Transformers/PyTorch dependencies isolated to their nox job.
# pylint: disable=import-error

from braintrust.integrations.test_utils import run_auto_smoke


def _call(memory_logger):
    from transformers import pipeline

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


run_auto_smoke(
    "transformers",
    auto_instrument_kwargs={"transformers": True},
    use_vcr=False,
    run=_call,
)
print("SUCCESS")
