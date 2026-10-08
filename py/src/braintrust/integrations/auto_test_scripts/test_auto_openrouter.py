"""Test auto_instrument for OpenRouter."""

import importlib.util
import os

import openrouter
from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


results = auto_instrument()
assert results.get("openrouter") == True

results2 = auto_instrument()
assert results2.get("openrouter") == True

with autoinstrument_test_context("test_auto_openrouter", integration="openrouter") as memory_logger:
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

if importlib.util.find_spec("openrouter.beta_responses") is not None:
    with autoinstrument_test_context(
        "test_wrap_openrouter_beta_responses_send", integration="openrouter"
    ) as memory_logger:
        client = openrouter.OpenRouter(api_key=os.environ.get("OPENROUTER_API_KEY"))
        response = client.beta.responses.send(
            model="openai/gpt-4o-mini",
            input="Say one short sentence about observability.",
            max_output_tokens=64,
            temperature=0,
        )
        assert response.output

        spans = memory_logger.pop()
        assert len(spans) == 1, f"Expected 1 beta Responses span, got {len(spans)}"
        span = spans[0]
        assert span["span_attributes"]["name"] == "openrouter.beta.responses.send"
        assert span["input"] == "Say one short sentence about observability."
        assert span["metadata"]["model"] == "gpt-4o-mini"
        assert span["metadata"]["provider"] == "openai"
        assert span["output"][0]["type"] == "message"
        assert span["metrics"]["tokens"] > 0

print("SUCCESS")
