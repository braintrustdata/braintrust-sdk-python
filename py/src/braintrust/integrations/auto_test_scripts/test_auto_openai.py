"""Test auto_instrument for OpenAI."""

import inspect

import openai
from braintrust.integrations.test_utils import run_auto_smoke
from wrapt import FunctionWrapper


def _is_patched() -> bool:
    attr = inspect.getattr_static(openai.resources.chat.completions.Completions, "create", None)
    return isinstance(attr, FunctionWrapper)


def _call(memory_logger):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi"}],
    )
    assert response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert span["metadata"]["provider"] == "openai"
    assert "gpt-4o-mini" in span["metadata"]["model"]


run_auto_smoke("openai", is_patched=_is_patched, integration="openai", run=_call)
print("SUCCESS")
