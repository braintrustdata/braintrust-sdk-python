"""Test auto_instrument for OpenAI."""

import inspect

import openai
from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context
from wrapt import FunctionWrapper


def _is_braintrust_wrapped(resource) -> bool:
    attr = inspect.getattr_static(resource, "create", None)
    return isinstance(attr, FunctionWrapper)


resources = [openai.resources.chat.completions.Completions]
try:
    from openai.resources.beta.agents.sessions.sessions import Sessions

    resources.append(Sessions)
except ImportError:
    pass


# 1. Verify not patched initially
assert all(not _is_braintrust_wrapped(resource) for resource in resources)

# 2. Instrument
results = auto_instrument()
assert results.get("openai") == True
assert all(_is_braintrust_wrapped(resource) for resource in resources)

# 3. Idempotent
results2 = auto_instrument()
assert results2.get("openai") == True

# 4. Make API call and verify span
with autoinstrument_test_context("test_auto_openai", integration="openai") as memory_logger:
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

print("SUCCESS")
