"""Test auto_instrument for Google GenAI (no uninstrument available)."""

import os

from braintrust.integrations.test_utils import run_auto_smoke


def _call(memory_logger):
    from google.genai import types
    from google.genai.client import Client

    client = Client()
    response = client.models.generate_content(
        model=(
            "gemini-2.5-flash-lite"
            if os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") == "latest"
            else "gemini-2.0-flash-001"
        ),
        contents="Say hi",
        config=types.GenerateContentConfig(max_output_tokens=100),
    )
    assert response.text

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    assert "gemini" in span["metadata"]["model"]


run_auto_smoke("google_genai", integration="google_genai", run=_call)
print("SUCCESS")
