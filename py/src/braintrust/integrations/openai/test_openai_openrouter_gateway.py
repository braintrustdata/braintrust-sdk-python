"""
Tests to ensure wrap_openai works correctly with OpenRouter.

OpenRouter is a popular API gateway that provides access to multiple LLM providers
through an OpenAI-compatible interface. This test validates that our wrapper handles
OpenRouter-specific response fields correctly (e.g., boolean `is_byok` in usage).
"""

import os
import time

import pytest
from braintrust import logger, wrap_openai
from braintrust.integrations.test_utils import assert_metrics_are_valid
from braintrust.test_helpers import init_test_logger
from openai import OpenAI


PROJECT_NAME = "test-openrouter"
TEST_MODEL = "openai/gpt-4o-mini"


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


@pytest.mark.parametrize(
    "stream,prompt,expected",
    (
        (False, "What is 2+2? Reply with just the number.", "4"),
        (True, "What is 5+5? Reply with just the number.", "10"),
    ),
    ids=("non_stream", "stream"),
)
@pytest.mark.vcr
def test_openrouter_chat_completion(memory_logger, stream, prompt, expected):
    """Test that wrap_openai works with OpenRouter's streaming and non-streaming responses."""
    assert not memory_logger.pop()

    client = wrap_openai(_get_client())

    start = time.time()
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        stream=stream,
    )
    if stream:
        chunks = list(response)
        assert chunks
        content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    else:
        assert response
        content = response.choices[0].message.content
    end = time.time()

    assert content
    assert expected in content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)

    # Ensure no boolean values in metrics (the original bug with is_byok)
    for key, value in metrics.items():
        assert not isinstance(value, bool), f"Metric {key} should not be a boolean"
