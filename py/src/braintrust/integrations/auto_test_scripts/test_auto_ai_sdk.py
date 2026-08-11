"""Verify Vercel AI SDK instrumentation when ai is imported first."""
# pylint: disable=import-error,no-member

import asyncio

import ai
from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


async def main():
    results = auto_instrument()
    assert results.get("ai_sdk") is True
    assert auto_instrument().get("ai_sdk") is True

    with autoinstrument_test_context("test_auto_ai_sdk", integration="ai_sdk") as memory_logger:
        async with ai.stream(
            ai.get_model("openai:gpt-4o-mini"),
            [ai.user_message("Reply with the single word hello.")],
        ) as stream:
            async for _ in stream:
                pass

        spans = memory_logger.pop()
        assert len(spans) == 2, f"Expected AI SDK and provider spans, got: {spans!r}"
        ai_span = next(span for span in spans if span["span_attributes"]["name"] == "ai.stream")
        provider_span = next(span for span in spans if span["span_attributes"].get("type") == "llm")
        assert ai_span["span_attributes"]["type"] == "task"
        for token_metric in ("tokens", "prompt_tokens", "completion_tokens"):
            assert token_metric not in ai_span["metrics"]
        assert provider_span["metrics"]["tokens"] > 0
        assert provider_span["metrics"]["completion_reasoning_tokens"] >= 0
        assert provider_span["metrics"]["prompt_cached_tokens"] >= 0
        assert ai_span["context"]["span_origin"]["instrumentation"]["name"] == "ai-sdk-auto"
        assert provider_span["context"]["span_origin"]["instrumentation"]["name"] == "openai-auto"


asyncio.run(main())
print("SUCCESS")
