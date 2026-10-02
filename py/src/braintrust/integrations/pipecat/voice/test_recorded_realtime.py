# pylint: disable=import-error
# Pipecat is installed in the integration test session, not the shared lint environment.
"""Replay observed provider responses through Pipecat's actual event handler."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext  # pylint: disable=import-error
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,  # pylint: disable=import-error
)
from pipecat.processors.frame_processor import FrameDirection  # pylint: disable=import-error
from pipecat.services.openai.realtime import events  # pylint: disable=import-error
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService  # pylint: disable=import-error

from .instrumentation import NativeObserver
from .realtime import RealtimeCapture
from .test_instrumentation import Span


@pytest.mark.asyncio
async def test_recorded_tool_response_and_spoken_continuation():
    responses = json.loads((Path(__file__).parent / "fixtures/realtime-responses.json").read_text())
    observer = NativeObserver(Span())
    pair = LLMContextAggregatorPair(LLMContext())
    service = OpenAIRealtimeLLMService(api_key="test-unused")

    async def output(frame, direction=FrameDirection.DOWNSTREAM):
        await observer.on_push_frame(
            SimpleNamespace(
                frame=frame,
                first_push=True,
                source=service,
                destination=pair.assistant(),
                direction=direction,
                timestamp=0,
            )
        )

    service._setup = SimpleNamespace(enable_usage_metrics=True, enable_metrics=False)
    service.push_frame = output
    RealtimeCapture(observer, service, pair.user())
    calls = []
    for response in responses:
        span = observer.root.start_span(name="llm_response")
        observer.llm = span
        observer.ttfb.start("llm", service, span.log)
        event = events.ResponseDone(
            type="response.done", event_id="fixture-envelope", response=events.Response.model_validate(response)
        )
        await service._handle_evt_response_done(event)
        metadata = {key: value for row in span.rows for key, value in row.get("metadata", {}).items()}
        standard_metrics = {key: value for row in span.rows for key, value in row.get("metrics", {}).items()}
        assert standard_metrics["tokens"] == response["usage"]["total_tokens"]
        assert metadata["openai.response.id"] == response["id"]
        assert metadata["openai.response"]["usage"]["total_tokens"] == response["usage"]["total_tokens"]
        final_output = [row["output"] for row in span.rows if "output" in row][-1]
        for message in final_output:
            calls.extend(message.get("tool_calls", []))
        if any(item["type"] == "function_call" for item in response["output"]):
            assert final_output[0]["content"] is None
            assert metadata.get("pipecat.text") != ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup_order"
    assert "1042" in calls[0]["function"]["arguments"]
    await observer.finish()
