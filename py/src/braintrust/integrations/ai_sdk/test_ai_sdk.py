# pylint: disable=import-error,no-member
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import base64

import ai
import pytest
from braintrust import Attachment, logger, setup_ai_sdk
from braintrust.integrations.ai_sdk import patch_ai_sdk, unpatch_ai_sdk
from braintrust.integrations.ai_sdk.tracing import _shape_input_messages
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from pydantic import BaseModel


PROJECT_NAME = "test-ai-sdk"
MODEL = "openai:gpt-4o-mini"
_RED_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


class ImageAnswer(BaseModel):
    color: str


@pytest.fixture(scope="module", autouse=True)
def setup_integration():
    assert setup_ai_sdk(project_name=PROJECT_NAME)


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as background_logger:
        background_logger.pop()
        yield background_logger


def _span_named(spans, name):
    return next(span for span in spans if span["span_attributes"]["name"] == name)


def _contains_attachment(value):
    if isinstance(value, Attachment):
        return True
    if isinstance(value, dict):
        return any(_contains_attachment(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_attachment(item) for item in value)
    return False


def _assert_ai_sdk_origin(span):
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "ai-sdk-auto"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_stream_structured_multimodal_response(memory_logger):
    model = ai.get_model(MODEL)
    messages = [
        ai.system_message("Return the requested value exactly."),
        ai.user_message(
            "What is the dominant color of this image?",
            ai.file_part(_RED_PIXEL_PNG, media_type="image/png", filename="red.png"),
        ),
    ]

    chunks = []
    async with ai.stream(model, messages, output_type=ImageAnswer) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                chunks.append(event.chunk)

    assert chunks
    assert stream.output.color.lower() == "red"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = _span_named(spans, "ai.stream")
    assert span["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert _contains_attachment(span["input"])
    assert span["output"][0]["message"]["role"] == "assistant"
    assert "red" in span["output"][0]["message"]["content"].lower()
    assert span["metadata"]["provider"] == "openai"
    assert "gpt-4o-mini" in span["metadata"]["model"]
    assert span["metadata"]["response_id"]
    assert span["metadata"]["output_type"] == "ImageAnswer"
    for token_metric in ("tokens", "prompt_tokens", "completion_tokens"):
        assert token_metric not in span["metrics"]
    assert span["metrics"]["time_to_first_token"] >= 0
    _assert_ai_sdk_origin(span)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_tool_loop_emits_run_model_and_tool_hierarchy(memory_logger):
    @ai.tool
    async def get_temperature(city: str) -> dict[str, object]:
        """Get the current temperature for a city."""
        return {"city": city, "temperature": 21, "unit": "celsius"}

    model = ai.get_model(MODEL)
    agent = ai.Agent(tools=[get_temperature])
    messages = [
        ai.system_message("Always use get_temperature once before answering. Be concise."),
        ai.user_message("What is the temperature in Paris?"),
    ]

    text_chunks = []
    async with agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                text_chunks.append(event.chunk)

    assert "21" in "".join(text_chunks)

    spans = memory_logger.pop()
    run_span = _span_named(spans, "Agent.run")
    stream_spans = [span for span in spans if span["span_attributes"]["name"] == "ai.stream"]
    tool_span = _span_named(spans, "get_temperature")
    turn_spans = [span for span in spans if span["span_attributes"]["name"] == "Loop Turn"]

    assert run_span["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert len(stream_spans) == 2
    assert not [span for span in spans if span["span_attributes"].get("type") == SpanTypeAttribute.LLM]
    assert len(turn_spans) == 2
    assert tool_span["span_attributes"]["type"] == SpanTypeAttribute.TOOL
    assert tool_span["input"]["city"].lower() == "paris"
    assert tool_span["output"] == {"city": "Paris", "temperature": 21, "unit": "celsius"}
    assert tool_span["metadata"]["tool_call_id"]

    span_by_id = {span["span_id"]: span for span in spans}
    for turn_span in turn_spans:
        assert turn_span["span_parents"] == [run_span["span_id"]]
    for child_span in [*stream_spans, tool_span]:
        (parent_id,) = child_span["span_parents"]
        assert span_by_id[parent_id]["span_attributes"]["name"] == "Loop Turn"

    for stream_span in stream_spans:
        assert stream_span["span_attributes"]["type"] == SpanTypeAttribute.TASK
        assert stream_span["metadata"]["provider"] == "openai"
        assert stream_span["metadata"]["tools"] == [{"type": "function", "function": {"name": "get_temperature"}}]
        for token_metric in ("tokens", "prompt_tokens", "completion_tokens"):
            assert token_metric not in stream_span["metrics"]
        _assert_ai_sdk_origin(stream_span)

    tool_call_stream = next(span for span in stream_spans if span["output"][0]["finish_reason"] == "tool_calls")
    tool_call = tool_call_stream["output"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_temperature"
    assert isinstance(tool_call["function"]["arguments"], str)
    followup_stream = next(span for span in stream_spans if span is not tool_call_stream)
    tool_result_message = next(message for message in followup_stream["input"] if message["role"] == "tool")
    assert tool_result_message["tool_call_id"] == tool_call["id"]
    assert isinstance(tool_result_message["content"], str)

    for token_metric in ("tokens", "prompt_tokens", "completion_tokens"):
        assert token_metric not in run_span.get("metrics", {})
    assert "21" in str(run_span["output"])
    for span in spans:
        _assert_ai_sdk_origin(span)


def test_tool_result_input_uses_model_facing_value():
    result = ai.types.messages.ToolResultPart(
        tool_call_id="call_1",
        tool_name="search",
        result={"documents": ["full document contents"]},
    )
    result.set_model_input({"summary": "compact model context"})
    message = ai.types.messages.Message(role="tool", parts=[result])

    expected = '{"summary":"compact model context"}'
    assert _shape_input_messages([message])[0]["content"] == expected
    assert _shape_input_messages([message.model_dump(mode="python")])[0]["content"] == expected


@pytest.mark.asyncio
async def test_durable_replay_preserves_parent_hierarchy(memory_logger):
    sink = ai.experimental_telemetry.DictSink()
    async with ai.experimental_telemetry.use_sink(sink):
        async with ai.experimental_telemetry.span(
            ai.experimental_telemetry.RunSpanData(
                agent="DurableAgent",
                model="gpt-4o-mini",
                messages=[ai.user_message("hello")],
                provider="openai",
            )
        ):
            async with ai.experimental_telemetry.span(ai.experimental_telemetry.LoopTurnSpanData()):
                pass

    payload = [span.model_dump(mode="json") for span in sink.finished_spans]
    await ai.experimental_telemetry.push_all(payload)

    spans = memory_logger.pop()
    run_span = _span_named(spans, "Agent.run")
    turn_span = _span_named(spans, "Loop Turn")
    assert turn_span["span_parents"] == [run_span["span_id"]]
    assert turn_span["root_span_id"] == run_span["root_span_id"]


@pytest.mark.asyncio
async def test_custom_and_hook_spans_preserve_attributes(memory_logger):
    async with ai.experimental_telemetry.span("retrieve_context") as custom_span:
        custom_span.set_attrs(query="Paris", count=2)

    hook_data = ai.experimental_telemetry.HookSpanData(
        label="approve_call_1",
        hook_type="approval",
        metadata={"tool": "send_email"},
        tool_call_id="call_1",
    )
    async with ai.experimental_telemetry.span(hook_data):
        hook_data.status = "resolved"
        hook_data.resolution = {"granted": True}

    spans = memory_logger.pop()
    custom = _span_named(spans, "retrieve_context")
    hook = _span_named(spans, "Hook: approve_call_1")
    assert custom["metadata"] == {"query": "Paris", "count": 2}
    assert hook["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert hook["metadata"]["tool_call_id"] == "call_1"
    assert hook["output"] == {"status": "resolved", "resolution": {"granted": True}}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_provider_error_propagates_and_is_logged(memory_logger):
    model = ai.get_model("openai:braintrust-model-that-does-not-exist")

    with pytest.raises(Exception) as exc_info:  # provider exception type is intentionally preserved
        async with ai.stream(model, [ai.user_message("hello")]) as stream:
            async for _ in stream:
                pass

    assert "model" in str(exc_info.value).lower()
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = _span_named(spans, "ai.stream")
    assert span["error"]
    assert "model" in span["error"].lower()
    _assert_ai_sdk_origin(span)


def test_registration_and_unregistration_are_idempotent():
    assert unpatch_ai_sdk()
    assert unpatch_ai_sdk()
    assert patch_ai_sdk()
    assert patch_ai_sdk()


def test_auto_instrument_ai_sdk_subprocess():
    verify_autoinstrument_script("test_auto_ai_sdk.py")
