# pylint: disable=import-error,no-member
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import asyncio
import base64
import contextlib
import contextvars
import random

import ai
import pytest
from braintrust import Attachment, logger, setup_ai_sdk
from braintrust.integrations.ai_sdk import patch_ai_sdk, unpatch_ai_sdk
from braintrust.integrations.ai_sdk.tracing import _shape_input_messages
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from pydantic import BaseModel


TEST_MODEL = "gpt-6-luna"

PROJECT_NAME = "test-ai-sdk"
MODEL = f"openai:{TEST_MODEL}"


def _model():
    from ai.providers import openai as provider

    # Older AI SDKs cannot disable Responses item references for ZDR accounts.
    if not hasattr(provider, "OpenAIParams"):
        return ai.get_model(MODEL, protocol=provider.OpenAIChatCompletionsProtocol())
    return ai.get_model(MODEL)


def _model_params():
    from ai.providers import openai as provider

    params = ai.InferenceRequestParams().with_reasoning_effort(None)
    if hasattr(provider, "OpenAIParams"):
        params = params.with_provider_params(provider.OpenAIParams(store=False))
    return params


_RED_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


@ai.tool
async def get_temperature(city: str) -> dict[str, object]:
    """Get the current temperature for a city."""
    return {"city": city, "temperature": 21, "unit": "celsius"}


def _temperature_messages():
    return [
        ai.system_message("Always use get_temperature once before answering. Be concise."),
        ai.user_message("What is the temperature in Paris?"),
    ]


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


def _assert_tool_round_trip(stream_spans):
    """The tool call from one ``ai.stream`` turn feeds the next turn's input."""
    tool_call_stream = next(span for span in stream_spans if span["output"][0]["finish_reason"] == "tool_calls")
    tool_call = tool_call_stream["output"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_temperature"
    assert isinstance(tool_call["function"]["arguments"], str)
    followup_stream = next(span for span in stream_spans if span is not tool_call_stream)
    tool_result_message = next(message for message in followup_stream["input"] if message["role"] == "tool")
    assert tool_result_message["tool_call_id"] == tool_call["id"]
    assert isinstance(tool_result_message["content"], str)
    return tool_call


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_stream_structured_multimodal_response(memory_logger):
    model = _model()
    messages = [
        ai.system_message("Return the requested value exactly."),
        ai.user_message(
            "What is the dominant color of this image?",
            ai.file_part(_RED_PIXEL_PNG, media_type="image/png", filename="red.png"),
        ),
    ]

    chunks = []
    async with ai.stream(model, messages, output_type=ImageAnswer, params=_model_params()) as stream:
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
    assert TEST_MODEL in span["metadata"]["model"]
    assert span["metadata"]["response_id"]
    assert span["metadata"]["output_type"] == "ImageAnswer"
    for token_metric in ("tokens", "prompt_tokens", "completion_tokens"):
        assert token_metric not in span["metrics"]
    assert span["metrics"]["time_to_first_token"] >= 0
    _assert_ai_sdk_origin(span)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_tool_loop_emits_run_model_and_tool_hierarchy(memory_logger):
    model = _model()
    agent = ai.Agent(tools=[get_temperature])
    messages = _temperature_messages()

    text_chunks = []
    async with agent.run(model, messages, params=_model_params()) as stream:
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

    _assert_tool_round_trip(stream_spans)

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


class _WorkerCrash(Exception):
    pass


class _DurableEngine:
    """The replay contract of a durable runtime such as Temporal.

    Each step runs once, in a fresh context like an activity worker, and its
    JSON result is recorded. Later executions of the workflow replay recorded
    results instead of running the step again.
    """

    def __init__(self):
        self.history = []
        self._cursor = 0
        self._crash_before_step = None

    def start_execution(self, *, crash_before_step=None):
        self._cursor = 0
        self._crash_before_step = crash_before_step

    async def step(self, fn, *args):
        if self._cursor < len(self.history):
            result = self.history[self._cursor]
        else:
            if self._cursor == self._crash_before_step:
                raise _WorkerCrash()
            # The task copies the empty context it is created in. (``create_task(context=)``
            # is 3.11+, and pylint also lints this file on 3.10.)
            result = await contextvars.Context().run(asyncio.create_task, fn(*args))
            self.history.append(result)
        self._cursor += 1
        return result


async def _durable_llm_step(model_data, messages_data, tools_data, parent_data):
    """Model call activity: collect its spans and return them with the message."""
    telemetry = ai.experimental_telemetry
    model = ai.Model.model_validate(model_data)
    messages = [ai.messages.Message.model_validate(message) for message in messages_data]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]
    sink = telemetry.DictSink()
    async with telemetry.use_sink(sink), telemetry.use_span(telemetry.Span.model_validate(parent_data)):
        async with ai.stream(model, messages, tools=tools, params=_model_params()) as stream:
            async for _ in stream:
                pass
    return {
        "message": stream.message.model_dump(mode="json"),
        # Works around an ai 0.9 bug: its ``ModelProviderDefault`` params sentinel
        # is not JSON-serializable, so dump it as null.
        "spans": [span.model_dump(mode="json", fallback=lambda _: None) for span in sink.finished_spans],
    }


async def _run_durable_workflow(engine, *, crash_before_step=None):
    telemetry = ai.experimental_telemetry
    engine.start_execution(crash_before_step=crash_before_step)
    sink = telemetry.DictSink()

    @contextlib.asynccontextmanager
    async def streamer(*, context):
        result = await engine.step(
            _durable_llm_step,
            context.model.model_dump(mode="json"),
            [message.model_dump(mode="json") for message in context.messages],
            [tool.model_dump(mode="json") for tool in context.tools],
            telemetry.current_span().model_dump(mode="json"),
        )
        await telemetry.push_all(result["spans"])
        async with ai.Stream.replay_message(ai.messages.Message.model_validate(result["message"])) as replay:
            yield replay

    # Deterministic ids let a replayed execution mint the same span ids, so
    # spans recorded by an earlier execution still find their parents.
    async with ai.messages.use_random(random.Random(1086)), telemetry.use_sink(sink):
        agent = ai.durable.DurableAgent(tools=[get_temperature], streamer=streamer)
        async with agent.run(_model(), _temperature_messages()) as run:
            async for _ in run:
                pass
    # Telemetry activity: the one place spans reach Braintrust.
    await engine.step(telemetry.push_all, [span.model_dump(mode="json") for span in sink.finished_spans])
    return run.messages


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_durable_agent_replay_preserves_parent_hierarchy(memory_logger):
    pytest.importorskip("ai.durable", reason="ai.durable.DurableAgent requires ai 0.9+")
    engine = _DurableEngine()

    # The worker crashes after the first model turn; the crash surfaces through
    # the agent loop's task group.
    with pytest.RaisesGroup(_WorkerCrash, flatten_subgroups=True):
        await _run_durable_workflow(engine, crash_before_step=1)
    assert len(engine.history) == 1

    # The resumed execution replays turn one and runs turn two live. Only its
    # spans are flushed, so the crashed execution leaves no partial trace.
    resumed_messages = await _run_durable_workflow(engine)
    assert len(engine.history) == 3
    assert "21" in resumed_messages[-1].text

    spans = memory_logger.pop()
    assert sorted(span["span_attributes"]["name"] for span in spans) == [
        "Agent.run",
        "ai.stream",
        "ai.stream",
        "get_temperature",
    ]
    run_span = _span_named(spans, "Agent.run")
    stream_spans = [span for span in spans if span["span_attributes"]["name"] == "ai.stream"]
    tool_span = _span_named(spans, "get_temperature")
    assert run_span["span_parents"] is None
    assert run_span["metadata"]["agent"] == "DurableAgent"
    for child_span in [*stream_spans, tool_span]:
        assert child_span["span_parents"] == [run_span["span_id"]]
        assert child_span["root_span_id"] == run_span["root_span_id"]
    for span in spans:
        _assert_ai_sdk_origin(span)

    # Turn one's stream span was recorded by the crashed execution; the tool
    # span comes from the resumed one.
    tool_call = _assert_tool_round_trip(stream_spans)
    assert tool_span["metadata"]["tool_call_id"] == tool_call["id"]
    assert tool_span["output"] == {"city": "Paris", "temperature": 21, "unit": "celsius"}
    assert "21" in str(run_span["output"])


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_typesafe_provider_evaluate_nests_typesafe_span(memory_logger):
    typesafe_provider = pytest.importorskip(
        "ai.providers.typesafe", reason="The native TypeSafe provider requires ai 0.9+"
    )
    from braintrust.integrations.typesafe import setup_typesafe

    assert setup_typesafe()
    provider = typesafe_provider.TypeSafeProvider()
    try:
        await ai.ops.experimental.evaluate(
            ai.Model(id="jev-latest", provider=provider),
            "The package arrived intact and on time.",
            {"positive": ai.ops.experimental.NoulQuestion(instructions="Is this feedback positive?")},
        )
    finally:
        await provider.aclose()

    spans = memory_logger.pop()
    assert len(spans) == 2
    evaluate_span = _span_named(spans, "ai.evaluate")
    typesafe_span = _span_named(spans, "typesafe.systemOne")

    assert evaluate_span["span_attributes"]["type"] == SpanTypeAttribute.TASK
    assert evaluate_span["metadata"] == {
        "model": "jev-latest",
        "provider": "typesafe",
        "question_count": 1,
        "answer_count": 1,
    }
    _assert_ai_sdk_origin(evaluate_span)

    # The TypeSafe integration owns the single model span and its tokens; its
    # span shape is covered by typesafe/test_typesafe.py.
    assert typesafe_span["span_attributes"]["type"] == SpanTypeAttribute.DECISION
    assert typesafe_span["span_parents"] == [evaluate_span["span_id"]]
    assert typesafe_span["root_span_id"] == evaluate_span["root_span_id"]
    assert [span for span in spans if "tokens" in span.get("metrics", {})] == [typesafe_span]


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


@pytest.mark.vcr(cassette_name="test_agent_tool_loop_emits_run_model_and_tool_hierarchy")
@pytest.mark.asyncio
async def test_agent_descendants_recheck_provider_privacy_at_export(memory_logger):
    from braintrust._instrumentation import provider_content_allowed

    allowed = True
    token = provider_content_allowed.set(lambda: allowed)
    try:
        agent = ai.Agent(tools=[get_temperature])
        async with agent.run(_model(), _temperature_messages(), params=_model_params()) as stream:
            async for _ in stream:
                pass
    finally:
        provider_content_allowed.reset(token)
    with logger.start_span(name="customer", input="unrelated customer content"):
        pass
    allowed = False
    rows = memory_logger.pop()
    provider_rows = [row for row in rows if row["span_attributes"]["name"] != "customer"]
    assert len(provider_rows) > 3
    assert _span_named(provider_rows, "get_temperature")
    assert all("input" not in row and "output" not in row and "error" not in row for row in provider_rows)
    assert _span_named(rows, "customer")["input"] == "unrelated customer content"
