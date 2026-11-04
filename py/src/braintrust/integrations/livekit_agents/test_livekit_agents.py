import pytest
from braintrust import logger
from braintrust.audio import AudioEncoding, AudioRecordingOptions
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.livekit_agents.compat import STATE_ATTR
from braintrust.integrations.livekit_agents.privacy import Privacy, opted_in
from braintrust.integrations.test_utils import verify_autoinstrument_script


def named(logs, name):
    return [row for row in logs if row.get("span_attributes", {}).get("name") == name]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_ids", [False, True])
@pytest.mark.vcr(cassette_name="test_livekit_agents_function_tool_e2e")
async def test_livekit_agents_function_tool_e2e(memory_logger, monkeypatch, legacy_ids):
    from livekit.agents import Agent, AgentSession, function_tool
    from livekit.plugins import openai

    monkeypatch.setenv("BRAINTRUST_LEGACY_IDS", str(legacy_ids).lower())
    logger._state._reset_id_generator()

    @function_tool
    async def lookup_weather(city: str) -> str:
        with logger.start_span(name="customer_tool_detail"):
            return f"sunny in {city}"

    assert setup_livekit_agents()
    llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=32)
    async with llm, AgentSession() as session:
        await session.start(
            Agent(
                instructions="Use lookup_weather for weather questions, then answer briefly.",
                llm=llm,
                tools=[lookup_weather],
            )
        )
        result = await session.run(user_input="What is the weather in Paris?")
        result.expect.next_event().is_function_call(name="lookup_weather")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_message(role="assistant")
    rows = memory_logger.pop()
    assert len(named(rows, "session")) == 1
    turns = named(rows, "agent_turn")
    models = named(rows, "llm_response")
    tools = named(rows, "lookup_weather")
    assert len(turns) == 1
    assert len(models) == 2
    assert len(tools) == 1
    assert tools[0]["input"] == {"city": "Paris"}
    assert tools[0]["output"] == "sunny in Paris"
    assert named(rows, "customer_tool_detail")[0]["span_parents"] == [tools[0]["span_id"]]
    assert all(row["span_parents"] == [turns[0]["span_id"]] for row in models + tools)
    assert all(row["metrics"]["end"] >= row["metrics"]["start"] for row in rows)
    assert all("tokens" not in row.get("metrics", {}) for row in turns + tools)
    assert models[0]["metrics"]["tokens"] == 69
    initial_input = [
        {"role": "system", "content": "Use lookup_weather for weather questions, then answer briefly."},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    assert models[0]["input"] == initial_input
    call = models[0]["output"][0]["tool_calls"][0]
    # LiveKit serializes the parsed arguments when adding the tool to context.
    observed_call = {
        "id": call["id"],
        "type": "function",
        "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
    }
    assert models[1]["input"] == initial_input + [
        {"role": "assistant", "content": None, "tool_calls": [observed_call]},
        {"role": "tool", "content": "sunny in Paris", "tool_call_id": call["id"]},
    ]
    if legacy_ids:
        assert all("links" not in row.get("metadata", {}) for row in models + tools)
    else:
        for row, target, relationship in (
            (tools[0], models[0], "requested_by"),
            (models[1], tools[0], "consumes_tool_result"),
        ):
            assert row["metadata"]["links"] == [
                {
                    "context": {
                        "traceId": target["root_span_id"],
                        "spanId": target["span_id"],
                        "traceFlags": 1,
                    },
                    "attributes": {"braintrust.relationship.type": relationship},
                }
            ]
    assert not hasattr(session, STATE_ATTR)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_livekit_agents_agent_speaking_e2e(memory_logger):
    # Real roomless session and recorded HTTP provider responses. Playback is
    # qualified independently using LiveKit's real RoomIO sink in test_playback.
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import openai

    assert setup_livekit_agents()
    llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8)
    async with llm, AgentSession() as session:
        with logger.start_span(name="outer"):
            await session.start(Agent(instructions="Reply with exactly: hi there", llm=llm))
            result = await session.run(user_input="say hi")
            result.expect.next_event().is_message(role="assistant")
    rows = memory_logger.pop()
    outer, root = named(rows, "outer")[0], named(rows, "session")[0]
    assert root["span_parents"] == [outer["span_id"]]
    assert named(rows, "llm_response")[0]["output"][0]["content"] == "hi there"
    assert root["metadata"]["audio.schema.version"] == "3.0.0"
    assert "audio.segments" not in root["metadata"]
    user = named(rows, "user_turn")[0]
    assert named(rows, "agent_turn")[0]["metadata"]["turn.reply_to"] == user["span_id"]


@pytest.mark.asyncio
async def test_real_session_start_failure_cleans_up(memory_logger):
    from livekit.agents import Agent, AgentSession

    assert setup_livekit_agents()
    session = AgentSession()
    # Real framework rejection, without a stand-in start function.
    with pytest.raises(Exception):
        await session.start(Agent(instructions="test"), room=object())
    assert not hasattr(session, STATE_ATTR)
    assert named(memory_logger.pop(), "session")[0]["metrics"]["end"]


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE ", ""])
def test_audio_dedicated_opt_in_false_forms(monkeypatch, value):
    monkeypatch.setenv("BRAINTRUST_CAPTURE_ATTACHMENTS", "true")
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", value)
    assert not opted_in("AGENT")


@pytest.mark.asyncio
async def test_provider_privacy_configured_before_setup_is_effective():
    from livekit.agents import AgentSession, telemetry
    from livekit.agents.telemetry import traces
    from opentelemetry.sdk.trace import TracerProvider

    original = traces.tracer._tracer_provider
    provider = TracerProvider()
    try:
        telemetry.set_tracer_provider(provider, allow_pii=False)
        assert setup_livekit_agents()
        assert not Privacy(AgentSession(), None).allows_content()
    finally:
        traces.tracer.set_provider(original)
        provider.shutdown()


def test_explicit_setup_survives_automatic_setup():
    from braintrust.integrations.livekit_agents import tracing

    original = tracing._OPTIONS
    try:
        options = AudioRecordingOptions(encoding=AudioEncoding(codec="pcm", container="wav"))
        assert setup_livekit_agents(audio=options)
        assert setup_livekit_agents()
        assert tracing._OPTIONS is options
    finally:
        tracing._OPTIONS = original


def test_auto_instrument_livekit_agents_subprocess():
    verify_autoinstrument_script("test_auto_livekit_agents.py")
    verify_autoinstrument_script("test_auto_livekit_agents_before_import.py")
