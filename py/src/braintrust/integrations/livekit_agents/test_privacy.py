import asyncio
import json

import pytest
from braintrust import logger
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.openai import OpenAIIntegration


def named(rows, name):
    return [row for row in rows if row.get("span_attributes", {}).get("name") == name]


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["provider_first", "livekit_first"])
@pytest.mark.parametrize("restricted", [True, False])
async def test_provider_ownership_and_privacy_are_session_local(memory_logger, vcr, order, restricted):
    import openai as vendor
    from livekit.agents import Agent, AgentSession, telemetry
    from livekit.agents.telemetry import traces
    from livekit.plugins import openai
    from opentelemetry.sdk.trace import TracerProvider

    original = traces.tracer._tracer_provider
    provider = TracerProvider()
    try:
        if order == "provider_first":
            OpenAIIntegration.setup()
            telemetry.set_tracer_provider(provider, allow_pii=not restricted)
            setup_livekit_agents()
        else:
            setup_livekit_agents()
            OpenAIIntegration.setup()
            telemetry.set_tracer_provider(provider, allow_pii=not restricted)

        async def standalone():
            async with vendor.AsyncOpenAI() as client:
                stream = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    stream=True,
                    max_completion_tokens=8,
                    messages=[
                        {"role": "system", "content": "Reply with exactly: hi there"},
                        {"role": "user", "content": "say hi"},
                    ],
                    stream_options={"include_usage": True},
                )
                async for _ in stream:
                    pass

        with vcr.use_cassette("test_livekit_agents_agent_speaking_e2e.yaml", allow_playback_repeats=True):
            llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8)
            async with llm, AgentSession() as session:
                await session.start(Agent(instructions="Reply with exactly: hi there", llm=llm))
                await asyncio.gather(session.run(user_input="say hi"), standalone())
        rows = memory_logger.pop()
        canonical = named(rows, "llm_response")
        independent = named(rows, "Chat Completion")
        assert len(canonical) == 1
        assert canonical[0]["span_attributes"]["type"] == "task"
        assert "tokens" not in canonical[0]["metrics"]
        assert len(independent) == (1 if restricted else 2)
        if not restricted:
            nested = next(row for row in independent if row.get("span_parents") == [canonical[0]["span_id"]])
            assert nested["metrics"]["tokens"] > 0
        assert "output" in independent[0]
        if restricted:
            assert "input" not in canonical[0]
            assert "output" not in canonical[0]
            assert "say hi" not in json.dumps([row for row in rows if row not in independent])
        else:
            assert canonical[0]["output"][0]["content"] == "hi there"
    finally:
        traces.tracer.set_provider(original)
        provider.shutdown()


@pytest.mark.asyncio
async def test_late_session_redaction_filters_queued_content(memory_logger, vcr):
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import openai

    setup_livekit_agents()
    with vcr.use_cassette("test_livekit_agents_agent_speaking_e2e.yaml"):
        llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8)
        async with llm, AgentSession() as session:
            await session.start(Agent(instructions="Reply with exactly: hi there", llm=llm))
            await session.run(user_input="say hi")
            session._redaction_enabled = True
    rows = memory_logger.pop()
    assert "say hi" not in json.dumps(rows)
    assert "hi there" not in json.dumps(rows)
    assert "tokens" not in named(rows, "llm_response")[0]["metrics"]


@pytest.mark.asyncio
async def test_late_redaction_filters_supplemental_provider_content_only(memory_logger, vcr):
    import openai as vendor
    from livekit.agents import Agent, AgentSession, function_tool
    from livekit.plugins import openai

    OpenAIIntegration.setup()
    setup_livekit_agents()

    async def provider_call():
        with vcr.use_cassette("test_livekit_agents_agent_speaking_e2e.yaml", allow_playback_repeats=True):
            async with vendor.AsyncOpenAI() as client:
                stream = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    stream=True,
                    max_completion_tokens=8,
                    messages=[
                        {"role": "system", "content": "Reply with exactly: hi there"},
                        {"role": "user", "content": "say hi"},
                    ],
                    stream_options={"include_usage": True},
                )
                async for _ in stream:
                    pass

    @function_tool
    async def lookup_weather(city: str) -> str:
        with logger.start_span(name="customer_detail") as detail:
            detail.log(input="customer-owned detail")
            await provider_call()
        return f"sunny in {city}"

    with logger.start_span(name="unrelated"):
        await provider_call()
    with vcr.use_cassette("test_livekit_agents_function_tool_e2e.yaml"):
        llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=32)
        async with llm, AgentSession() as session:
            await session.start(
                Agent(
                    instructions="Use lookup_weather for weather questions, then answer briefly.",
                    llm=llm,
                    tools=[lookup_weather],
                )
            )
            await session.run(user_input="What is the weather in Paris?")
            session._redaction_enabled = True

    rows = memory_logger.pop()
    detail = named(rows, "customer_detail")[0]
    unrelated = named(rows, "unrelated")[0]
    providers = named(rows, "Chat Completion")
    assert len(providers) == 4
    supplemental = next(row for row in providers if row["span_parents"] == [detail["span_id"]])
    independent = next(row for row in providers if row["span_parents"] == [unrelated["span_id"]])
    assert "input" not in supplemental
    assert "output" not in supplemental
    assert supplemental["metrics"]["tokens"] > 0
    assert independent["input"] == [
        {"role": "system", "content": "Reply with exactly: hi there"},
        {"role": "user", "content": "say hi"},
    ]
    assert independent["output"][0]["message"]["content"] == "hi there"
    assert detail["input"] == "customer-owned detail"


@pytest.mark.asyncio
async def test_two_livekit_sessions_keep_explicit_parents(memory_logger, vcr):
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import openai

    setup_livekit_agents()

    async def conversation(label):
        with logger.start_span(name=label):
            llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8)
            async with llm, AgentSession() as session:
                await session.start(Agent(instructions="Reply with exactly: hi there", llm=llm))
                await session.run(user_input="say hi")

    with vcr.use_cassette("test_livekit_agents_agent_speaking_e2e.yaml", allow_playback_repeats=True):
        await asyncio.gather(conversation("call-A"), conversation("call-B"))
    rows = memory_logger.pop()
    sessions = named(rows, "session")
    assert len(sessions) == 2
    outer = {row["span_id"] for row in named(rows, "call-A") + named(rows, "call-B")}
    assert {row["span_parents"][0] for row in sessions} == outer
    assert len({row["root_span_id"] for row in sessions}) == 2
    models = named(rows, "llm_response")
    assert len(models) == 2
    assert {row["root_span_id"] for row in models} == {row["root_span_id"] for row in sessions}
