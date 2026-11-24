import asyncio
import json
import os
import struct

import pytest
from braintrust import logger
from braintrust.voice import LiveKitAgent, livekit_agent
from braintrust.voice._livekit import run_livekit
from braintrust.voice.test_session import simulator, terminal
from braintrust.voice.test_session import tracing as tracing


pytest.importorskip("livekit.agents")


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_error", [False, True])
async def test_native_livekit_bridge_propagates_parent_audio_result_and_errors(tracing, agent_error):
    # A real instrumented AgentSession must report completion or errors back through the native room.
    from braintrust.integrations.livekit_agents import setup_livekit_agents
    from livekit import api, rtc
    from livekit.agents import Agent, AgentSession

    url = os.environ.get("LIVEKIT_URL")
    if not url:
        pytest.skip("Set LIVEKIT_URL to a local test server")
    key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
    setup_livekit_agents()
    agent_audio = []
    acknowledgements = []
    ready = asyncio.Event()

    async def handler(socket):
        await socket.send(struct.pack("<480h", *([100] * 480)))
        async for message in socket:
            if isinstance(message, bytes):
                agent_audio.append(message)
            else:
                event = json.loads(message)
                if event["type"] == "playback.ack":
                    acknowledgements.append(event)
                if event["type"] == "stop":
                    await socket.send(json.dumps(terminal()))
                    return

    async def receive_call(call):
        room = rtc.Room()
        token = (
            api.AccessToken(key, secret)
            .with_identity("support-agent")
            .with_kind("agent")
            .with_grants(api.VideoGrants(room_join=True, room=call["session_id"], agent=True))
        )
        await room.connect(url, token.to_jwt())
        ready.set()
        session = AgentSession()
        try:
            async with livekit_agent(session, room) as outcome:
                await session.start(agent=Agent(instructions="Handle the call"), room=room, record=False)
                if agent_error:
                    raise RuntimeError("Agent TTS failed")

                async def greeting():
                    for _ in range(10):
                        yield rtc.AudioFrame(struct.pack("<480h", *([2000] * 480)), 24000, 1, 480)

                await session.say("Hello", audio=greeting())
                outcome.agent_output = {"greeted": True}
        finally:
            await room.disconnect()

    with logger.start_span(name="Task") as parent:
        async with simulator(handler, parent.export()) as call:
            receiving = asyncio.create_task(receive_call(call))
            await asyncio.wait_for(ready.wait(), 10)
            try:
                agent = LiveKitAgent(server_url=url, agent_name="support-agent", api_key=key, api_secret=secret)
                if agent_error:
                    with pytest.raises(RuntimeError, match="Agent TTS failed"):
                        await asyncio.wait_for(run_livekit(call, agent), 20)
                else:
                    result = await asyncio.wait_for(run_livekit(call, agent), 20)
                    assert result["agent_output"] == {"greeted": True}
                    assert any(any(audio) for audio in agent_audio)
                    assert acknowledgements
            finally:
                if not receiving.done():
                    receiving.cancel()
                await asyncio.gather(receiving, return_exceptions=True)
    rows = tracing.pop()
    agent_span = next(row for row in rows if row["span_attributes"]["name"] == "Voice agent")
    parent_span = next(row for row in rows if row["span_attributes"]["name"] == "Task")
    assert agent_span["span_parents"] == [parent_span["span_id"]]
