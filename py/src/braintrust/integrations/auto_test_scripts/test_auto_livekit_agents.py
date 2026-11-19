"""Auto setup uses the same framework-level cascade hooks after imports."""

import asyncio
import inspect
import os
import uuid

from braintrust.auto import auto_instrument
from livekit.agents import AgentSession
from livekit.agents.voice import generation
from livekit.agents.voice.agent_activity import AgentActivity
from wrapt import FunctionWrapper


async def assert_exported_eou_wait():
    """Drive normal RTC/VAD/STT turn completion using the existing cascade tape."""
    from braintrust.integrations.test_utils import autoinstrument_test_context
    from livekit import api, rtc
    from livekit.agents import Agent
    from livekit.plugins import openai, silero

    with autoinstrument_test_context(
        "test_livekit_agents_e2e_parenting_under_custom_span", integration="livekit_agents"
    ) as memory_logger:
        room_name = f"braintrust-auto-eou-{uuid.uuid4()}"
        agent_room, caller_room = rtc.Room(), rtc.Room()
        source = rtc.AudioSource(24000, 1)
        try:
            for room, identity in ((agent_room, "agent"), (caller_room, "caller")):
                token = (
                    api.AccessToken(os.getenv("LIVEKIT_API_KEY", "devkey"), os.getenv("LIVEKIT_API_SECRET", "secret"))
                    .with_identity(identity)
                    .with_grants(api.VideoGrants(room_join=True, room=room_name))
                    .to_jwt()
                )
                await asyncio.wait_for(room.connect(os.getenv("LIVEKIT_URL", "ws://localhost:7880"), token), 15)
            track = rtc.LocalAudioTrack.create_audio_track("caller-microphone", source)
            await caller_room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            replied = asyncio.Event()
            eou_metrics = []
            async with (
                openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8) as llm,
                openai.TTS(model="gpt-4o-mini-tts", voice="ash") as tts,
                openai.STT(model="gpt-4o-mini-transcribe") as stt,
                AgentSession(vad=silero.VAD.load(), stt=stt, tts=tts, llm=llm) as session,
            ):
                speech = [chunk.frame async for chunk in tts.synthesize("hello there")]

                def conversation(event):
                    if event.item.type == "message" and event.item.role == "assistant":
                        replied.set()

                session.on("conversation_item_added", conversation)
                session.on(
                    "metrics_collected",
                    lambda event: eou_metrics.append(event.metrics) if event.metrics.type == "eou_metrics" else None,
                )
                await session.start(
                    Agent(instructions="Reply with exactly: hi there", llm=llm), room=agent_room, record=False
                )
                silence = rtc.AudioFrame(
                    data=bytes(2400 * 2), sample_rate=24000, num_channels=1, samples_per_channel=2400
                )
                for frame in [silence] * 5 + speech + [silence] * 15:
                    await source.capture_frame(frame)
                await asyncio.wait_for(replied.wait(), 15)
                await session.drain()
        finally:
            await source.aclose()
            await asyncio.gather(agent_room.disconnect(), caller_room.disconnect())

        rows = memory_logger.pop()
        waits = [row for row in rows if row.get("span_attributes", {}).get("name") == "eou_wait"]
        assert len(waits) == 1, f"Expected one exported eou_wait, got {len(waits)}"
        wait = waits[0]
        user = next(
            row for row in rows if row.get("span_attributes", {}).get("name") == "user_turn" and row.get("input")
        )
        speech = next(row for row in rows if row.get("span_attributes", {}).get("name") == "user_speech")
        assert wait["span_attributes"]["type"] == "task"
        assert wait["span_parents"] == [user["span_id"]]
        assert wait["metadata"]["turn.id"] == user["span_id"]
        assert wait["metadata"]["eou.outcome"] == "committed"
        assert wait["metadata"]["eou.speech_end_source"] == "vad"
        (reported,) = eou_metrics
        assert speech["metrics"]["start"] <= wait["metrics"]["start"]
        # LiveKit sets its speech-end anchor after the speech callback returns;
        # that callback can take arbitrary time. Compare with native EOU timing,
        # rather than imposing a scheduling budget against user_speech.end.
        assert wait["metrics"]["start"] + reported.end_of_utterance_delay <= wait["metrics"]["end"]
        assert wait["metrics"]["end"] <= reported.timestamp
        assert user["metadata"]["contrib.livekit.end_of_turn_delay"] == reported.end_of_utterance_delay
        assert wait["metrics"]["start"] < wait["metrics"]["end"] <= user["metrics"]["end"]
        assert wait["context"]["span_origin"]["instrumentation"]["name"] == "livekit-agents-auto"
        assert "eou.end_reason" not in wait["metadata"]
        assert "eou.detection_methods" not in wait["metadata"]
        assert not any(row.get("span_attributes", {}).get("name") == "livekit.eou_wait" for row in rows)


if __name__ == "__main__":
    assert not isinstance(inspect.getattr_static(AgentSession, "start"), FunctionWrapper)
    assert auto_instrument().get("livekit_agents") is True
    assert isinstance(inspect.getattr_static(AgentSession, "start"), FunctionWrapper)
    assert isinstance(inspect.getattr_static(AgentSession, "_aclose_impl"), FunctionWrapper)
    assert isinstance(inspect.getattr_static(AgentActivity, "push_audio"), FunctionWrapper)
    assert isinstance(generation._llm_inference_task, FunctionWrapper)
    assert isinstance(generation._tts_inference_task, FunctionWrapper)
    assert auto_instrument().get("livekit_agents") is True
    asyncio.run(assert_exported_eou_wait())
    print("SUCCESS")
