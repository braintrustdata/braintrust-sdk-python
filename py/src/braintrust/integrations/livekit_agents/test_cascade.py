import asyncio

import pytest
from braintrust import logger
from braintrust.audio import AudioEncoding, AudioRecordingOptions
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.livekit_agents.compat import STATE_ATTR


def named(rows, name):
    return [row for row in rows if row.get("span_attributes", {}).get("name") == name]


@pytest.mark.asyncio
@pytest.mark.parametrize("preemptive,directions", [(False, "both"), (True, "both"), (False, "user")])
@pytest.mark.vcr(cassette_name="test_livekit_agents_e2e_parenting_under_custom_span")
async def test_consumed_input_drives_real_cascade(rooms, memory_logger, monkeypatch, preemptive, directions):
    from livekit import rtc
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import openai, silero

    monkeypatch.setenv("BRAINTRUST_LEGACY_IDS", "false")
    logger._state._reset_id_generator()
    monkeypatch.setenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", "true")
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", "true" if directions == "both" else "false")
    uploads = []

    def upload(attachment):
        uploads.append(attachment)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    setup_livekit_agents(
        audio=AudioRecordingOptions(
            encoding=AudioEncoding(codec="pcm", container=None),
            segment_duration_seconds=0.5,
        )
    )
    tts = openai.TTS(model="gpt-4o-mini-tts", voice="ash")
    stt = openai.STT(model="gpt-4o-mini-transcribe")
    llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8)
    vad = silero.VAD.load()
    heard = asyncio.Event()
    input_source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("caller-microphone", input_source)
    await rooms[1].local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    async with llm, tts, stt, AgentSession(vad=vad, stt=stt, tts=tts, preemptive_generation=preemptive) as session:
        frames = [chunk.frame async for chunk in tts.synthesize("hello there")]
        assert frames

        def conversation(event):
            if event.item.role == "assistant":
                heard.set()

        session.on("conversation_item_added", conversation)
        await session.start(Agent(instructions="Reply with exactly: hi there", llm=llm), room=rooms[0], record=False)
        owner = getattr(session, STATE_ATTR)
        try:
            silence = rtc.AudioFrame(data=bytes(2400 * 2), sample_rate=24000, num_channels=1, samples_per_channel=2400)
            for _ in range(5):
                await input_source.capture_frame(silence)
            for frame in frames:
                await input_source.capture_frame(frame)
            for _ in range(15):
                await input_source.capture_frame(silence)
            await asyncio.wait_for(heard.wait(), 15)
            await session.drain()
            assert owner.input_end is not None
        finally:
            await input_source.aclose()
    rows = memory_logger.pop()
    assert named(rows, "user_speech")
    assert named(rows, "stt")
    assert named(rows, "llm_response")
    assert named(rows, "tts")
    assert named(rows, "agent_speech")
    tts_ids = {row["span_id"] for row in named(rows, "tts")}
    for speech in named(rows, "agent_speech"):
        relationships = speech["metadata"]["links"]
        assert len(relationships) == 1
        assert relationships[0]["context"]["spanId"] in tts_ids
        assert relationships[0]["attributes"] == {"braintrust.relationship.type": "generated_by"}
    user = next(row for row in named(rows, "user_turn") if row.get("input"))
    assert user["input"][0]["content"] == "Hello there."
    assert user["metadata"]["audio.selections"]
    by_id = {row["span_id"]: row for row in rows}
    for speech in named(rows, "user_speech"):
        parent = by_id[speech["span_parents"][0]]
        assert parent["metrics"]["start"] <= speech["metrics"]["start"]
    assert any(row["metadata"].get("turn.reply_to") == user["span_id"] for row in named(rows, "agent_turn"))
    root = named(rows, "session")[0]
    assert uploads
    assert all(
        len(segment["channels"]) == (2 if directions == "both" else 1)
        for segment in root["metadata"]["audio.segments"].values()
    )
    assert owner.native is None
