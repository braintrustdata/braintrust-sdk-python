import asyncio
import time
from array import array

import pytest
from braintrust import logger
from braintrust._audio.config import AudioEncoding, AudioRecordingOptions
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.livekit_agents.compat import STATE_ATTR


def named(rows, name):
    return [row for row in rows if row.get("span_attributes", {}).get("name") == name]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "full",
        "interrupted",
        "resumed",
        "handoff",
        "capture_cancel",
        "unknown_timing",
        "duplicate",
        "early_progress",
        "progress_only",
        "endpoints",
        "paused_endpoints",
    ],
)
async def test_actual_roomio_playback_records_only_confirmed_audio(rooms, memory_logger, monkeypatch, mode):
    from livekit import rtc
    from livekit.agents import Agent, AgentSession

    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", "true")
    monkeypatch.delenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", raising=False)
    uploads: list[logger.Attachment] = []
    native_discarded_seconds = 0.0

    def upload(attachment: logger.Attachment) -> dict[str, str]:
        uploads.append(attachment)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    assert setup_livekit_agents(
        audio=AudioRecordingOptions(
            encoding=AudioEncoding(codec="pcm", container=None),
            segment_duration_seconds=0.1,
        )
    )

    async def audio():
        for _ in range(30):
            yield rtc.AudioFrame(
                data=array("h", [1200] * 480).tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=480
            )

    session = AgentSession()
    try:
        await asyncio.wait_for(session.start(Agent(instructions="test"), room=rooms[0], record=False), 15)
        owner = getattr(session, STATE_ATTR)
        if mode == "resumed":
            leaf = owner.tap.next_in_chain
            while leaf.next_in_chain is not None:
                leaf = leaf.next_in_chain

            def observe_discard(_event):
                nonlocal native_discarded_seconds
                # RoomIO 1.8.10 intentionally drops the device queue on pause.
                # Observe its counter before the final callback resets it.
                native_discarded_seconds = max(native_discarded_seconds, leaf._source_discarded_duration)

            leaf.on("playback_progressed", observe_discard)
        if mode == "duplicate":
            owner.tap.off("playback_progressed", owner.tap._progressed)

            def twice(event):
                owner.tap._progressed(event)
                owner.tap._progressed(event)

            owner.tap.on("playback_progressed", twice)
        if mode == "early_progress":
            downstream = owner.tap.next_in_chain
            capture = downstream.capture_frame
            first = True

            async def delayed_ack(frame):
                nonlocal first
                await capture(frame)
                if first:
                    first = False
                    await asyncio.sleep(frame.duration + 0.03)
                    leaf = downstream
                    while leaf.next_in_chain is not None:
                        leaf = leaf.next_in_chain
                    # Report the actual native device playhead while the tap's
                    # capture acknowledgement is still outstanding.
                    leaf._report_run(
                        offset=leaf._source_pushed_duration - leaf._audio_source.queued_duration, ended_at=time.time()
                    )

            monkeypatch.setattr(downstream, "capture_frame", delayed_ack)
        if mode == "capture_cancel":
            # Rare transport race: the real sink accepted a frame, but its
            # caller is cancelled before the acknowledgement returns. Keep all
            # real RoomIO/RTC behavior and inject only the blocked return path.
            downstream = owner.tap.next_in_chain
            capture = downstream.capture_frame

            async def blocked_capture(frame):
                await capture(frame)
                await asyncio.Event().wait()

            monkeypatch.setattr(downstream, "capture_frame", blocked_capture)
        if mode == "unknown_timing":
            # Exercise the actual endpoint-only event path without a timing
            # signal at the adapter's sole observation point.
            owner.tap.off("playback_started", owner.tap._started)
            owner.tap.off("playback_progressed", owner.tap._progressed)
        if mode == "progress_only":
            owner.tap.off("playback_started", owner.tap._started)
        if mode in ("endpoints", "paused_endpoints"):
            owner.tap.off("playback_progressed", owner.tap._progressed)
        started = asyncio.Event()
        session.output.audio.on("playback_started", lambda _: started.set())
        handle = session.say("heard text", audio=audio())
        if mode in ("interrupted", "resumed", "capture_cancel", "paused_endpoints"):
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.sleep(0.12)
            if mode in ("interrupted", "capture_cancel"):
                handle.interrupt(force=True)
            else:
                session.output.audio.pause()
                await asyncio.sleep(0.1)
                session.output.audio.resume()
        await asyncio.wait_for(handle.wait_for_playout(), 5)
        if mode == "handoff":
            replacement = Agent(instructions="Replacement agent")
            session.update_agent(replacement)
            # current_agent changes before the outgoing activity finishes
            # draining. Wait for the native handoff before scheduling speech.
            await asyncio.wait_for(asyncio.shield(session._update_activity_atask), 5)
            assert session.current_agent is replacement
            assert session.output.audio is owner.tap
            await asyncio.wait_for(session.say("second agent", audio=audio()).wait_for_playout(), 5)
    finally:
        await session.aclose()
    rows = memory_logger.pop()
    speech = named(rows, "agent_speech")
    if mode in ("capture_cancel", "unknown_timing", "paused_endpoints"):
        assert not speech
        segments = named(rows, "session")[0]["metadata"]["audio.segments"]
        assert any(segment["state"] == "omitted" for segment in segments.values())
        assert not any(
            segment.get("channels", {}).get("0", {}).get("sources") == ["agent-output"]
            for segment in segments.values()
        )
        return
    assert speech, rows
    selections = [selection for span in speech for selection in span["metadata"]["audio.selections"].values()]
    played_ms = sum(s["end_offset_ms"] - s["start_offset_ms"] for s in selections)
    if mode == "interrupted":
        assert 0 < played_ms < 600
    elif mode in ("full", "handoff", "duplicate", "early_progress", "progress_only", "endpoints"):
        assert abs(played_ms - (1200 if mode == "handoff" else 600)) <= 2
    else:
        assert len(selections) >= 2
        # The pinned sink queues 200 ms plus an in-flight 50 ms native frame.
        assert 0 <= native_discarded_seconds <= 0.25
        expected_ms = 600 - native_discarded_seconds * 1000
        # Each rounded selection endpoint can contribute at most 0.5 ms.
        assert abs(played_ms - expected_ms) <= len(selections)
    root = named(rows, "session")[0]
    segments = root["metadata"]["audio.segments"]
    assert all(segment["state"] == "ready" for segment in segments.values()), segments
    assert len(uploads) == len(segments)
    assert all(
        segment["channels"] in ({"0": {"sources": ["agent-output"]}}, {"0": {"sources": []}})
        for segment in segments.values()
    )
    assert any(segment["channels"]["0"]["sources"] == ["agent-output"] for segment in segments.values())
    if mode in ("full", "resumed"):
        audible_samples = 0
        for attachment in uploads:
            identifier = attachment.reference["filename"].removesuffix(".pcm")
            segment = segments[identifier]
            samples = array("h")
            samples.frombytes(attachment.data)
            # Placement is rounded to milliseconds; allow one ms at each end.
            assert abs(len(samples) - segment["duration_ms"] * 48) <= 48
            assert set(samples) <= {0, 1200}
            for index, value in enumerate(samples):
                position_ms = segment["start_offset_ms"] + index / 48
                inside = any(
                    selection["start_offset_ms"] + 1 <= position_ms < selection["end_offset_ms"] - 1
                    for selection in selections
                )
                outside = all(
                    position_ms < selection["start_offset_ms"] - 1 or position_ms >= selection["end_offset_ms"] + 1
                    for selection in selections
                )
                if inside:
                    assert value == 1200
                if outside:
                    assert value == 0
                audible_samples += value == 1200
        # Supplied PCM is 14,400 samples at 24 kHz, normalized to 48 kHz.
        # Exclude only the independently observed native pause discard.
        expected_samples = round((0.6 - native_discarded_seconds) * 48000)
        assert abs(audible_samples - expected_samples) <= len(selections)
    # Conversation timestamps end before upload/drain and state ownership is gone.
    assert owner.native is None
    assert not hasattr(session, STATE_ATTR)
