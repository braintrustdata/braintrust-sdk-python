import asyncio
import multiprocessing
from array import array

import pytest
from braintrust import logger
from braintrust.audio import AudioEncoding, AudioRecordingOptions
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.livekit_agents.compat import STATE_ATTR


@pytest.mark.asyncio
@pytest.mark.parametrize("project_redaction", [False, True])
async def test_real_job_shutdown_callbacks_settle_playback_before_sealing(
    rooms, memory_logger, monkeypatch, project_redaction
):
    from livekit import rtc
    from livekit.agents import Agent, AgentSession
    from livekit.agents.ipc.inference_proc_executor import InferenceProcExecutor
    from livekit.agents.job import (
        JobAcceptArguments,
        JobContext,
        JobExecutorType,
        JobProcess,
        RunningJobInfo,
        _JobContextVar,
    )
    from livekit.protocol import agent, models

    # Use the real job context, process descriptor, protocol messages and
    # inference executor. The local job needs no inference subprocess.
    executor = InferenceProcExecutor(
        runners={},
        initialize_timeout=10,
        close_timeout=10,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=1,
        ping_timeout=10,
        high_ping_threshold=1,
        mp_ctx=multiprocessing.get_context("spawn"),
        loop=asyncio.get_running_loop(),
        http_proxy=None,
    )
    context = JobContext(
        proc=JobProcess(executor_type=JobExecutorType.THREAD, user_arguments=None, http_proxy=None),
        info=RunningJobInfo(
            accept_arguments=JobAcceptArguments(name="agent", identity="agent", metadata=""),
            # Protocol classes are generated dynamically by protobuf.
            job=agent.Job(  # pylint: disable=no-member
                id="JB_cascade_test",
                room=models.Room(sid=await rooms[0].sid, name=rooms[0].name),  # pylint: disable=no-member
                enable_redaction=project_redaction,
            ),
            url="ws://localhost:7880",
            token="",
            worker_id="local-test",
            fake_job=False,
        ),
        room=rooms[0],
        on_connect=lambda: None,
        on_shutdown=lambda _: None,
        inference_executor=executor,
    )
    context._connected = True  # The fixture has already established the real transport.
    token = _JobContextVar.set(context)
    uploads = []

    def upload(attachment):
        uploads.append(attachment)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", "true")
    monkeypatch.delenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", raising=False)
    setup_livekit_agents(audio=AudioRecordingOptions(encoding=AudioEncoding(codec="pcm", container=None)))
    session = AgentSession()

    async def audio():
        for _ in range(60):
            yield rtc.AudioFrame(
                data=array("h", [900] * 480).tobytes(), sample_rate=24000, num_channels=1, samples_per_channel=480
            )

    try:
        await session.start(Agent(instructions="test"), room=rooms[0], record=False)
        owner = getattr(session, STATE_ATTR)
        started = asyncio.Event()
        session.output.audio.on("playback_started", lambda _: started.set())
        session.say("private spoken content", audio=audio())
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.sleep(0.08)
        callbacks = tuple(context._shutdown_callbacks)
        assert len(callbacks) >= 2
        # This is LiveKit's actual job-runner dispatch pattern. Our fallback is
        # registered before LiveKit's own callback and must not seal early.
        await asyncio.wait_for(asyncio.gather(*(callback("test shutdown") for callback in callbacks)), 10)
        await asyncio.gather(*(callback("repeated shutdown") for callback in callbacks))
        assert owner.native is None
        assert not hasattr(session, STATE_ATTR)
        rows = memory_logger.pop()
        root = next(row for row in rows if row.get("span_attributes", {}).get("name") == "session")
        if project_redaction:
            assert not uploads
            assert "input" not in root
            assert all("input" not in row and "output" not in row for row in rows)
        else:
            assert uploads
            assert all(segment["state"] == "ready" for segment in root["metadata"]["audio.segments"].values())
    finally:
        await session.aclose()
        _JobContextVar.reset(token)
        rooms[0].off("participant_connected", context._participant_available)
        await context._on_cleanup()
