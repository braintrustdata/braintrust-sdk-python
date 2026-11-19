"""Synthetic lifecycle events on real LiveKit sessions, supplemental to VCR cascade.

No provider response is fabricated. Explicit native hooks let us exercise
rearms, corrections and teardown independently of audio/scheduler timing.
"""

import asyncio
import inspect
import time

import pytest
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.livekit_agents.compat import STATE_ATTR


@pytest.mark.parametrize("hook", range(6))
@pytest.mark.parametrize("change", ["missing", "signature", "async"])
def test_eou_optional_hooks_are_gated_together(monkeypatch, hook, change):
    """Pure patch routing: incompatible native layouts must disable the feature."""
    import importlib

    from livekit.agents.voice.audio_recognition import AudioRecognition

    from .patchers import EOUWaitPatcher

    patcher = EOUWaitPatcher.sub_patchers[hook]
    module = importlib.import_module(patcher.target_module)
    parent, name = patcher.target_path.rsplit(".", 1)
    target = getattr(module, parent)
    method = getattr(target, name, None)
    if method is None:
        pytest.skip("Manual snapshots are unavailable in this framework version")
    assert EOUWaitPatcher.applies(None, None)
    replacement = None if change == "missing" else dict.get
    if change == "async" and not inspect.iscoroutinefunction(method):
        replacement = AudioRecognition._on_vad_event
    monkeypatch.setattr(target, name, replacement)
    assert not EOUWaitPatcher.applies(None, None)


async def vad(recognition, kind, *, speech=0.0, silence=0.0):
    from livekit.agents.vad import VADEvent, VADEventType

    await recognition._on_vad_event(
        VADEvent(
            type=VADEventType(kind),
            samples_index=0,
            timestamp=time.time(),
            speech_duration=speech,
            silence_duration=silence,
        )
    )


def waits(rows):
    return sorted(
        (r for r in rows if r.get("span_attributes", {}).get("name") == "eou_wait"),
        key=lambda r: r["metrics"]["start"],
    )


def ensure(recognition, anchor=None):
    return recognition._ensure_eou_wait_span(
        recognition._ensure_user_turn_span(),
        trigger="vad",
        last_speaking_time=recognition._last_speaking_time if anchor is None else anchor,
        endpointing_delay=0.5,
    )


@pytest.fixture(params=[False, True])
def native_tracing(request):
    from livekit.agents.telemetry import traces
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import NoOpTracerProvider

    original = traces.tracer._tracer_provider
    provider = TracerProvider() if request.param else NoOpTracerProvider()
    traces.tracer.set_provider(provider)
    try:
        yield request.param
    finally:
        traces.tracer.set_provider(original)
        if request.param:
            provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["committed", "dropped", "shutdown"])
async def test_eou_pause_resume_rearm_and_terminal(memory_logger, terminal):
    from livekit.agents import Agent, AgentSession
    from livekit.agents.telemetry import traces
    from opentelemetry.sdk.trace import TracerProvider

    setup_livekit_agents()
    # Real OTel spans are needed to inspect LiveKit's cleanup floor and attrs.
    original = traces.tracer._tracer_provider
    provider = TracerProvider()
    traces.tracer.set_provider(provider)
    try:
        async with AgentSession(turn_detection="manual") as session:
            await session.start(Agent(instructions="lifecycle test"))
            owner = getattr(session, STATE_ATTR)
            recognition = session._activity._audio_recognition
            await asyncio.sleep(0.02)
            await vad(recognition, "start_of_speech", speech=0.01)
            await vad(recognition, "end_of_speech", silence=0.005)
            first_start = recognition._last_speaking_time
            first = ensure(recognition)
            ensure(recognition)
            # LiveKit wraps OTel's contextmanager decorator.
            with traces.tracer.start_as_current_span("eou_detection", end_on_exit=False) as detection:  # pylint: disable=not-context-manager
                with recognition._track_eou_detection_span(detection):
                    # Detector cleanup runs after the actual reported resume.
                    await asyncio.sleep(0.01)
                    await vad(recognition, "start_of_speech", speech=0.005)
            first_native_end = first.end_time / 1e9
            resume = recognition._speech_start_time
            assert first_native_end > resume
            await asyncio.sleep(0.01)
            await vad(recognition, "end_of_speech", silence=0.005)
            last_start = recognition._last_speaking_time
            last = ensure(recognition)
            ensure(recognition)
            if terminal != "shutdown":
                if terminal == "dropped":
                    recognition._clear_user_turn()
                else:
                    recognition._end_eou_wait_span(terminal)
                recognition._end_eou_wait_span(terminal)  # idempotent
            # Shutdown must not turn LiveKit's cleanup label into abandonment.
        assert not owner.eou_active
    finally:
        traces.tracer.set_provider(original)
        provider.shutdown()
    rows = memory_logger.pop()
    emitted = waits(rows)
    assert len(emitted) == 2
    first_row, last_row = emitted
    assert first_row["metrics"]["start"] == first_start
    assert first_row["metrics"]["end"] == resume
    assert first_row["metadata"]["eou.outcome"] == "user_resumed"
    assert first_row["metadata"]["contrib.livekit.lk.eou.outcome"] == "user_resumed"
    assert first_row["metadata"]["contrib.livekit.native_end_time_ns"] == first.end_time
    assert last_row["metrics"]["start"] == last_start
    if terminal == "shutdown":
        assert "eou.outcome" not in last_row["metadata"]
        assert last_row["metadata"]["contrib.livekit.lk.eou.outcome"] == "dropped"
    else:
        assert last_row["metadata"]["eou.outcome"] == terminal
        assert last_row["metadata"]["contrib.livekit.outcome"] == terminal
    for row in emitted:
        assert row["metadata"]["eou.speech_end_source"] == "vad"
        assert "eou.end_reason" not in row["metadata"]
        assert "eou.detection_methods" not in row["metadata"]
        assert row["span_parents"] == [row["metadata"]["turn.id"]]
        assert row["context"]["span_origin"]["instrumentation"]["name"] == "livekit-agents-auto"
    framework = next(r for r in rows if r.get("span_attributes", {}).get("name") == "livekit.eou_detection")
    assert framework["metrics"]["end"] > first_row["metrics"]["end"]
    assert all("end" in row["metrics"] for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", [None, 0, -1, True, float("inf")])
async def test_eou_requires_observed_speech_anchor(memory_logger, anchor):
    from livekit.agents import Agent, AgentSession

    setup_livekit_agents()
    async with AgentSession(turn_detection="manual") as session:
        await session.start(Agent(instructions="lifecycle test"))
        recognition = session._activity._audio_recognition
        # No speech-end observation, even when native uses a plausible 'now'.
        ensure(recognition, anchor)
    assert not waits(memory_logger.pop())


@pytest.mark.asyncio
async def test_eou_unknown_turn_membership_and_corrected_anchor(memory_logger):
    from livekit.agents import Agent, AgentSession
    from livekit.agents.telemetry import traces
    from opentelemetry.sdk.trace import TracerProvider

    original = traces.tracer._tracer_provider
    provider = TracerProvider()
    traces.tracer.set_provider(provider)
    setup_livekit_agents()
    async with AgentSession(turn_detection="manual") as session:
        await session.start(Agent(instructions="lifecycle test"))
        owner = getattr(session, STATE_ATTR)
        recognition = session._activity._audio_recognition
        await vad(recognition, "end_of_speech", silence=0.2)
        # A native span with no observed user-turn association stays at session.
        with traces.tracer.start_as_current_span("unassociated turn") as native_user:  # pylint: disable=not-context-manager
            first = recognition._last_speaking_time
            recognition._ensure_eou_wait_span(
                native_user, trigger="vad", last_speaking_time=first, endpointing_delay=0.5
            )
            await vad(recognition, "end_of_speech", silence=0.1)
            corrected = recognition._last_speaking_time
            recognition._ensure_eou_wait_span(
                native_user, trigger="stt", last_speaking_time=corrected, endpointing_delay=0.5
            )
            recognition._end_eou_wait_span("committed")
    traces.tracer.set_provider(original)
    provider.shutdown()
    (row,) = waits(memory_logger.pop())
    assert row["span_parents"] == [owner.span.span_id]
    assert "turn.id" not in row["metadata"]
    assert row["metrics"]["start"] == corrected
    # An STT evaluation trigger does not change the timestamp's VAD origin.
    assert row["metadata"]["eou.speech_end_source"] == "vad"


@pytest.mark.asyncio
async def test_eou_waits_for_transcript_before_native_detector_exists(memory_logger):
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import openai, silero

    setup_livekit_agents()
    # Real segmented STT and VAD; no PCM is submitted, so no provider request
    # is made. Synthetic native VAD events expose the pending-transcript path.
    async with openai.STT() as stt, AgentSession(vad=silero.VAD.load(), stt=stt, turn_detection="vad") as session:
        await session.start(Agent(instructions="lifecycle test"))
        owner = getattr(session, STATE_ATTR)
        recognition = session._activity._audio_recognition
        await asyncio.sleep(0.02)
        await vad(recognition, "start_of_speech", speech=0.01)
        await vad(recognition, "end_of_speech", silence=0.005)
        assert recognition._eou_wait_span is None
        assert recognition in owner.eou_active
        await asyncio.sleep(0.01)
        await vad(recognition, "start_of_speech", speech=0.005)
        await asyncio.sleep(0.01)
        await vad(recognition, "end_of_speech", silence=0.005)
    assert not owner.eou_active and not owner.eou_snapshots
    first, second = waits(memory_logger.pop())
    assert first["metadata"]["eou.outcome"] == "user_resumed"
    assert "eou.outcome" not in second["metadata"]
    assert first["metrics"]["end"] < second["metrics"]["start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("second_turn", [False, True])
async def test_eou_pending_manual_snapshot_survives_buffer_clear(memory_logger, second_turn, native_tracing):
    from livekit.agents import Agent, AgentSession, stt
    from livekit.agents.voice.audio_recognition import AudioRecognition
    from livekit.plugins import openai, silero

    if not hasattr(AudioRecognition, "_take_turn_snapshot"):
        pytest.skip("Native manual snapshots require LiveKit 1.8.12")
    setup_livekit_agents()
    async with (
        openai.STT() as service,
        AgentSession(
            stt=service, vad=silero.VAD.load(), turn_detection="manual", min_endpointing_delay=0.4
        ) as session,
    ):
        await session.start(Agent(instructions="manual snapshot regression"))
        recognition = session._activity._audio_recognition
        owner = getattr(session, STATE_ATTR)
        await vad(recognition, "start_of_speech")
        recognition._process_stt_event(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="en", text="Keep this turn")],
            )
        )
        await vad(recognition, "end_of_speech")
        anchor = recognition._last_speaking_time
        first_turn = owner.user
        await session.commit_user_turn(transcript_timeout=0, stt_flush_duration=0, skip_reply=True)
        pending = recognition._end_of_turn_task
        assert pending is not None and not pending.done()
        session.clear_user_turn()
        if second_turn:
            await vad(recognition, "start_of_speech")
            recognition._process_stt_event(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language="en", text="A different turn")],
                )
            )
            await vad(recognition, "end_of_speech")
            next_turn = owner.user
        await asyncio.wait_for(asyncio.shield(pending), 2)
        await asyncio.sleep(0)  # native snapshot done callback
        if second_turn:
            assert owner.user is next_turn
            await session.commit_user_turn(transcript_timeout=0, stt_flush_duration=0, skip_reply=True)
            await asyncio.wait_for(asyncio.shield(recognition._end_of_turn_task), 2)
            await asyncio.sleep(0)
        assert not owner.eou_snapshots and not owner.eou_active
    rows = memory_logger.pop()
    emitted = waits(rows)
    assert len(emitted) == (2 if second_turn else 1)
    row = emitted[0]
    assert row["metrics"]["start"] == anchor
    assert row["metadata"]["eou.outcome"] == "committed"
    if native_tracing:
        assert row["metadata"]["contrib.livekit.lk.eou.outcome"] == "committed"
    assert row["metrics"]["end"] - anchor >= 0.35
    assert row["span_parents"] == [first_turn.span_id]
    assert row["metadata"]["turn.id"] == first_turn.span_id
    if second_turn:
        assert emitted[1]["span_parents"] == [next_turn.span_id]
        assert emitted[1]["metadata"]["eou.outcome"] == "committed"
    assert all("end" in span["metrics"] for span in rows)
