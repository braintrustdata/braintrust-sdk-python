import asyncio
import inspect
import shutil
import socket
import subprocess
import time

import pytest
from braintrust import logger
from braintrust.auto import auto_instrument
from braintrust.integrations.livekit_agents import (
    LiveKitAgentsIntegration,
    setup_livekit_agents,
    tracing,
    wrap_livekit_agents,
)
from braintrust.integrations.livekit_agents.tracing import (
    _SESSION_PARENT_ATTR,
    _SESSION_SPAN_ATTR,
    traced_llm_stream_run,
    traced_session_start,
)
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


@pytest.fixture
def memory_logger():
    init_test_logger("test-project-livekit-agents-py-tracing")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _span_name(log):
    return log.get("span_attributes", {}).get("name")


def _span_names(logs):
    return {_span_name(log) for log in logs}


def _spans_named(logs, name):
    return [log for log in logs if _span_name(log) == name]


def _single_span(logs, name):
    matches = _spans_named(logs, name)
    assert len(matches) == 1, (name, matches)
    return matches[0]


def _assert_any_span(spans, predicate):
    assert any(predicate(span) for span in spans), spans


def _assert_all_spans(spans, predicate):
    assert all(predicate(span) for span in spans), spans


def _assert_all_spans_have_end_time(logs):
    spans_without_end = [
        log for log in logs if log.get("span_attributes") is not None and "end" not in log.get("metrics", {})
    ]
    assert not spans_without_end, spans_without_end


def _assert_no_metrics(spans, *metric_names):
    assert all(not (set(metric_names) & set(span.get("metrics", {}))) for span in spans), spans


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_in_use(port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"LiveKit server did not start within {timeout}s")


@pytest.fixture(autouse=True)
def isolate_livekit_agents_tracing_state():
    # pylint: disable=protected-access
    latest_session_parent = tracing._LATEST_SESSION_PARENT
    outer_parent_to_session_parent = tracing._OUTER_PARENT_TO_SESSION_PARENT.copy()
    outer_span_id_to_session_parent = tracing._OUTER_SPAN_ID_TO_SESSION_PARENT.copy()
    active_session_parent_token = tracing._ACTIVE_SESSION_PARENT.set(None)

    tracing._LATEST_SESSION_PARENT = None
    tracing._OUTER_PARENT_TO_SESSION_PARENT.clear()
    tracing._OUTER_SPAN_ID_TO_SESSION_PARENT.clear()
    try:
        yield
    finally:
        tracing._ACTIVE_SESSION_PARENT.reset(active_session_parent_token)
        tracing._LATEST_SESSION_PARENT = latest_session_parent
        tracing._OUTER_PARENT_TO_SESSION_PARENT.clear()
        tracing._OUTER_PARENT_TO_SESSION_PARENT.update(outer_parent_to_session_parent)
        tracing._OUTER_SPAN_ID_TO_SESSION_PARENT.clear()
        tracing._OUTER_SPAN_ID_TO_SESSION_PARENT.update(outer_span_id_to_session_parent)
    # pylint: enable=protected-access


@pytest.fixture(scope="session")
def livekit_server():
    port = 7880
    if _port_in_use(port):
        yield
        return

    livekit_server_bin = shutil.which("livekit-server")
    if not livekit_server_bin:
        pytest.skip("livekit-server is required for LiveKit e2e tests")

    proc = subprocess.Popen([livekit_server_bin, "--dev"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_port(port)
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


@pytest.mark.asyncio
async def test_llm_stream_run_does_not_create_user_turn_on_cancellation(memory_logger):
    async def cancelled_run():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await traced_llm_stream_run(cancelled_run, object(), (), {})

    memory_logger.pop()


@pytest.mark.asyncio
async def test_session_start_cleans_up_span_on_cancellation(memory_logger):
    pytest.importorskip("livekit.agents")

    from livekit.agents import AgentSession

    async def cancelled_start():
        raise asyncio.CancelledError

    session = AgentSession()
    with pytest.raises(asyncio.CancelledError):
        await traced_session_start(cancelled_start, session, (), {})

    assert getattr(session, _SESSION_SPAN_ATTR) is None
    assert getattr(session, _SESSION_PARENT_ATTR) is None
    logs = memory_logger.pop()
    session_logs = _spans_named(logs, "livekit_agent_session")
    assert len(session_logs) == 1


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_openai_tts_stream_closes_span_when_closed_early(memory_logger):
    from livekit.plugins import openai

    assert setup_livekit_agents()
    tts = openai.TTS(model="gpt-4o-mini-tts", voice="ash")
    stream = tts.synthesize("hello there")

    async for chunk in stream:
        assert chunk.frame
        break
    await stream.aclose()

    logs = memory_logger.pop()
    tts_logs = _spans_named(logs, "tts_request")
    assert tts_logs
    assert any(log.get("metrics", {}).get("time_to_first_token", 0) > 0 for log in tts_logs), tts_logs


def test_livekit_agents_integration_min_version():
    assert LiveKitAgentsIntegration.min_version == "1.3.1"


def test_auto_instrument_livekit_agents_subprocess():
    pytest.importorskip("livekit.agents")
    verify_autoinstrument_script("test_auto_livekit_agents.py")


def test_wrap_livekit_agents_wraps_real_agent_session():
    pytest.importorskip("livekit.agents")

    from livekit.agents import AgentSession

    WrappedAgentSession = wrap_livekit_agents(AgentSession)

    assert WrappedAgentSession is AgentSession
    assert wrap_livekit_agents(AgentSession) is AgentSession
    assert hasattr(inspect.getattr_static(AgentSession, "start"), "__wrapped__")
    assert hasattr(inspect.getattr_static(AgentSession, "run"), "__wrapped__")
    assert hasattr(inspect.getattr_static(AgentSession, "say"), "__wrapped__")


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_livekit_agents_agent_speaking_e2e(memory_logger, livekit_server):
    assert setup_livekit_agents()

    from livekit.agents import Agent, AgentSession
    from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities
    from livekit.plugins import openai

    class CapturingAudioOutput(AudioOutput):
        def __init__(self):
            super().__init__(
                label="test_audio_output", capabilities=AudioOutputCapabilities(pause=False), sample_rate=24000
            )
            self.playback_position = 0.0

        async def capture_frame(self, frame):
            await super().capture_frame(frame)
            self.playback_position += frame.samples_per_channel / frame.sample_rate

        def flush(self):
            super().flush()
            self.on_playback_finished(
                playback_position=self.playback_position,
                interrupted=False,
                synchronized_transcript="hi there",
            )

        def clear_buffer(self):
            self.flush()

    class Assistant(Agent):
        def __init__(self, llm):
            super().__init__(instructions="Reply with exactly: hi there", llm=llm)

    llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8)
    tts = openai.TTS(model="gpt-4o-mini-tts", voice="ash")
    audio_output = CapturingAudioOutput()

    async with llm, tts, AgentSession(tts=tts) as session:
        session.output.audio = audio_output
        await session.start(Assistant(llm))
        result = await session.run(user_input="say hi")
        result.expect.next_event().is_message(role="assistant")

    logs = memory_logger.pop()
    span_names = _span_names(logs)
    agent_speaking_logs = _spans_named(logs, "agent_speaking")
    assert agent_speaking_logs
    audio_attachments = [
        log.get("output", {}).get("audio") for log in agent_speaking_logs if log.get("output", {}).get("audio")
    ]
    assert audio_attachments, agent_speaking_logs
    assert any(attachment.reference["content_type"] == "audio/wav" for attachment in audio_attachments), (
        agent_speaking_logs
    )
    assert any(log.get("output", {}).get("text") == "hi there" for log in agent_speaking_logs), agent_speaking_logs


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_livekit_agents_e2e_parenting_under_custom_span(memory_logger, livekit_server):
    assert setup_livekit_agents()
    await _run_livekit_agents_openai_e2e_voice_turn(memory_logger, outer_parent_name="custom_livekit_parent")


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_livekit_agents_function_tool_e2e(memory_logger, livekit_server):
    assert setup_livekit_agents()

    from livekit.agents import Agent, AgentSession, function_tool
    from livekit.plugins import openai

    @function_tool
    async def lookup_weather(city: str) -> str:
        return f"sunny in {city}"

    class Assistant(Agent):
        def __init__(self, llm):
            super().__init__(
                instructions="Use lookup_weather for weather questions, then answer briefly.",
                llm=llm,
                tools=[lookup_weather],
            )

    llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=32)

    async with llm, AgentSession() as session:
        await session.start(Assistant(llm))
        result = await session.run(user_input="What is the weather in Paris?")
        result.expect.next_event().is_function_call(name="lookup_weather")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_message(role="assistant")

    logs = memory_logger.pop()
    span_names = _span_names(logs)
    assert "function_tool" in span_names, sorted(str(name) for name in span_names)
    function_tool_logs = _spans_named(logs, "function_tool")
    assert all("tokens" not in log.get("metrics", {}) for log in function_tool_logs)
    assert any(log.get("input", {}).get("name") == "lookup_weather" for log in function_tool_logs), function_tool_logs
    assert any(log.get("input", {}).get("arguments", {}).get("city") == "Paris" for log in function_tool_logs), (
        function_tool_logs
    )
    assert any(log.get("output", {}).get("output") == "sunny in Paris" for log in function_tool_logs), (
        function_tool_logs
    )
    session_logs = _spans_named(logs, "livekit_agent_session")
    assert session_logs, logs
    assert all(log.get("root_span_id") == session_logs[0].get("root_span_id") for log in function_tool_logs), logs


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_setup_livekit_agents_openai_e2e_voice_turn(memory_logger, livekit_server):
    assert setup_livekit_agents()
    await _run_livekit_agents_openai_e2e_voice_turn(memory_logger)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_auto_instrument_livekit_agents_openai_e2e_voice_turn(memory_logger, livekit_server):
    results = auto_instrument()
    assert results.get("livekit_agents") is True
    await _run_livekit_agents_openai_e2e_voice_turn(memory_logger)


async def _run_livekit_agents_openai_e2e_voice_turn(memory_logger, outer_parent_name=None):
    from livekit import rtc
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import openai

    silero = pytest.importorskip("livekit.plugins.silero")

    class Assistant(Agent):
        def __init__(self, llm):
            super().__init__(instructions="Reply with exactly: hi there", llm=llm)

    tts, stt, llm, vad = _create_openai_voice_turn_components(openai, silero)
    audio_frames, speech_text = await _prepare_openai_speech_input(tts, stt)

    async def run_session():
        await _run_openai_voice_turn_session(
            agent_session_cls=AgentSession,
            assistant_cls=Assistant,
            rtc=rtc,
            llm=llm,
            tts=tts,
            stt=stt,
            vad=vad,
            audio_frames=audio_frames,
            user_input=speech_text,
        )

    if outer_parent_name:
        with logger.start_span(name=outer_parent_name):
            await run_session()
    else:
        await run_session()

    logs = memory_logger.pop()
    _assert_openai_voice_turn_spans(logs, speech_text=speech_text, outer_parent_name=outer_parent_name)


def _create_openai_voice_turn_components(openai, silero):
    tts = openai.TTS(model="gpt-4o-mini-tts", voice="ash")
    stt = openai.STT(model="gpt-4o-mini-transcribe")
    llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8)
    vad = silero.VAD.load()
    return tts, stt, llm, vad


async def _prepare_openai_speech_input(tts, stt):
    tts_stream = tts.synthesize("hello there")
    audio_frames = [chunk.frame async for chunk in tts_stream]
    assert audio_frames

    speech_event = await stt.recognize(audio_frames)
    assert speech_event.alternatives
    speech_text = speech_event.alternatives[0].text
    assert speech_text
    return audio_frames, speech_text


async def _run_openai_voice_turn_session(
    *, agent_session_cls, assistant_cls, rtc, llm, tts, stt, vad, audio_frames, user_input
):
    async with llm, tts, stt, agent_session_cls(vad=vad) as session:
        await session.start(assistant_cls(llm))
        session._update_user_state("speaking")
        await asyncio.sleep(0.01)
        session._update_user_state("listening")
        await _run_silero_vad(vad, rtc, audio_frames)
        result = await session.run(user_input=user_input)
        result.expect.next_event().is_message(role="assistant")


def _assert_openai_voice_turn_spans(logs, *, speech_text, outer_parent_name):
    _assert_all_spans_have_end_time(logs)
    _assert_required_voice_turn_spans(logs)
    _assert_tts_spans(logs)
    _assert_vad_endpointing_spans(logs)
    _assert_eou_spans(logs)
    _assert_stt_spans(logs, speech_text)
    _assert_session_spans(logs)
    _assert_custom_metrics_stay_in_metadata(logs)

    if outer_parent_name:
        _assert_voice_turn_parenting(logs, outer_parent_name)


def _assert_required_voice_turn_spans(logs):
    span_names = _span_names(logs)
    span_name_list = sorted(str(name) for name in span_names)
    for span_name in ("llm_request_run", "user_speaking", "vad_endpointing", "tts_request", "stt_processing"):
        assert span_name in span_names, span_name_list


def _assert_tts_spans(logs):
    tts_logs = _spans_named(logs, "tts_request")
    _assert_any_span(tts_logs, lambda log: log.get("input", {}).get("text") == "hello there")
    _assert_any_span(tts_logs, lambda log: log.get("metrics", {}).get("time_to_first_token", 0) > 0)
    _assert_all_spans(tts_logs, lambda log: "audio_duration" not in log.get("metrics", {}))
    _assert_any_span(
        tts_logs, lambda log: log.get("metadata", {}).get("livekit_metrics", {}).get("audio_duration", 0) > 0
    )
    _assert_all_spans(tts_logs, lambda log: "ttfb" not in log.get("metadata", {}).get("livekit_metrics", {}))


def _assert_vad_endpointing_spans(logs):
    vad_endpointing_logs = _spans_named(logs, "vad_endpointing")
    _assert_any_span(vad_endpointing_logs, lambda log: log.get("metadata", {}).get("silence_duration", 0) > 0)
    _assert_any_span(vad_endpointing_logs, lambda log: log.get("metadata", {}).get("speech_duration", 0) > 0)
    _assert_all_spans(vad_endpointing_logs, lambda log: "duration" not in log.get("metrics", {}))
    _assert_all_spans(vad_endpointing_logs, lambda log: "event_type" not in log.get("metadata", {}))
    _assert_all_spans(vad_endpointing_logs, lambda log: "timestamp" not in log.get("metadata", {}))


def _assert_eou_spans(logs):
    eou_logs = _spans_named(logs, "eou_detection")
    if eou_logs:
        _assert_any_span(eou_logs, lambda log: log.get("input", {}).get("text"))
        _assert_any_span(eou_logs, lambda log: "is_end_of_turn" in log.get("output", {}))


def _assert_stt_spans(logs, speech_text):
    stt_logs = _spans_named(logs, "stt_processing")
    _assert_any_span(stt_logs, lambda log: log.get("metrics", {}).get("duration", 0) > 0)
    _assert_any_span(stt_logs, lambda log: log.get("output", {}).get("text") == speech_text)
    _assert_any_span(
        stt_logs, lambda log: log.get("metadata", {}).get("livekit_metrics", {}).get("audio_duration", 0) > 0
    )
    _assert_all_spans(stt_logs, lambda log: "audio_duration" not in log.get("metrics", {}))

    stt_log = stt_logs[0]
    assert "prompt_tokens" not in stt_log.get("metrics", {}), stt_log
    assert "completion_tokens" not in stt_log.get("metrics", {}), stt_log
    compact_livekit_metrics = stt_log.get("metadata", {}).get("livekit_metrics", {})
    assert "duration" not in compact_livekit_metrics, stt_log
    assert "label" not in compact_livekit_metrics, stt_log
    assert "metadata" not in compact_livekit_metrics, stt_log
    assert "request_id" not in compact_livekit_metrics, stt_log


def _assert_session_spans(logs):
    session_logs = _spans_named(logs, "livekit_agent_session")
    _assert_any_span(session_logs, lambda log: log.get("root_span_id"))
    _assert_no_metrics(session_logs, "prompt_tokens", "completion_tokens", "tokens")


def _assert_voice_turn_parenting(logs, outer_parent_name):
    outer_parent_log = _single_span(logs, outer_parent_name)
    session_log = _single_span(logs, "livekit_agent_session")
    assert session_log.get("span_parents") == [outer_parent_log.get("span_id")]
    assert "end" in session_log.get("metrics", {})

    for child_name in ("function_tool", "llm_request_run", "user_speaking", "vad_endpointing"):
        child_log = _single_span(logs, child_name)
        assert child_log.get("span_parents") == [session_log.get("span_id")], child_name
    assert _single_span(logs, "llm_request_run")["span_attributes"]["type"].value == "task"


def _assert_custom_metrics_stay_in_metadata(logs):
    custom_metric_names = {
        "audio_duration",
        "end_of_turn_delay",
        "input_audio_tokens",
        "input_cached_tokens",
        "input_text_tokens",
        "on_user_turn_completed_delay",
        "output_audio_tokens",
        "output_text_tokens",
        "tokens_per_second",
        "transcription_delay",
    }
    _assert_all_spans(logs, lambda log: not (custom_metric_names & set(log.get("metrics", {}))))


async def _run_silero_vad(vad, rtc, speech_frames):
    stream = vad.stream()

    async def consume():
        async for _ in stream:
            pass

    consume_task = asyncio.create_task(consume())
    try:
        for frame in speech_frames[:80]:
            stream.push_frame(frame)
            await asyncio.sleep(0.01)
        sample_rate = getattr(speech_frames[0], "sample_rate", 16000) if speech_frames else 16000
        num_channels = getattr(speech_frames[0], "num_channels", 1) if speech_frames else 1
        samples_per_channel = max(1, sample_rate // 100)
        for _ in range(600):
            stream.push_frame(
                rtc.AudioFrame(
                    data=b"\0\0" * samples_per_channel * num_channels,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=samples_per_channel,
                )
            )
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
    finally:
        await stream.aclose()
        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass
