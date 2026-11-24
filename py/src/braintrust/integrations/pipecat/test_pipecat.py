# pylint: disable=protected-access
"""Pipecat tracing, exercised through real pipelines and the public setup.

Every test instruments Pipecat with ``setup_pipecat`` and runs a real
``PipelineWorker`` with real Pipecat processors and
OpenAI services replayed from cassettes. Voice tests stream a speech fixture
from a websocket client through Silero VAD, exactly as a caller would. Tests
assert only on exported trace rows.
"""

import asyncio
import importlib
import inspect
import json
import logging
import math
import os
import socket
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import pytest
import pytest_asyncio
from braintrust import SpanCustomizer, logger, set_span_customizers
from braintrust.integrations.pipecat import setup_pipecat, wrap_pipeline_worker
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


if TYPE_CHECKING:
    from .tracing._eou import ControllerWait
    from .tracing.observer import BraintrustPipecatObserver, _Session


TEST_MODEL = "gpt-6-luna"
PROJECT = "test-project-pipecat-py-tracing"
SPEECH = Path(__file__).parents[2] / "fixtures/test_speech.wav"
# Recorded "Reply with exactly: Private words." generation and synthesis.
PRIVATE_WORDS_CASSETTE = "test_synthesis_without_output_transport_is_not_playback_and_respects_customizers.yaml"
INPUT_RATE = 16000
CHUNK_SECONDS = 0.02


def _scrub_recording_upload(request):
    url = urlsplit(request.uri)
    if request.method == "PUT" and ".amazonaws.com" in (url.hostname or ""):
        # Outbound session PCM varies with wall-clock latency. Playback checks
        # its placement and metadata; the upload body need not be retained.
        request.body = b""
        request.uri = urlunsplit((url.scheme, url.netloc, url.path, "", ""))
    return request


@pytest.fixture(scope="module")
def vcr_config(vcr_config):
    config = dict(vcr_config)
    scrub_response = config["before_record_response"]

    def scrub_audio_response(response):
        response = scrub_response(response)
        body = response.get("body", {}).get("string")
        if body:
            try:
                data = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                return response
            if isinstance(data, dict) and "signedUrl" in data:
                data["signedUrl"] = data["signedUrl"].split("?", 1)[0]
                response["body"]["string"] = json.dumps(data).encode()
        return response

    config.update(before_record_request=_scrub_recording_upload, before_record_response=scrub_audio_response)
    return config


@pytest.fixture
def memory_logger(monkeypatch):
    # init_test_logger replaces this global; restore it after each test.
    monkeypatch.setattr(logger, "_compute_logger_metadata", logger._compute_logger_metadata)
    init_test_logger(PROJECT)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture
def attachment_uploads():
    # Real attachment uploads replay through VCR; rows stay in the memory logger.
    logger.login(api_key=os.environ.get("BRAINTRUST_API_KEY") or "sk-bt-pipecat-vcr")


def _instrument(**options):
    """Instrument Pipecat through the public setup, with this test's options."""
    assert setup_pipecat(project_name=PROJECT, **options)
    # Setup initializes a real project logger; trace into the local one instead.
    init_test_logger(PROJECT)


def _import(path):
    # Pipecat's sentence aggregation loads NLTK data on import.
    data_dir = Path(tempfile.gettempdir()) / "braintrust-pipecat-nltk-data"
    (data_dir / "tokenizers" / "punkt_tab").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NLTK_DATA", str(data_dir))
    module_name, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def _private_words_context():
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    return LLMContext(messages=[{"role": "user", "content": "Reply with exactly: Private words."}])


def _frames():
    _import("pipecat.frames.frames.Frame")
    return importlib.import_module("pipecat.frames.frames")


def _spans_named(logs, name):
    return sorted(
        (log for log in logs if log.get("span_attributes", {}).get("name") == name),
        key=lambda log: log["metrics"]["start"],
    )


def _single_span(logs, name):
    matches = _spans_named(logs, name)
    assert len(matches) == 1, (name, matches)
    return matches[0]


def _link_targets(span, relationship):
    return [
        link["context"]["spanId"]
        for link in span["metadata"].get("links", [])
        if link["attributes"]["braintrust.relationship.type"] == relationship
    ]


def _assert_all_ended(logs):
    for row in logs:
        assert row["metrics"].get("end") is not None, row
        if row["span_attributes"].get("name") in {"llm_response", "stt", "tts"}:
            assert set(row["metrics"]) <= {"start", "end"}, row


def _make_worker(pipeline, **overrides):
    PipelineWorker = _import("pipecat.pipeline.worker.PipelineWorker")
    signature = inspect.signature(PipelineWorker)
    kwargs = {"idle_timeout_secs": None}
    for name, value in {"enable_turn_tracking": False, "enable_rtvi": False, "check_dangling_tasks": False}.items():
        if name in signature.parameters:
            kwargs[name] = value
    kwargs.update(overrides)
    return PipelineWorker(pipeline, **kwargs)


@asynccontextmanager
async def _running(worker, *, cancel=False):
    """Run a worker; on exit, end it with an ``EndFrame`` and wait for shutdown."""
    WorkerRunner = _import("pipecat.workers.runner.WorkerRunner")
    kwargs = {"handle_sigint": False}
    if "check_dangling_tasks" in inspect.signature(WorkerRunner).parameters:
        kwargs["check_dangling_tasks"] = False
    runner = WorkerRunner(**kwargs)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        yield
        if cancel:
            await worker.cancel()
        else:
            await worker.queue_frame(_frames().EndFrame(reason="pipeline complete"))
        await asyncio.wait_for(running, 20)
    finally:
        if not running.done():
            await worker.cancel()
            await asyncio.wait_for(running, 10)


def _openai_llm(max_tokens=20, model=TEST_MODEL, **settings):
    OpenAILLMService = _import("pipecat.services.openai.llm.OpenAILLMService")
    return OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model=model,
            temperature=1.0,
            extra={"reasoning_effort": "none"},
            max_completion_tokens=max_tokens,
            **settings,
        ),
    )


def _openai_tts():
    OpenAITTSService = _import("pipecat.services.openai.tts.OpenAITTSService")
    return OpenAITTSService(
        api_key=os.environ["OPENAI_API_KEY"], settings=OpenAITTSService.Settings(model="tts-1", voice="alloy")
    )


async def _run_text_pipeline(processors, steps, *, until="LLMFullResponseEndFrame"):
    """Run each step in turn, waiting for ``until`` to reach the sink.

    A step is a context to submit, or a callable run once the previous step
    completes that returns the frames to queue, as an application would.
    """
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    PipelineParams = _import("pipecat.pipeline.worker.PipelineParams")
    frames = _frames()
    worker = _make_worker(
        Pipeline(processors),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    boundary = getattr(frames, until)
    worker.set_reached_downstream_filter((boundary,))
    reached = asyncio.Queue()
    errors = []

    @worker.event_handler("on_frame_reached_downstream")
    async def on_reached(_worker, frame):
        reached.put_nowait(frame)

    @worker.event_handler("on_pipeline_error")
    async def on_error(_worker, frame):
        errors.append(frame)
        reached.put_nowait(frame)

    async with _running(worker):
        for step in steps:
            await worker.queue_frames(step() if callable(step) else [frames.LLMContextFrame(step)])
            await asyncio.wait_for(reached.get(), 20)
    return errors


async def _run_frame_sequence(processors, conversation, *, expressive=False, metrics=False):
    """Submit native input through the real runner, waiting for downstream delivery."""
    frames = _frames()
    options = {}
    if expressive or metrics:
        options["params"] = _import("pipecat.pipeline.worker.PipelineParams")(
            enable_expressive_conversation=expressive, enable_metrics=metrics, enable_usage_metrics=metrics
        )
    worker = _make_worker(_import("pipecat.pipeline.pipeline.Pipeline")(processors), **options)
    worker.set_reached_downstream_filter((frames.Frame,))
    reached, started = asyncio.Queue(), asyncio.Event()
    retained = []

    @worker.event_handler("on_pipeline_started")
    async def on_started(_worker, _frame):
        retained.append(worker._braintrust_pipecat_observer._session)
        started.set()

    @worker.event_handler("on_frame_reached_downstream")
    async def on_reached(_worker, frame):
        reached.put_nowait(frame)

    @worker.event_handler("on_pipeline_error")
    async def on_error(_worker, frame):
        reached.put_nowait(frame)

    async def send(frame, *, until=None, wait=True):
        await worker.queue_frame(frame)
        if not wait:
            return None
        while True:
            received = await asyncio.wait_for(reached.get(), 20)
            assert not isinstance(received, frames.ErrorFrame), received.error
            matches = received is frame if until is None else isinstance(received, until)
            if matches:
                return received

    async with _running(worker):
        await asyncio.wait_for(started.wait(), 10)
        await conversation(send)
    observer = worker._braintrust_pipecat_observer
    _assert_cascade_released(observer, retained[0])


def _assert_cascade_released(observer, session):
    assert observer._session is None
    assert session.user._turn is session.user._speech is None
    assert not session.user._recognitions and not session.user._transcribed and not session.user._recognized_until
    assert session.agent._turn is session.agent._response is session.agent._playback is None
    assert not session.agent._synthesis and not session.agent._tools and not session.agent._requesters
    assert not session.agent._finished_tools
    assert session.linker.messages is session.linker.reply_to is None
    assert not session.linker._raw and not session.linker._submitted and not session.linker._transcripts
    assert not session.linker._provenance and not session.linker._tool_results
    assert not observer._response_frames and not observer._aggregation_parts and not observer._handled_metrics
    assert (
        not observer._response_waiters and not observer._response_executions and not observer._native_turn_processors
    )


@dataclass
class _VoiceSession:
    """A websocket caller talking to STT -> LLM -> TTS -> playback."""

    system: str
    max_tokens: int = 20
    speaking_frames: bool = True
    inference_gate: bool = False
    smart_turn: bool = False
    delay_observer: bool = False
    hold_inflight_audio: bool = False
    cancel: bool = False
    observer: "BraintrustPipecatObserver | None" = field(init=False, default=None)
    tracker: "_Session | None" = field(init=False, default=None)
    controllers: "tuple[ControllerWait, ...]" = field(init=False, default=())
    assistant_turns: list = field(default_factory=list)
    played: bytearray = field(default_factory=bytearray)

    async def run(self, *, utterances=1, barge_in=False, services=None, shutdown_during_playback=False):
        frames = _frames()
        Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
        PipelineParams = _import("pipecat.pipeline.worker.PipelineParams")
        LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
        universal = "pipecat.processors.aggregators.llm_response_universal"
        LLMContextAggregatorPair = _import(f"{universal}.LLMContextAggregatorPair")
        LLMUserAggregatorParams = _import(f"{universal}.LLMUserAggregatorParams")
        UserTurnStrategies = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")
        SpeechTimeout = _import("pipecat.turns.user_stop.SpeechTimeoutUserTurnStopStrategy")
        VADStart = _import("pipecat.turns.user_start.VADUserTurnStartStrategy")
        SileroVADAnalyzer = _import("pipecat.audio.vad.silero.SileroVADAnalyzer")
        VADParams = _import("pipecat.audio.vad.vad_analyzer.VADParams")
        OpenAISTTService = _import("pipecat.services.openai.stt.OpenAISTTService")
        llm_factory, tts_factory = _openai_llm, _openai_tts
        if services is not None:
            OpenAISTTService, llm_factory, tts_factory = services
        WebsocketServerTransport = _import("pipecat.transports.websocket.server.WebsocketServerTransport")
        WebsocketServerParams = _import("pipecat.transports.websocket.server.WebsocketServerParams")
        ProtobufFrameSerializer = _import("pipecat.serializers.protobuf.ProtobufFrameSerializer")
        connect = _import("websockets.asyncio.client.connect")

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        serializer = ProtobufFrameSerializer()
        transport = WebsocketServerTransport(
            host="127.0.0.1",
            port=port,
            params=WebsocketServerParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=INPUT_RATE,
                audio_out_sample_rate=24000,
                serializer=serializer,
            ),
        )
        stt = OpenAISTTService(
            api_key=os.environ["OPENAI_API_KEY"], settings=OpenAISTTService.Settings(model="whisper-1")
        )
        stop = [SpeechTimeout(user_speech_timeout=0.5, enable_user_speaking_frames=self.speaking_frames)]
        if self.smart_turn:
            Analyzer = _import("pipecat.audio.turn.smart_turn.local_smart_turn_v3.LocalSmartTurnAnalyzerV3")
            AnalyzerStrategy = _import("pipecat.turns.user_stop.TurnAnalyzerUserTurnStopStrategy")
            stop = [AnalyzerStrategy(turn_analyzer=Analyzer())]
        if self.inference_gate:
            Gate = _import("pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy")
            stop.append(Gate())
        aggregators = LLMContextAggregatorPair(
            LLMContext(messages=[{"role": "system", "content": self.system}]),
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
                user_turn_strategies=UserTurnStrategies(
                    start=None if self.speaking_frames else [VADStart(enable_user_speaking_frames=False)],
                    stop=stop,
                ),
            ),
        )
        worker = _make_worker(
            Pipeline(
                [
                    transport.input(),
                    stt,
                    aggregators.user(),
                    llm_factory(self.max_tokens),
                    tts_factory(),
                    transport.output(),
                    aggregators.assistant(),
                ]
            ),
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )
        ready, turn_stopped = asyncio.Event(), asyncio.Event()
        self.observer = getattr(worker, "_braintrust_pipecat_observer")
        release_observer = asyncio.Event()
        if self.delay_observer:
            on_push = self.observer.on_push_frame
            started_turns = 0

            @aggregators.user().event_handler("on_user_turn_started")
            async def on_user_started(*_args):
                nonlocal started_turns
                started_turns += 1
                # Hold first-turn observations through commitment and the next
                # native turn's start, reproducing the ownership race through
                # the real audio pipeline without changing turn-taking.
                if started_turns == 2:
                    release_observer.set()

            async def delayed_push(data):
                if isinstance(data.frame, frames.VADUserStartedSpeakingFrame):
                    await release_observer.wait()
                await on_push(data)

            self.observer.on_push_frame = delayed_push
        retained = []

        @worker.event_handler("on_pipeline_started")
        async def on_started(_worker, _frame):
            retained.append(worker._braintrust_pipecat_observer._session)

        @transport.event_handler("on_websocket_ready")
        async def on_ready(_transport):
            ready.set()

        @aggregators.assistant().event_handler("on_assistant_turn_stopped")
        async def on_stopped(_aggregator, message, *_args):
            self.assistant_turns.append(message)
            turn_stopped.set()

        with wave.open(str(SPEECH)) as audio:
            assert audio.getframerate() == INPUT_RATE
            pcm = audio.readframes(audio.getnframes())

        async with _running(worker, cancel=self.cancel):
            await asyncio.wait_for(ready.wait(), 10)
            self.tracker = self.observer._session
            hold_audio = self._hold_inflight_audio(transport.output()) if self.hold_inflight_audio else None
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                agent_audio = asyncio.Event()

                async def receive():
                    async for payload in websocket:
                        frame = await serializer.deserialize(payload)
                        if isinstance(frame, frames.InputAudioRawFrame):
                            self.played.extend(frame.audio)
                            agent_audio.set()

                reading = asyncio.create_task(receive())
                for index in range(utterances):
                    if index and barge_in:
                        # Speak over the agent once its reply is audible.
                        await asyncio.wait_for(agent_audio.wait(), 30)
                        if hold_audio is not None:
                            hold_audio.set()
                    elif index:
                        await self._wait_for_turns(turn_stopped, index)
                    if shutdown_during_playback:
                        speaking = asyncio.create_task(self._speak(websocket, serializer, pcm))
                        try:
                            await asyncio.wait_for(agent_audio.wait(), 30)
                            await worker.cancel()
                        finally:
                            speaking.cancel()
                            await asyncio.gather(speaking, return_exceptions=True)
                        break
                    await self._speak(websocket, serializer, pcm)
                if not shutdown_during_playback:
                    await self._wait_for_turns(turn_stopped, utterances)
                release_observer.set()
                if self.observer._controller_queue is not None:
                    await asyncio.wait_for(self.observer._controller_queue.join(), 5)
                self.controllers = tuple(self.observer._controllers.values())
                reading.cancel()
                await asyncio.gather(reading, return_exceptions=True)
        _assert_cascade_released(worker._braintrust_pipecat_observer, retained[0])

    @staticmethod
    def _hold_inflight_audio(output):
        """Once armed, hold the next chunk the transport dequeues until it handles the interruption.

        Current Pipecat keeps the output audio task running through an
        interruption, so a chunk it already dequeued is still written and
        restarts bot speech. A loaded runner hits that window by chance; this
        reaches it on every run without replacing any Pipecat object.
        """
        OutputAudioRawFrame = _frames().OutputAudioRawFrame
        sender = output._media_senders[None]
        handle_frame, handle_interruptions = sender._handle_frame, sender.handle_interruptions
        armed, interrupted = asyncio.Event(), asyncio.Event()

        async def held_handle_frame(frame):
            if armed.is_set() and not interrupted.is_set() and isinstance(frame, OutputAudioRawFrame):
                await interrupted.wait()
            await handle_frame(frame)

        async def tracked_handle_interruptions(frame):
            await handle_interruptions(frame)
            if armed.is_set():
                interrupted.set()

        sender._handle_frame = held_handle_frame
        sender.handle_interruptions = tracked_handle_interruptions
        return armed

    async def _wait_for_turns(self, turn_stopped, count):
        while len(self.assistant_turns) < count:
            turn_stopped.clear()
            await asyncio.wait_for(turn_stopped.wait(), 30)

    @staticmethod
    async def _speak(websocket, serializer, pcm):
        # Real-time pacing lets VAD and turn-stop timers behave as in a call.
        # The wire has one audio message: a client's outgoing audio is the
        # server's input, serialized like any outgoing audio frame.
        chunk = int(INPUT_RATE * CHUNK_SECONDS) * 2
        audio = [pcm[offset : offset + chunk] for offset in range(0, len(pcm), chunk)]
        audio += [bytes(chunk)] * int(1.5 / CHUNK_SECONDS)
        OutputAudioRawFrame = _frames().OutputAudioRawFrame
        for samples in audio:
            await websocket.send(await serializer.serialize(OutputAudioRawFrame(samples, INPUT_RATE, 1)))
            await asyncio.sleep(CHUNK_SECONDS)


# Setup


@pytest.mark.parametrize("hook", range(4))
@pytest.mark.parametrize("change", ["missing", "signature", "async"])
def test_eou_optional_hooks_are_gated_together(monkeypatch, hook, change):
    """Pure patch routing, independent of provider response payloads."""
    from .patchers import EOUWaitPatcher

    patcher = EOUWaitPatcher.sub_patchers[hook]
    parent, name = patcher.target_path.rsplit(".", 1)
    target = _import(f"{patcher.target_module}.{parent}")
    method = getattr(target, name)
    assert EOUWaitPatcher.applies(None, None)
    replacement = None if change == "missing" else dict.get
    if change == "async" and not inspect.iscoroutinefunction(method):
        replacement = _import("pipecat.turns.user_turn_controller.UserTurnController").process_frame
    monkeypatch.setattr(target, name, replacement)
    assert not EOUWaitPatcher.applies(None, None)


def test_setup_and_wrap_pipeline_worker_are_idempotent(memory_logger):
    PipelineWorker = _import("pipecat.pipeline.worker.PipelineWorker")
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    IdentityFilter = _import("pipecat.processors.filters.identity_filter.IdentityFilter")
    _instrument()
    _instrument()
    assert wrap_pipeline_worker(PipelineWorker) is PipelineWorker

    async def run():
        async with _running(_make_worker(Pipeline([IdentityFilter()]))):
            pass

    # Repeated setup and wrapping still trace each worker exactly once.
    asyncio.run(run())
    asyncio.run(run())
    sessions = _spans_named(memory_logger.pop(), "session")
    assert len(sessions) == 2
    assert all(session["metadata"]["contrib.pipecat.terminal_frame"] == "EndFrame" for session in sessions)


@pytest.mark.vcr
@pytest.mark.skipif(__import__("sys").version_info < (3, 11), reason="Pipecat AI 1.x requires Python 3.11+")
@pytest.mark.parametrize("entrypoint", ["auto", "wrap"])
def test_auto_instrument_pipecat_subprocess(monkeypatch, entrypoint):
    pytest.importorskip("pipecat")
    monkeypatch.setenv("BRAINTRUST_TEST_PIPECAT_ENTRYPOINT", entrypoint)
    verify_autoinstrument_script("test_auto_pipecat.py")


# Voice sessions


@pytest.mark.vcr(cassette_name="test_voice_session_traces_turns_speech_recognition_generation_and_playback")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "speaking_frames,inference_gate,smart_turn,cancel",
    [
        (True, False, False, False),
        (False, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
        (True, False, False, True),
    ],
)
async def test_voice_session_traces_turns_speech_recognition_generation_and_playback(
    memory_logger, attachment_uploads, speaking_frames, inference_gate, smart_turn, cancel
):
    from braintrust._audio.config import AudioEncoding, AudioRecordingOptions
    from braintrust.integrations.versioning import detect_module_version, version_satisfies

    pipecat = pytest.importorskip("pipecat")
    if inference_gate and not version_satisfies(detect_module_version(pipecat, ("pipecat",)), ">=1.16.0"):
        pytest.skip("ALL stop-strategy composition is qualified on Pipecat 1.16.0")
    _instrument(
        capture_audio_attachments=True,
        audio_recording_options=AudioRecordingOptions(encoding=AudioEncoding(codec="pcm", container="wav")),
    )
    session = _VoiceSession(
        system="Reply with exactly: Hi.",
        speaking_frames=speaking_frames,
        inference_gate=inference_gate,
        smart_turn=smart_turn,
        cancel=cancel,
    )
    await session.run()

    logs = memory_logger.pop()
    root = _single_span(logs, "session")
    assert root["metadata"]["audio.schema.version"] == "3.1.0"
    assert root["metadata"]["contrib.pipecat.terminal_frame"] == ("CancelFrame" if cancel else "EndFrame")
    if not cancel:
        assert root["metadata"]["contrib.pipecat.reason"] == "pipeline complete"
    assert session.observer._controllers == {}
    assert session.observer._controller_queue is None
    for controller in session.controllers:
        assert controller.span is None and controller.turn is None
        assert controller.session is None and controller.controller is None
    user_turn = _single_span(logs, "user_turn")
    agent_turn = _single_span(logs, "agent_turn")
    speech = _single_span(logs, "user_speech")
    wait = _single_span(logs, "eou_wait")
    assert wait["metadata"]["eou.outcome"] == "committed"
    assert wait["metadata"]["eou.speech_end_source"] == "vad"
    assert wait["span_parents"] == [user_turn["span_id"]]
    assert wait["metadata"]["turn.id"] == user_turn["span_id"]
    assert wait["metrics"]["start"] == pytest.approx(speech["metrics"]["end"], abs=0.03)
    assert "eou.detection_methods" not in wait["metadata"]
    recognition = _single_span(logs, "stt")
    model = _single_span(logs, "llm_response")
    if inference_gate:
        assert wait["metrics"]["start"] < model["metrics"]["start"] < wait["metrics"]["end"]
        assert "eou.end_reason" not in wait["metadata"]
    synthesis = _single_span(logs, "tts")
    playback = _single_span(logs, "agent_speech")

    # User side: VAD speech within the native turn, recognized from its start.
    assert speech["span_parents"] == [user_turn["span_id"]]
    assert user_turn["metadata"]["turn.incomplete"] is False
    assert (
        user_turn["metadata"]["audio.selections"]["selection-0"]
        == speech["metadata"]["audio.selections"]["selection-0"]
    )
    assert recognition["span_parents"] == [user_turn["span_id"]]
    assert recognition["metrics"]["start"] == pytest.approx(speech["metrics"]["start"], abs=1e-6)
    assert recognition["output"][0]["role"] == "user" and recognition["output"][0]["content"]
    assert recognition["metadata"]["provider"] == "openai"
    assert _link_targets(recognition, "transcribes") == [speech["span_id"]]
    assert user_turn["input"] == recognition["output"]

    # Agent side: the reply turn owns generation, synthesis and playback.
    assert agent_turn["metadata"]["turn.reply_to"] == user_turn["span_id"]
    assert agent_turn["metadata"]["turn.incomplete"] is False
    for operation in (model, synthesis, playback):
        assert operation["span_parents"] == [agent_turn["span_id"]]
        assert operation["metadata"]["turn.id"] == agent_turn["span_id"]
        assert operation["metrics"]["end"] <= agent_turn["metrics"]["end"]
    assert model["span_attributes"]["type"] == "task"
    assert model["metadata"]["model"] == TEST_MODEL
    assert model["metadata"]["provider"] == "openai"
    assert model["input"][-1] == recognition["output"][0]
    if inference_gate:
        assert any(message.get("content") == session.system for message in model["input"])
    else:
        assert model["input"][0] == {"role": "system", "content": session.system}
    assert model["output"][0]["role"] == "assistant"
    assert model["output"][0]["content"].strip() == "Hi."
    assert set(model["output"][0]) == {"role", "content"}
    # Token usage belongs to provider spans; the request settings are here.
    assert not {"prompt_tokens", "completion_tokens", "tokens"} & set(model["metrics"])
    assert "time_to_first_token" not in model["metrics"]
    assert {key: model["metadata"][key] for key in ("temperature", "max_completion_tokens", "reasoning_effort")} == {
        "temperature": 1.0,
        "max_completion_tokens": 20,
        "reasoning_effort": "none",
    }
    assert _link_targets(model, "uses_context") == [recognition["span_id"]]
    assert synthesis["metadata"]["model"] == "tts-1"
    assert synthesis["input"]["text"].strip() == "Hi."
    assert "characters" not in synthesis["metrics"]
    assert _link_targets(synthesis, "generated_by") == [model["span_id"]]
    assert _link_targets(playback, "generated_by") == [synthesis["span_id"]]
    assert playback["metadata"]["audio.speaker_ids"] == synthesis["metadata"]["audio.speaker_ids"]
    assert playback["metadata"]["contrib.pipecat.interrupted"] is False
    assert playback["output"] == [{"role": "assistant", "content": synthesis["input"]["text"]}]
    assert "audio.participant_ids" in playback["metadata"]
    # Covered clients without their provider integration emit only Pipecat wrappers.
    assert not [row for row in logs if row["span_attributes"].get("type") == "llm"]

    # Both directions land in shared recordings; playback is what was sent.
    assert session.played
    segments = root["metadata"]["audio.segments"]
    assert segments and all(segment["state"] == "ready" for segment in segments.values()), segments
    for segment in segments.values():
        assert segment["channels"] == {"0": {"sources": ["session-input"]}, "1": {"sources": ["agent-output"]}}
        assert segment["attachment"]["ref"].startswith("/input/audio/")
    _assert_all_ended(logs)


@pytest.mark.vcr(cassette_name="test_consecutive_voice_turns_reply_to_and_link_only_their_own_speech")
@pytest.mark.asyncio
@pytest.mark.parametrize("delayed", [False, True])
async def test_consecutive_voice_turns_reply_to_and_link_only_their_own_speech(
    memory_logger, attachment_uploads, delayed
):
    from braintrust._audio.config import AudioEncoding, AudioRecordingOptions

    # User audio only, configured when wrapping: agent playback is not recorded.
    _instrument()
    assert wrap_pipeline_worker(
        _import("pipecat.pipeline.worker.PipelineWorker"),
        capture_user_audio_attachments=True,
        capture_agent_audio_attachments=False,
        audio_recording_options=AudioRecordingOptions(encoding=AudioEncoding(codec="pcm", container="wav")),
    ) is _import("pipecat.pipeline.worker.PipelineWorker")
    await _VoiceSession(system="Reply with exactly one word: Okay.", delay_observer=delayed).run(utterances=2)

    logs = memory_logger.pop()
    user_turns = _spans_named(logs, "user_turn")
    agent_turns = _spans_named(logs, "agent_turn")
    speeches = _spans_named(logs, "user_speech")
    recognitions = _spans_named(logs, "stt")
    models = _spans_named(logs, "llm_response")
    waits = _spans_named(logs, "eou_wait")
    assert len(waits) == 2
    for wait, turn in zip(waits, user_turns):
        assert wait["span_parents"] == [turn["span_id"]]
        assert wait["metadata"]["turn.id"] == turn["span_id"]
    for synthesis in _spans_named(logs, "tts"):
        assert "characters" not in synthesis["metrics"]
    assert len(user_turns) == len(agent_turns) == len(speeches) == len(recognitions) == len(models) == 2
    for index, (user_turn, agent_turn, speech, recognition, model) in enumerate(
        zip(user_turns, agent_turns, speeches, recognitions, models)
    ):
        assert agent_turn["metadata"]["turn.incomplete"] is False
        if delayed and index == 0:
            # Provider execution preceded the queued speech/turn observations.
            # Keep the native context-item link below without guessing which
            # speech interval or turn that execution belonged to.
            assert "turn.reply_to" not in agent_turn["metadata"]
            assert recognition["span_parents"] == [_single_span(logs, "session")["span_id"]]
            assert "turn.id" not in recognition["metadata"]
            assert not _link_targets(recognition, "transcribes")
        else:
            assert agent_turn["metadata"]["turn.reply_to"] == user_turn["span_id"]
            assert recognition["span_parents"] == [user_turn["span_id"]]
            assert recognition["metrics"]["start"] == pytest.approx(speech["metrics"]["start"], abs=1e-6)
            assert _link_targets(recognition, "transcribes") == [speech["span_id"]]
        # The second request repeats the first utterance in its context, but
        # links only the transcript it newly consumes.
        assert _link_targets(model, "uses_context") == [recognition["span_id"]]
        assert model["metadata"]["turn.id"] == agent_turn["span_id"]
    assert user_turns[0]["metrics"]["end"] <= user_turns[1]["metrics"]["start"]
    assert agent_turns[0]["metrics"]["end"] <= agent_turns[1]["metrics"]["start"]
    segments = _single_span(logs, "session")["metadata"]["audio.segments"]
    assert segments
    for segment in segments.values():
        assert segment["channels"] == {"0": {"sources": ["session-input"]}}
    _assert_all_ended(logs)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_user_barge_in_interrupts_agent_playback(memory_logger):
    _instrument(capture_audio_attachments=False)
    session = _VoiceSession(
        system="Count from one to ten in words, then stop. If interrupted, reply exactly: Okay.",
        max_tokens=30,
        # The transport still writes the chunk it dequeued before the
        # interruption. That audio must not open speech without a turn.
        hold_inflight_audio=True,
    )
    await session.run(utterances=2, barge_in=True)

    logs = memory_logger.pop()
    user_turns = _spans_named(logs, "user_turn")
    interrupted, reply = _spans_named(logs, "agent_turn")
    assert len(user_turns) == 2
    # The interrupted reply ends incomplete, with its playback cut short.
    assert interrupted["metadata"]["turn.incomplete"] is True
    assert interrupted["metadata"]["turn.reply_to"] == user_turns[0]["span_id"]
    speech = _spans_named(logs, "agent_speech")
    assert [row["metadata"].get("turn.id") for row in speech] == [interrupted["span_id"], reply["span_id"]]
    cut = speech[0]
    assert cut["metadata"]["contrib.pipecat.interrupted"] is True
    assert interrupted["metrics"]["end"] <= user_turns[1]["metrics"]["end"]
    # The answer to the interruption is a separate, complete turn.
    assert reply["metadata"]["turn.incomplete"] is False
    assert reply["metrics"]["start"] >= interrupted["metrics"]["end"]
    _assert_all_ended(logs)


# Text pipelines


@pytest.mark.vcr(cassette_name="test_responses_without_playback_or_assistant_aggregator_are_separate_turns")
@pytest.mark.asyncio
async def test_responses_without_assistant_aggregator_remain_under_session(memory_logger):
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    first = LLMContext(messages=[{"role": "user", "content": "Reply with exactly: One."}])
    second = LLMContext(messages=[{"role": "user", "content": "Reply with exactly: Two."}])
    _instrument()
    await _run_text_pipeline([_openai_llm()], [first, second])

    logs = memory_logger.pop()
    assert not _spans_named(logs, "agent_turn")
    responses = _spans_named(logs, "llm_response")
    assert len(responses) == 2
    for response in responses:
        assert response["span_parents"] == [_single_span(logs, "session")["span_id"]]
        assert "turn.id" not in response["metadata"]
        assert "turn.reply_to" not in response["metadata"]
    assert [row["output"][0]["content"].strip() for row in responses] == ["One.", "Two."]
    _assert_all_ended(logs)


@pytest.mark.vcr(cassette_name="test_context_edited_in_place_links_the_transcripts_it_now_includes")
@pytest.mark.asyncio
async def test_unrelated_transcript_text_does_not_establish_context_provenance(memory_logger):
    frames = _frames()
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])

    def first():
        # A text application feeds transcripts natively, then requests a reply.
        return [frames.TranscriptionFrame("Hello", "caller", "0"), frames.LLMContextFrame(context)]

    def edited():
        # The application extends the existing user message in place.
        # Its text matches the two transcripts joined, which is not provenance.
        context.get_messages()[0]["content"] = "Hello again"
        return [frames.TranscriptionFrame("again", "caller", "1"), frames.LLMContextFrame(context)]

    _instrument()
    await _run_text_pipeline([_openai_llm()], [first, edited])

    logs = memory_logger.pop()
    transcripts = {row["output"][0]["content"]: row["span_id"] for row in _spans_named(logs, "transcript_received")}
    first_response, second_response = _spans_named(logs, "llm_response")
    assert set(transcripts) == {"Hello", "again"}
    assert _link_targets(first_response, "uses_context") == []
    assert _link_targets(second_response, "uses_context") == []
    _assert_all_ended(logs)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_synthesis_without_output_transport_is_not_playback_and_respects_customizers(memory_logger):
    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            if "input" in data:
                data["input"] = "[redacted]"
            return data

    context = _private_words_context()
    # Synthesized buffers never reached a transport, so nobody heard them.
    _instrument(capture_agent_audio_attachments=True)
    set_span_customizers([Redact()])
    try:
        await _run_text_pipeline([_openai_llm(), _openai_tts()], [context], until="TTSStoppedFrame")
    finally:
        set_span_customizers(None)

    logs = memory_logger.pop()
    synthesis = [row for row in logs if row.get("span_attributes", {}).get("name") == "tts"]
    assert synthesis and all(row["input"] == "[redacted]" for row in synthesis if "input" in row)
    assert "characters" not in _single_span(logs, "tts")["metrics"]
    assert not _spans_named(logs, "agent_speech")
    assert "audio.segments" not in _single_span(logs, "session")["metadata"]
    _assert_all_ended(logs)


@pytest.mark.asyncio
async def test_failed_transport_writes_do_not_emit_agent_speech(memory_logger, vcr):
    WebsocketServerTransport = _import("pipecat.transports.websocket.server.WebsocketServerTransport")
    WebsocketServerParams = _import("pipecat.transports.websocket.server.WebsocketServerParams")
    ProtobufFrameSerializer = _import("pipecat.serializers.protobuf.ProtobufFrameSerializer")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    transport = WebsocketServerTransport(
        host="127.0.0.1",
        port=port,
        params=WebsocketServerParams(
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            serializer=ProtobufFrameSerializer(),
        ),
    )
    context = _private_words_context()
    # No client connects: the real transport rejects every write after Pipecat
    # has emitted BotStartedSpeakingFrame. Reuse the same provider calls as the
    # synthesis-only test; no replacement transport or write method is needed.
    _instrument(capture_agent_audio_attachments=True)
    with vcr.use_cassette(PRIVATE_WORDS_CASSETTE):
        await _run_text_pipeline(
            [transport.input(), _openai_llm(), _openai_tts(), transport.output()],
            [context],
            until="TTSStoppedFrame",
        )
    logs = memory_logger.pop()
    assert _spans_named(logs, "llm_response")
    assert _spans_named(logs, "tts")
    assert not _spans_named(logs, "agent_speech")
    assert "audio.segments" not in _single_span(logs, "session")["metadata"]
    assert not _spans_named(logs, "agent_turn")
    _assert_all_ended(logs)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_service_errors_are_attributed_to_the_failing_service(memory_logger):
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    context = LLMContext(messages=[{"role": "user", "content": "Hello."}])
    _instrument()
    errors = await _run_text_pipeline([_openai_llm(model="braintrust-nonexistent-model")], [context])
    assert errors

    logs = memory_logger.pop()
    response = _single_span(logs, "llm_response")
    assert "braintrust-nonexistent-model" in response["error"]
    assert "braintrust-nonexistent-model" in _single_span(logs, "session")["error"]
    _assert_all_ended(logs)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_tracing_failures_never_break_the_pipeline_or_session(memory_logger, monkeypatch, caplog):
    from braintrust.logger import SpanImpl

    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    end = SpanImpl.end

    # Narrow injected failure: exporting a model span fails during the run and
    # again at shutdown. Recordings cannot provoke this.
    def failing_end(span, *args, **kwargs):
        if span._name == "llm_response":
            raise RuntimeError("span export failed")
        return end(span, *args, **kwargs)

    monkeypatch.setattr(SpanImpl, "end", failing_end)
    context = LLMContext(messages=[{"role": "user", "content": "Reply with exactly: Fine."}])
    _instrument()
    with caplog.at_level(logging.WARNING, logger="braintrust"):
        await _run_text_pipeline([_openai_llm()], [context])

    assert "Unable to trace Pipecat" in caplog.text
    assert "Unable to close Pipecat trace" in caplog.text
    logs = memory_logger.pop()
    root = _single_span(logs, "session")
    assert root["metrics"].get("end") is not None
    assert root["metadata"]["contrib.pipecat.terminal_frame"] == "EndFrame"


LIVE_MODEL = "gpt-live-1"
LIVE_BACKEND_MODEL = "gpt-5.4-mini"
LIVE_INSTRUCTIONS = (
    "Speak English. When the caller asks about an order, use lookup_order. "
    "If it is unavailable, briefly say you cannot look it up right now."
)
LIVE_CALLER_PROMPT = "Hi, can you check the status of order demo 1042 for me?"
LIVE_TOOL = {
    "name": "lookup_order",
    "description": "Look up a demo order.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
}


async def _record_live_capture(path):
    """Capture a real GPT-Live session for ``test_live_observations``.

    A local proxy forwards Pipecat's real ``OpenAILiveLLMService`` to the Live
    API and logs every JSON event in order, dropping audio payloads. The caller
    speaks a TTS-synthesized order question; Pipecat's real missing-tool handler
    answers ``lookup_order``. Like VCR's ``once`` mode, this runs only when the
    capture is missing, and never in CI.
    """
    from openai import AsyncOpenAI
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed

    live = _import("pipecat.services.openai.live.llm")
    FunctionSchema = _import("pipecat.adapters.llm.schemas.function_schema.FunctionSchema")
    ToolsSchema = _import("pipecat.adapters.llm.schemas.tools_schema.ToolsSchema")
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    LLMContextAggregatorPair = _import(
        "pipecat.processors.aggregators.llm_response_universal.LLMContextAggregatorPair"
    )
    OpenAIResponsesLLMService = _import("pipecat.services.openai.responses.llm.OpenAIResponsesLLMService")
    WorkerRunner = _import("pipecat.workers.runner.WorkerRunner")
    frames = _frames()

    api_key = os.environ["OPENAI_API_KEY"]
    sample_rate = 24000
    speech = await AsyncOpenAI(api_key=api_key).audio.speech.create(
        model="tts-1", voice="alloy", input=LIVE_CALLER_PROMPT, response_format="pcm"
    )
    pcm = speech.content + b"\x00" * (sample_rate * 2 * 2)  # 2s of trailing silence ends the turn.

    records = []
    started, replied = asyncio.Event(), asyncio.Event()
    last_output = [0.0]

    def log(direction, raw):
        event = json.loads(raw)
        kind = event.get("type", "")
        if kind == "session.started":
            started.set()
        if kind == "session.output_transcript.delta":
            last_output[0] = time.monotonic()
        if (
            kind == "response.event"
            and event["event"]["type"] == "response.completed"
            and any(r["event"]["type"] == "response.create" for r in records)
        ):
            replied.set()
        if "audio" not in kind:
            records.append({"direction": direction, "event": event})

    async def proxy(client):
        upstream_url = "wss://api.openai.com/v1/live/sessions"
        async with connect(upstream_url, additional_headers={"Authorization": f"Bearer {api_key}"}) as upstream:

            async def pump(source, sink, direction):
                try:
                    async for raw in source:
                        log(direction, raw)
                        await sink.send(raw)
                except ConnectionClosed:
                    pass
                finally:
                    await sink.close()

            await asyncio.gather(pump(client, upstream, "client"), pump(upstream, client, "server"))

    async with serve(proxy, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = live.OpenAILiveLLMService(
            api_key="sent-by-the-proxy",
            base_url=f"ws://127.0.0.1:{port}",
            settings=live.OpenAILiveLLMService.Settings(model=LIVE_MODEL, system_instruction=LIVE_INSTRUCTIONS),
            delegation=live.ResponsesDelegation(
                settings=OpenAIResponsesLLMService.Settings(
                    model=LIVE_BACKEND_MODEL, system_instruction=LIVE_INSTRUCTIONS
                )
            ),
        )
        context = LLMContext(
            messages=[],
            tools=ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name=LIVE_TOOL["name"],
                        description=LIVE_TOOL["description"],
                        properties=LIVE_TOOL["parameters"]["properties"],
                        required=LIVE_TOOL["parameters"]["required"],
                    )
                ]
            ),
        )
        aggregators = LLMContextAggregatorPair(context)
        worker = _make_worker(Pipeline([aggregators.user(), service, aggregators.assistant()]))
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            await worker.queue_frame(frames.LLMContextFrame(context))
            await asyncio.wait_for(started.wait(), 30)
            chunk = int(sample_rate * CHUNK_SECONDS) * 2
            for offset in range(0, len(pcm), chunk):
                await worker.queue_frame(frames.InputAudioRawFrame(pcm[offset : offset + chunk], sample_rate, 1))
                await asyncio.sleep(CHUNK_SECONDS)
            await asyncio.wait_for(replied.wait(), 60)
            # Let the spoken reply's transcript finish before closing.
            while time.monotonic() - last_output[0] < 3:
                await asyncio.sleep(0.25)
            await worker.queue_frame(frames.EndFrame())
            await asyncio.wait_for(running, 30)
        finally:
            if not running.done():
                await worker.cancel()
                await asyncio.wait_for(running, 10)

    kinds = [r["event"]["type"] for r in records]
    for required in ("session.start", "response.item.create", "response.create", "session.close", "session.closed"):
        assert required in kinds, (required, kinds)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["late_transcript", "cancel", "cancel_callback"])
async def test_live_observations(memory_logger, vcr_cassette_dir, boundary):
    live = pytest.importorskip("pipecat.services.openai.live.llm")
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed

    FunctionSchema = _import("pipecat.adapters.llm.schemas.function_schema.FunctionSchema")
    ToolsSchema = _import("pipecat.adapters.llm.schemas.tools_schema.ToolsSchema")
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    LLMContextAggregatorPair = _import(
        "pipecat.processors.aggregators.llm_response_universal.LLMContextAggregatorPair"
    )
    OpenAIResponsesLLMService = _import("pipecat.services.openai.responses.llm.OpenAIResponsesLLMService")
    WorkerRunner = _import("pipecat.workers.runner.WorkerRunner")

    # Native WebSocket capture, including the real built-in missing-tool result.
    # Audio payloads are omitted. A missing capture is recorded live, like VCR's
    # default "once" mode; CI only replays.
    capture = Path(vcr_cassette_dir) / "test_live_observations.jsonl"
    if not capture.exists() and not os.environ.get("CI") and not os.environ.get("GITHUB_ACTIONS"):
        capture.parent.mkdir(parents=True, exist_ok=True)
        await _record_live_capture(capture)
    recording = [json.loads(line) for line in capture.read_text().splitlines()]
    config = recording[0]["event"]["session"]
    backend = config["delegation"]["responses"]
    server_events = [r["event"] for r in recording if r["direction"] == "server"]
    transcripts = [
        e for e in server_events if e["type"] in ("session.input_transcript.delta", "session.output_transcript.delta")
    ]
    late = transcripts[-1]
    sent, outgoing = [], []
    ready, callback_cancelled = asyncio.Event(), asyncio.Event()
    replay_errors = []

    async def replay(websocket):
        delayed_transcripts = []
        tool_arguments_started = False
        try:
            for record in recording:
                event = record["event"]
                if record["direction"] == "client":
                    actual = json.loads(await websocket.recv())
                    outgoing.append(actual)
                    assert actual["type"] == event["type"]
                    if actual["type"] == "session.start":
                        assert actual["session"]["instructions"] == event["session"]["instructions"]
                    if actual["type"] == "response.item.create":
                        assert actual["item"] == event["item"]
                    if actual["type"] == "session.close":
                        # Deliver a captured fragment late, while the real
                        # service awaits the server's close acknowledgment.
                        await websocket.send(json.dumps(late))
                        sent.append(late)
                else:
                    if event == late:
                        ready.set()
                        continue
                    if (
                        boundary == "cancel"
                        and event["type"] == "session.output_transcript.delta"
                        and not tool_arguments_started
                    ):
                        # Live speech is independent of the backend; captures
                        # can interleave it before the tool call starts. Hold it
                        # so cancellation always lands mid-function-call.
                        delayed_transcripts.append(event)
                        continue
                    if boundary == "cancel_callback" and event["type"] == "session.output_transcript.delta":
                        # Delay the independent speech stream until backend
                        # text has arrived, then cancel inside its public callback.
                        delayed_transcripts.append(event)
                        continue
                    if (
                        boundary == "cancel_callback"
                        and event["type"] == "response.event"
                        and event["event"]["type"] == "response.output_text.done"
                    ):
                        await websocket.send(json.dumps(delayed_transcripts[0]))
                        sent.append(delayed_transcripts[0])
                        await websocket.wait_closed()
                        return
                    await websocket.send(json.dumps(event))
                    sent.append(event)
                    if event["type"] == "response.event" and event["event"]["type"] == (
                        "response.function_call_arguments.delta"
                    ):
                        tool_arguments_started = True
                        if boundary == "cancel" and delayed_transcripts:
                            await websocket.send(json.dumps(delayed_transcripts[0]))
                            sent.append(delayed_transcripts[0])
                            await websocket.wait_closed()
                            return
                    if boundary != "late_transcript" and event["type"] == "session.output_transcript.delta":
                        await websocket.wait_closed()
                        return
        except ConnectionClosed:
            if boundary == "late_transcript":
                raise
        except Exception as exc:
            replay_errors.append(exc)
            ready.set()

    _instrument()
    async with serve(replay, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = live.OpenAILiveLLMService(
            api_key="unused-in-replay",
            base_url=f"ws://127.0.0.1:{port}",
            settings=live.OpenAILiveLLMService.Settings(
                model=config["model"], system_instruction=config["instructions"]
            ),
            delegation=live.ResponsesDelegation(
                settings=OpenAIResponsesLLMService.Settings(
                    model=backend["model"], system_instruction=backend["instructions"]
                )
            ),
        )
        context = LLMContext(
            messages=[{"role": "system", "content": "Superseded context instructions."}]
            if boundary == "cancel_callback"
            else [],
            tools=ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name=t["name"],
                        description=t["description"],
                        properties=t["parameters"]["properties"],
                        required=t["parameters"]["required"],
                    )
                    for t in backend["tools"]
                ]
            ),
        )

        # Pipecat executes its real missing-tool handler. No replacement tool,
        # provider client, frame processor, or observer stands in for the SDK.
        @service.event_handler("on_after_push_frame")
        async def on_frame(_service, frame):
            if isinstance(frame, _frames().LLMTextFrame) and boundary != "late_transcript":
                ready.set()
                if boundary == "cancel_callback":
                    try:
                        await asyncio.Event().wait()
                    finally:
                        callback_cancelled.set()

        # Live's proposed turn frames also reach the universal user controller.
        # Those callbacks must not create a second, cascade-owned user turn.
        aggregators = LLMContextAggregatorPair(context)
        worker = _make_worker(Pipeline([aggregators.user(), service, aggregators.assistant()]))
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            await worker.queue_frame(_frames().LLMContextFrame(context))
            await asyncio.wait_for(ready.wait(), 10)
            if replay_errors:
                raise replay_errors[0]
            tracker = worker._braintrust_pipecat_observer._session.live
            assert tracker is not None
            if boundary == "late_transcript":
                await worker.queue_frame(_frames().EndFrame())
            else:
                await worker.cancel()
            await asyncio.wait_for(running, 10)
        finally:
            if not running.done():
                await worker.cancel()
                await asyncio.wait_for(running, 10)
    assert not replay_errors
    # Cleanup releases the tracker's own payload buffers and correlation
    # caches, even if a caller retains the tracker after pipeline shutdown.
    assert not {name: len(value) for name, value in vars(tracker).items() if isinstance(value, (dict, list)) and value}
    if boundary == "cancel_callback":
        assert callback_cancelled.is_set()

    logs = memory_logger.pop()
    session = _single_span(logs, "session")
    model = _single_span(logs, "pipecat.live")
    assert model["span_attributes"]["type"] == "llm"
    assert model["metadata"]["audio.scope"] == "session"
    assert model["span_parents"] == [session["span_id"]]
    assert model["metadata"]["contrib.pipecat.live.finalization_confirmed"] is (boundary == "late_transcript")
    assert all(row["metrics"].get("end") is not None for row in logs)
    turns = _spans_named(logs, "user_turn") + _spans_named(logs, "agent_turn")
    by_id = {row["span_id"]: row for row in turns}
    received = _spans_named(logs, "transcript_received")
    assert turns and len(turns) == len(received)
    assert not _spans_named(logs, "eou_wait")
    assert all(row["metadata"].get("contrib.pipecat.live.grouping") == "pipecat_transcript_gap" for row in turns)
    for role in ("user", "assistant"):
        kind = "session.input_transcript.delta" if role == "user" else "session.output_transcript.delta"
        observed = "".join(e["delta"] for e in sent if e["type"] == kind)
        assert "".join(r["output"][0]["content"] for r in received if r["output"][0]["role"] == role) == observed
    for row in received:
        turn = by_id[row["metadata"]["turn.id"]]
        assert row["span_parents"] == [turn["span_id"]]
        assert turn["metadata"]["turn.id"] == turn["span_id"]
        assert turn["span_attributes"]["name"] == ("user_turn" if row["output"][0]["role"] == "user" else "agent_turn")
        assert turn["metrics"]["start"] <= row["metrics"]["start"] <= row["metrics"]["end"] <= turn["metrics"]["end"]
        assert "audio.selections" not in row["metadata"]
        assert "audio.selections" not in turn["metadata"]
        assert "turn.reply_to" not in turn["metadata"]
    # Cancellation, or a group opened by late text after native stop, must not
    # be presented as a normally completed framework turn.
    assert any(row["metadata"].get("turn.incomplete") for row in turns)
    assert all("turn.id" not in row.get("metadata", {}) for row in logs if row not in turns + received)
    responses = _spans_named(logs, "llm_response")
    assert all(row["span_attributes"]["type"] == "task" and "tokens" not in row["metrics"] for row in responses)
    assert all(not _link_targets(row, "consumes_tool_result") for row in responses)
    llms = _spans_named(logs, "pipecat.live.backend_response")
    if boundary == "late_transcript":
        assert len(responses) == 2
        assert (
            responses[1]["metadata"]["contrib.pipecat.live.previous_response_id"]
            == responses[0]["metadata"]["contrib.pipecat.live.response_id"]
        )
        delegation = next(e for e in server_events if e["type"] == "session.delegation.created")
        assert all(
            row["metadata"]["contrib.pipecat.live.delegation_offset_ms"] == delegation["offset_ms"]
            for row in responses
        )
        tool = _single_span(logs, "lookup_order")
        assert tool["span_parents"] == [responses[0]["span_id"]]
        assert _link_targets(tool, "requested_by")
        assert responses[0]["metrics"]["end"] >= tool["metrics"]["end"]
        assert any(e["type"] == "response.item.create" for e in outgoing)
        completions = [
            e["event"]["response"]
            for e in server_events
            if e["type"] == "response.event" and e["event"]["type"] == "response.completed"
        ]
        assert [r["metrics"]["tokens"] for r in llms] == [r["usage"]["total_tokens"] for r in completions]
        assert [r["metrics"]["prompt_cache_creation_tokens"] for r in llms] == [
            r["usage"]["input_tokens_details"]["cache_write_tokens"] for r in completions
        ]
        completed_items = [
            e["event"]["item"]
            for e in sent
            if e["type"] == "response.event" and e["event"]["type"] == "response.output_item.done"
        ]
        assert [r["output"] for r in llms] == [[item] for item in completed_items]
        closed = next(e for e in server_events if e["type"] == "session.closed")
        assert model["metadata"]["contrib.pipecat.live.usage_seconds"] == closed["usage"]["seconds"]
    else:
        assert responses[-1]["metadata"].get("contrib.pipecat.incomplete")
        assert llms[-1]["metadata"].get("contrib.pipecat.incomplete")
        events = [e["event"] for e in sent if e["type"] == "response.event"]
        item = [e["item"] for e in events if e["type"] == "response.output_item.added"][-1]
        assert not any(e["type"] == "response.output_item.done" and e["item"]["id"] == item["id"] for e in events)
        if boundary == "cancel":
            arguments = "".join(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
            assert arguments
            expected = {**item, "arguments": arguments}
        else:
            part = next(e["part"] for e in events if e["type"] == "response.content_part.added")
            text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
            assert text
            expected = {**item, "content": [{**part, "text": text}]}
        assert llms[-1]["output"] == responses[-1]["output"] == [expected]
    started = next(e for e in server_events if e["type"] == "session.started")
    assert [message for message in model["input"] if message.get("role") == "system"] == [
        {"role": "system", "content": started["session"]["instructions"]}
    ]
    by_id = {row["span_id"]: row for row in logs}
    for row in logs:
        for parent_id in row.get("span_parents") or []:
            parent = by_id[parent_id]
            parent_start = parent["metrics"]["start"]
            if parent_id == session["span_id"]:
                # The session uses time.time(); recorder-derived child times
                # can round the same Windows clock tick one float apart.
                parent_start = math.nextafter(parent_start, -math.inf)
            assert parent_start <= row["metrics"]["start"], row["span_attributes"]["name"]
            assert row["metrics"]["end"] <= parent["metrics"]["end"], row["span_attributes"]["name"]


@pytest.mark.asyncio
async def test_realtime_cached_audio_usage(memory_logger):
    realtime = pytest.importorskip("pipecat.services.openai.realtime.llm")
    from websockets.asyncio.server import serve

    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    # Replay the same provider WebSocket payloads used by the LiveKit integration.
    cassette = Path(__file__).parent.parent / "livekit_agents" / "cassettes" / "latest" / "realtime-conversation.json"
    received = [entry["message"] for entry in json.loads(cassette.read_text()) if entry["direction"] == "receive"]
    events = {
        kind: next(event for event in received if event["type"] == kind)
        for kind in ("session.created", "session.updated", "response.created", "response.done")
    }
    usage = events["response.done"]["response"]["usage"]
    input_details = usage["input_token_details"]
    output_details = usage["output_token_details"]
    cached_audio_tokens = input_details["cached_tokens_details"]["audio_tokens"]

    async def replay(websocket):
        await websocket.send(json.dumps(events["session.created"]))
        async for message in websocket:
            kind = json.loads(message)["type"]
            if kind == "session.update":
                await websocket.send(json.dumps(events["session.updated"]))
            elif kind == "response.create":
                await websocket.send(json.dumps(events["response.created"]))
                await websocket.send(json.dumps(events["response.done"]))

    _instrument()
    async with serve(replay, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        service = realtime.OpenAIRealtimeLLMService(api_key="test", base_url=f"ws://127.0.0.1:{port}")
        context = LLMContext(messages=[{"role": "user", "content": "Tell me about my order."}])
        assert not await _run_text_pipeline([service], [context])

    logs = memory_logger.pop()
    task = _single_span(logs, "llm_response")
    # The Realtime completion reports into the one general Pipecat model child.
    span = _single_span(logs, "pipecat.llm_response")
    assert [row for row in logs if row["span_attributes"].get("type") == "llm"] == [span]
    assert task["span_attributes"]["type"] == "task"
    assert span["span_attributes"]["type"] == "llm"
    assert span["span_parents"] == [task["span_id"]]
    assert span["metadata"]["provider"] == "openai"
    assert span["metadata"]["model"]
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "pipecat-auto"
    assert span["output"] == task["output"]
    assert span["output"][0]["role"] == "assistant"
    assert task["metrics"]["start"] <= span["metrics"]["start"] <= span["metrics"]["end"] <= task["metrics"]["end"]
    expected = {
        "tokens": usage["total_tokens"],
        "prompt_tokens": usage["input_tokens"],
        "completion_tokens": usage["output_tokens"],
        "prompt_audio_tokens": input_details["audio_tokens"],
        "completion_audio_tokens": output_details["audio_tokens"],
        "prompt_cached_tokens": input_details["cached_tokens"],
        "prompt_cached_audio_tokens": cached_audio_tokens,
    }
    assert not expected.keys() & task["metrics"].keys()
    assert {key: span["metrics"][key] for key in expected} == expected


# Keep provider tests last: enabling OpenAI patches the client for the process.
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_provider_llm_spans_nest_under_llm_response_and_own_tokens(memory_logger):
    import braintrust
    import openai
    from braintrust.integrations.openai import OpenAIIntegration

    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    LLMAssistantAggregator = _import("pipecat.processors.aggregators.llm_response_universal.LLMAssistantAggregator")
    ToolsSchema = _import("pipecat.adapters.schemas.tools_schema.ToolsSchema")
    FunctionSchema = _import("pipecat.adapters.schemas.function_schema.FunctionSchema")
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    _instrument()
    OpenAIIntegration.setup()

    async def lookup(params):
        # Application code traced with plain Braintrust spans.
        with braintrust.start_span(name="lookup database"):
            await params.result_callback({"answer": "42"})

    instruction = "Call lookup exactly once. After its result reply exactly: Done."
    llm = _openai_llm(max_tokens=60, system_instruction=instruction)
    llm.register_function("lookup", lookup)
    context = LLMContext(
        messages=[
            {"role": "user", "content": "Look it up."},
        ],
        tools=ToolsSchema(
            standard_tools=[FunctionSchema(name="lookup", description="Look up.", properties={}, required=[])]
        ),
    )
    aggregator = LLMAssistantAggregator(context)
    worker = _make_worker(Pipeline([llm, aggregator]))
    completed = asyncio.Event()

    @aggregator.event_handler("on_assistant_turn_stopped")
    async def on_stopped(_aggregator, message, *_args):
        if message.content:
            completed.set()

    async def conversation():
        async with _running(worker):
            await worker.queue_frame(_frames().LLMContextFrame(context))
            await asyncio.wait_for(completed.wait(), 20)

    async def unrelated():
        # Application code outside the pipeline, running concurrently.
        async with openai.AsyncOpenAI() as client:
            await client.embeddings.create(model="text-embedding-3-small", input="unrelated")

    await asyncio.gather(conversation(), unrelated())

    logs = memory_logger.pop()
    session = _single_span(logs, "session")
    responses = _spans_named(logs, "llm_response")
    provider = _spans_named(logs, "Chat Completion")
    # Each model call has exactly one LLM span: the provider's, under its
    # llm_response, carrying the tokens llm_response never logs.
    assert len(provider) == len(responses) == 2
    for response, call in zip(responses, provider):
        assert response["input"][0] == {"role": "system", "content": instruction}
        assert response["input"] == call["input"]
        assert response["span_attributes"]["type"] == "task"
        assert not {"prompt_tokens", "completion_tokens", "tokens"} & set(response["metrics"])
        assert call["span_attributes"]["type"] == "llm"
        assert call["span_parents"] == [response["span_id"]]
        assert call["metrics"]["prompt_tokens"] > 0 and call["metrics"]["completion_tokens"] > 0
    # Pipecat runs tools outside the response, so tool code joins the session.
    custom = _single_span(logs, "lookup database")
    assert custom["root_span_id"] == session["root_span_id"]
    assert custom["span_parents"] == [session["span_id"]]
    # Concurrent application calls outside the pipeline stay independent.
    embedding = _single_span(logs, "Embedding")
    assert embedding["root_span_id"] != session["root_span_id"]
    assert not embedding.get("span_parents")


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["setup", "wrap"])
async def test_provider_transcriptions_nest_under_their_stt_spans(memory_logger, vcr, entrypoint):
    from braintrust.integrations.openai import OpenAIIntegration

    options = {"capture_audio_attachments": False}
    if entrypoint == "setup":
        _instrument(**options)
    else:
        wrap_pipeline_worker(_import("pipecat.pipeline.worker.PipelineWorker"), **options)
    OpenAIIntegration.setup()

    # Same real speech/provider exchanges, with provider tracing enabled.
    with vcr.use_cassette("test_consecutive_voice_turns_reply_to_and_link_only_their_own_speech.yaml"):
        await _VoiceSession(system="Reply with exactly one word: Okay.").run(utterances=2)

    logs = memory_logger.pop()
    recognitions = _spans_named(logs, "stt")
    provider = _spans_named(logs, "Transcription")
    turns = _spans_named(logs, "user_turn")
    assert len(recognitions) == len(provider) == len(turns) == 2
    for recognition, call, turn in zip(recognitions, provider, turns):
        assert recognition["span_parents"] == [turn["span_id"]]
        assert call["span_parents"] == [recognition["span_id"]]
        # Pipecat maps monotonic timestamps to wall time; provider spans use
        # time.time(). Allow one coarse Windows clock tick between those clocks.
        clock_tolerance = 0.02
        assert recognition["metrics"]["start"] <= call["metrics"]["start"] + clock_tolerance
        assert call["metrics"]["end"] <= recognition["metrics"]["end"] + clock_tolerance
        assert "tokens" not in recognition["metrics"]
    # Subsequent model calls must retain their own scope, never the STT scope.
    for response, call in zip(_spans_named(logs, "llm_response"), _spans_named(logs, "Chat Completion")):
        assert call["span_parents"] == [response["span_id"]]
    assert all(row["metrics"].get("end") is not None for row in logs)


@pytest_asyncio.fixture
async def tracker_observer(memory_logger):
    """Supplemental timestamp/layout cases; recorded pipelines cover behavior."""
    _frames()
    from braintrust.integrations.pipecat.tracing.observer import BraintrustPipecatObserver

    observer = BraintrustPipecatObserver(capture_audio_attachments=False)
    yield observer
    await observer.cleanup()


@asynccontextmanager
async def _native_eou_controller(observer, strategies, *, observe_start=True, begin_turn=True):
    """Synthetic native frames + real controller/strategies/timers, no provider IO."""
    from braintrust.integrations.pipecat.patchers import _ACTIVE_OBSERVER

    Controller = _import("pipecat.turns.user_turn_controller.UserTurnController")
    TaskManager = _import("pipecat.utils.asyncio.task_manager.TaskManager")
    TaskManagerParams = _import("pipecat.utils.asyncio.task_manager.TaskManagerParams")
    controller = Controller(user_turn_strategies=strategies, user_turn_stop_timeout=30)
    manager = TaskManager()
    manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    if "task_manager" in inspect.signature(controller.setup).parameters:
        await controller.setup(manager)
    else:
        Setup = _import("pipecat.processors.frame_processor.FrameProcessorSetup")
        Clock = _import("pipecat.clocks.system_clock.SystemClock")
        Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
        Identity = _import("pipecat.processors.filters.identity_filter.IdentityFilter")
        await controller.setup(
            Setup(clock=Clock(), task_manager=manager, pipeline_worker=_make_worker(Pipeline([Identity()])))
        )
    token = _ACTIVE_OBSERVER.set(observer if observe_start else None)
    try:
        if begin_turn:
            await controller.process_frame(_frames().VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.02)
        if not observe_start:
            _ACTIVE_OBSERVER.reset(token)
            token = _ACTIVE_OBSERVER.set(observer)
        yield controller
    finally:
        await controller.cleanup()
        _ACTIVE_OBSERVER.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "committed",
        "resumed",
        "transcript_late",
        "finalization_frame",
        "shutdown",
        "unassociated",
        "multiple",
        "audio_only",
        "text_late",
        "gate_changed",
    ],
)
async def test_eou_native_controller_lifecycle(memory_logger, tracker_observer, mode):

    _instrument()
    frames = _frames()
    if mode == "finalization_frame" and not hasattr(frames, "TranscriptionFinalizedFrame"):
        pytest.skip("Pipecat does not support a separate transcription finalization frame")
    Strategies = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")
    SpeechTimeout = _import("pipecat.turns.user_stop.SpeechTimeoutUserTurnStopStrategy")
    options = {}
    if mode in ("audio_only", "gate_changed"):
        if "wait_for_transcript" not in inspect.signature(SpeechTimeout).parameters:
            pytest.skip("wait_for_transcript=False requires current Pipecat")
        options["wait_for_transcript"] = False
    strategies = [SpeechTimeout(user_speech_timeout=0.03, **options)]
    if mode == "multiple":
        strategies.append(SpeechTimeout(user_speech_timeout=0.06))
    stopped = asyncio.Event()
    async with _native_eou_controller(
        tracker_observer, Strategies(stop=strategies), observe_start=mode != "unassociated"
    ) as controller:

        @controller.event_handler("on_user_turn_stopped")
        async def on_stopped(*_args):
            stopped.set()

        await controller.process_frame(
            frames.STTMetadataFrame(
                service_name="local-test",
                ttfs_p99_latency=0 if mode in ("audio_only", "text_late", "gate_changed") else 1.0,
            )
        )
        pause = frames.VADUserStoppedSpeakingFrame(stop_secs=0.01)
        await controller.process_frame(pause)
        if mode == "gate_changed":
            strategies[0].wait_for_transcript = True
            await asyncio.sleep(0.06)
            assert not stopped.is_set(), "The controller must still require text"
        if mode == "resumed":
            resume = frames.VADUserStartedSpeakingFrame(start_secs=0.001)
            await controller.process_frame(resume)
            await asyncio.sleep(0.02)
            pause = frames.VADUserStoppedSpeakingFrame(stop_secs=0.01)
            await controller.process_frame(pause)
        # Multiple transcriptions/evaluations during this pause must not split it.
        await controller.process_frame(frames.InterimTranscriptionFrame("Hi", "caller", "0"))
        await controller.process_frame(pause)  # Re-evaluate/re-arm this same pause.
        if mode in ("transcript_late", "text_late", "finalization_frame"):
            await asyncio.sleep(0.06)
        if mode != "shutdown":
            if mode == "finalization_frame":
                await controller.process_frame(frames.TranscriptionFrame("Hi", "caller", "0"))
                assert not stopped.is_set()
                await controller.process_frame(frames.TranscriptionFinalizedFrame())
            elif mode != "audio_only":
                await controller.process_frame(frames.TranscriptionFrame("Hi", "caller", "0", finalized=True))
            await asyncio.wait_for(stopped.wait(), 2)
        else:
            assert not stopped.is_set()
    await tracker_observer.cleanup()
    logs = memory_logger.pop()
    emitted = _spans_named(logs, "eou_wait")
    assert len(emitted) == (2 if mode == "resumed" else 1)
    final = emitted[-1]
    assert final["metrics"]["start"] == pause.timestamp - pause.stop_secs
    assert final["metadata"]["contrib.pipecat.vad_timestamp"] == pause.timestamp
    assert final["metadata"]["contrib.pipecat.vad_stop_secs"] == pause.stop_secs
    assert len(final["metadata"]["contrib.pipecat.configured_stop_strategies"]) == len(strategies)
    if mode == "shutdown":
        assert "eou.outcome" not in final["metadata"]
        assert "eou.end_reason" not in final["metadata"]
    else:
        assert final["metadata"]["eou.outcome"] == "committed"
    if mode in ("transcript_late", "finalization_frame"):
        assert final["metadata"]["eou.end_reason"] == "transcription_finalized"
    if mode == "text_late":
        assert "eou.end_reason" not in final["metadata"]
    if mode == "resumed":
        assert emitted[0]["metadata"]["eou.outcome"] == "user_resumed"
        assert emitted[0]["metrics"]["end"] == resume.timestamp - resume.start_secs
        assert "eou.end_reason" not in emitted[0]["metadata"]
    if mode == "unassociated":
        assert "turn.id" not in final["metadata"]
        assert final["span_parents"] == [_single_span(logs, "session")["span_id"]]
    else:
        assert final["span_parents"] == [final["metadata"]["turn.id"]]
    for wait in emitted:
        assert wait["metadata"]["eou.speech_end_source"] == "vad"
        if mode == "audio_only":
            assert wait["metadata"]["eou.end_reason"] == "silence_timeout"
        assert "eou.detection_methods" not in wait["metadata"]
        assert wait["context"]["span_origin"]["instrumentation"]["name"] == "pipecat-auto"
    _assert_all_ended(logs)


def _http_services(*, sdk_configuration=False, remove_followup_tools=False):
    """Real custom services using OpenAI's HTTP APIs without the OpenAI client.

    Native Pipecat parsing/aggregation is retained. API responses are recorded,
    parsed into the provider's actual response classes, and streamed normally.
    The inherited SDK clients deliberately remain unused: ownership must follow
    the overridden execution path rather than a service label or attribute.
    """
    httpx = importlib.import_module("httpx2" if importlib.util.find_spec("httpx2") else "httpx")
    from openai._utils import strip_not_given
    from openai.types.audio import Transcription
    from openai.types.chat import ChatCompletionChunk

    OpenAILLM = _import("pipecat.services.openai.llm.OpenAILLMService")
    OpenAISTT = _import("pipecat.services.openai.stt.OpenAISTTService")
    OpenAITTS = _import("pipecat.services.openai.tts.OpenAITTSService")

    class HTTPCompletions(OpenAILLM):
        def build_chat_completion_params(self, invocation):
            params = super().build_chat_completion_params(invocation)
            if remove_followup_tools and any(message.get("role") == "tool" for message in params["messages"]):
                # A custom service can change its actual request after adapter
                # conversion; the fallback must observe this final snapshot.
                params["tools"] = []
                params["tool_choice"] = "none"
            return params

        async def _process_context(self, context):
            await super()._process_context(context)
            # Some custom services enqueue completion before reporting their
            # final usage. Repeat the real provider snapshot to catch summing
            # and association with a subsequent response.
            usage = getattr(self, "_http_usage", None)
            if usage is not None:
                await self.push_frame(_frames().LLMFullResponseEndFrame())
                await asyncio.sleep(0)
                from pipecat.metrics.metrics import LLMTokenUsage  # pylint: disable=import-error

                for _ in range(2):
                    await self.start_llm_usage_metrics(LLMTokenUsage.model_validate(usage.model_dump()))

        async def get_chat_completions(self, context):
            invocation = self.get_llm_adapter().get_llm_invocation_params(
                context,
                system_instruction=self._settings.system_instruction
                if isinstance(self._settings.system_instruction, str)
                else None,
                convert_developer_to_user=not self.supports_developer_role,
            )
            if inspect.isawaitable(invocation):
                invocation = await invocation
            params = strip_not_given(self.build_chat_completion_params(invocation))

            async def chunks():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        getattr(self, "_http_base_url", "https://api.openai.com/v1/") + "chat/completions",
                        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                        json=params,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                chunk = ChatCompletionChunk.model_validate_json(line[6:])
                                if chunk.usage is not None:
                                    self._http_usage = chunk.usage
                                yield chunk

            return chunks()

    class ConfiguredHTTPCompletions(HTTPCompletions):
        async def get_chat_completions(self, context):
            # This SDK read supplies configuration; HTTP still executes the request.
            self._http_base_url = str(self._client.base_url)
            return await super().get_chat_completions(context)

    class HTTPTranscription(OpenAISTT):
        async def _transcribe(self, audio):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                    data={"model": self._settings.model},
                    files={"file": ("audio.wav", audio, "audio/wav")},
                )
                response.raise_for_status()
                return Transcription.model_validate(response.json())

    class HTTPSpeech(OpenAITTS):
        async def run_tts(self, text, context_id):
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                    json={
                        "model": self._settings.model,
                        "voice": self._settings.voice,
                        "input": text,
                        "response_format": "pcm",
                    },
                ) as response:
                    response.raise_for_status()
                    await self.start_tts_usage_metrics(text)
                    async for chunk in response.aiter_bytes(self.chunk_size):
                        if chunk:
                            await self.stop_ttfb_metrics()
                            yield _frames().TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)

    def llm(max_tokens=20):
        service = ConfiguredHTTPCompletions if sdk_configuration else HTTPCompletions
        return service(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=HTTPCompletions.Settings(
                model=TEST_MODEL,
                temperature=1.0,
                extra={"reasoning_effort": "none"},
                max_completion_tokens=max_tokens,
            ),
        )

    def tts():
        return HTTPSpeech(
            api_key=os.environ["OPENAI_API_KEY"], settings=HTTPSpeech.Settings(model="tts-1", voice="alloy")
        )

    return HTTPTranscription, llm, tts


@pytest.mark.vcr(cassette_name="test_consecutive_voice_turns_reply_to_and_link_only_their_own_speech")
@pytest.mark.asyncio
# capture and sdk_configuration are independent; two cases cover both branches.
@pytest.mark.parametrize("capture,sdk_configuration", [(False, False), (True, True)])
async def test_uncovered_http_clients_own_model_children_and_usage(
    memory_logger, attachment_uploads, capture, sdk_configuration
):
    from braintrust.audio import AudioEncoding, AudioRecordingOptions

    _instrument(
        capture_audio_attachments=capture,
        audio_recording_options=AudioRecordingOptions(encoding=AudioEncoding(codec="pcm", container="wav")),
    )
    await _VoiceSession(system="Reply with exactly one word: Okay.").run(
        utterances=2, services=_http_services(sdk_configuration=sdk_configuration)
    )
    logs = memory_logger.pop()
    for wrapper, child in (("llm_response", "pipecat.llm_response"), ("stt", "pipecat.stt"), ("tts", "pipecat.tts")):
        parents, children = _spans_named(logs, wrapper), _spans_named(logs, child)
        assert len(parents) == len(children) == 2
        for parent, model in zip(parents, children):
            assert model["span_attributes"]["type"] == "llm"
            assert model["span_parents"] == [parent["span_id"]]
            assert model["metadata"]["provider"] == "openai"
            assert model["metadata"]["model"]
            assert set(parent["metrics"]) == {"start", "end"}
            assert "characters" not in model["metrics"]
            assert model["metrics"].get("time_to_first_token", -1) >= 0, model
            if wrapper == "llm_response":
                assert model["input"] == parent["input"]
                assert model["output"] == parent["output"]
                assert model["metrics"]["prompt_tokens"] > 0
                assert model["metrics"]["completion_tokens"] > 0
                assert (
                    model["metrics"]["tokens"]
                    == model["metrics"]["prompt_tokens"] + model["metrics"]["completion_tokens"]
                )
            elif wrapper == "stt":
                assert model["input"]["operation"] == "transcribe"
                if capture:
                    part = model["input"]["content"][0]["file"]
                    assert part["file_data"].reference["type"] == "braintrust_attachment"
                    assert part["byte_size"] > 44
                else:
                    assert model["input"] == {"operation": "transcribe"}
                assert model["output"] == {"content": [{"type": "text", "text": parent["output"][0]["content"]}]}
                assert "tokens" not in model["metrics"]
            else:
                assert model["input"]["operation"] == "speech"
                assert model["input"]["prompt"] == parent["input"]["text"]
                if capture:
                    part = model["output"]["content"][0]["file"]
                    assert part["file_data"].reference["type"] == "braintrust_attachment"
                    assert part["byte_size"] > 44
                else:
                    assert model["output"] == {"content": []}
                assert "tokens" not in model["metrics"]
    assert not any(row["context"]["span_origin"]["instrumentation"]["name"] == "openai-auto" for row in logs)
    _assert_all_ended(logs)


async def _exercise_speech_stream(tts, mode):
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    worker = _make_worker(Pipeline([tts]))
    finished = asyncio.Event()
    failures = []

    @worker.event_handler("on_pipeline_started")
    async def consume(_worker, _frame):
        caller = logger.current_span()
        stream = tts.run_tts("Private words.", "native-synthesis")
        try:
            if mode == "full":
                chunks = [frame async for frame in stream]
                assert chunks
            else:
                chunk = await anext(stream)
                assert chunk.audio
                # Yielding audio must restore the consumer's parent context.
                assert logger.current_span() is caller
                if mode == "error":
                    with pytest.raises(RuntimeError, match="consumer stopped"):
                        await stream.athrow(RuntimeError("consumer stopped"))
                elif mode == "cancel":
                    with pytest.raises(asyncio.CancelledError):
                        await stream.athrow(asyncio.CancelledError())
        except BaseException as exc:
            failures.append(exc)
        finally:
            await stream.aclose()
            assert logger.current_span() is caller
            finished.set()

    async with _running(worker):
        await asyncio.wait_for(finished.wait(), 10)
    if failures:
        raise failures[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "close", "error", "cancel"])
async def test_provider_speech_stream_cleanup_and_simultaneous_sessions(memory_logger, vcr, mode):
    from braintrust.integrations.openai import OpenAIIntegration

    _instrument(capture_audio_attachments=False)
    OpenAIIntegration.setup()
    with vcr.use_cassette(
        PRIVATE_WORDS_CASSETTE,
        allow_playback_repeats=True,
    ):
        await asyncio.gather(*(_exercise_speech_stream(_openai_tts(), mode) for _ in range(2)))
    logs = memory_logger.pop()
    sessions, wrappers, models = (_spans_named(logs, name) for name in ("session", "tts", "Speech"))
    assert len(sessions) == len(wrappers) == len(models) == 2
    assert len({session["root_span_id"] for session in sessions}) == 2
    for model in models:
        wrapper = next(wrapper for wrapper in wrappers if wrapper["span_id"] in model["span_parents"])
        assert model["span_parents"] == [wrapper["span_id"]]
        assert model["root_span_id"] == wrapper["root_span_id"]
        assert model["span_attributes"]["type"] == "llm"
        assert model["input"]["operation"] == "speech"
        assert model["input"]["prompt"] == "Private words."
        assert wrapper["input"] == {"text": "Private words."}
    assert not _spans_named(logs, "pipecat.tts")
    assert not _spans_named(logs, "agent_speech")
    _assert_all_ended(logs)


@pytest.mark.asyncio
async def test_native_tts_transform_is_captured_at_the_model_boundary(memory_logger, vcr):
    _instrument(capture_audio_attachments=False)
    _, _, tts_factory = _http_services()
    tts = tts_factory()

    async def spoken_text(_text, _aggregation_type):
        return "Private words."

    tts.add_text_transformer(spoken_text)
    context = _private_words_context()
    with vcr.use_cassette(
        PRIVATE_WORDS_CASSETTE,
        allow_playback_repeats=True,
    ):
        await _run_text_pipeline(
            [_openai_llm(), tts],
            [context, lambda: [_frames().TTSSpeakFrame("placeholder")]],
            until="TTSStoppedFrame",
        )
    logs = memory_logger.pop()
    models, wrappers = _spans_named(logs, "pipecat.tts"), _spans_named(logs, "tts")
    assert len(models) == len(wrappers) == 2
    assert all(model["input"]["prompt"] == "Private words." for model in models)
    assert wrappers[1]["input"] == {"text": "placeholder"}
    assert _link_targets(wrappers[0], "generated_by") == [_single_span(logs, "llm_response")["span_id"]]
    # Independent application speech does not consume the previous response.
    assert not _link_targets(wrappers[1], "generated_by")
    assert not _spans_named(logs, "agent_speech")
    _assert_all_ended(logs)


def _observe_response_end(monkeypatch, *, delay_start):
    """Return an event set once the observer has handled a response end.

    With ``delay_start``, the observer handles each response start late, after
    the service has already executed its request.
    """
    from braintrust.integrations.pipecat.tracing.observer import BraintrustPipecatObserver

    original = BraintrustPipecatObserver.on_push_frame
    observed = asyncio.Event()

    async def observe(observer, data):
        if delay_start and isinstance(data.frame, _frames().LLMFullResponseStartFrame):
            await asyncio.sleep(0.1)
        await original(observer, data)
        if isinstance(data.frame, _frames().LLMFullResponseEndFrame):
            observed.set()

    monkeypatch.setattr(BraintrustPipecatObserver, "on_push_frame", observe)
    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_observer", [False, True])
async def test_fallback_llm_input_observes_native_adapter_transform(memory_logger, vcr, monkeypatch, delayed_observer):
    _instrument(capture_audio_attachments=False)
    _, llm_factory, _ = _http_services()
    llm = llm_factory()
    llm.supports_developer_role = False
    context = _import("pipecat.processors.aggregators.llm_context.LLMContext")(
        messages=[{"role": "developer", "content": "Reply with exactly: Private words."}]
    )
    processed = _observe_response_end(monkeypatch, delay_start=delayed_observer)

    async def converse(send):
        await send(_frames().LLMContextFrame(context), until=_frames().LLMFullResponseEndFrame)
        await asyncio.wait_for(processed.wait(), 5)

    with vcr.use_cassette(PRIVATE_WORDS_CASSETTE):
        await _run_frame_sequence([llm], converse, metrics=True)
    logs = memory_logger.pop()
    assert _single_span(logs, "llm_response")["input"][0]["content"] == "Reply with exactly: Private words."
    assert _single_span(logs, "pipecat.llm_response")["input"] == [
        {"role": "user", "content": "Reply with exactly: Private words."}
    ]
    assert _single_span(logs, "pipecat.llm_response")["metrics"]["prompt_tokens"] > 0
    _assert_all_ended(logs)


@pytest.mark.asyncio
async def test_streaming_stt_silence_after_a_transcript_opens_no_recognition(memory_logger):
    """Streaming STT receives every input frame, including silence between turns."""
    _instrument(capture_user_audio_attachments=True)
    frames = _frames()
    STTService = _import("pipecat.services.stt_service.STTService")
    speech, silence = b"\x01\x00" * 320, b"\x00\x00" * 320

    class StreamingSTT(STTService):
        # A local streaming service without a model client, so it owns a
        # pipecat.stt child. The bug is local buffering, not a response shape.
        async def run_stt(self, audio):
            if audio == speech:
                yield frames.TranscriptionFrame("hello", "caller", "final", finalized=True)
            else:
                yield None

    async def conversation(send):
        await send(frames.UserStartedSpeakingFrame())
        await send(frames.VADUserStartedSpeakingFrame())
        await send(frames.InputAudioRawFrame(silence, 16000, 1), wait=False)
        await send(frames.InputAudioRawFrame(speech, 16000, 1), until=frames.TranscriptionFrame)
        await send(frames.VADUserStoppedSpeakingFrame())
        await send(frames.UserStoppedSpeakingFrame())
        # Idle silence after the turn must neither open a recognition nor buffer.
        for _ in range(20):
            await send(frames.InputAudioRawFrame(silence, 16000, 1), wait=False)
        await send(frames.UserStartedSpeakingFrame())

    await _run_frame_sequence([StreamingSTT()], conversation)
    logs = memory_logger.pop()
    recognition = _single_span(logs, "stt")
    assert recognition["output"] == [{"role": "user", "content": "hello"}]
    model = _single_span(logs, "pipecat.stt")
    assert model["span_parents"] == [recognition["span_id"]]
    # The first turn's two frames only: 44-byte header plus 2 x 640 bytes.
    assert model["input"]["content"][0]["file"]["byte_size"] == 44 + 2 * 640
    _assert_all_ended(logs)


@pytest.mark.asyncio
async def test_eou_cleanup_continues_after_wait_close_fails(memory_logger, tracker_observer, monkeypatch):
    from braintrust.integrations.pipecat.tracing._eou import ControllerWait

    _instrument()
    Strategies = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")
    Timeout = _import("pipecat.turns.user_stop.SpeechTimeoutUserTurnStopStrategy")

    def fail_close(_wait, _ended):
        raise RuntimeError("wait cleanup failed")

    async with _native_eou_controller(tracker_observer, Strategies(stop=[Timeout()])):
        monkeypatch.setattr(ControllerWait, "close", fail_close)
        await tracker_observer.cleanup()

    assert not tracker_observer._controllers
    rows = memory_logger.pop()
    _single_span(rows, "session")
    _single_span(rows, "user_turn")
    _assert_all_ended(rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed", [False, True])
async def test_eou_queued_observations_keep_turn_ownership(memory_logger, tracker_observer, monkeypatch, delayed):
    """Real WorkerObserver queue falls behind two real native controller turns."""
    _instrument()
    WorkerObserver = _import("pipecat.pipeline.worker_observer.WorkerObserver")
    FramePushed = _import("pipecat.observers.base_observer.FramePushed")
    Direction = _import("pipecat.processors.frame_processor.FrameDirection")
    Identity = _import("pipecat.processors.filters.identity_filter.IdentityFilter")
    Manager = _import("pipecat.utils.asyncio.task_manager.TaskManager")
    Params = _import("pipecat.utils.asyncio.task_manager.TaskManagerParams")
    Strategies = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")
    Timeout = _import("pipecat.turns.user_stop.SpeechTimeoutUserTurnStopStrategy")
    manager = Manager()
    manager.setup(Params(loop=asyncio.get_running_loop()))
    worker = WorkerObserver(observers=[tracker_observer])
    release = asyncio.Event()
    entered = asyncio.Event()
    original = tracker_observer.on_push_frame

    async def on_push(data):
        if delayed and not entered.is_set():
            entered.set()
            await release.wait()
        await original(data)

    monkeypatch.setattr(tracker_observer, "on_push_frame", on_push)
    await worker.setup(manager)
    if hasattr(worker, "start"):
        await worker.start()
    source, destination = Identity(), Identity()
    session = tracker_observer._start_session()
    with session.clock.event(time.monotonic_ns()):
        pass

    async def push(frame):
        await worker.on_push_frame(FramePushed(source, destination, frame, Direction.DOWNSTREAM, time.monotonic_ns()))

    try:
        async with _native_eou_controller(
            tracker_observer, Strategies(stop=[Timeout(user_speech_timeout=0.03)]), begin_turn=False
        ) as controller:
            committed = asyncio.Queue()
            committed_at = []

            @controller.event_handler("on_user_turn_stopped")
            async def on_stopped(*_args):
                committed_at.append(time.time())
                committed.put_nowait(True)

            async def process(frame):
                await push(frame)
                await controller.process_frame(frame)

            for index in range(2):
                await process(_frames().VADUserStartedSpeakingFrame())
                await asyncio.sleep(0.005)  # Speech must outlast the stop correction.
                await process(_frames().VADUserStoppedSpeakingFrame(stop_secs=0.001))
                await process(_frames().TranscriptionFrame(f"Turn {index}", "caller", "0", finalized=True))
                await asyncio.wait_for(committed.get(), 2)
            if delayed:
                await asyncio.wait_for(entered.wait(), 2)
            release.set()
            await asyncio.wait_for(worker._proxies[tracker_observer].queue.join(), 2)
    finally:
        release.set()
        if hasattr(worker, "stop"):
            await worker.stop()
        await worker.cleanup()
    logs = memory_logger.pop()
    turns = _spans_named(logs, "user_turn")
    assert len(turns) == 2
    for name in ("user_speech", "eou_wait"):
        rows = _spans_named(logs, name)
        assert len(rows) == 2
        for row, turn in zip(rows, turns):
            assert row["span_parents"] == [turn["span_id"]], (name, row, turn)
            assert row["metadata"]["turn.id"] == turn["span_id"]
    # The controller establishes these turns, but these receipts have no STT
    # operation or native item lineage establishing their membership.
    receipts = _spans_named(logs, "transcript_received")
    assert len(receipts) == 2
    for receipt in receipts:
        assert receipt["span_parents"] == [_single_span(logs, "session")["span_id"]]
        assert "turn.id" not in receipt["metadata"]
        assert not _link_targets(receipt, "transcribes")
    for wait, turn, accepted_at in zip(_spans_named(logs, "eou_wait"), turns, committed_at):
        assert wait["metrics"]["end"] <= accepted_at
        assert turn["metrics"]["end"] == pytest.approx(wait["metrics"]["end"], abs=1e-6, rel=0)
    _assert_all_ended(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("new_turn", [False, True])
async def test_native_user_boundaries_and_delayed_receipts_through_pipeline(memory_logger, vcr, new_turn):
    _instrument(capture_audio_attachments=False)
    frames = _frames()
    context = _private_words_context()

    async def conversation(send):
        await send(frames.UserStartedSpeakingFrame())
        for _ in range(2):
            await send(frames.VADUserStartedSpeakingFrame())
            await asyncio.sleep(0.02)
            await send(frames.VADUserStoppedSpeakingFrame())
            await asyncio.sleep(0.02)
        await send(frames.UserStoppedSpeakingFrame())
        # No speech in this native turn: it must not inherit the previous one.
        await send(frames.UserStartedSpeakingFrame())
        await send(frames.UserStoppedSpeakingFrame())
        if new_turn:
            await send(frames.UserStartedSpeakingFrame())
        await send(frames.TranscriptionFrame("yes", "caller", "delayed"))
        await send(frames.LLMContextFrame(context), until=frames.LLMFullResponseEndFrame)

    with vcr.use_cassette(PRIVATE_WORDS_CASSETTE):
        await _run_frame_sequence([_openai_llm()], conversation)
    logs = memory_logger.pop()
    turns = _spans_named(logs, "user_turn")
    speech = _spans_named(logs, "user_speech")
    assert len(speech) == 2 and len(turns) == 2 + new_turn
    intervals = [row["metadata"]["audio.selections"]["selection-0"] for row in speech]
    assert intervals[0]["end_offset_ms"] < intervals[1]["start_offset_ms"]
    assert turns[0]["metadata"]["audio.selections"] == {
        "selection-0": {
            "start_offset_ms": intervals[0]["start_offset_ms"],
            "end_offset_ms": intervals[1]["end_offset_ms"],
        }
    }
    assert turns[0]["metrics"]["end"] > speech[-1]["metrics"]["end"]
    assert all("audio.selections" not in row["metadata"] for row in turns[1:])
    receipt = _single_span(logs, "transcript_received")
    assert receipt["span_parents"] == [_single_span(logs, "session")["span_id"]]
    assert "turn.id" not in receipt["metadata"] and not _link_targets(receipt, "transcribes")
    response = _single_span(logs, "llm_response")
    assert not _link_targets(response, "uses_context") and "turn.id" not in response["metadata"]
    assert not _spans_named(logs, "agent_turn")
    _assert_all_ended(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stamp,correction", [(0, 0), (-1, 0), (float("nan"), 0), (float("inf"), 0), (None, 0), (None, -1)]
)
async def test_eou_omits_invalid_speech_timestamps(memory_logger, tracker_observer, stamp, correction):
    _instrument()
    Strategies = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")
    SpeechTimeout = _import("pipecat.turns.user_stop.SpeechTimeoutUserTurnStopStrategy")
    async with _native_eou_controller(
        tracker_observer, Strategies(stop=[SpeechTimeout(user_speech_timeout=1)])
    ) as controller:
        frame = _frames().VADUserStoppedSpeakingFrame(timestamp=stamp, stop_secs=correction)
        await controller.process_frame(frame)
    await tracker_observer.cleanup()
    assert not _spans_named(memory_logger.pop(), "eou_wait")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["external", "inactive"])
async def test_eou_external_turns_do_not_make_a_wait_from_vad(memory_logger, tracker_observer, mode):
    _instrument()
    Strategies = _import("pipecat.turns.user_turn_strategies.ExternalUserTurnStrategies")
    if mode == "external":
        strategies = Strategies()
    else:
        Local = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")
        Timeout = _import("pipecat.turns.user_stop.SpeechTimeoutUserTurnStopStrategy")
        strategies = Local(stop=[Timeout()])
    async with _native_eou_controller(tracker_observer, strategies, begin_turn=mode != "inactive") as controller:
        await controller.process_frame(_frames().VADUserStoppedSpeakingFrame())
    await tracker_observer.cleanup()
    assert not _spans_named(memory_logger.pop(), "eou_wait")


@pytest.mark.asyncio
async def test_shutdown_during_native_playback_keeps_interruption_unknown(memory_logger, vcr):
    _instrument(capture_audio_attachments=False)
    with vcr.use_cassette("test_voice_session_traces_turns_speech_recognition_generation_and_playback.yaml"):
        await _VoiceSession(system="Reply with exactly: Hi.").run(shutdown_during_playback=True)
    logs = memory_logger.pop()
    playback = _single_span(logs, "agent_speech")
    assert playback["metadata"]["contrib.pipecat.incomplete"] is True
    assert "contrib.pipecat.interrupted" not in playback["metadata"]
    _assert_all_ended(logs)


@pytest.mark.parametrize(
    "missing,incompatible", [("run_tts", "absent"), ("push_frame", "signature"), ("process_frame", "absent")]
)
def test_tts_execution_hooks_require_compatible_complete_surface(monkeypatch, missing, incompatible):
    from braintrust.integrations.pipecat.patchers import TTSExecutionPatcher

    service = _openai_tts()
    assert TTSExecutionPatcher.applies(None, None, target=service)
    # A local compatibility gate test, not a simulated provider response.
    monkeypatch.setattr(service, missing, None if incompatible == "absent" else service.cleanup)
    assert not TTSExecutionPatcher.applies(None, None, target=service)
    assert not TTSExecutionPatcher.is_patched(None, None, target=service)


@pytest.mark.asyncio
@pytest.mark.parametrize("expressive,ignored_transcript", [(False, False), (False, True), (True, False)])
async def test_native_formatted_context_retains_observed_transcript_provenance(
    memory_logger, vcr, expressive, ignored_transcript
):
    perception = pytest.importorskip("pipecat.expressive.perception") if expressive else None
    _instrument(capture_audio_attachments=False)
    frames = _frames()
    universal = "pipecat.processors.aggregators.llm_response_universal"
    context = _import("pipecat.processors.aggregators.llm_context.LLMContext")()
    StartStrategy = _import("pipecat.turns.user_start.ExternalUserTurnStartStrategy")
    start_options = (
        {"enable_interruptions": False}
        if "enable_interruptions" in inspect.signature(StartStrategy).parameters
        else {}
    )
    strategies = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")(
        start=[StartStrategy(**start_options)],
        stop=[_import("pipecat.turns.user_stop.ExternalUserTurnStopStrategy")(wait_for_transcript=False)],
    )
    pair = _import(f"{universal}.LLMContextAggregatorPair")(
        context, user_params=_import(f"{universal}.LLMUserAggregatorParams")(user_turn_strategies=strategies)
    )

    started = asyncio.Event()
    pair.user().add_event_handler("on_user_turn_started", lambda *_args: started.set())

    async def conversation(send):
        await send(getattr(frames, "ProposedUserStartedSpeakingFrame", frames.UserStartedSpeakingFrame)(), wait=False)
        await asyncio.wait_for(started.wait(), 5)
        await send(
            frames.TranscriptionFrame(
                "Reply with exactly: Private words.",
                "caller",
                "external-transcript",
                **(
                    {"perception": [perception.PerceptionSignal("tone", labels=[("neutral", 1.0)])]}
                    if perception
                    else {}
                ),
            ),
            wait=False,
        )
        if ignored_transcript:
            await send(frames.TranscriptionFrame(" ", "caller", "ignored-transcript"), wait=False)
        await send(
            getattr(frames, "ProposedUserStoppedSpeakingFrame", frames.UserStoppedSpeakingFrame)(),
            until=frames.LLMFullResponseEndFrame,
        )

    with vcr.use_cassette(PRIVATE_WORDS_CASSETTE):
        await _run_frame_sequence([pair.user(), _openai_llm()], conversation, expressive=expressive)
    logs = memory_logger.pop()
    receipts = _spans_named(logs, "transcript_received")
    receipt = next(row for row in receipts if row["output"][0]["content"].strip())
    response = _single_span(logs, "llm_response")
    assert (
        response["input"][-1]["content"]
        == ("([tone neutral=1.00]) " if expressive else "") + receipt["output"][0]["content"]
    )
    assert _link_targets(response, "uses_context") == [receipt["span_id"]]
    _assert_all_ended(logs)


@pytest.mark.vcr(cassette_name="test_tool_request_result_and_followup_use_native_missing_tool_handler")
@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sdk", "http"])
async def test_tool_request_result_and_followup_use_native_missing_tool_handler(memory_logger, transport):
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    LLMAssistantAggregator = _import("pipecat.processors.aggregators.llm_response_universal.LLMAssistantAggregator")
    ToolsSchema = _import("pipecat.adapters.schemas.tools_schema.ToolsSchema")
    FunctionSchema = _import("pipecat.adapters.schemas.function_schema.FunctionSchema")
    # Pipecat's built-in missing-tool handler executes the call and returns its
    # terminal result, so no replacement tool implementation is needed.
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": "Call unavailable_tool exactly once with query='hello'. After its result reply exactly: Done.",
            },
            {"role": "user", "content": "Run the tool."},
        ],
        tools=ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="unavailable_tool",
                    description="An unavailable tool.",
                    properties={"query": {"type": "string"}},
                    required=["query"],
                )
            ]
        ),
        tool_choice="auto",
    )
    llm = (_openai_llm if transport == "sdk" else _http_services(remove_followup_tools=True)[1])(max_tokens=60)
    aggregator = LLMAssistantAggregator(context)
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    PipelineParams = _import("pipecat.pipeline.worker.PipelineParams")
    _instrument()
    if transport == "sdk":
        from braintrust.integrations.openai import OpenAIIntegration

        OpenAIIntegration.setup()
    worker = _make_worker(
        Pipeline([llm, aggregator]),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    completed = asyncio.Event()

    @aggregator.event_handler("on_assistant_turn_stopped")
    async def on_stopped(_aggregator, message, *_args):
        if message.content:
            completed.set()

    async with _running(worker):
        await worker.queue_frame(_frames().LLMContextFrame(context))
        await asyncio.wait_for(completed.wait(), 20)

    logs = memory_logger.pop()
    responses = _spans_named(logs, "llm_response")
    assert len(responses) == 2
    requester, followup = responses
    tool = _single_span(logs, "unavailable_tool")
    assert requester["output"][0]["role"] == "assistant"
    assert set(requester["output"][0]) == {"role", "content", "tool_calls"}
    calls = requester["output"][0]["tool_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert json.loads(call["function"]["arguments"]) == {"query": "hello"}
    assert tool["input"] == {"query": "hello"}
    assert tool["output"]
    assert tool["span_parents"] == [requester["span_id"]]
    assert tool["metadata"]["gen_ai.tool.call.id"] == call["id"]
    assert _link_targets(tool, "requested_by") == [requester["span_id"]]
    assert _link_targets(followup, "consumes_tool_result") == [tool["span_id"]]
    assert requester["metadata"]["model"] == followup["metadata"]["model"] == TEST_MODEL
    models = _spans_named(logs, "Chat Completion" if transport == "sdk" else "pipecat.llm_response")
    assert len(models) == 2
    for response, model in zip(responses, models):
        assert model["span_parents"] == [response["span_id"]]
        if transport == "http" and response is followup:
            assert "tools" not in model["metadata"]
            assert model["metadata"]["tool_choice"] == "none"
            assert response["metadata"]["tools"]  # The final builder removed these.
        else:
            assert model["metadata"]["tools"] == response["metadata"]["tools"]
            assert model["metadata"]["tool_choice"] == "auto"
    # One agent turn spans the request, the tool and the follow-up.
    agent_turn = _single_span(logs, "agent_turn")
    assert agent_turn["metadata"]["turn.incomplete"] is False
    for operation in (requester, tool, followup):
        assert operation["metadata"]["turn.id"] == agent_turn["span_id"]
        assert operation["metrics"]["end"] <= agent_turn["metrics"]["end"]
    _assert_all_ended(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("context_id", [None, "reused-context"])
async def test_deferred_speech_with_reused_context_captures_each_operation(memory_logger, vcr, context_id):
    """A custom HTTP service delivers recorded audio after run_tts returns."""
    _instrument(capture_agent_audio_attachments=True)
    _, _, factory = _http_services()
    HTTPService = type(factory())
    frames = _frames()

    class DeferredSpeech(HTTPService):
        async def process_frame(self, frame, direction):
            if isinstance(frame, frames.TTSSpeakFrame):
                await self.process_generator(self.run_tts(frame.text, context_id))
            else:
                await super().process_frame(frame, direction)

        async def run_tts(self, text, context_id):
            async def produce():
                await self.push_frame(frames.TTSStartedFrame(context_id=context_id))
                async for frame in super(DeferredSpeech, self).run_tts(text, context_id):
                    await self.push_frame(frame)
                await self.push_frame(frames.TTSStoppedFrame(context_id=context_id))

            self.create_task(produce())
            yield None

    service = DeferredSpeech(
        api_key=os.environ["OPENAI_API_KEY"], settings=HTTPService.Settings(model="tts-1", voice="alloy")
    )

    async def converse(send):
        for _ in range(2):
            await send(frames.TTSSpeakFrame("Private words."), until=frames.TTSStoppedFrame)

    with vcr.use_cassette(
        PRIVATE_WORDS_CASSETTE,
        allow_playback_repeats=True,
    ):
        await _run_frame_sequence([service], converse)
    logs = memory_logger.pop()
    wrappers, models = _spans_named(logs, "tts"), _spans_named(logs, "pipecat.tts")
    assert len(wrappers) == len(models) == 2
    for wrapper, model in zip(wrappers, models):
        assert model["span_parents"] == [wrapper["span_id"]]
        assert wrapper["metrics"]["start"] <= model["metrics"]["start"] <= model["metrics"]["end"]
        assert model["output"]["content"][0]["file"]["byte_size"] > 44
        assert model["input"]["prompt"] == "Private words."
    assert not _spans_named(logs, "agent_speech")
    _assert_all_ended(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["bare", "native_turn", "nested_native_turn"])
async def test_provider_response_parent_survives_delayed_observation(memory_logger, vcr, monkeypatch, layout):
    from braintrust.integrations.openai import OpenAIIntegration

    _instrument(capture_audio_attachments=False)
    OpenAIIntegration.setup()
    observed = _observe_response_end(monkeypatch, delay_start=True)
    context = _private_words_context()
    processors = [_openai_llm()]
    if layout == "nested_native_turn":
        # The model runs inside a nested pipeline, upstream of the aggregator.
        processors = [_import("pipecat.pipeline.pipeline.Pipeline")(processors)]
    native_turn = layout != "bare"
    if native_turn:
        processors.extend(
            [
                _openai_tts(),
                _import("pipecat.processors.aggregators.llm_response_universal.LLMAssistantAggregator")(context),
            ]
        )

    async def converse(send):
        await send(
            _frames().LLMContextFrame(context),
            until=_frames().TTSStoppedFrame if native_turn else _frames().LLMFullResponseEndFrame,
        )
        await asyncio.wait_for(observed.wait(), 5)

    with vcr.use_cassette(PRIVATE_WORDS_CASSETTE):
        await _run_frame_sequence(processors, converse)
    logs = memory_logger.pop()
    wrapper, model = _single_span(logs, "llm_response"), _single_span(logs, "Chat Completion")
    assert model["span_parents"] == [wrapper["span_id"]]
    assert model["metrics"]["tokens"] > 0
    assert set(wrapper["metrics"]) == {"start", "end"}
    parent = _single_span(logs, "agent_turn" if native_turn else "session")
    assert wrapper["span_parents"] == [parent["span_id"]]
    if native_turn:
        assert parent["metadata"]["turn.incomplete"] is False
    _assert_all_ended(logs)
