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
import os
import socket
import tempfile
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest
import pytest_asyncio
from braintrust import SpanCustomizer, logger, set_span_customizers
from braintrust.integrations.pipecat import setup_pipecat, wrap_pipeline_worker
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.test_helpers import init_test_logger


TEST_MODEL = "gpt-6-luna"
PROJECT = "test-project-pipecat-py-tracing"
SPEECH = Path(__file__).parents[2] / "fixtures/test_speech.wav"
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
        # Pipecat never emits LLM spans: provider integrations own those.
        assert row["span_attributes"].get("type") != "llm", row


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
async def _running(worker):
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


@dataclass
class _VoiceSession:
    """A websocket caller talking to STT -> LLM -> TTS -> playback."""

    system: str
    max_tokens: int = 20
    assistant_turns: list = field(default_factory=list)
    played: bytearray = field(default_factory=bytearray)

    async def run(self, *, utterances=1, barge_in=False):
        frames = _frames()
        Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
        PipelineParams = _import("pipecat.pipeline.worker.PipelineParams")
        LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
        universal = "pipecat.processors.aggregators.llm_response_universal"
        LLMContextAggregatorPair = _import(f"{universal}.LLMContextAggregatorPair")
        LLMUserAggregatorParams = _import(f"{universal}.LLMUserAggregatorParams")
        UserTurnStrategies = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")
        SpeechTimeout = _import("pipecat.turns.user_stop.SpeechTimeoutUserTurnStopStrategy")
        SileroVADAnalyzer = _import("pipecat.audio.vad.silero.SileroVADAnalyzer")
        VADParams = _import("pipecat.audio.vad.vad_analyzer.VADParams")
        OpenAISTTService = _import("pipecat.services.openai.stt.OpenAISTTService")
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
        aggregators = LLMContextAggregatorPair(
            LLMContext(messages=[{"role": "system", "content": self.system}]),
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
                user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeout(user_speech_timeout=0.5)]),
            ),
        )
        worker = _make_worker(
            Pipeline(
                [
                    transport.input(),
                    stt,
                    aggregators.user(),
                    _openai_llm(self.max_tokens),
                    _openai_tts(),
                    transport.output(),
                    aggregators.assistant(),
                ]
            ),
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )
        ready, turn_stopped = asyncio.Event(), asyncio.Event()

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

        async with _running(worker):
            await asyncio.wait_for(ready.wait(), 10)
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
                    elif index:
                        await self._wait_for_turns(turn_stopped, index)
                    await self._speak(websocket, serializer, pcm)
                await self._wait_for_turns(turn_stopped, utterances)
                reading.cancel()

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
def test_auto_instrument_pipecat_subprocess():
    pytest.importorskip("pipecat")
    verify_autoinstrument_script("test_auto_pipecat.py")


# Voice sessions


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_voice_session_traces_turns_speech_recognition_generation_and_playback(
    memory_logger, attachment_uploads
):
    from braintrust._audio.config import AudioEncoding, AudioRecordingOptions

    _instrument(
        capture_audio_attachments=True,
        audio_recording_options=AudioRecordingOptions(encoding=AudioEncoding(codec="pcm", container="wav")),
    )
    session = _VoiceSession(system="Reply with exactly: Hi.")
    await session.run()

    logs = memory_logger.pop()
    root = _single_span(logs, "session")
    assert root["metadata"]["audio.schema.version"] == "3.0.0"
    assert root["metadata"]["contrib.pipecat.terminal_frame"] == "EndFrame"
    assert root["metadata"]["contrib.pipecat.reason"] == "pipeline complete"
    user_turn = _single_span(logs, "user_turn")
    agent_turn = _single_span(logs, "agent_turn")
    speech = _single_span(logs, "user_speech")
    recognition = _single_span(logs, "stt")
    model = _single_span(logs, "llm_response")
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
    assert model["input"][0] == {"role": "system", "content": session.system}
    assert model["output"][0]["role"] == "assistant"
    assert model["output"][0]["content"].strip() == "Hi."
    assert set(model["output"][0]) == {"role", "content"}
    # Token usage belongs to provider spans; the request settings are here.
    assert not {"prompt_tokens", "completion_tokens", "tokens"} & set(model["metrics"])
    assert model["metrics"]["time_to_first_token"] >= 0
    assert {key: model["metadata"][key] for key in ("temperature", "max_completion_tokens", "reasoning_effort")} == {
        "temperature": 1.0,
        "max_completion_tokens": 20,
        "reasoning_effort": "none",
    }
    assert _link_targets(model, "uses_context") == [recognition["span_id"]]
    assert synthesis["metadata"]["model"] == "tts-1"
    assert synthesis["input"]["text"].strip() == "Hi."
    # Speech synthesis usage: no provider span carries it, so tts does.
    assert synthesis["metrics"]["characters"] == len(synthesis["input"]["text"])
    assert _link_targets(synthesis, "generated_by") == [model["span_id"]]
    assert _link_targets(playback, "generated_by") == [synthesis["span_id"]]
    assert playback["metadata"]["audio.speaker_ids"] == synthesis["metadata"]["audio.speaker_ids"]
    assert playback["metadata"]["contrib.pipecat.interrupted"] is False
    assert "audio.participant_ids" in playback["metadata"]

    # Both directions land in shared recordings; playback is what was sent.
    assert session.played
    segments = root["metadata"]["audio.segments"]
    assert segments and all(segment["state"] == "ready" for segment in segments.values()), segments
    for segment in segments.values():
        assert segment["channels"] == {"0": {"sources": ["session-input"]}, "1": {"sources": ["agent-output"]}}
        assert segment["attachment"]["ref"].startswith("/input/audio/")
    _assert_all_ended(logs)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_consecutive_voice_turns_reply_to_and_link_only_their_own_speech(memory_logger, attachment_uploads):
    from braintrust._audio.config import AudioEncoding, AudioRecordingOptions

    # User audio only, configured when wrapping: agent playback is not recorded.
    _instrument()
    assert wrap_pipeline_worker(
        _import("pipecat.pipeline.worker.PipelineWorker"),
        capture_user_audio_attachments=True,
        capture_agent_audio_attachments=False,
        audio_recording_options=AudioRecordingOptions(encoding=AudioEncoding(codec="pcm", container="wav")),
    ) is _import("pipecat.pipeline.worker.PipelineWorker")
    await _VoiceSession(system="Reply with exactly one word: Okay.").run(utterances=2)

    logs = memory_logger.pop()
    user_turns = _spans_named(logs, "user_turn")
    agent_turns = _spans_named(logs, "agent_turn")
    speeches = _spans_named(logs, "user_speech")
    recognitions = _spans_named(logs, "stt")
    models = _spans_named(logs, "llm_response")
    for synthesis in _spans_named(logs, "tts"):
        assert synthesis["metrics"]["characters"] == len(synthesis["input"]["text"])
    assert len(user_turns) == len(agent_turns) == len(speeches) == len(recognitions) == len(models) == 2
    for user_turn, agent_turn, speech, recognition, model in zip(
        user_turns, agent_turns, speeches, recognitions, models
    ):
        assert agent_turn["metadata"]["turn.reply_to"] == user_turn["span_id"]
        assert agent_turn["metadata"]["turn.incomplete"] is False
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
    )
    await session.run(utterances=2, barge_in=True)

    logs = memory_logger.pop()
    user_turns = _spans_named(logs, "user_turn")
    interrupted, reply = _spans_named(logs, "agent_turn")
    assert len(user_turns) == 2
    # The interrupted reply ends incomplete, with its playback cut short.
    assert interrupted["metadata"]["turn.incomplete"] is True
    assert interrupted["metadata"]["turn.reply_to"] == user_turns[0]["span_id"]
    cut = [row for row in _spans_named(logs, "agent_speech") if row["metadata"]["turn.id"] == interrupted["span_id"]]
    assert cut and cut[-1]["metadata"]["contrib.pipecat.interrupted"] is True
    assert interrupted["metrics"]["end"] <= user_turns[1]["metrics"]["end"]
    # The answer to the interruption is a separate, complete turn.
    assert reply["metadata"]["turn.incomplete"] is False
    assert reply["metrics"]["start"] >= interrupted["metrics"]["end"]
    _assert_all_ended(logs)


# Text pipelines


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_tool_request_result_and_followup_use_native_missing_tool_handler(memory_logger):
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
    )
    llm = _openai_llm(max_tokens=60)
    aggregator = LLMAssistantAggregator(context)
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    PipelineParams = _import("pipecat.pipeline.worker.PipelineParams")
    _instrument()
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
    # One agent turn spans the request, the tool and the follow-up.
    agent_turn = _single_span(logs, "agent_turn")
    assert agent_turn["metadata"]["turn.incomplete"] is False
    for operation in (requester, tool, followup):
        assert operation["metadata"]["turn.id"] == agent_turn["span_id"]
        assert operation["metrics"]["end"] <= agent_turn["metrics"]["end"]
    _assert_all_ended(logs)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_without_playback_or_assistant_aggregator_are_separate_turns(memory_logger):
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    first = LLMContext(messages=[{"role": "user", "content": "Reply with exactly: One."}])
    second = LLMContext(messages=[{"role": "user", "content": "Reply with exactly: Two."}])
    _instrument()
    await _run_text_pipeline([_openai_llm()], [first, second])

    logs = memory_logger.pop()
    turns = _spans_named(logs, "agent_turn")
    responses = _spans_named(logs, "llm_response")
    assert len(turns) == len(responses) == 2
    for turn, response in zip(turns, responses):
        assert turn["metadata"]["turn.incomplete"] is False
        assert response["metadata"]["turn.id"] == turn["span_id"]
        assert "turn.reply_to" not in turn["metadata"]
    assert [row["output"][0]["content"].strip() for row in responses] == ["One.", "Two."]
    _assert_all_ended(logs)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_context_edited_in_place_links_the_transcripts_it_now_includes(memory_logger):
    frames = _frames()
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])

    def first():
        # A text application feeds transcripts natively, then requests a reply.
        return [frames.TranscriptionFrame("Hello", "caller", "0"), frames.LLMContextFrame(context)]

    def edited():
        # The application extends the existing user message in place.
        context.get_messages()[0]["content"] = "Hello again"
        return [frames.TranscriptionFrame("again", "caller", "1"), frames.LLMContextFrame(context)]

    _instrument()
    await _run_text_pipeline([_openai_llm()], [first, edited])

    logs = memory_logger.pop()
    transcripts = {row["output"][0]["content"]: row["span_id"] for row in _spans_named(logs, "transcript_received")}
    first_response, second_response = _spans_named(logs, "llm_response")
    assert _link_targets(first_response, "uses_context") == [transcripts["Hello"]]
    assert _link_targets(second_response, "uses_context") == [transcripts["again"]]
    _assert_all_ended(logs)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_synthesis_without_output_transport_is_not_playback_and_respects_customizers(memory_logger):
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            if "input" in data:
                data["input"] = "[redacted]"
            return data

    context = LLMContext(messages=[{"role": "user", "content": "Reply with exactly: Private words."}])
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
    # Usage is counted whether or not content is redacted.
    assert _single_span(logs, "tts")["metrics"]["characters"] == len("Private words.")
    assert not _spans_named(logs, "agent_speech")
    assert "audio.segments" not in _single_span(logs, "session")["metadata"]
    _assert_all_ended(logs)


@pytest.mark.asyncio
async def test_failed_transport_writes_do_not_emit_agent_speech(memory_logger, vcr):
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
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
    context = LLMContext(messages=[{"role": "user", "content": "Reply with exactly: Private words."}])
    # No client connects: the real transport rejects every write after Pipecat
    # has emitted BotStartedSpeakingFrame. Reuse the same provider calls as the
    # synthesis-only test; no replacement transport or write method is needed.
    _instrument(capture_agent_audio_attachments=True)
    with vcr.use_cassette("test_synthesis_without_output_transport_is_not_playback_and_respects_customizers.yaml"):
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
    assert _single_span(logs, "agent_turn")["metadata"]["turn.incomplete"] is False
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


@pytest_asyncio.fixture
async def tracker_observer(memory_logger):
    # These regressions concern local frame bookkeeping, not provider payloads.
    # Use the real trackers, clock, recorder, services and native frame classes.
    _frames()
    from braintrust.integrations.pipecat.tracing.observer import BraintrustPipecatObserver

    observer = BraintrustPipecatObserver(capture_audio_attachments=False)
    yield observer
    await observer.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["turn", "response", "shutdown"])
async def test_queued_transcript_retains_text_and_receipt_time(memory_logger, tracker_observer, boundary):
    frames = _frames()
    STT = _import("pipecat.services.openai.stt.OpenAISTTService")
    stt = STT(api_key="unused")
    session = tracker_observer._start_session()
    with session.clock.event(0):
        session.user.start_speech(session.clock.now_ms())
        session.user.consume_audio(stt)
    with session.clock.event(100_000_000):
        session.user.stop_speech()
        session.user.transcribe(frames.TranscriptionFrame("Hello", "caller", "0"), stt)
        received = session.clock.wall()
    with session.clock.event(200_000_000):
        if boundary == "turn":
            session.user.start_turn()
        elif boundary == "response":
            session.user.flush_transcribed()
        else:
            session.user.close(session.clock.now_ms())
    recognition = _single_span(memory_logger.pop(), "stt")
    assert recognition["output"] == [{"role": "user", "content": "Hello"}]
    assert recognition["metrics"]["end"] == received


@pytest.mark.asyncio
async def test_untranscribed_stt_ends_with_its_user_turn(memory_logger, tracker_observer):
    STT = _import("pipecat.services.openai.stt.OpenAISTTService")
    session = tracker_observer._start_session()
    with session.clock.event(0):
        session.user.start_turn()
        session.user.start_speech(session.clock.now_ms())
        session.user.consume_audio(STT(api_key="unused"))
    with session.clock.event(100_000_000):
        session.user.stop_speech()
        session.user.stop_turn()
    with session.clock.event(10_000_000_000):
        session.user.close(session.clock.now_ms())
    logs = memory_logger.pop()
    recognition = _single_span(logs, "stt")
    turn = _single_span(logs, "user_turn")
    assert recognition["metrics"]["end"] == turn["metrics"]["end"]
    assert recognition["metadata"]["contrib.pipecat.incomplete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts,spaced,expected",
    [
        (["Hello.", "How are you?"], False, "Hello. How are you?"),
        (["Hello", " there", "!"], True, "Hello there!"),
    ],
)
async def test_tts_and_played_text_preserve_native_spacing(memory_logger, tracker_observer, parts, spaced, expected):
    frames = _frames()
    session = tracker_observer._start_session()
    session.agent.start_tts(frames.TTSStartedFrame(), _openai_tts())
    session.agent.start_playback()
    session.agent.played_audio(frames.OutputAudioRawFrame(bytes(320), 16000, 1))
    for text in parts:
        frame = frames.TTSTextFrame(text, aggregated_by="sentence")
        frame.includes_inter_frame_spaces = spaced
        session.agent.tts_text(frame)
        # Feed the same text through the actual observer playback callback.
        tracker_observer._observe_playback(session, frame)
    session.agent.stop_tts(frames.TTSStoppedFrame())
    session.agent.stop_playback()
    logs = memory_logger.pop()
    assert _single_span(logs, "tts")["input"]["text"] == expected
    assert _single_span(logs, "agent_speech")["output"] == [{"role": "assistant", "content": expected}]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["completed", "interrupted", "shutdown"])
async def test_playback_completion_distinguishes_shutdown_from_interruption(memory_logger, tracker_observer, boundary):
    frames = _frames()
    session = tracker_observer._start_session()
    with session.clock.event(0):
        session.agent.start_tts(frames.TTSStartedFrame(), _openai_tts())
        session.agent.start_playback()
        session.agent.played_audio(frames.OutputAudioRawFrame(bytes(320), 16000, 1))
    with session.clock.event(100_000_000):
        # Synthesis can finish while its audio is still playing.
        session.agent.stop_tts(frames.TTSStoppedFrame())
    with session.clock.event(200_000_000):
        if boundary == "completed":
            session.agent.stop_playback()
        elif boundary == "interrupted":
            session.agent.interrupt()
        await tracker_observer._finish()

    logs = memory_logger.pop()
    playback = _single_span(logs, "agent_speech")
    assert playback["metadata"]["contrib.pipecat.interrupted"] is (boundary == "interrupted")
    if boundary == "shutdown":
        assert playback["metadata"]["contrib.pipecat.incomplete"] is True
    else:
        assert "contrib.pipecat.incomplete" not in playback["metadata"]
    assert playback["metadata"]["audio.selections"]["selection-0"] == {
        "start_offset_ms": 0,
        "end_offset_ms": 200,
    }
    _assert_all_ended(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_recognition_scope_restores_context_after_errors_and_cancellation(tracker_observer, failure):
    STT = _import("pipecat.services.openai.stt.OpenAISTTService")
    stt = STT(api_key="unused")
    session = tracker_observer._start_session()
    session.user.start_turn()
    session.user.consume_audio(stt)

    with logger.start_span(name="caller") as caller:
        with pytest.raises(failure):
            with tracker_observer._current_recognition(stt):
                recognition = logger.current_span()
                assert recognition is not caller
                # Reentrant wrapping must also leave the outer scope intact.
                with tracker_observer._current_recognition(stt):
                    assert logger.current_span() is recognition
                assert logger.current_span() is recognition
                await asyncio.sleep(0)
                raise failure("operation stopped")
        assert logger.current_span() is caller


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
        assert recognition["metrics"]["start"] <= call["metrics"]["start"]
        assert call["metrics"]["end"] <= recognition["metrics"]["end"]
        assert "tokens" not in recognition["metrics"]
    # Subsequent model calls must retain their own scope, never the STT scope.
    for response, call in zip(_spans_named(logs, "llm_response"), _spans_named(logs, "Chat Completion")):
        assert call["span_parents"] == [response["span_id"]]
    assert all(row["metrics"].get("end") is not None for row in logs)
