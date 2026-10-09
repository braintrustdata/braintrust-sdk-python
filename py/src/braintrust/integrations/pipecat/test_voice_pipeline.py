"""Provider-backed voice contract: HTTP cassettes, real Pipecat pipeline and export."""

# pylint: disable=import-error
import asyncio
import io
import json
import os
import wave
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from braintrust._audio import RecordingOptions
from braintrust.integrations.pipecat import setup_pipecat
from braintrust.integrations.pipecat.test_pipecat import (
    _make_worker,
    _single_span,
    _worker_runner_kwargs,
)
from braintrust.integrations.pipecat.test_pipecat import memory_logger as memory_logger
from braintrust.test_helpers import init_test_logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner


def _request_payload(request):
    content_type = request.headers.get("Content-Type", "")
    body = request.body
    if "multipart/form-data" in content_type:
        message = BytesParser(policy=default).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
        return [(part.get("Content-Disposition"), part.get_payload(decode=True)) for part in message.iter_parts()]
    return json.loads(body)


@pytest.fixture(scope="module")
def vcr(vcr):
    # Compare audio bytes, prompts, model/settings and tool results. Multipart
    # boundaries are random HTTP framing, not part of the request's meaning.
    def payload_matches(actual, expected):
        assert _request_payload(actual) == _request_payload(expected)

    from braintrust.integrations.pipecat._test_audio_cassettes import AudioPersister

    vcr.register_persister(AudioPersister)
    vcr.register_matcher("voice_payload", payload_matches)
    return vcr


@pytest.fixture
def vcr_cassette_name(request, vcr_cassette_dir, vcr):
    name = request.node.originalname or request.node.name
    if vcr.record_mode != "all":
        yield name
        return
    from braintrust.integrations.pipecat._test_audio_cassettes import publish_cassette

    # A fresh temporary path avoids VCR appending old interactions. The
    # dependent VCR fixture saves here before this fixture's teardown runs.
    target = Path(vcr_cassette_dir) / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".recording-", dir=target.parent) as directory:
        staged = Path(directory) / target.name
        yield str(staged)
        if getattr(request.node, "voice_recording_validated", False):
            publish_cassette(staged, target)


class MemoryInput(BaseInputTransport):
    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)


class MemoryOutput(BaseOutputTransport):
    """Local application transport: accept output without a telephone or playback delay."""

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        self.last_audio_frame = frame
        return True


async def _run_conversation(worker, finished):
    runner = WorkerRunner(**_worker_runner_kwargs())
    task = asyncio.create_task(runner.run(worker))
    try:
        await asyncio.wait_for(finished.wait(), 45)
        await worker.stop_when_done()
        await asyncio.wait_for(task, 10)
    finally:
        if not task.done():
            await worker.cancel()
            await task


def _decode_recordings(rows):
    spans = {row["span_id"]: row for row in rows}
    recordings = {}
    for owner in rows:
        for descriptor in owner.get("metadata", {}).get("audio.recordings", []):
            assert descriptor["state"] == "ready", descriptor
            attachment = descriptor["attachment"]
            value = spans[attachment["span_id"]]
            for part in attachment["ref"].strip("/").split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                value = value[int(part)] if isinstance(value, list) else value[part]
            with wave.open(io.BytesIO(value.data)) as audio:
                rate = audio.getframerate()
                channels = audio.getnchannels()
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").reshape(-1, channels)
            assert len(samples) and np.any(samples), "recording must contain speech"
            assert descriptor["duration_ms"] == pytest.approx(len(samples) / rate * 1000, abs=1000 / rate)
            assert descriptor["channel_count"] == channels
            key = (owner["span_id"], descriptor["id"])
            assert key not in recordings, "recording IDs must be unique within a span"
            recordings[key] = (samples, rate)
    assert recordings
    return recordings


def _assert_selections(owner, root, recordings, channel):
    selections = owner["metadata"]["audio.selections"]
    assert selections
    pieces = []
    for selection in selections:
        assert selection["recording_span_id"] == root["span_id"]
        assert selection["channel_index"] == channel
        samples, rate = recordings[(selection["recording_span_id"], selection["recording_id"])]
        start, end = selection["start_offset_ms"], selection["end_offset_ms"]
        assert 0 <= start < end <= len(samples) / rate * 1000
        assert 0 <= channel < samples.shape[1]
        pieces.append(samples[round(start * rate / 1000) : round(end * rate / 1000), channel])
    selected = np.concatenate(pieces)
    assert np.any(selected), "selection must point to speech, not silence"
    if channel == 1:
        # At 24 kHz no resampling is needed: navigating to the agent span
        # must select the same samples as its independently attached clip.
        clips = owner["metadata"]["audio.recordings"]
        assert len(clips) == 1
        clip, clip_rate = recordings[(owner["span_id"], clips[0]["id"])]
        assert clip_rate == rate
        np.testing.assert_array_equal(selected, clip[:, 0])
    if not owner["metadata"].get("audio.recordings"):
        return
    descriptor = owner["metadata"]["audio.recordings"][0]
    timeline = descriptor["timeline"]
    assert timeline["origin_unix_ms"] == root["metadata"]["audio.recordings"][0]["timeline"]["origin_unix_ms"]
    clip, clip_rate = recordings[(owner["span_id"], descriptor["id"])]
    assert timeline["ranges"]
    for interval in timeline["ranges"]:
        a, b = interval["recording_start_ms"], interval["recording_end_ms"]
        assert 0 <= a < b <= len(clip) / clip_rate * 1000 + 0.001
        assert abs((b - a) - (interval["timeline_end_ms"] - interval["timeline_start_ms"])) <= 1000 / rate
    if channel == 0:
        assert timeline["ranges"][-1]["recording_end_ms"] < len(clip) / clip_rate * 1000  # STT padding


def _assert_shutdown_audio(output, root, recordings):
    # Exercise Pipecat's real EndFrame path, including its default single
    # two-second silence frame. The exported recording must retain that tail.
    frame = output.last_audio_frame
    assert len(frame.audio) == 2 * frame.sample_rate * frame.num_channels * 2
    assert not any(frame.audio)
    descriptor = root["metadata"]["audio.recordings"][-1]
    samples, rate = recordings[(root["span_id"], descriptor["id"])]
    assert len(samples) > rate * 2
    assert np.any(samples[: -rate * 2, 1]), "agent speech must survive shutdown"
    assert not np.any(samples[-rate * 2 :, 1]), "closing silence must be retained"


@pytest.mark.vcr(match_on=["method", "uri", "voice_payload"])
@pytest.mark.asyncio
@pytest.mark.parametrize("early_metrics", [False, True])
async def test_cascade_voice_conversation(memory_logger, request, monkeypatch, early_metrics):
    setup_pipecat(capture_audio_attachments=True, audio_format="wav", recording_options=RecordingOptions())
    init_test_logger("test-project-pipecat-py-tracing")
    tool_calls = []

    async def lookup_order(params):
        tool_calls.append(params.arguments)
        await params.result_callback({"order_id": params.arguments["order_id"], "delivery": "Friday"})

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": "You are an English order assistant. Always call lookup_order for the supplied order number. "
                "After its result, answer only: Your order arrives Friday.",
            }
        ],
        tools=ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="lookup_order",
                    description="Look up an order",
                    properties={"order_id": {"type": "string"}},
                    required=["order_id"],
                    handler=lookup_order,
                )
            ]
        ),
    )
    pair = LLMContextAggregatorPair(context)
    params = TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    source, output = MemoryInput(params), MemoryOutput(params)
    key = os.environ["OPENAI_API_KEY"]
    stt = OpenAISTTService(api_key=key)
    llm = OpenAILLMService(api_key=key, settings=OpenAILLMService.Settings(model="gpt-4.1-mini", temperature=0))
    tts = OpenAITTSService(
        api_key=key,
        settings=OpenAITTSService.Settings(model="gpt-4o-mini-tts", voice="alloy"),
    )
    if early_metrics:
        # Exercise native audio-queue scheduling without replacing the service,
        # metric emission, or provider responses.
        usage_emitted = asyncio.Event()
        original_push = tts.push_frame

        async def delayed_start(frame, *args, **kwargs):
            if type(frame).__name__ == "TTSStartedFrame":
                await usage_emitted.wait()  # Covered by the conversation's overall deadline.
                await asyncio.sleep(0)
            result = await original_push(frame, *args, **kwargs)
            if type(frame).__name__ == "MetricsFrame" and any(
                type(metric).__name__ == "TTSUsageMetricsData" for metric in frame.data
            ):
                usage_emitted.set()
            return result

        monkeypatch.setattr(tts, "push_frame", delayed_start)
    worker = _make_worker(
        Pipeline([source, stt, pair.user(), llm, tts, output, pair.assistant()]),
        params=PipelineParams(
            audio_in_sample_rate=16000, audio_out_sample_rate=24000, enable_metrics=True, enable_usage_metrics=True
        ),
    )
    finished = asyncio.Event()

    @pair.assistant().event_handler("on_assistant_turn_stopped")
    async def turn_finished(aggregator, message):
        if "friday" in message.content.lower():
            finished.set()

    @worker.event_handler("on_pipeline_started")
    async def feed(worker, frame):
        # A fixed recording and speech boundaries are the test input. Recognition,
        # model/tool behavior, synthesis and turn callbacks all run through Pipecat.
        with wave.open(str(Path(__file__).parent / "voice/fixtures/order.wav")) as audio:
            pcm = audio.readframes(audio.getnframes())
        await source.push_frame(VADUserStartedSpeakingFrame())
        for offset in range(0, len(pcm), 640):
            await source.push_audio_frame(InputAudioRawFrame(pcm[offset : offset + 640], 16000, 1))
            await asyncio.sleep(0.02)
        await source.push_frame(VADUserStoppedSpeakingFrame())

    await _run_conversation(worker, finished)
    rows = memory_logger.pop()
    assert not [r for r in rows if r.get("error")]
    assert len(tool_calls) == 1
    assert "1042" in tool_calls[0]["order_id"].replace(" ", "")
    root = _single_span(rows, "pipecat.pipeline")
    recordings = _decode_recordings(rows)
    _assert_shutdown_audio(output, root, recordings)
    user = _single_span(rows, "user_turn")
    recognition = _single_span(rows, "stt")
    tool = _single_span(rows, "lookup_order")
    synthesis = _single_span(rows, "tts")
    assert synthesis["input"]["text"]
    # Request identity must survive early metrics, without root duplicates.
    tts_measurements = [
        (row, measurement)
        for row in rows
        for measurement in row.get("metadata", {}).get("contrib.pipecat.measurements", [])
        if measurement["processor"].startswith("OpenAITTSService")
    ]
    for metric_type in ("TTSUsageMetricsData", "ProcessingMetricsData"):
        matching = [(row, metric) for row, metric in tts_measurements if metric["type"] == metric_type]
        assert len(matching) == 1
        owner, metric = matching[0]
        assert owner["span_id"] == synthesis["span_id"]
        assert metric["value"] > 0
    assert all(
        metric["processor"].startswith("OpenAITTSService")
        for metric in synthesis.get("metadata", {}).get("contrib.pipecat.measurements", [])
    )
    for row in rows:
        assert not any(key.startswith("pipecat.") for key in row.get("metadata", {}))
        events = row.get("metadata", {}).get("contrib.pipecat.events", [])
        assert not any(e["contrib.pipecat.frame.type"] in {"StartFrame", "EndFrame", "MetricsFrame"} for e in events)
    models = [r for r in rows if r.get("span_attributes", {}).get("name") == "llm_response"]
    assert models
    assert recognition["span_parents"] == [user["span_id"]]
    assert any(tool["span_parents"] == [r["span_id"]] for r in models)
    assert any(
        r["span_id"] not in tool["span_parents"] and "friday" in json.dumps(r["output"]).lower() for r in models
    ), "a model response must deliver the tool result to the caller"
    assert "friday" in synthesis["metadata"]["contrib.pipecat.text"].lower()
    for model in models:
        assert model["metadata"]["model"] == "gpt-4.1-mini"
        assert model["metrics"]["tokens"] > 0
    _assert_selections(user, root, recordings, 0)
    _assert_selections(synthesis, root, recordings, 1)
    request.node.voice_recording_validated = True


@pytest.mark.asyncio
async def test_realtime_voice_conversation(memory_logger, monkeypatch, request, vcr_cassette_dir):
    from braintrust.integrations.pipecat._test_websocket import WebSocketCassette
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.services.openai.realtime import events
    from pipecat.services.openai.realtime import llm as realtime_module

    cassette = WebSocketCassette(
        Path(vcr_cassette_dir) / "test_realtime_voice_conversation.json",
        record=request.config.getoption("--vcr-record") == "all",
    )
    original_connect = realtime_module.websocket_connect

    async def connect(**kwargs):
        return await cassette.connect(original_connect, **kwargs)

    monkeypatch.setattr(realtime_module, "websocket_connect", connect)
    setup_pipecat(capture_audio_attachments=True, audio_format="wav")
    init_test_logger("test-project-pipecat-py-tracing")
    tool_calls = []

    async def lookup_order(params):
        tool_calls.append(params.arguments)
        await params.result_callback({"order_id": params.arguments["order_id"], "delivery": "Friday"})

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": "Speak English. Start by saying exactly: Hello, what is your order number? "
                "When the user gives an order number, always call lookup_order. "
                "After its result, say only: Your order arrives Friday.",
            }
        ],
        tools=ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="lookup_order",
                    description="Look up an order",
                    properties={"order_id": {"type": "string"}},
                    required=["order_id"],
                    handler=lookup_order,
                )
            ]
        ),
    )
    pair = LLMContextAggregatorPair(context)
    params = TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    source, output = MemoryInput(params), MemoryOutput(params)
    service = realtime_module.OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=realtime_module.OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime",
            session_properties=events.SessionProperties(
                audio=events.AudioConfiguration(
                    input=events.AudioInput(
                        transcription=events.InputAudioTranscription(model="gpt-4o-mini-transcribe", language="en"),
                        turn_detection=events.TurnDetection(),
                    ),
                    output=events.AudioOutput(voice="alloy"),
                )
            ),
        ),
    )
    worker = _make_worker(
        Pipeline([source, pair.user(), service, output, pair.assistant()]),
        params=PipelineParams(
            audio_in_sample_rate=24000, audio_out_sample_rate=24000, enable_metrics=True, enable_usage_metrics=True
        ),
    )
    greeted, finished = asyncio.Event(), asyncio.Event()

    @pair.assistant().event_handler("on_assistant_turn_stopped")
    async def turn_finished(aggregator, message):
        if "friday" in message.content.lower():
            finished.set()
        else:
            greeted.set()

    @worker.event_handler("on_pipeline_started")
    async def feed(worker, frame):
        await worker.queue_frame(LLMRunFrame())
        await asyncio.wait_for(greeted.wait(), 20)
        # Fixed wire-rate input keeps strict cassette matching independent of
        # platform-specific resampler rounding. Recording uses this same input.
        with wave.open(str(Path(__file__).parent / "voice/fixtures/order-24khz.wav")) as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (24000, 1, 2)
            pcm = audio.readframes(audio.getnframes())
        pcm += b"\0" * 48000  # Server VAD detects the end of the spoken request.
        for offset in range(0, len(pcm), 960):
            await source.push_audio_frame(InputAudioRawFrame(pcm[offset : offset + 960], 24000, 1))
            await asyncio.sleep(0.02)

    await _run_conversation(worker, finished)
    cassette.verify()
    rows = memory_logger.pop()
    assert not [r for r in rows if r.get("error")]
    assert len(tool_calls) == 1
    assert "1042" in tool_calls[0]["order_id"].replace(" ", "")
    root = _single_span(rows, "pipecat.pipeline")
    recordings = _decode_recordings(rows)
    _assert_shutdown_audio(output, root, recordings)
    user = _single_span(rows, "user_turn")
    tool = _single_span(rows, "lookup_order")
    models = [r for r in rows if r.get("span_attributes", {}).get("name") == "llm_response"]
    assert models
    assert any(tool["span_parents"] == [r["span_id"]] for r in models)
    assert any(
        r["span_id"] not in tool["span_parents"] and "friday" in json.dumps(r["output"]).lower() for r in models
    ), "a model response must deliver the tool result to the caller"
    _assert_selections(user, root, recordings, 0)
    for model in models:
        assert model["metadata"]["provider"] == "openai"
        assert model["metadata"]["model"].startswith("gpt-realtime")
        assert model["metrics"]["tokens"] > 0
        assert model["metadata"].get("contrib.pipecat.text") != ""
        assert model["output"]
    assert not [r for r in rows if r.get("span_attributes", {}).get("name") in ("stt", "tts")]
    audio_spans = [r for r in rows if r.get("span_attributes", {}).get("name") == "pipecat.audio_output"]
    assert audio_spans
    for audio_span in audio_spans:
        assert audio_span["metadata"]["openai.response.id"]
        _assert_selections(audio_span, root, recordings, 1)
    if cassette.record:
        cassette.save()
