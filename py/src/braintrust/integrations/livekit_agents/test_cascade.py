import asyncio
import inspect

import pytest
from braintrust import logger
from braintrust._audio.config import AudioEncoding, AudioRecordingOptions
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.livekit_agents.compat import STATE_ATTR
from braintrust.integrations.openai import OpenAIIntegration


def named(rows, name):
    return [row for row in rows if row.get("span_attributes", {}).get("name") == name]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preemptive,directions,provider_mode,manual",
    [
        (False, "both", "enabled", False),
        (True, "both", "enabled", False),
        (False, "user", "enabled", False),
        (False, "both", "disabled", False),
        (False, "both", "enabled", True),
    ],
)
@pytest.mark.vcr(cassette_name="test_livekit_agents_e2e_parenting_under_custom_span")
async def test_consumed_input_drives_real_cascade(
    rooms, memory_logger, monkeypatch, preemptive, directions, provider_mode, manual
):
    from livekit import rtc
    from livekit.agents import Agent, AgentSession
    from livekit.plugins import openai, silero

    if manual:
        from livekit.agents.voice.audio_recognition import AudioRecognition

        if not hasattr(AudioRecognition, "_take_turn_snapshot"):
            pytest.skip("Pending manual snapshots require LiveKit 1.8.12")

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
    OpenAIIntegration.setup()
    if provider_mode != "enabled":
        from openai.resources.audio.transcriptions import AsyncTranscriptions
        from openai.resources.chat.completions import AsyncCompletions

        monkeypatch.setattr(AsyncTranscriptions, "create", inspect.unwrap(AsyncTranscriptions.create))
        monkeypatch.setattr(AsyncCompletions, "create", inspect.unwrap(AsyncCompletions.create))
    tts = openai.TTS(model="gpt-4o-mini-tts", voice="ash")
    stt = openai.STT(model="gpt-4o-mini-transcribe")
    recognition_metrics = []
    stt.on("metrics_collected", recognition_metrics.append)
    llm = openai.LLM(model="gpt-4.1-mini", max_completion_tokens=8)
    model_metrics = []
    llm.on("metrics_collected", model_metrics.append)
    vad = silero.VAD.load()
    heard = asyncio.Event()
    eou_metrics = []
    input_source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("caller-microphone", input_source)
    await rooms[1].local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    with logger.start_span(name="voice_call") as outer:
        options = {"turn_detection": "manual", "min_endpointing_delay": 0.8} if manual else {}
        async with (
            llm,
            tts,
            stt,
            AgentSession(vad=vad, stt=stt, tts=tts, preemptive_generation=preemptive, **options) as session,
        ):
            frames = [chunk.frame async for chunk in tts.synthesize("hello there")]
            assert frames

            def conversation(event):
                if event.item.type == "message" and event.item.role == "assistant":
                    heard.set()

            session.on("conversation_item_added", conversation)
            session.on(
                "metrics_collected",
                lambda event: eou_metrics.append(event.metrics) if event.metrics.type == "eou_metrics" else None,
            )
            await session.start(
                Agent(instructions="Reply with exactly: hi there", llm=llm), room=rooms[0], record=False
            )
            owner = getattr(session, STATE_ATTR)
            committed = None
            if manual:
                paused = asyncio.Event()
                transcribed = asyncio.Event()

                def user_state(event):
                    if event.new_state == "listening":
                        paused.set()

                session.on("user_state_changed", user_state)

                def transcript(event):
                    if event.is_final:
                        transcribed.set()

                session.on("user_input_transcribed", transcript)

                async def commit_and_clear():
                    await paused.wait()
                    await transcribed.wait()
                    await session.commit_user_turn(transcript_timeout=0, stt_flush_duration=0)
                    session.clear_user_turn()

                committed = asyncio.create_task(commit_and_clear())
            try:
                silence = rtc.AudioFrame(
                    data=bytes(2400 * 2), sample_rate=24000, num_channels=1, samples_per_channel=2400
                )
                for _ in range(5):
                    await input_source.capture_frame(silence)
                for frame in frames:
                    await input_source.capture_frame(frame)
                for _ in range(15):
                    await input_source.capture_frame(silence)
                await asyncio.wait_for(heard.wait(), 15)
                if committed is not None:
                    await committed
                await session.drain()
                assert owner.input_end is not None
            finally:
                if committed is not None and not committed.done():
                    committed.cancel()
                    await asyncio.gather(committed, return_exceptions=True)
                await input_source.aclose()
        assert logger.current_span() is outer
    memory_logger.flush()
    rows = memory_logger.pop()
    assert named(rows, "user_speech")
    recognitions = named(rows, "stt")
    assert len(recognitions) == len(recognition_metrics) == 1
    recognition, measured = recognitions[0], recognition_metrics[0]
    assert not measured.streamed
    # Bracket the real provider request, rather than timing transcript delivery.
    start, end = recognition["metrics"]["start"], recognition["metrics"]["end"]
    assert start < measured.timestamp <= end
    assert start <= measured.timestamp - measured.duration + 0.001
    # The wrapper includes span setup and scheduling around LiveKit's measured
    # request. Assert containment, not a fixed overhead budget on shared CI.
    assert recognition["metadata"]["model"] == stt.model
    assert recognition["metadata"]["provider"] == stt.provider
    transcriptions = named(rows, "Transcription")
    assert len(transcriptions) == (1 if provider_mode == "enabled" else 0)
    if transcriptions:
        assert transcriptions[0]["span_parents"] == [recognition["span_id"]]
        assert start <= transcriptions[0]["metrics"]["start"]
        assert transcriptions[0]["metrics"]["end"] <= end
    fallback_stt = named(rows, "livekit.stt")
    fallback_tts = named(rows, "livekit.tts")
    assert not fallback_stt and not fallback_tts
    assert recognition["output"] == [{"role": "user", "content": "Hello there."}]
    models = named(rows, "llm_response")
    assert models and len(models) == len(model_metrics)
    requests = named(rows, "Chat Completion")
    assert len(requests) == (len(models) if provider_mode == "enabled" else 0)
    if requests:
        assert {row["span_parents"][0] for row in requests} == {row["span_id"] for row in models}
        assert sum(row["metrics"].get("prompt_cached_tokens", 0) for row in requests) == sum(
            measured.prompt_cached_tokens for measured in model_metrics
        )
    assert all("prompt_tokens" not in row["metrics"] for row in models)
    assert all("cache_read_input_tokens" not in row["metrics"] for row in requests)
    assert named(rows, "tts")
    assert all(set(row["metrics"]) <= {"start", "end"} for row in recognitions + models + named(rows, "tts"))
    assert named(rows, "agent_speech")
    tts_ids = {row["span_id"] for row in named(rows, "tts")}
    for speech in named(rows, "agent_speech"):
        relationships = speech["metadata"]["links"]
        assert len(relationships) == 1
        assert relationships[0]["context"]["spanId"] in tts_ids
        assert relationships[0]["attributes"] == {"braintrust.relationship.type": "generated_by"}
    user = next(row for row in named(rows, "user_turn") if row.get("input"))
    assert user["input"][0]["content"] == "Hello there."
    if manual:
        # clear_user_turn detaches the buffer before the pending snapshot commits.
        # Speech retains its audio selection; the detached turn has no buffer selection.
        assert all(row["metadata"]["audio.selections"] for row in named(rows, "user_speech"))
    else:
        assert user["metadata"]["audio.selections"]
    if not preemptive:
        assert len(eou_metrics) == 1
    if eou_metrics:
        reported = eou_metrics[0]
        assert user["metadata"]["contrib.livekit.end_of_turn_delay"] == pytest.approx(reported.end_of_utterance_delay)
        assert user["metadata"]["contrib.livekit.transcription_delay"] == pytest.approx(reported.transcription_delay)
        assert (
            user["metadata"]["contrib.livekit.on_user_turn_completed_delay"] == reported.on_user_turn_completed_delay
        )
    (wait,) = named(rows, "eou_wait")
    assert wait["span_parents"] == [user["span_id"]]
    assert wait["span_attributes"]["type"] == "task"
    assert wait["metadata"]["eou.outcome"] == "committed"
    assert wait["metadata"]["eou.speech_end_source"] == "vad"
    assert wait["metadata"]["turn.id"] == user["span_id"]
    assert not named(rows, "livekit.eou_wait")
    if manual:
        assert wait["metrics"]["end"] - wait["metrics"]["start"] >= 0.8
    if preemptive:
        assert any(model["metrics"]["start"] < wait["metrics"]["end"] for model in models)
    assert "eou.end_reason" not in wait["metadata"]
    assert "eou.detection_methods" not in wait["metadata"]
    assert wait["metrics"]["end"] - wait["metrics"]["start"] == pytest.approx(
        user["metadata"]["contrib.livekit.end_of_turn_delay"], abs=0.02
    )
    for speech in named(rows, "user_speech"):
        assert speech["metadata"]["contrib.livekit.vad_silence_duration"] >= 0
        assert speech["metadata"]["contrib.livekit.vad_inference_duration"] >= 0
    by_id = {row["span_id"]: row for row in rows}
    for speech in named(rows, "user_speech"):
        parent = by_id[speech["span_parents"][0]]
        assert parent["metrics"]["start"] <= speech["metrics"]["start"]
    assert any(row["metadata"].get("turn.reply_to") == user["span_id"] for row in named(rows, "agent_turn"))
    root = named(rows, "session")[0]
    assert root["span_parents"] == [outer.span_id]
    assert root["metadata"]["audio.schema.version"] == "3.2.0"
    assert isinstance(root["metadata"]["audio.timeline"]["origin_unix_ms"], int)
    for row in rows:
        if row["span_id"] != root["span_id"]:
            assert "audio.schema.version" not in row.get("metadata", {}), row["span_attributes"]["name"]
    assert uploads
    assert all(
        len(segment["channels"]) == (2 if directions == "both" else 1)
        for segment in root["metadata"]["audio.segments"].values()
    )
    assert owner.native is None
    assert not owner.eou_snapshots and not owner.eou_active
    assert not owner.native_users and not owner.speech_ends
    assert not owner.pending_users and not owner.requests and not owner.tool_results
    assert owner.user is owner.user_speech is owner.stt is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["openai", "inference"])
@pytest.mark.parametrize(
    "capture,cancel_stream",
    [
        pytest.param("off", False, id="off"),
        pytest.param("full", False, id="full"),
        pytest.param("bounded", False, id="bounded"),
        pytest.param("off", True, id="cancelled"),
    ],
)
async def test_native_streaming_stt_owns_model_child(
    memory_logger, monkeypatch, vcr_cassette_dir, backend, capture, cancel_stream
):
    import wave
    from pathlib import Path

    from livekit import rtc
    from livekit.agents import Agent, AgentSession, inference
    from livekit.agents.utils import http_context
    from livekit.plugins import openai
    from openai.resources.audio.transcriptions import AsyncTranscriptions

    from . import tracing
    from .test_realtime_helpers import RealtimeCassette

    limit = 4096 if capture == "bounded" else 4 * 1024 * 1024
    monkeypatch.setattr(tracing, "_MAX_STT_INPUT_BYTES", limit)
    setup_livekit_agents()
    if capture == "off":
        OpenAIIntegration.setup()
    else:
        monkeypatch.setattr(AsyncTranscriptions, "create", inspect.unwrap(AsyncTranscriptions.create))
    monkeypatch.setenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", "false" if capture == "off" else "true")
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", "false")
    uploads = []

    def upload(attachment):
        uploads.append(attachment)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    heard = asyncio.Event()
    measured = asyncio.Event()
    transcripts = []
    measurements = []

    def transcription(event):
        if event.is_final:
            transcripts.append(event.transcript)
            heard.set()

    def metrics(event):
        if event.streamed:
            measurements.append(event)
            measured.set()

    filename = "native-stt.json" if backend == "openai" else "inference-stt.json"
    operation = "stt" if backend == "inference" else None
    async with http_context.open(), RealtimeCassette(Path(vcr_cassette_dir) / filename, operation=operation) as tape:
        stt = (
            openai.STT(model="gpt-4o-mini-transcribe", use_realtime=True, base_url=tape.base_url)
            if backend == "openai"
            else inference.STT(model="deepgram/nova-3", base_url=tape.base_url, **tape.inference_credentials())
        )
        async with stt:
            stt.on("metrics_collected", metrics)
            async with AgentSession(stt=stt, turn_detection="manual") as session:
                session.on("user_input_transcribed", transcription)
                await session.start(Agent(instructions="Listen to the caller."))
                with wave.open(str(Path(__file__).parents[2] / "fixtures/test_speech.wav"), "rb") as speech:
                    rate, channels = speech.getframerate(), speech.getnchannels()
                    pcm = speech.readframes(speech.getnframes())
                pcm = bytes(rate * channels) + pcm + bytes(rate * channels * 4)
                for offset in range(0, len(pcm), rate // 50 * channels * 2):
                    data = pcm[offset : offset + rate // 50 * channels * 2]
                    session._activity.push_audio(
                        rtc.AudioFrame(
                            data=data,
                            sample_rate=rate,
                            num_channels=channels,
                            samples_per_channel=len(data) // (channels * 2),
                        )
                    )
                    await asyncio.sleep(0)
                await asyncio.wait_for(heard.wait(), 30)
                await asyncio.wait_for(measured.wait(), 10)
                if cancel_stream:
                    # Cancel the real STT pump while the session is still open.
                    assert not getattr(session, STATE_ATTR).closing
                    pump = session._activity._audio_recognition._stt_pipeline._pump_task
                    assert not pump.done()
                    pump.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pump
    memory_logger.flush()
    rows = memory_logger.pop()
    wrappers, children = named(rows, "stt"), named(rows, "livekit.stt")
    assert len(wrappers) == len(children) == 1
    wrapper, child = wrappers[0], children[0]
    for completed in (wrapper, child):
        if cancel_stream:
            assert completed["metadata"]["contrib.livekit.cancelled"] is True
            assert completed["metadata"]["contrib.livekit.incomplete"] is True
        else:
            assert "contrib.livekit.cancelled" not in completed["metadata"]
            assert "contrib.livekit.incomplete" not in completed["metadata"]
        assert completed["metrics"]["end"] >= completed["metrics"]["start"]
    assert wrapper["span_parents"] == [named(rows, "session")[0]["span_id"]]
    assert child["span_parents"] == [wrapper["span_id"]]
    assert child["span_attributes"]["type"] == "llm"
    assert child["context"]["span_origin"]["instrumentation"]["name"] == "livekit-agents-auto"
    assert child["metadata"]["model"] == stt.model
    assert child["metadata"]["provider"] == stt.provider
    reports = child["metadata"]["contrib.livekit.recognition_usage"]
    assert sum(report["audio_duration"] for report in reports) == pytest.approx(
        sum(event.audio_duration for event in measurements)
    )
    assert "usage_seconds" not in child["metrics"]
    assert child["output"]["content"] == [{"type": "text", "text": text} for text in transcripts]
    if backend == "openai":
        assert child["metrics"]["prompt_tokens"] == sum(event.input_tokens for event in measurements)
    else:
        assert all(event.input_tokens == event.output_tokens == 0 for event in measurements)
        assert not {"prompt_tokens", "completion_tokens", "tokens"} & child["metrics"].keys()
    assert set(wrapper["metrics"]) <= {"start", "end"}
    assert not named(rows, "Transcription")
    assert child["input"]["operation"] == "transcribe"
    if capture == "off":
        assert child["input"] == {"operation": "transcribe"}
        assert not uploads and not memory_logger.upload_attempts
    else:
        import io

        attachment = child["input"]["content"][0]["file"]["file_data"]
        reference = attachment.reference if isinstance(attachment, logger.Attachment) else attachment
        uploaded = next(item for item in uploads + memory_logger.upload_attempts if item.reference == reference)
        with wave.open(io.BytesIO(uploaded.data), "rb") as captured:
            assert captured.getframerate() == rate
            assert captured.getnchannels() == channels
            captured_pcm = captured.readframes(captured.getnframes())
        assert captured_pcm == pcm[:limit]
        assert len(captured_pcm) <= limit
        assert child["metadata"].get("contrib.livekit.input_audio_truncated", False) == (capture == "bounded")
    assert child["metrics"]["time_to_first_token"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, True])
async def test_native_tts_observes_transformed_input(rooms, memory_logger, monkeypatch, vcr_cassette_dir, capture):
    from pathlib import Path

    from livekit.agents import Agent, AgentSession, inference
    from livekit.agents.utils import http_context
    from livekit.agents.voice.transcription.text_transforms import replace

    from .test_realtime_helpers import RealtimeCassette

    monkeypatch.setenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", "false")
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", str(capture).lower())
    uploads = []

    def upload(attachment):
        uploads.append(attachment)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", upload)
    setup_livekit_agents()
    async with (
        http_context.open(),
        RealtimeCassette(Path(vcr_cassette_dir) / "inference-tts.json", operation="tts") as tape,
    ):
        async with (
            inference.TTS(model="cartesia/sonic-3", base_url=tape.base_url, **tape.inference_credentials()) as tts,
            AgentSession(tts=tts, tts_text_transforms=[replace({"placeholder": "Hello there"})]) as session,
        ):
            await session.start(Agent(instructions="Say hello."), room=rooms[0], record=False)
            await session.say("placeholder.", allow_interruptions=False)
            await session.drain()
    memory_logger.flush()
    rows = memory_logger.pop()
    children = named(rows, "livekit.tts")
    assert len(children) == 1
    child = children[0]
    assert child["input"] == {"operation": "speech", "prompt": "Hello there."}
    sent_text = "".join(
        entry["message"]["transcript"]
        for entry in tape.events
        if entry["direction"] == "send" and entry["message"]["type"] == "input_transcript"
    )
    assert sent_text.strip() == child["input"]["prompt"]
    assert named(rows, "tts")[0]["input"] == "placeholder."
    assert child["span_parents"] == [named(rows, "tts")[0]["span_id"]]
    assert child["span_attributes"]["type"] == "llm"
    assert child["context"]["span_origin"]["instrumentation"]["name"] == "livekit-agents-auto"
    assert child["metadata"]["model"] == tts.model
    assert child["metadata"]["provider"] == tts.provider
    assert child["metrics"]["time_to_first_token"] >= 0
    assert child["metrics"]["end"] >= child["metrics"]["start"]
    assert bool(child.get("output", {}).get("content")) == capture
    assert bool(uploads or memory_logger.upload_attempts) == capture
    if capture:
        assert child["output"]["content"][0]["type"] == "file"
