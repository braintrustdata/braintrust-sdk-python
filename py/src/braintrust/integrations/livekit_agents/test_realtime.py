import asyncio
import base64
import io
import os
from contextlib import nullcontext
from pathlib import Path

import pytest
from braintrust import logger
from braintrust._audio.config import AudioEncoding, AudioRecordingOptions
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.livekit_agents.test_realtime_helpers import RealtimeCassette


def named(rows, name):
    return [row for row in rows if row.get("span_attributes", {}).get("name") == name]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "native",
        "late_burst",
        "late_metrics",
        "unknown_timing",
        "no_transcription_interrupt",
        "restricted_initial",
        "restricted_late",
        "hybrid",
    ],
)
async def test_realtime_conversation(rooms, memory_logger, monkeypatch, vcr_cassette_dir, vcr, mode):
    from braintrust.integrations.livekit_agents.realtime import Realtime, Response
    from livekit import rtc
    from livekit.agents import Agent, AgentSession, APIConnectOptions, function_tool
    from livekit.plugins import openai
    from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

    timing_lost = mode == "unknown_timing"
    instructions = (
        "You help with fictional demo orders. Use lookup_order for order status. Speak English in one short sentence."
    )
    observed_stops = {}
    original_speech = Realtime.speech

    def observe_speech(instance, event, now):
        if timing_lost and event["type"] == "input_audio_buffer.speech_stopped":
            # Lose timing after the first start, covering both a known start
            # with an unknown end and subsequent wholly unpositioned utterances.
            instance.clock.valid = False
            observed_stops[event["item_id"]] = now
        return original_speech(instance, event, now)

    if timing_lost:
        monkeypatch.setattr(Realtime, "speech", observe_speech)

    early_binding = []
    native_metrics = {}
    delayed_metrics = []
    original_metrics = Realtime.metrics

    def observe_metrics(instance, event):
        if getattr(event, "type", None) != "realtime_model_metrics" or not event.request_id:
            return original_metrics(instance, event)
        native_metrics[event.request_id] = event
        if mode == "late_metrics":

            async def deliver():
                while (record := instance.response(event.request_id)).handle is None or not record.handle.done():
                    await asyncio.sleep(0.01)
                original_metrics(instance, event)

            delayed_metrics.append(asyncio.create_task(deliver()))
        else:
            original_metrics(instance, event)

    monkeypatch.setattr(Realtime, "metrics", observe_metrics)
    original_bind = Response.bind

    def observe_bind(instance, *args, **kwargs):
        if instance.span is None:
            early_binding.append(instance.ended is not None)
        return original_bind(instance, *args, **kwargs)

    monkeypatch.setattr(Response, "bind", observe_bind)
    folder = Path(vcr_cassette_dir)
    restricted = mode.startswith("restricted")
    no_transcription = mode == "no_transcription_interrupt"
    hybrid = mode == "hybrid"
    cassette = (
        "realtime-no-transcription-interrupted"
        if no_transcription
        else "realtime-hybrid"
        if hybrid
        else "realtime-conversation"
    )
    with logger.start_span(name="unrelated") as unrelated:
        unrelated.log(input="independent customer record")
    uploads = []
    encoded = {}
    from braintrust._audio.session import AudioSessionRecorder

    original_export = AudioSessionRecorder._export

    async def observe_export(instance, result):
        encoded[f"part-{result.meta.index}"] = result.meta
        return await original_export(instance, result)

    monkeypatch.setattr(AudioSessionRecorder, "_export", observe_export)

    def uploaded(attachment):
        uploads.append(attachment)
        return {"upload_status": "done"}

    monkeypatch.setattr(logger.Attachment, "upload", uploaded)
    monkeypatch.setenv("BRAINTRUST_LEGACY_IDS", "false")
    logger._state._reset_id_generator()
    monkeypatch.setenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", "true")
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", "true")
    setup_livekit_agents(audio=AudioRecordingOptions(encoding=AudioEncoding()))

    @function_tool
    async def lookup_order(order_id: str) -> dict:
        return {"order_id": order_id, "status": "in_transit", "delivery": "Friday"}

    async with RealtimeCassette(
        folder / (cassette + ".json"), late_transcript=mode == "late_burst", burst=mode == "late_burst"
    ) as tape:
        model_options = {"modalities": ["text"]} if hybrid else {}
        if no_transcription:
            model_options["input_audio_transcription"] = None
        model = openai.realtime.RealtimeModel(
            **model_options,
            model="gpt-realtime",
            voice="marin",
            api_key="test-key",
            base_url=tape.base_url,
            conn_options=APIConnectOptions(max_retry=0),
            turn_detection=ServerVad(type="server_vad", silence_duration_ms=300, prefix_padding_ms=200),
        )
        source = rtc.AudioSource(24000, 1)
        track = rtc.LocalAudioTrack.create_audio_track("caller", source)
        assistant = []
        added = asyncio.Event()
        queue = asyncio.Queue()
        pump = None
        tts = (
            openai.TTS(model="tts-1", voice="alloy", api_key=tape.recording_key() if tape.record else "test-key")
            if hybrid
            else None
        )
        session = AgentSession(llm=model, tts=tts)
        if mode == "restricted_initial":
            session._redaction_enabled = True
        http = vcr.use_cassette("realtime-hybrid-tts.yaml", ignore_hosts=["127.0.0.1"]) if hybrid else nullcontext()
        http.__enter__()

        def conversation(event):
            if event.item.type == "message" and event.item.role == "assistant":
                assistant.append(event.item)
                added.set()

        session.on("conversation_item_added", conversation)

        async def wait_responses(count):
            while len(assistant) < count:
                added.clear()
                await asyncio.wait_for(added.wait(), 40)

        async def feed():
            while True:
                data = queue.get_nowait() if not queue.empty() else bytes(480)
                await source.capture_frame(
                    rtc.AudioFrame(data=data, sample_rate=24000, num_channels=1, samples_per_channel=len(data) // 2)
                )

        def say(filename):
            data = (folder / filename).read_bytes() + bytes(24000)
            for offset in range(0, len(data), 480):
                queue.put_nowait(data[offset : offset + 480])

        try:
            await session.start(
                Agent(
                    instructions=instructions,
                    tools=[lookup_order],
                ),
                room=rooms[0],
                record=False,
            )
            await session.generate_reply(instructions="Say exactly: Hello.").wait_for_playout()
            await wait_responses(1)
            if mode == "restricted_late":
                session._redaction_enabled = True
            await asyncio.sleep(1.2)  # Input-clock origin differs from the session origin.
            await rooms[1].local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            pump = asyncio.create_task(feed())
            for _ in range(100):
                queue.put_nowait(bytes(480))
            say("order-24khz.pcm")
            await wait_responses(2)
            if no_transcription:
                from braintrust.integrations.livekit_agents.compat import STATE_ATTR

                controller = next(iter(getattr(session, STATE_ATTR).realtime.values()))
                handle = session.generate_reply(instructions="Count slowly from one to twenty.")

                async def generated():
                    while not any(
                        record.handle is handle and record.ended is not None
                        for record in controller.responses.values()
                    ):
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(generated(), 30)
                handle.interrupt(force=True)
                await handle.wait_for_playout()
            else:
                say("followup-24khz.pcm")
                await wait_responses(3)
            await session.drain()
            if delayed_metrics:
                await asyncio.wait_for(asyncio.gather(*delayed_metrics), 15)
            from braintrust.integrations.livekit_agents.compat import STATE_ATTR

            owner = getattr(session, STATE_ATTR)
            controllers = list(owner.realtime.values())
            assert len(controllers) == 1
            controller = controllers[0]
            origin = controller.clock.origin
            assert controller.clock.valid != timing_lost and origin is not None
            if timing_lost:
                assert observed_stops
                assert all(item.stopped for item in controller.inputs.values()), (
                    "Observed stops must settle user speech"
                )
            if restricted:
                assert controller.instructions is None
                assert not controller.response_options
        finally:
            if pump is not None:
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
            await session.aclose()
            await source.aclose()
            await model.aclose()
            if tts is not None:
                await tts.aclose()
            http.__exit__(None, None, None)
    rows = memory_logger.pop()
    if artifact_dir := os.getenv("BRAINTRUST_LIVEKIT_REALTIME_ARTIFACT_DIR"):
        from braintrust.bt_json import bt_dumps

        destination = Path(artifact_dir) / mode
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "trace.json").write_text(bt_dumps(rows))
        for attachment in uploads:
            (destination / attachment.reference["filename"]).write_bytes(attachment.data)

    server = [entry["message"] for entry in tape.events if entry["direction"] == "receive"]
    starts = {
        event["item_id"]: event["audio_start_ms"]
        for event in server
        if event["type"] == "input_audio_buffer.speech_started"
    }
    stops = {
        event["item_id"]: event["audio_end_ms"]
        for event in server
        if event["type"] == "input_audio_buffer.speech_stopped"
    }
    finals = {
        event["item_id"]: event["transcript"]
        for event in server
        if event["type"] == "conversation.item.input_audio_transcription.completed"
    }
    responses = {event["response"]["id"]: event["response"] for event in server if event["type"] == "response.done"}
    session_tools = next(
        event["session"]["tools"]
        for event in server
        if event["type"] == "session.updated" and event["session"].get("tools")
    )
    requested_instructions = {
        entry["message"]["response"]["metadata"]["client_event_id"]: entry["message"]["response"].get("instructions")
        for entry in tape.events
        if entry["direction"] == "send" and entry["message"]["type"] == "response.create"
    }
    users = named(rows, "user_turn")
    speeches = named(rows, "user_speech")
    receipts = named(rows, "transcript_received")
    models = named(rows, "llm_response")
    llms = named(rows, "Realtime request")
    assert len(llms) == len(models)
    leaves = {row["span_parents"][0]: row for row in llms}
    for task in models:
        assert task["span_attributes"]["type"] == "task"
        assert leaves[task["span_id"]]["span_attributes"]["type"] == "llm"
        assert "tokens" not in task["metrics"]
    assert len(users) == len(speeches) == len(starts)
    assert len(receipts) == len(finals)
    assert bool(receipts) != no_transcription
    assert len(models) == len(responses)
    assert not named(rows, "stt")
    assert bool(named(rows, "tts")) == hybrid
    if mode == "late_burst":
        assert any(early_binding), "Replay must exercise completion before async model-span binding"
    if restricted:
        assert not uploads
        related = [row for row in rows if row["span_attributes"]["name"] != "unrelated"]
        assert all("input" not in row and "output" not in row for row in related)
        assert named(rows, "unrelated")[0]["input"] == "independent customer record"
        assert sum(row["metrics"]["tokens"] for row in llms) == sum(
            r["usage"]["total_tokens"] for r in responses.values()
        )
        assert not owner.realtime and not controller.listeners
        return
    by_id = {row["span_id"]: row for row in rows}
    for user in users:
        item = user["metadata"]["contrib.livekit.item_id"]
        if item in finals:
            assert user["input"] == [{"role": "user", "content": finals[item]}]
        else:
            assert "input" not in user
        speech = next(row for row in speeches if row["span_parents"] == [user["span_id"]])
        if timing_lost:
            for row in (user, speech):
                assert "audio.selections" not in row["metadata"]
                assert row["metadata"]["contrib.livekit.timing"] == "unavailable"
                assert not row["metadata"].get("turn.incomplete")
                assert row["metrics"]["end"] == observed_stops[item]
                assert row["metrics"]["start"] < row["metrics"]["end"]
            assert speech["metrics"] == user["metrics"]
        else:
            selection = user["metadata"]["audio.selections"]["selection-0"]
            assert abs(selection["start_offset_ms"] - (float(origin) / 1e6 + starts[item])) <= 1
            assert abs(selection["end_offset_ms"] - (float(origin) / 1e6 + stops[item])) <= 1
            assert selection["start_offset_ms"] > starts[item] + 1200
            assert speech["metadata"]["audio.selections"] == user["metadata"]["audio.selections"]
        assert speech["metadata"]["turn.id"] == user["span_id"]
        if item in finals:
            receipt = next(row for row in receipts if row["span_parents"] == [user["span_id"]])
            assert receipt["output"] == user["input"]
            assert receipt["metrics"]["start"] == receipt["metrics"]["end"]
            assert receipt["metadata"]["links"][0]["context"]["spanId"] == speech["span_id"]
            assert receipt["metadata"]["turn.id"] == user["span_id"]
    for model in models:
        leaf = leaves[model["span_id"]]
        assert leaf["input"] == model["input"]
        assert leaf["metadata"]["tools"] == [
            {"type": "function", "function": {k: v for k, v in tool.items() if k != "type" and v is not None}}
            for tool in session_tools
        ]
        assert leaf["metadata"]["tool_choice"] == "auto"
        native = responses[model["metadata"]["contrib.livekit.response_id"]]
        override = requested_instructions.get((native.get("metadata") or {}).get("client_event_id"))
        expected_instructions = override if isinstance(override, str) else instructions
        assert model["input"][0] == {"role": "system", "content": expected_instructions}
        assert [message for message in model["input"] if message["role"] == "system"] == [model["input"][0]]
        assert model["metadata"]["model"] == "gpt-realtime"
        assert model["metadata"]["contrib.livekit.response_status"] == native["status"]
        assert leaf["metrics"]["tokens"] == native["usage"]["total_tokens"]
        assert leaf["metrics"]["prompt_tokens"] == native["usage"]["input_tokens"]
        assert leaf["metrics"]["completion_tokens"] == native["usage"]["output_tokens"]
        assert leaf["metrics"]["prompt_audio_tokens"] == native["usage"]["input_token_details"]["audio_tokens"]
        assert leaf["metrics"]["completion_audio_tokens"] == native["usage"]["output_token_details"]["audio_tokens"]
        assert leaf["metrics"]["prompt_cached_tokens"] == native["usage"]["input_token_details"]["cached_tokens"]
        assert "cache_read_input_tokens" not in leaf["metrics"]
        assert leaf["metrics"]["start"] == pytest.approx(native_metrics[native["id"]].timestamp, abs=1e-6)
        assert leaf["metrics"]["end"] == pytest.approx(
            native_metrics[native["id"]].timestamp + native_metrics[native["id"]].duration, abs=1e-6
        )
        if any(item["type"] == "message" for item in native["output"]):
            assert "time_to_first_token" in leaf["metrics"]
        if native["status"] == "completed":
            expected = []
            for item in native["output"]:
                if item["type"] == "message":
                    expected.append("".join(part.get("transcript", part.get("text", "")) for part in item["content"]))
            assert [out["content"] for out in model.get("output", []) if out.get("content") is not None] == expected
        assert leaf.get("output", []) == [
            {
                "index": index,
                "message": message,
                "finish_reason": "tool_calls"
                if native["status"] == "completed" and message.get("tool_calls")
                else None,
            }
            for index, message in enumerate(model.get("output", []))
        ]
    assert sum(row["metrics"]["tokens"] for row in llms) == sum(r["usage"]["total_tokens"] for r in responses.values())
    tools = named(rows, "lookup_order")
    assert len(tools) == 1
    assert tools[0]["metadata"]["links"][0]["attributes"]["braintrust.relationship.type"] == "requested_by"
    sent_results = {
        entry["message"]["item"]["call_id"]: entry["message"]["item"]["output"]
        for entry in tape.events
        if entry["direction"] == "send"
        and entry["message"]["type"] == "conversation.item.create"
        and entry["message"]["item"]["type"] == "function_call_output"
    }
    observed_inputs = [row for row in models if row.get("input")]
    assert observed_inputs
    for model in observed_inputs:
        assert model["metadata"]["contrib.livekit.input_partial"] is True
        # This recording sends user audio, not its later ASR transcript as text.
        # Beyond instructions, only explicitly sent tool results belong here.
        for message in model["input"][1:]:
            assert message == {
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": sent_results[message["tool_call_id"]],
            }
    assert any(
        link["attributes"]["braintrust.relationship.type"] == "consumes_tool_result"
        for row in models
        for link in row["metadata"].get("links", [])
    )
    if mode == "native":
        # The later goodbye still uses the persistent conversation history,
        # including the earlier tool result, rather than a fresh empty context.
        goodbye = max(models, key=lambda row: row["metrics"]["start"])
        assert goodbye["input"][1:] == [
            {"role": "tool", "tool_call_id": call_id, "content": result} for call_id, result in sent_results.items()
        ]
        assert any(
            link["attributes"]["braintrust.relationship.type"] == "consumes_tool_result"
            and link["context"]["spanId"] == tools[0]["span_id"]
            for link in goodbye["metadata"]["links"]
        )
        assert leaves[goodbye["span_id"]]["metrics"]["prompt_cached_tokens"] > 0
    for speech in named(rows, "agent_speech"):
        target = by_id[speech["metadata"]["links"][0]["context"]["spanId"]]
        assert target in (named(rows, "tts") if hybrid else models)
        parent = by_id[speech["span_parents"][0]]
        assert speech["metadata"]["turn.id"] == parent["span_id"]
    root = named(rows, "session")[0]
    assert root["metadata"]["audio.schema.version"] == "3.0.0"
    assert all("audio.schema.version" not in row.get("metadata", {}) for row in rows if row is not root)
    assert not controller.listeners and not controller.inputs and not controller.responses
    assert controller.instructions is None and not controller.response_options
    assert not owner.realtime
    assert uploads
    if no_transcription:
        assert handle.interrupted
        generated_text = " ".join(out.get("content") or "" for row in models for out in row.get("output", []))
        heard_text = " ".join(
            out.get("content") or "" for row in named(rows, "agent_speech") for out in row.get("output", [])
        )
        assert len(generated_text) > len(heard_text)
    import av
    import numpy as np

    user_audio = []
    agent_audio = []
    for attachment in uploads:
        identifier = attachment.reference["filename"].removesuffix(".ogg")
        meta = encoded[identifier]
        with av.open(io.BytesIO(attachment.data)) as container:
            frames = list(container.decode(audio=0))
        assert sum(frame.samples for frame in frames) == meta.sample_count
        declared = root["metadata"]["audio.segments"][identifier]
        assert abs(declared["duration_ms"] - meta.sample_count / 48) <= 1
        assert declared["channels"] == {"0": {"sources": ["session-input"]}, "1": {"sources": ["agent-output"]}}
        user_audio.extend(np.concatenate([frame.to_ndarray()[0] for frame in frames]))
        agent_audio.extend(np.concatenate([frame.to_ndarray()[1] for frame in frames]))
    # Compare energy envelopes of known supplied audio through RTC+Opus loss.
    actual = np.asarray(user_audio)
    expected = np.frombuffer((folder / "order-24khz.pcm").read_bytes(), dtype="<i2").astype(float) / 32768
    actual = actual[: len(actual) // 480 * 480].reshape(-1, 480)
    expected = expected[: len(expected) // 240 * 240].reshape(-1, 240)
    envelope = np.sqrt((actual**2).mean(axis=1))
    template = np.sqrt((expected**2).mean(axis=1))
    lag = int(np.correlate(envelope, template, mode="valid").argmax())
    observed = envelope[lag : lag + len(template)]
    assert np.corrcoef(observed, template)[0, 1] > 0.8
    voiced = np.flatnonzero(template > template.max() * 0.15)
    if not timing_lost:
        selections = [next(iter(row["metadata"]["audio.selections"].values())) for row in users]
        assert any(
            s["start_offset_ms"] <= (lag + voiced[0]) * 10 + 30 and s["end_offset_ms"] >= (lag + voiced[-1]) * 10 - 30
            for s in selections
        )
    if not hybrid:
        # Compare actual played audio to the provider's generated PCM as well.
        # The interrupted case checks the count response's played prefix.
        response = (max if no_transcription else min)(models, key=lambda row: row["metrics"]["start"])
        speech = next(
            row
            for row in named(rows, "agent_speech")
            if row["metadata"]["links"][0]["context"]["spanId"] == response["span_id"]
        )
        selection = speech["metadata"]["audio.selections"]["selection-0"]
        played_ms = selection["end_offset_ms"] - selection["start_offset_ms"]
        pcm = b"".join(
            base64.b64decode(event["delta"])
            for event in server
            if event["type"] == "response.output_audio.delta"
            and event["response_id"] == response["metadata"]["contrib.livekit.response_id"]
        )
        if no_transcription:
            assert played_ms < len(pcm) / 48 - 200
        # Exclude the final 50 ms, where interrupted frame trimming can occur.
        expected = np.frombuffer(pcm, dtype="<i2")[: max(0, played_ms - 50) * 24].astype(float) / 32768
        expected = expected[: len(expected) // 240 * 240].reshape(-1, 240)
        actual = np.asarray(agent_audio)
        actual = actual[: len(actual) // 480 * 480].reshape(-1, 480)
        template = np.sqrt((expected**2).mean(axis=1))
        envelope = np.sqrt((actual**2).mean(axis=1))
        # Verify the declared placement within 50 ms for RTC frames, energy bins
        # and codec delay. Normalize correlation so louder later speech cannot
        # outweigh the expected audio at its known position.
        center = round(selection["start_offset_ms"] / 10)
        candidates = range(max(0, center - 5), min(len(envelope) - len(template) + 1, center + 6))
        assert max(np.corrcoef(envelope[i : i + len(template)], template)[0, 1] for i in candidates) > 0.8


def test_realtime_input_clock_uses_sample_position_and_fences_discontinuities():
    from fractions import Fraction

    from braintrust.integrations.livekit_agents.realtime import InputClock

    clock = InputClock()
    clock.push(Fraction(1_200_000_000), Fraction(1_000_000_000))
    clock.appended = 24000
    assert clock.offset(400) == 1_600_000_000
    assert clock.offset(1002) is None  # Never place audio not forwarded yet.
    clock.push(Fraction(3_000_000_000), Fraction(20_000_000))
    assert clock.offset(400) is None
