import asyncio
import json
import struct
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from uuid import uuid4

import pytest
from braintrust import logger
from braintrust.test_helpers import init_test_logger
from braintrust.util import LazyValue
from braintrust.voice import connect


websockets = pytest.importorskip("websockets.asyncio.server")


@pytest.fixture
def tracing(monkeypatch):
    monkeypatch.setattr(logger, "_compute_logger_metadata", logger._compute_logger_metadata)
    init_test_logger("voice-test")
    with logger._internal_with_memory_background_logger() as memory:
        monkeypatch.setattr(
            logger._internal_get_global_state(), "_global_bg_logger", LazyValue(lambda: memory, use_mutex=False)
        )
        yield memory


@asynccontextmanager
async def simulator(handler, parent, **options):
    async def connection(socket):
        assert socket.request.headers["Authorization"] == "Bearer test-session-token"
        await socket.send(
            json.dumps({"type": "ready", "sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le", "epoch": 0})
        )
        await handler(socket)

    async with websockets.serve(connection, "127.0.0.1", 0, **options) as server:
        port = server.sockets[0].getsockname()[1]
        yield {
            "connection_url": f"ws://127.0.0.1:{port}",
            "session_token": "test-session-token",
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
            "session_id": str(uuid4()),
            "span_id": "simulator-span",
            "parent": parent,
        }


def terminal(**overrides):
    return {
        "type": "ended",
        "output": {
            "transcript": [{"id": "turn", "speaker": "caller", "text": "Hello"}],
            "reason": "Client stopped session",
            "recording": {"audio": {}, "metadata": {}},
        },
        **overrides,
    }


@pytest.mark.asyncio
async def test_consumption_interruptions_resampling_and_trace_scope(tracing):
    # Playback acknowledgements follow actual consumption, reset on clear, and never mix call contexts.
    acknowledgements = []
    agent_audio = []
    consumed = []
    heard = asyncio.Event()
    release = asyncio.Event()

    async def handler(socket):
        await socket.send(struct.pack("<480h", *([100] * 480)))
        async for message in socket:
            if isinstance(message, bytes):
                agent_audio.append(message)
                continue
            event = json.loads(message)
            if event["type"] == "playback.ack":
                acknowledgements.append(event)
                if event["epoch"] == 0:
                    await socket.send(json.dumps({"type": "audio.clear", "epoch": 1}))
                    await socket.send(struct.pack("<480h", *([200] * 480)))
                else:
                    heard.set()
            if event["type"] == "stop":
                await socket.send(json.dumps(terminal()))
                break

    async def consume(audio):
        if any(audio):
            consumed.append(audio)
            await release.wait()

    async def noop():
        return None

    with logger.start_span(name="Task") as task:
        async with simulator(handler, task.export()) as call:
            async with connect(call) as session:
                assert logger.current_span().span_id != task.span_id
                assert "test-session-token" not in repr(session)
                session.set_audio_sink(consume, clear=noop, ended=noop)
                await asyncio.sleep(0.04)
                assert acknowledgements == []
                release.set()
                await asyncio.wait_for(heard.wait(), 2)
                await session.send_audio(struct.pack("<320h", *([300] * 320)), 16000, 1)
            result = session.result(agent_output={"transferred_to": "billing"})
            assert logger.current_span() is task
    assert [ack["played_samples"] for ack in acknowledgements[:2]] == [480, 480]
    assert [ack["epoch"] for ack in acknowledgements[:2]] == [0, 1]
    assert any(any(audio) for audio in agent_audio)
    assert all(len(audio) == 960 for audio in agent_audio)
    assert result["agent_output"] == {"transferred_to": "billing"}
    assert result["caller_transcript"][0]["speaker"] == "caller"
    rows = tracing.pop()
    root = next(row for row in rows if row["span_attributes"]["name"] == "Task")
    agent = next(row for row in rows if row["span_attributes"]["name"] == "Agent session")
    assert agent["span_parents"] == [root["span_id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disconnect", "session", "recording", "missing-recording", "agent", "cancel"])
async def test_failures_never_return_success_and_restore_parent(tracing, failure):
    # A clean simulator response must not mask agent errors, cancellation, or missing recording evidence.
    async def handler(socket):
        if failure == "disconnect":
            return
        if failure == "session":
            await socket.send(json.dumps({"type": "error", "message": "Simulator failed"}))
            return
        async for message in socket:
            if isinstance(message, str) and json.loads(message)["type"] == "stop":
                event = terminal()
                if failure == "recording":
                    event["output"]["recording_error"] = "Upload failed"
                if failure == "missing-recording":
                    del event["output"]["recording"]
                await socket.send(json.dumps(event))
                return

    with logger.start_span(name="Task") as parent:
        async with simulator(handler, parent.export()) as call:
            exception = asyncio.CancelledError if failure == "cancel" else RuntimeError
            with pytest.raises(exception):
                async with connect(call) as session:
                    if failure == "agent":
                        raise RuntimeError("Agent TTS failed")
                    if failure == "cancel":
                        raise asyncio.CancelledError()
            assert logger.current_span() is parent
            with pytest.raises(RuntimeError):
                session.result()


@pytest.mark.asyncio
async def test_concurrent_call_tracing_is_isolated(tracing):
    # Distributed parents must stay local to each async call even while their lifetimes overlap.
    async def handler(socket):
        async for message in socket:
            if isinstance(message, str) and json.loads(message)["type"] == "stop":
                await socket.send(json.dumps(terminal()))
                return

    async def run(index):
        with logger.start_span(name=f"Case {index}") as parent:
            async with simulator(handler, parent.export()) as call:
                async with connect(call):
                    with logger.start_span(name=f"Tool {index}"):
                        await asyncio.sleep(0.01)
                assert logger.current_span() is parent

    await asyncio.gather(run(1), run(2))
    rows = tracing.pop()
    cases = {row["span_id"]: row for row in rows if row["span_attributes"]["name"].startswith("Case")}
    agents = {row["span_id"]: row for row in rows if row["span_attributes"]["name"] == "Agent session"}
    for row in rows:
        if row["span_attributes"]["name"].startswith("Tool"):
            agent = agents[row["span_parents"][0]]
            case = cases[agent["span_parents"][0]]
            assert row["span_attributes"]["name"][-1] == case["span_attributes"]["name"][-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.UNAUTHORIZED])
async def test_retries_only_explicit_capacity_rejections_before_connection(tracing, status):
    # A one-use connection may retry a capacity rejection but must never retry authentication errors.
    from websockets.exceptions import InvalidStatus

    attempts = []

    def process_request(socket, request):
        attempts.append(request.headers["Authorization"])
        if len(attempts) == 1:
            return socket.respond(status, "Rejected")

    async def handler(socket):
        async for message in socket:
            if isinstance(message, str) and json.loads(message)["type"] == "stop":
                await socket.send(json.dumps(terminal()))
                return

    with logger.start_span(name="Task") as parent:
        async with simulator(handler, parent.export(), process_request=process_request) as call:
            if status == HTTPStatus.UNAUTHORIZED:
                with pytest.raises(InvalidStatus):
                    async with connect(call):
                        pytest.fail("Authentication failure must not enter the agent body")
            else:
                async with connect(call):
                    pass
    assert len(attempts) == (2 if status == HTTPStatus.SERVICE_UNAVAILABLE else 1)
