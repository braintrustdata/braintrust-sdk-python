"""Test-only record/replay for the harness WebSocket, using real SDK parsers.

HTTP VCR cannot intercept the bundled Go runtime's Gemini requests. Record
actual WebSocket frames instead, then replay through LocalConnection's real
event processor. No provider responses are synthesized.
"""

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path

from braintrust.integrations.conftest import _versioned_cassette_dir
from google.antigravity.connections.local.event_processor import parse_initialize_response
from google.antigravity.connections.local.local_connection import LocalConnection, LocalConnectionStrategy
from google.antigravity.proto import localharness_pb2
from google.protobuf import json_format


def cassette_path(name):
    base = Path(os.environ.get("BRAINTRUST_INTEGRATIONS_DIR", Path(__file__).resolve().parent.parent))
    return Path(_versioned_cassette_dir(str(base / "antigravity" / "cassettes"))) / f"{name}.json"


def _sanitize(value):
    if isinstance(value, dict):
        return {k: "REDACTED" if k in ("apiKey", "api_key") else _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, str):
        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            secret = os.environ.get(key)
            if secret:
                value = value.replace(secret, "REDACTED")
        value = value.replace(str(Path.cwd()), "WORKSPACE")
        if value.startswith(("/private/var/folders/", "/tmp/pytest-", "C:\\Users\\")):
            value = "TEST_WORKSPACE"
    return value


class _RecordingSocket:
    def __init__(self, ws, frames):
        self.ws, self.frames = ws, frames

    async def send(self, data):
        self.frames.append({"send": _sanitize(json.loads(data))})
        await self.ws.send(data)

    async def recv(self):
        data = await self.ws.recv()
        self.frames.append({"recv": _sanitize(json.loads(data))})
        return data

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.recv()

    async def close(self):
        await self.ws.close()


class _ReplaySocket:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.next_frame = next(self.frames, None)
        self.changed = asyncio.Event()

    def _advance(self):
        self.next_frame = next(self.frames, None)
        self.changed.set()

    async def send(self, data):
        assert self.next_frame is not None and "send" in self.next_frame
        actual = json.loads(data)
        expected = self.next_frame["send"]
        # Handshake has machine-specific workspace, credentials and config.
        # Prompts must match exactly; other events retain their protocol type.
        assert actual.keys() == expected.keys()
        if "userInput" in actual:
            assert actual == expected
        self._advance()

    async def recv(self):
        while self.next_frame is None or "recv" not in self.next_frame:
            self.changed.clear()
            await self.changed.wait()
        data = json.dumps(self.next_frame["recv"])
        self._advance()
        await asyncio.sleep(0)
        return data

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.recv()

    async def close(self):
        pass


@contextmanager
def cassette_transport(name):
    path = cassette_path(name)
    mode = os.environ.get("BRAINTRUST_ANTIGRAVITY_RECORD_MODE", "none" if os.getenv("CI") else "once")
    recording = mode == "all" or (mode == "once" and not path.exists())
    original_spawn = LocalConnectionStrategy._spawn_harness_and_connect_ws
    original_enter = LocalConnectionStrategy.__aenter__
    frames = []

    async def spawn(strategy, **kwargs):
        process, ws, url = await original_spawn(strategy, **kwargs)
        return process, _RecordingSocket(ws, frames), url

    async def enter(strategy):
        ws = _ReplaySocket(json.loads(path.read_text(encoding="utf-8")))
        init = localharness_pb2.InitializeConversationEvent(config=strategy._build_harness_config())  # pylint: disable=no-member
        await ws.send(json_format.MessageToJson(init))
        event = localharness_pb2.OutputEvent()  # pylint: disable=no-member
        json_format.Parse(await ws.recv(), event)
        parsed = parse_initialize_response(event.initialize_conversation_response)
        strategy._connection = LocalConnection(
            process=None,
            ws=ws,
            tool_runner=strategy._tool_runner,
            hook_runner=strategy._hook_runner,
            initial_history=parsed.history,
            initial_usage=parsed.cumulative_usage,
            initial_trajectory_usages=parsed.trajectory_usages,
        )

    if recording:
        LocalConnectionStrategy._spawn_harness_and_connect_ws = spawn
    else:
        if not path.exists():
            raise FileNotFoundError(f"Missing Antigravity recording: {path}")
        LocalConnectionStrategy.__aenter__ = enter
    try:
        yield
    finally:
        LocalConnectionStrategy.__aenter__ = original_enter
        LocalConnectionStrategy._spawn_harness_and_connect_ws = original_spawn
    if recording:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(frames, indent=2) + "\n", encoding="utf-8")
