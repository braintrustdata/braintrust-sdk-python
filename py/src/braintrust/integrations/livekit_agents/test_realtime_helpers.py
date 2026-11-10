"""Real aiohttp WebSocket protocol recording/replay, local to LiveKit tests."""

import asyncio
import base64
import json
import os
import re
from pathlib import Path

import aiohttp
from aiohttp import web


class RealtimeCassette:
    def __init__(self, path: Path, *, late_transcript=False, burst=False):
        self.path = path
        self.record = os.getenv("BRAINTRUST_LIVEKIT_RECORD_REALTIME") == "1"
        if self.record and os.getenv("CI", "").lower() == "true":
            raise RuntimeError("Realtime recording is disabled in CI")
        self.events = [] if self.record else json.loads(path.read_text())
        self.generated_samples = {}
        for entry in self.events:
            message = entry["message"]
            if entry["direction"] == "receive" and message["type"] == "response.output_audio.delta":
                identifier = message["item_id"]
                self.generated_samples[identifier] = (
                    self.generated_samples.get(identifier, 0) + len(base64.b64decode(message["delta"])) // 2
                )
        self.audio_bytes = 0
        self.index = 0
        self.ids = {}
        self.error = None
        self.late_transcript = late_transcript
        self.burst = burst
        self.response_audio_gate = 0
        self.held = []
        self.client = None
        self.runner = None

    async def __aenter__(self):
        app = web.Application()
        app.router.add_get("/v1/realtime", self._connect)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
        return self

    async def __aexit__(self, *_):
        await self.runner.cleanup()
        if self.record:
            self.path.write_text(json.dumps(self.events, separators=(",", ":")) + "\n")
        if self.error is not None:
            raise self.error

    def _match(self, actual, expected, key=""):
        if isinstance(expected, dict):
            assert isinstance(actual, dict)
            if expected.get("type") == "conversation.item.truncate":
                # Playback progress is client-local and varies under replay.
                # Keep identity strict and bound it by the real generated PCM.
                position = actual["audio_end_ms"]
                assert type(position) is int and 0 <= position <= self.generated_samples[expected["item_id"]] / 24 + 1
                actual = {name: value for name, value in actual.items() if name != "audio_end_ms"}
                expected = {name: value for name, value in expected.items() if name != "audio_end_ms"}
            assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
            for name, value in expected.items():
                self._match(actual[name], value, name)
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                self._match(a, b)
        elif isinstance(expected, str) and key in ("event_id", "id", "client_event_id"):
            assert actual == self.ids.setdefault(expected, actual)
        else:
            assert actual == expected, (key, actual, expected)

    def _remap(self, obj):
        if isinstance(obj, dict):
            return {key: self._remap(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._remap(value) for value in obj]
        return self.ids.get(obj, obj) if isinstance(obj, str) else obj

    async def _pump(self):
        while self.index < len(self.events):
            entry = self.events[self.index]
            message = entry["message"]
            gate = entry["audio_bytes"]
            if self.burst and entry["direction"] == "receive":
                if message["type"] == "response.created":
                    self.response_audio_gate = gate
                elif message["type"].startswith("response.") or (
                    message["type"].startswith("conversation.item.")
                    and message.get("item", {}).get("role") == "assistant"
                ):
                    gate = self.response_audio_gate
            if entry["direction"] == "send" or gate > self.audio_bytes:
                break
            self.index += 1
            message = self._remap(entry["message"])
            if self.late_transcript and message["type"] == "conversation.item.input_audio_transcription.completed":
                self.held.append(message)
                continue
            await self.client.send_json(message)
            if message["type"] == "response.done":
                for held in self.held:
                    await self.client.send_json(held)
                self.held.clear()

    async def _connect(self, request):
        self.client = web.WebSocketResponse()
        await self.client.prepare(request)
        try:
            if self.record:
                await self._record(request)
            else:
                await self._pump()
                async for item in self.client:
                    if item.type != aiohttp.WSMsgType.TEXT:
                        continue
                    message = json.loads(item.data)
                    if message["type"] == "input_audio_buffer.append":
                        data = message["audio"]
                        self.audio_bytes += len(data) * 3 // 4 - (len(data) - len(data.rstrip("=")))
                    else:
                        entry = self.events[self.index]
                        assert entry["direction"] == "send", (message["type"], entry["message"]["type"])
                        self._match(message, entry["message"])
                        self.index += 1
                    await self._pump()
        except Exception as error:
            self.error = error
            await self.client.close()
        return self.client

    async def _record(self, request):
        key = self.recording_key()
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                "wss://api.openai.com/v1/realtime?model=" + request.query["model"],
                headers={"Authorization": "Bearer " + key},
            ) as upstream:

                async def caller():
                    async for item in self.client:
                        if item.type != aiohttp.WSMsgType.TEXT:
                            continue
                        message = json.loads(item.data)
                        if message["type"] == "input_audio_buffer.append":
                            data = message["audio"]
                            self.audio_bytes += len(data) * 3 // 4 - (len(data) - len(data.rstrip("=")))
                        else:
                            self.events.append(
                                {"direction": "send", "audio_bytes": self.audio_bytes, "message": message}
                            )
                        await upstream.send_str(item.data)

                async def provider():
                    async for item in upstream:
                        if item.type == aiohttp.WSMsgType.TEXT:
                            message = json.loads(item.data)
                            if isinstance(message.get("session"), dict):
                                message["session"].pop("client_secret", None)
                            self.events.append(
                                {"direction": "receive", "audio_bytes": self.audio_bytes, "message": message}
                            )
                            await self.client.send_json(message)

                tasks = [asyncio.create_task(caller()), asyncio.create_task(provider())]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def recording_key():
        key_path = Path(os.environ["BRAINTRUST_LIVEKIT_RECORD_KEY_FILE"])
        found = re.search(r"^(?:export\s+)?OPENAI_API_KEY\s*=\s*(.+)$", key_path.read_text(), re.M)
        if found is None:
            raise RuntimeError("No recording key configured")
        return found[1].strip().strip("\"'")
