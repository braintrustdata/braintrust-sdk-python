"""Test-only, ordered recording/replay at Pipecat's real WebSocket boundary.

Only transport I/O is replaced. Requests must match; native service handlers,
frames, aggregators and instrumentation are unchanged. No headers are stored.
"""

import asyncio
import json
from pathlib import Path

from ._test_audio_cassettes import load_websocket, save_websocket


class WebSocketCassette:
    def __init__(self, path: Path, record=False):
        self.path, self.record = path, record
        saved = {} if record else load_websocket(path)
        self.events = saved.get("events", [])
        self.index = 0
        self.condition = asyncio.Condition()
        self.closed = False
        self.socket = None
        self.client_ids = {}
        self.failure = None
        self.endpoint = saved.get("endpoint")

    async def connect(self, real_connect, **kwargs):
        if self.record:
            self.endpoint = kwargs["uri"]
            self.socket = await real_connect(**kwargs)
        else:
            assert kwargs["uri"] == self.endpoint, "WebSocket endpoint changed; re-record cassette"
        return self

    def _normalize(self, message):
        value = json.loads(json.dumps(message))
        value.pop("event_id", None)
        item = value.get("item", {})
        if item.get("id") in self.client_ids:
            item["id"] = self.client_ids[item["id"]]
        return value

    async def send(self, data):
        message = json.loads(data)
        if self.record:
            self.events.append({"direction": "send", "message": message})
            await self.socket.send(data)
            return
        try:
            async with self.condition:
                await asyncio.wait_for(
                    self.condition.wait_for(
                        lambda: (
                            self.closed
                            or self.index == len(self.events)
                            or self.events[self.index]["direction"] == "send"
                        )
                    ),
                    5,
                )
                assert not self.closed and self.index < len(self.events), "Unexpected WebSocket send"
                expected = self.events[self.index]["message"]
                # Initial conversation item IDs are client-generated. Preserve
                # their correspondence in incoming echoes, not just in matching.
                if message.get("type") == "conversation.item.create":
                    actual_id = message["item"].get("id")
                    expected_id = expected.get("item", {}).get("id")
                    if actual_id and expected_id:
                        self.client_ids[actual_id] = expected_id
                assert self._normalize(message) == self._normalize(expected), (
                    f"WebSocket request mismatch at event {self.index}: {message.get('type')}"
                )
                self.index += 1
                self.condition.notify_all()
        except Exception as error:
            self.failure = error
            raise

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.record:
            data = await self.socket.recv()
            self.events.append({"direction": "receive", "message": json.loads(data)})
            return data
        async with self.condition:
            await self.condition.wait_for(
                lambda: (
                    self.closed
                    or (self.index < len(self.events) and self.events[self.index]["direction"] == "receive")
                )
            )
            if self.closed:
                raise StopAsyncIteration
            message = self.events[self.index]["message"]
            self.index += 1
            self.condition.notify_all()
        # Rewrite only values identical to client IDs learned from outgoing
        # requests. Provider-generated IDs and all payloads are preserved.
        reverse = {recorded: actual for actual, recorded in self.client_ids.items()}

        def remap(value):
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            if isinstance(value, list):
                return [remap(item) for item in value]
            return reverse.get(value, value) if isinstance(value, str) else value

        await asyncio.sleep(0)
        return json.dumps(remap(message))

    async def close(self):
        self.closed = True
        if self.socket:
            await self.socket.close()
        async with self.condition:
            self.condition.notify_all()

    def verify(self):
        assert self.failure is None, f"WebSocket replay failed: {self.failure}"
        if not self.record:
            assert self.index == len(self.events), "WebSocket cassette was not fully consumed"

    def save(self):
        assert self.record and self.events
        save_websocket(self.path, self.endpoint, self.events)
