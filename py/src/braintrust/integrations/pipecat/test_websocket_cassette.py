"""Guard the recorder itself: changed requests must never silently replay."""

import json

import pytest

from ._test_websocket import WebSocketCassette


@pytest.mark.asyncio
async def test_websocket_cassette_rejects_changed_payload(tmp_path):
    path = tmp_path / "socket.json"
    path.write_text(
        json.dumps(
            {
                "endpoint": "wss://example.test",
                "events": [
                    {
                        "direction": "send",
                        "message": {"type": "response.create", "response": {"modalities": ["audio"]}},
                    },
                ],
            }
        )
    )
    cassette = WebSocketCassette(path)
    with pytest.raises(AssertionError, match="request mismatch"):
        await cassette.send(json.dumps({"type": "response.create", "response": {"modalities": ["text"]}}))
    with pytest.raises(AssertionError, match="replay failed"):
        cassette.verify()
    await cassette.close()
