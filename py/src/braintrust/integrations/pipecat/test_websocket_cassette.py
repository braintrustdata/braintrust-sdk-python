"""Guard the recorder itself: changed requests must never silently replay."""

import json

import pytest

from ._test_websocket import WebSocketCassette


@pytest.mark.asyncio
async def test_websocket_cassette_preserves_ids_and_rejects_changed_payload(tmp_path):
    path = tmp_path / "socket.json"
    path.write_text(
        json.dumps(
            {
                "endpoint": "wss://example.test",
                "events": [
                    {
                        "direction": "send",
                        "message": {
                            "type": "conversation.item.create",
                            "event_id": "old-event",
                            "item": {"id": "old-item", "text": "hello"},
                        },
                    },
                    {
                        "direction": "receive",
                        "message": {"type": "conversation.item.created", "item": {"id": "old-item"}},
                    },
                    {
                        "direction": "send",
                        "message": {"type": "response.create", "response": {"modalities": ["audio"]}},
                    },
                ],
            }
        )
    )
    cassette = WebSocketCassette(path)
    await cassette.send(
        json.dumps(
            {"type": "conversation.item.create", "event_id": "new-event", "item": {"id": "new-item", "text": "hello"}}
        )
    )
    assert json.loads(await anext(cassette))["item"]["id"] == "new-item"
    with pytest.raises(AssertionError, match="not fully consumed"):
        cassette.verify()
    with pytest.raises(AssertionError, match="request mismatch"):
        await cassette.send(json.dumps({"type": "response.create", "response": {"modalities": ["text"]}}))
    with pytest.raises(AssertionError, match="replay failed"):
        cassette.verify()
    await cassette.close()
