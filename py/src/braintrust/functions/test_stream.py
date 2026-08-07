"""Tests for synchronous and asynchronous Braintrust streams."""

import asyncio
import queue
import threading

import pytest
from braintrust.functions.stream import BraintrustStream, BraintrustTextChunk


def test_braintrust_stream_supports_sync_for():
    stream = BraintrustStream(
        [
            BraintrustTextChunk(data="Hello"),
            BraintrustTextChunk(data=" world"),
        ]
    )

    assert [chunk.data for chunk in stream] == ["Hello", " world"]


@pytest.mark.asyncio
async def test_braintrust_stream_supports_async_for():
    stream = BraintrustStream(
        [
            BraintrustTextChunk(data="Hello"),
            BraintrustTextChunk(data=" world"),
        ]
    )

    chunks = [chunk async for chunk in stream]

    assert [chunk.data for chunk in chunks] == ["Hello", " world"]


@pytest.mark.asyncio
async def test_braintrust_stream_cancellation_preserves_pending_chunk():
    chunk_queue: queue.Queue[BraintrustTextChunk] = queue.Queue()
    read_started = threading.Event()
    first_chunk = BraintrustTextChunk(data="first")

    def chunks():
        read_started.set()
        yield chunk_queue.get(timeout=2)
        yield BraintrustTextChunk(data="second")

    stream = BraintrustStream(chunks())  # type: ignore[arg-type]
    pending_read = asyncio.create_task(anext(stream))
    assert await asyncio.wait_for(asyncio.to_thread(read_started.wait), timeout=1)

    pending_read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_read

    chunk_queue.put(first_chunk)
    assert await anext(stream) is first_chunk


@pytest.mark.asyncio
async def test_braintrust_stream_final_value_async():
    stream = BraintrustStream(
        [
            BraintrustTextChunk(data="Hello"),
            BraintrustTextChunk(data=" world"),
        ]
    )

    assert await stream.final_value_async() == "Hello world"
    assert await stream.final_value_async() == "Hello world"
