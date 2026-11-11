import asyncio
import os
import shutil
import socket
import subprocess
import uuid

import pytest
import pytest_asyncio
from braintrust import logger
from braintrust.test_helpers import init_test_logger


@pytest.fixture
def memory_logger(monkeypatch):
    monkeypatch.setattr(logger, "_compute_logger_metadata", logger._compute_logger_metadata)
    init_test_logger("test-project-livekit-cascade")
    with logger._internal_with_memory_background_logger() as bgl:
        monkeypatch.setattr(logger._state, "_global_bg_logger", logger.LazyValue(lambda: bgl, use_mutex=False))
        yield bgl


@pytest.fixture(scope="session")
def livekit_server():
    def available():
        with socket.socket() as sock:
            sock.settimeout(0.1)
            return sock.connect_ex(("127.0.0.1", 7880)) == 0

    if available():
        yield
        return
    binary = shutil.which("livekit-server")
    if binary is None:
        pytest.skip("RoomIO tests require a local LiveKit server")
    import time

    process = subprocess.Popen([binary, "--dev"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while not available():
            if time.monotonic() > deadline:
                raise TimeoutError("LiveKit server failed to start")
            time.sleep(0.1)
        yield
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest_asyncio.fixture
async def rooms(livekit_server):
    from livekit import api, rtc

    name = f"braintrust-cascade-{uuid.uuid4()}"
    agent, caller = rtc.Room(), rtc.Room()

    def token(identity):
        return (
            api.AccessToken(os.getenv("LIVEKIT_API_KEY", "devkey"), os.getenv("LIVEKIT_API_SECRET", "secret"))
            .with_identity(identity)
            .with_grants(api.VideoGrants(room_join=True, room=name))
            .to_jwt()
        )

    try:
        url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
        await asyncio.wait_for(agent.connect(url, token("agent")), 15)
        await asyncio.wait_for(caller.connect(url, token("caller")), 15)
        yield agent, caller
    finally:
        await asyncio.gather(agent.disconnect(), caller.disconnect())
