"""Bound native submission; cancellation keeps worker-owned buffers alive."""

import asyncio
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial


_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_pid: int | None = None
_slots: threading.Semaphore | None = None
_waiters: deque | None = None


async def run_native(function, *args):
    global _executor, _pid, _slots, _waiters
    with _lock:
        if _executor is None or _pid != os.getpid():
            _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="braintrust-audio")
            _slots = threading.Semaphore(2)
            _waiters = deque()
            _pid = os.getpid()
        executor, slots, waiters = _executor, _slots, _waiters
    assert slots is not None
    assert waiters is not None
    # Waiting sessions retain their buffers. Never queue unbounded native jobs
    # or block the voice loop on a threading semaphore.
    request = object()
    with _lock:
        waiters.append(request)
    try:
        while True:
            with _lock:
                if waiters[0] is request and slots.acquire(blocking=False):
                    waiters.popleft()
                    break
            await asyncio.sleep(0.005)
    except BaseException:
        with _lock:
            waiters.remove(request)
        raise
    try:
        future = executor.submit(partial(function, *args))
    except BaseException:
        slots.release()
        raise
    future.add_done_callback(lambda _: slots.release())
    wrapped = asyncio.wrap_future(future)
    # A cancelled waiter cannot consume a late native failure. Retrieve it
    # without logging potentially private exception details.
    wrapped.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
    return await asyncio.shield(wrapped)
