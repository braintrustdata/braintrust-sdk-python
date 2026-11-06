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
_waiters: deque[asyncio.Future[None]] | None = None


def _admit(request: asyncio.Future[None], slots: threading.Semaphore, waiters: deque[asyncio.Future[None]]) -> None:
    # This callback runs on the waiter's event loop, including cancellation checks.
    if request.cancelled():
        _release(slots, waiters)
    else:
        request.set_result(None)


def _release(slots: threading.Semaphore, waiters: deque[asyncio.Future[None]]) -> None:
    # Transfer the occupied slot directly to the oldest waiter.
    while True:
        with _lock:
            if not waiters:
                slots.release()
                return
            request = waiters.popleft()
        try:
            request.get_loop().call_soon_threadsafe(_admit, request, slots, waiters)
            return
        except RuntimeError:
            # A closed event loop cannot accept the slot. Try the next waiter.
            continue


async def run_native(function, *args):
    global _executor, _pid, _slots, _waiters
    request = asyncio.get_running_loop().create_future()
    with _lock:
        if _executor is None or _pid != os.getpid():
            _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="braintrust-audio")
            _slots = threading.Semaphore(2)
            _waiters = deque()
            _pid = os.getpid()
        executor, slots, waiters = _executor, _slots, _waiters
        assert slots is not None
        assert waiters is not None
        # Waiting sessions retain their buffers; only admitted jobs reach the executor.
        if slots.acquire(blocking=False):
            request.set_result(None)
        else:
            waiters.append(request)
    try:
        await request
    except BaseException:
        with _lock:
            try:
                waiters.remove(request)
            except ValueError:
                pass
        # If admission won the race with cancellation, return its reserved slot.
        # Otherwise a pending _admit callback owns the handoff, if any.
        if not request.cancelled():
            _release(slots, waiters)
        raise
    try:
        future = executor.submit(partial(function, *args))
    except BaseException:
        _release(slots, waiters)
        raise
    future.add_done_callback(lambda _: _release(slots, waiters))
    wrapped = asyncio.wrap_future(future)
    # A cancelled waiter cannot consume a late native failure. Retrieve it
    # without logging potentially private exception details.
    wrapped.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
    return await asyncio.shield(wrapped)
