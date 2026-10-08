"""Lazy process-wide recording worker with bounded admission.

At most one running and one queued job. Caller cancellation does not release
capacity while its native encoder is still running.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor


_slots = threading.BoundedSemaphore(2)
_lock = threading.Lock()
_executor = None


class RecordingBusy(RuntimeError):
    pass


async def encode_in_worker(function, *args):
    global _executor
    if not _slots.acquire(blocking=False):
        raise RecordingBusy("recording_worker_capacity")
    try:
        with _lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="braintrust-audio")

        def work():
            try:
                return function(*args)
            finally:
                _slots.release()

        future = _executor.submit(work)
    except BaseException:
        _slots.release()
        raise
    return await await_background(asyncio.wrap_future(future))


async def await_background(work):
    """Drain non-cancellable thread work before releasing its owning job."""
    wrapped = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(wrapped)
    except asyncio.CancelledError:
        # Native work cannot be cancelled. Keep its source lease alive until it
        # exits, even if shutdown cancels the task that owns the buffers.
        while not wrapped.done():
            try:
                await asyncio.shield(wrapped)
            except asyncio.CancelledError:
                continue
        if not wrapped.cancelled():
            wrapped.exception()
        raise
