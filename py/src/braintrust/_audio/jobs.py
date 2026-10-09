"""Own background recording tasks and release their input after work stops."""

import asyncio
import logging
from collections.abc import Awaitable, Callable


class RecordingJobs:
    def __init__(self):
        self._tasks: set[asyncio.Task] = set()

    def __len__(self):
        return sum(not task.done() for task in self._tasks)

    def submit(
        self,
        operation: Callable[[], Awaitable[None]],
        release: Callable[[], None],
        *,
        after_release: Callable[[], Awaitable[None]] | None = None,
    ) -> asyncio.Task:
        released = False

        def release_once():
            nonlocal released
            if not released:
                released = True
                release()

        async def run():
            try:
                await operation()
            finally:
                release_once()
            if after_release:
                await after_release()

        task = asyncio.create_task(run())
        self._tasks.add(task)

        def completed(task):
            # Also runs if cancelled before run() starts. The encoding worker
            # does not complete cancellation until native work has stopped.
            try:
                release_once()
            finally:
                self._tasks.discard(task)
            if not task.cancelled() and (error := task.exception()) is not None:
                logging.getLogger(__name__).warning(
                    "Recording publication failed", exc_info=(type(error), error, error.__traceback__)
                )

        task.add_done_callback(completed)
        return task

    async def drain(self) -> None:
        if self._tasks:
            # A cancelled waiter must not release another task's encoder input.
            await asyncio.shield(asyncio.gather(*tuple(self._tasks), return_exceptions=True))
