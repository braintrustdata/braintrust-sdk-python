import threading
from collections import deque
from time import monotonic
from typing import TypeVar

from .util import eprint


T = TypeVar("T")

DEFAULT_QUEUE_SIZE = 25000


class LogQueue:
    """A thread-safe queue with a fixed size that drops oldest items when full.

    When enforcement is disabled (default), drops happen silently.
    When enforcement is enabled, dropped items are tracked and returned.
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize the LogQueue.

        Args:
            maxsize: Maximum size of the queue. If 0 or negative, defaults to DEFAULT_QUEUE_SIZE.
        """
        if maxsize < 1:
            eprint(f"Queue maxsize {maxsize} is invalid, using default size {DEFAULT_QUEUE_SIZE}")
            maxsize = DEFAULT_QUEUE_SIZE

        self.maxsize = maxsize
        self._mutex = threading.Lock()
        self._queue: deque[T] = deque(maxlen=maxsize)
        self._condition = threading.Condition(self._mutex)
        self._batch_min_size: int | None = None
        self._pending_since: float | None = None
        self._total_dropped = 0
        self._enforce_size_limit = False

    def enforce_queue_size_limit(self, enforce: bool) -> None:
        """
        Set queue size limit enforcement. When enabled, the queue will drop oldest items
        when it reaches maxsize and return them from put(). When disabled (default),
        the queue will still drop oldest items when full but won't track or return them.

        Args:
            enforce: Whether to track and return dropped items.
        """
        with self._mutex:
            self._enforce_size_limit = enforce

    def put(self, item: T) -> list[T]:
        """
        Put an item in the queue.

        Args:
            item: The item to add to the queue.

        Returns:
            List of items that were dropped (empty if no items were dropped).
        """
        with self._mutex:
            dropped = []
            was_empty = not self._queue

            if not self._enforce_size_limit:
                # For queues with enforcement disabled, deque auto-drops silently
                self._queue.append(item)
            else:
                # For bounded queues with enforcement, explicitly track drops
                while len(self._queue) >= self.maxsize:
                    dropped_item = self._queue.popleft()
                    dropped.append(dropped_item)
                    self._total_dropped += 1
                self._queue.append(item)

            if was_empty:
                self._pending_since = monotonic()
                self._condition.notify_all()
            elif self._batch_min_size is not None and len(self._queue) >= self._batch_min_size:
                self._condition.notify_all()

        return dropped

    def drain_all(self) -> list[T]:
        """
        Drain all items from the queue.

        Returns:
            List of all items that were in the queue.
        """
        old_queue = None
        with self._mutex:
            if len(self._queue) == 0:
                return []

            old_queue = self._queue
            self._queue = deque(maxlen=self.maxsize)
            self._pending_since = None

            # An explicit flush may drain the queue while the publisher waits.
            if self._batch_min_size is not None:
                self._condition.notify_all()

        return list(old_queue) if old_queue else []

    def size(self) -> int:
        """
        Get the current size of the queue.

        Returns:
            Number of items currently in the queue.
        """
        return len(self._queue)

    def wait_for_items(self, timeout: float | None = None) -> bool:
        """
        Will block until the queue has at least one item in it. Might be empty by the time
        you read though.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if items became available, False if timeout occurred.
        """
        with self._condition:
            return self._condition.wait_for(lambda: bool(self._queue), timeout=timeout)

    def wait_for_batch(self, min_size: int, max_delay: float) -> None:
        """Wait for a full batch or the oldest pending row's deadline.

        Used by the single background publisher. New arrivals do not extend
        the deadline; draining the queue starts a new window for the next row.
        Like wait_for_items(), the queue may be empty by the time this returns.
        """
        with self._condition:
            self._batch_min_size = min_size
            try:
                while True:
                    if self._pending_since is None:
                        self._condition.wait()
                        continue
                    remaining = self._pending_since + max_delay - monotonic()
                    if len(self._queue) >= min_size or remaining <= 0:
                        return
                    self._condition.wait(timeout=remaining)
            finally:
                self._batch_min_size = None
