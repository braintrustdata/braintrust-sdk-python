"""Internal single-attempt ingestion service and destination admission control."""

import contextlib
import hashlib
import threading
import time
import weakref
from typing import Any

from requests.adapters import HTTPAdapter

from ..env import BraintrustEnv
from ._routing import EndpointRouter, RequestTarget
from ._service import ResourceAPI
from ._transport import RetryRequestExceptionsAdapter, Transport
from .errors import BraintrustHTTPError
from .policies import DEFAULT_RETRYABLE_STATUSES, RetryMode, RetryPolicy


class IngestionDeferred(Exception):
    """Admission changed after the writer scheduled an attempt."""


class IngestionDestination:
    def __init__(self, concurrency: int):
        self.lock = threading.Lock()
        self.concurrency = concurrency
        self.active = 0
        self.cooldown_until = 0.0
        self.next_start = 0.0
        self.recovery_starts = 0

    def delay(self) -> float:
        with self.lock:
            return max(0.0, self.cooldown_until - time.monotonic(), self.next_start - time.monotonic())

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            if self.active >= self.concurrency or now < max(self.cooldown_until, self.next_start):
                raise IngestionDeferred()
            self.active += 1
            if self.recovery_starts:
                self.next_start = now + 0.05
                self.recovery_starts -= 1

    def release(self, retry_after: float | None = None) -> None:
        with self.lock:
            if retry_after is not None:
                self.cooldown_until = max(self.cooldown_until, time.monotonic() + retry_after)
                self.recovery_starts = self.concurrency
            self.active -= 1


_destinations: weakref.WeakValueDictionary[tuple[str, str], IngestionDestination] = weakref.WeakValueDictionary()
_destinations_lock = threading.Lock()


class LogIngestionAPI(ResourceAPI):
    """Dedicated pools, existing router/auth, and no HTTP retries underneath the writer."""

    def __init__(self, router: EndpointRouter, api_key: str, *, concurrency: int, adapter: HTTPAdapter | None = None):
        if concurrency < 1:
            raise ValueError("Log concurrency must be positive")
        self._validate_adapter(adapter)
        timeout = BraintrustEnv.HTTP_TIMEOUT.get(60.0)
        super().__init__(
            Transport(adapter=adapter, request_timeout=timeout, persist_cookies=False, pool_maxsize=concurrency),
            router,
            api_key,
        )
        self.storage = Transport(
            adapter=adapter, request_timeout=timeout, persist_cookies=False, pool_maxsize=concurrency
        )
        key = (router.resolve(RequestTarget.API, "/logs3"), hashlib.sha256(api_key.encode()).hexdigest())
        with _destinations_lock:
            destination = _destinations.get(key)
            if destination is None:
                destination = IngestionDestination(concurrency)
                _destinations[key] = destination
            self.destination = destination

    @property
    def router(self) -> EndpointRouter:
        return self._router

    @property
    def transport(self) -> Transport:
        return self._transport

    @staticmethod
    def _validate_adapter(adapter: HTTPAdapter | None) -> None:
        if adapter is not None and (
            adapter.max_retries.total not in (0, False)
            or isinstance(adapter, RetryRequestExceptionsAdapter)
            and adapter.base_num_retries > 0
        ):
            raise ValueError("Log ingestion requires a single-attempt HTTP adapter; the writer owns retries")

    def request(self, method: str, path: str, **kwargs: Any):
        return self._request(RequestTarget.API, method, path, retry_mode=RetryMode.LOG_INGESTION, **kwargs)

    def request_json(self, method: str, path: str, **kwargs: Any):
        return self._request_json(RequestTarget.API, method, path, retry_mode=RetryMode.LOG_INGESTION, **kwargs)

    @contextlib.contextmanager
    def attempt(self, *, retry_after_on_429: float = 1.0):
        """Admit one writer attempt and publish any server cooldown before releasing it."""
        self.destination.acquire()
        retry_after = None
        try:
            yield
        except BraintrustHTTPError as error:
            if error.status_code in DEFAULT_RETRYABLE_STATUSES:
                retry_after = error.retry_after
                if retry_after is None and error.status_code == 429:
                    retry_after = retry_after_on_429
            raise
        finally:
            self.destination.release(retry_after)

    def version(self, policy: RetryPolicy | None = None):
        with self.attempt():
            return self._request_json(
                RequestTarget.API, "GET", "/version", retry_mode=RetryMode.LOG_INGESTION, retry_policy=policy
            )

    def close(self) -> None:
        self._transport.close()
        self.storage.close()
