"""Retry policies for Braintrust API operations."""

import enum
from dataclasses import dataclass

import requests


DEFAULT_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_MAX_ELAPSED_TIME = 60.0
DEFAULT_BACKOFF_FACTOR = 0.5
DEFAULT_MAX_BACKOFF = 10.0


def is_retryable_request_exception(exc: requests.exceptions.RequestException) -> bool:
    """Return whether a requests transport failure is safe to retry."""
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)) and not isinstance(
        exc, requests.exceptions.SSLError
    )


class RetryMode(enum.Enum):
    """The replay safety classification for an API operation."""

    NONE = "none"
    SAFE_READ = "safe_read"
    IDEMPOTENT_WRITE = "idempotent_write"
    LOG_INGESTION = "log_ingestion"


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry settings for one API operation."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_elapsed_time: float | None = DEFAULT_MAX_ELAPSED_TIME
    timeout: float = 20.0
    retryable_statuses: frozenset[int] = DEFAULT_RETRYABLE_STATUSES
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    max_backoff: float = DEFAULT_MAX_BACKOFF

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.max_elapsed_time is not None and self.max_elapsed_time <= 0:
            raise ValueError("max_elapsed_time must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_attempts > 1:
            if self.max_elapsed_time is None:
                raise ValueError("a retrying policy must have a maximum elapsed time")
            if self.max_elapsed_time <= self.timeout:
                raise ValueError("max_elapsed_time must be greater than its timeout")
        if self.backoff_factor < 0 or self.max_backoff < 0:
            raise ValueError("backoff settings cannot be negative")

    @classmethod
    def for_mode(cls, mode: RetryMode) -> "RetryPolicy":
        if mode in (RetryMode.NONE, RetryMode.LOG_INGESTION):
            return cls(max_attempts=1, max_elapsed_time=None, timeout=60.0)
        if mode in (RetryMode.SAFE_READ, RetryMode.IDEMPOTENT_WRITE):
            return cls()
        raise ValueError(f"Unknown retry mode: {mode!r}")
