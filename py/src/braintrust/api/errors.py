"""Structured errors raised by the Braintrust API client."""

from collections.abc import Mapping
from types import MappingProxyType

from ..util import AugmentedHTTPError


_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "content-type",
        "retry-after",
        "x-bt-internal-trace-id",
        "x-vercel-id",
        "request-id",
        "x-request-id",
    }
)
_REQUEST_ID_HEADERS = ("x-bt-internal-trace-id", "x-vercel-id", "x-request-id", "request-id")


class BraintrustAPIError(Exception):
    """Base class for errors raised by the Braintrust API client."""


class BraintrustHTTPError(BraintrustAPIError, AugmentedHTTPError):
    """An HTTP error with status, request, and retry context."""

    method: str
    url: str
    status_code: int
    response_body: str
    request_id: str | None
    request_id_header: str | None
    response_headers: Mapping[str, str]
    attempts: int
    retryable: bool
    retry_after: float | None
    retry_after_header: str | None

    def __init__(
        self,
        *,
        method: str,
        url: str,
        status_code: int,
        response_body: str,
        response_headers: Mapping[str, str],
        attempts: int,
        retryable: bool,
        retry_after: float | None = None,
    ):
        headers = {
            normalized_name: value
            for name, value in response_headers.items()
            if (normalized_name := name.lower()) in _RESPONSE_HEADER_ALLOWLIST
        }
        request_id_header = next((name for name in _REQUEST_ID_HEADERS if headers.get(name)), None)
        request_id = headers.get(request_id_header) if request_id_header else None
        self.method = method
        self.url = url
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id
        self.request_id_header = request_id_header
        self.response_headers = MappingProxyType(headers)
        self.attempts = attempts
        self.retryable = retryable
        self.retry_after = retry_after
        self.retry_after_header = headers.get("retry-after")
        message = f"{method} {url} failed with HTTP {status_code} after {attempts} attempt(s)"
        if request_id:
            message += f" (request ID: {request_id})"
        if self.response_body:
            message += f": {self.response_body}"
        super().__init__(message)


class BraintrustRetryExhaustedError(BraintrustHTTPError):
    """A retryable HTTP response exhausted its operation policy."""


class BraintrustTransportError(BraintrustAPIError):
    """A request failed before a usable HTTP response was received."""

    method: str
    url: str
    attempts: int
    retryable: bool

    def __init__(self, *, method: str, url: str, attempts: int, retryable: bool):
        self.method = method
        self.url = url
        self.attempts = attempts
        self.retryable = retryable
        super().__init__(f"{method} {url} failed after {attempts} attempt(s) without an HTTP response")


class BraintrustTransportRetryExhaustedError(BraintrustTransportError):
    """Transport exceptions exhausted the operation's retry policy."""


class BraintrustResponseError(BraintrustAPIError):
    """A successful HTTP response could not be decoded."""

    method: str
    url: str
    status_code: int
    response_body: str
    response_headers: Mapping[str, str]
    attempts: int

    def __init__(
        self,
        *,
        method: str,
        url: str,
        status_code: int,
        response_body: str,
        response_headers: Mapping[str, str],
        attempts: int,
    ):
        self.method = method
        self.url = url
        self.status_code = status_code
        self.response_body = response_body
        self.response_headers = MappingProxyType(
            {
                normalized_name: value
                for name, value in response_headers.items()
                if (normalized_name := name.lower()) in _RESPONSE_HEADER_ALLOWLIST
            }
        )
        self.attempts = attempts
        super().__init__(f"Could not decode the response from {method} {url} as JSON")
