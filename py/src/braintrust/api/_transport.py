"""Legacy and policy-aware HTTP transport primitives for the Braintrust SDK."""

import datetime
import logging
import sys
import time
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any, NoReturn

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..env import BraintrustEnv
from ..util import _urljoin, response_raise_for_status
from .errors import (
    BraintrustHTTPError,
    BraintrustResponseError,
    BraintrustRetryExhaustedError,
    BraintrustTransportError,
    BraintrustTransportRetryExhaustedError,
)
from .policies import RetryMode, RetryPolicy


logger = logging.getLogger(__name__)


class RetryRequestExceptionsAdapter(HTTPAdapter):
    """An HTTP adapter that automatically retries requests on connection exceptions.

    This adapter extends requests' HTTPAdapter to add retry logic for common network-related
    exceptions including connection errors, timeouts, and other HTTP errors. It implements
    an exponential backoff strategy between retries to avoid overwhelming servers during
    intermittent connectivity issues.

    Attributes:
        base_num_retries: Maximum number of retries before giving up and re-raising the exception.
        backoff_factor: A multiplier used to determine the time to wait between retries.
                       The actual wait time is calculated as: backoff_factor * (2 ** retry_count).
        default_timeout_secs: Default timeout in seconds for requests that don't specify one.
                             Prevents indefinite hangs on stale connections.
    """

    def __init__(
        self,
        *args: Any,
        base_num_retries: int = 0,
        backoff_factor: float = 0.5,
        default_timeout_secs: float = 60,
        **kwargs: Any,
    ):
        self.base_num_retries = base_num_retries
        self.backoff_factor = backoff_factor
        self.default_timeout_secs = default_timeout_secs
        super().__init__(*args, **kwargs)

    def send(self, *args, **kwargs):
        # Apply default timeout if none provided to prevent indefinite hangs
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.default_timeout_secs

        num_prev_retries = 0
        while True:
            try:
                response = super().send(*args, **kwargs)
                # Fully-download the content to ensure we catch any errors from
                # downloading.
                if not response.is_redirect and response.content:
                    pass
                return response
            except (urllib3.exceptions.HTTPError, requests.exceptions.RequestException) as e:
                if num_prev_retries < self.base_num_retries:
                    if isinstance(e, requests.exceptions.ReadTimeout):
                        # Clear all connection pools to discard stale connections. This
                        # fixes hangs caused by NAT gateways silently dropping idle TCP
                        # connections (e.g., Azure's ~4 min timeout). close() calls
                        # PoolManager.clear() which is thread-safe: in-flight requests
                        # keep their checked-out connections, and new requests create
                        # fresh pools on demand.
                        self.close()
                    # Emulates the sleeping logic in the backoff_factor of urllib3 Retry
                    sleep_s = self.backoff_factor * (2**num_prev_retries)
                    print("Retrying request after error:", e, file=sys.stderr)
                    print("Sleeping for", sleep_s, "seconds", file=sys.stderr)
                    time.sleep(sleep_s)
                    num_prev_retries += 1
                else:
                    raise e


class HTTPConnection:
    def __init__(self, base_url: str, adapter: HTTPAdapter | None = None):
        self.base_url = base_url
        self.token = None
        self.adapter = adapter

        self._reset(total=0)

    def ping(self) -> bool:
        try:
            resp = self.get("ping")
            return resp.ok
        except requests.exceptions.ConnectionError:
            return False

    def make_long_lived(self) -> None:
        if not self.adapter:
            timeout_secs = BraintrustEnv.HTTP_TIMEOUT.get(60.0)
            self.adapter = RetryRequestExceptionsAdapter(
                base_num_retries=10, backoff_factor=0.5, default_timeout_secs=timeout_secs
            )
        self._reset()

    @staticmethod
    def sanitize_token(token: str) -> str:
        return token.rstrip("\n")

    def set_token(self, token: str) -> None:
        token = HTTPConnection.sanitize_token(token)
        self.token = token
        self._set_session_token()

    def _set_adapter(self, adapter: HTTPAdapter | None) -> None:
        self.adapter = adapter

    def _reset(self, **retry_kwargs: Any) -> None:
        self.session = requests.Session()

        adapter = self.adapter
        if adapter is None:
            retry = Retry(**retry_kwargs)
            adapter = HTTPAdapter(max_retries=retry)

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._set_session_token()

    def _set_session_token(self) -> None:
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def get(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.get(_urljoin(self.base_url, path), *args, **kwargs)

    def post(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.post(_urljoin(self.base_url, path), *args, **kwargs)

    def patch(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.patch(_urljoin(self.base_url, path), *args, **kwargs)

    def put(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.put(_urljoin(self.base_url, path), *args, **kwargs)

    def delete(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.delete(_urljoin(self.base_url, path), *args, **kwargs)

    def get_json(self, object_type: str, args: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        resp = self.get(f"/{object_type}", params=args)
        response_raise_for_status(resp)
        return resp.json()

    def post_json(self, object_type: str, args: Mapping[str, Any] | None = None) -> Any:
        resp = self.post(f"/{object_type.lstrip('/')}", json=args)
        response_raise_for_status(resp)
        return resp.json()

    def patch_json(self, object_type: str, args: Mapping[str, Any] | None = None) -> Any:
        resp = self.patch(f"/{object_type.lstrip('/')}", json=args)
        response_raise_for_status(resp)
        return resp.json()


class Transport:
    """Policy-aware HTTP request engine for new API services.

    Existing ``HTTPConnection`` call sites deliberately do not use this class yet,
    so adding endpoint policies does not change their behavior during migration.
    Injected sessions and adapters own retries by default; callers may explicitly
    enable the SDK loop when they know the injected transport does not retry.
    """

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        adapter: HTTPAdapter | None = None,
        enable_sdk_retries: bool | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        custom_transport = session is not None or adapter is not None
        self._owns_session = session is None
        self.session = session if session is not None else requests.Session()
        self._sdk_retries_enabled = not custom_transport if enable_sdk_retries is None else enable_sdk_retries
        if adapter is not None:
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_clock = wall_clock

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
        retry_mode: RetryMode = RetryMode.NONE,
        retry_policy: RetryPolicy | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> requests.Response:
        method = method.upper()
        policy = retry_policy or RetryPolicy.for_mode(retry_mode)

        replay_safe = retry_mode in (RetryMode.SAFE_READ, RetryMode.IDEMPOTENT_WRITE)
        body_replayable = _request_body_is_replayable(data, kwargs.get("files"))
        sdk_retries_enabled = replay_safe and body_replayable and self._sdk_retries_enabled and policy.max_attempts > 1
        if replay_safe and not body_replayable:
            logger.debug("Disabling SDK retries for %s %s because its request body is not replayable", method, url)
        max_attempts = policy.max_attempts if sdk_retries_enabled else 1
        started_at = self._monotonic()

        for attempt in range(1, max_attempts + 1):
            remaining = self._remaining_budget(policy, started_at)
            if remaining is not None and remaining <= 0:
                # This is reachable only when an injected clock advances between
                # attempts; ordinary retry delays are checked before sleeping.
                error = BraintrustTransportRetryExhaustedError(
                    method=method, url=url, attempts=attempt - 1, retryable=True
                )
                raise error
            attempt_timeout = min(policy.timeout, remaining) if remaining is not None else policy.timeout

            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    data=data,
                    headers=headers,
                    timeout=attempt_timeout,
                    stream=stream,
                    **kwargs,
                )
            except requests.exceptions.RequestException as exc:
                if not _is_retryable_request_exception(exc):
                    error = BraintrustTransportError(method=method, url=url, attempts=attempt, retryable=False)
                    raise error from exc
                if attempt >= max_attempts:
                    error_type = (
                        BraintrustTransportRetryExhaustedError if sdk_retries_enabled else BraintrustTransportError
                    )
                    error = error_type(method=method, url=url, attempts=attempt, retryable=sdk_retries_enabled)
                    raise error from exc

                delay = _retry_delay(policy, attempt, None)
                if not self._can_wait(policy, started_at, delay):
                    error = BraintrustTransportRetryExhaustedError(
                        method=method, url=url, attempts=attempt, retryable=True
                    )
                    raise error from exc
                logger.debug(
                    "Retrying %s %s after transport error (attempt %d/%d) in %.3fs",
                    method,
                    url,
                    attempt,
                    max_attempts,
                    delay,
                )
                self._sleep(delay)
                continue

            if response.status_code < 400:
                setattr(response, "_braintrust_attempts", attempt)
                return response

            retry_after = _parse_retry_after(response.headers.get("Retry-After"), self._wall_clock())
            transient_status = response.status_code in policy.retryable_statuses
            retryable = replay_safe and body_replayable and transient_status
            if not (sdk_retries_enabled and transient_status):
                _raise_http_error(
                    response,
                    method=method,
                    url=url,
                    attempts=attempt,
                    retryable=retryable,
                    retry_after=retry_after,
                    exhausted=False,
                )

            if attempt >= max_attempts:
                _raise_http_error(
                    response,
                    method=method,
                    url=url,
                    attempts=attempt,
                    retryable=True,
                    retry_after=retry_after,
                    exhausted=True,
                )

            delay = _retry_delay(policy, attempt, retry_after)
            if not self._can_wait(policy, started_at, delay):
                _raise_http_error(
                    response,
                    method=method,
                    url=url,
                    attempts=attempt,
                    retryable=True,
                    retry_after=retry_after,
                    exhausted=True,
                    close_response=True,
                )

            logger.debug(
                "Retrying %s %s after HTTP %d (attempt %d/%d) in %.3fs; Retry-After=%r parsed=%r",
                method,
                url,
                response.status_code,
                attempt,
                max_attempts,
                delay,
                response.headers.get("Retry-After"),
                retry_after,
            )
            response.close()
            self._sleep(delay)

        raise AssertionError("retry loop exited unexpectedly")

    def request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        response = self.request(method, url, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            error = BraintrustResponseError(
                method=method.upper(),
                url=response.url or url,
                status_code=response.status_code,
                response_body=response.text,
                response_headers=response.headers,
                attempts=getattr(response, "_braintrust_attempts", 1),
            )
            raise error from exc

    def _remaining_budget(self, policy: RetryPolicy, started_at: float) -> float | None:
        if policy.max_elapsed_time is None:
            return None
        return policy.max_elapsed_time - (self._monotonic() - started_at)

    def _can_wait(self, policy: RetryPolicy, started_at: float, delay: float) -> bool:
        remaining = self._remaining_budget(policy, started_at)
        return remaining is None or delay < remaining


def _retry_delay(policy: RetryPolicy, attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return retry_after
    return min(policy.max_backoff, policy.backoff_factor * (2 ** (attempt - 1)))


def _request_body_is_replayable(data: Any, files: Any) -> bool:
    return files is None and (data is None or isinstance(data, (bytes, str)))


def _is_retryable_request_exception(exc: requests.exceptions.RequestException) -> bool:
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)) and not isinstance(
        exc, requests.exceptions.SSLError
    )


def _parse_retry_after(value: str | None, wall_time: float) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=datetime.timezone.utc)
        return max(0.0, retry_at.timestamp() - wall_time)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _raise_http_error(
    response: requests.Response,
    *,
    method: str,
    url: str,
    attempts: int,
    retryable: bool,
    retry_after: float | None,
    exhausted: bool,
    close_response: bool = False,
) -> NoReturn:
    error_type = BraintrustRetryExhaustedError if exhausted else BraintrustHTTPError
    error = error_type(
        method=method,
        url=response.url or url,
        status_code=response.status_code,
        response_body=response.text,
        response_headers=response.headers,
        attempts=attempts,
        retryable=retryable,
        retry_after=retry_after,
    )
    if close_response:
        response.close()
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as cause:
        raise error from cause
    raise error
