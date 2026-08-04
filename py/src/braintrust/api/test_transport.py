import contextlib
import datetime
import http.server
import io
import socketserver
import threading
import time
from email.utils import format_datetime

import pytest
import requests
from braintrust.api import (
    BraintrustHTTPError,
    BraintrustResponseError,
    BraintrustRetryExhaustedError,
    BraintrustTransportError,
    BraintrustTransportRetryExhaustedError,
    RetryMode,
    RetryPolicy,
)
from braintrust.api._transport import Transport
from braintrust.util import AugmentedHTTPError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class FakeClock:
    def __init__(self):
        self.monotonic_time = 0.0
        self.wall_time = 1_800_000_000.0
        self.sleeps = []

    def monotonic(self):
        return self.monotonic_time

    def time(self):
        return self.wall_time

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.monotonic_time += delay
        self.wall_time += delay


@contextlib.contextmanager
def scripted_server(script):
    class ScriptedHandler(http.server.BaseHTTPRequestHandler):
        request_count = 0
        requests = []

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle()

        def _handle(self):
            request_number = type(self).request_count
            type(self).request_count += 1
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length else b""
            type(self).requests.append((self.command, self.path, body))
            action = script[min(request_number, len(script) - 1)]

            if action == "close":
                self.connection.close()
                return

            if action[0] == "sleep":
                _, delay, status, headers, response_body = action
                time.sleep(delay)
            else:
                status, headers, response_body = action

            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            try:
                self.wfile.write(response_body)
            except BrokenPipeError:
                pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ScriptedHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", ScriptedHandler
    finally:
        server.shutdown()
        server.server_close()


def make_transport(clock=None, adapter=None):
    clock = clock or FakeClock()
    return Transport(
        adapter=adapter,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        wall_clock=clock.time,
    )


class TrackingAdapter(HTTPAdapter):
    def __init__(self):
        super().__init__()
        self.close_count = 0

    def close(self):
        self.close_count += 1
        super().close()


class TrackingSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.close_count = 0

    def close(self):
        self.close_count += 1
        super().close()


def test_transport_closes_owned_session():
    adapter = TrackingAdapter()

    with Transport(adapter=adapter) as transport:
        assert transport.session is not None

    assert adapter.close_count > 0


def test_transport_does_not_close_injected_session():
    session = TrackingSession()

    with Transport(session=session) as transport:
        assert transport.session is session

    assert session.close_count == 0
    session.close()


def test_retry_policy_defaults_and_validation():
    safe_read = RetryPolicy.for_mode(RetryMode.SAFE_READ)
    assert safe_read.max_attempts == 4
    assert safe_read.max_elapsed_time == 60
    assert safe_read.timeout == 20
    assert safe_read.retryable_statuses == frozenset({408, 429, 500, 502, 503, 504})

    for mode in (RetryMode.NONE, RetryMode.LOG_INGESTION):
        policy = RetryPolicy.for_mode(mode)
        assert policy.max_attempts == 1
        assert policy.timeout == 60

    with pytest.raises(ValueError, match="greater than its timeout"):
        RetryPolicy(max_attempts=2, max_elapsed_time=20, timeout=20)


def test_safe_logical_post_retries_transient_status():
    script = [
        (503, {}, b'{"error":"unavailable"}'),
        (200, {"Content-Type": "application/json"}, b'{"ok":true}'),
    ]
    clock = FakeClock()
    with scripted_server(script) as (url, handler):
        result = make_transport(clock).request_json(
            "POST", f"{url}/btql", json={"query": "select 1"}, retry_mode=RetryMode.SAFE_READ
        )

    assert result == {"ok": True}
    assert handler.request_count == 2
    assert [request[0] for request in handler.requests] == ["POST", "POST"]
    assert clock.sleeps == [0.5]


def test_none_and_log_ingestion_never_retry():
    for mode in (RetryMode.NONE, RetryMode.LOG_INGESTION):
        with scripted_server([(429, {"Retry-After": "0"}, b"limited"), (200, {}, b"ok")]) as (url, handler):
            with pytest.raises(BraintrustHTTPError) as exc_info:
                make_transport().request("POST", url, retry_mode=mode)

        assert not isinstance(exc_info.value, BraintrustRetryExhaustedError)
        assert exc_info.value.status_code == 429
        assert exc_info.value.attempts == 1
        assert handler.request_count == 1


@pytest.mark.parametrize("status", [400, 401, 403, 501, 505, 507, 508, 511])
def test_non_retryable_statuses_are_attempted_once(status):
    with scripted_server([(status, {}, b"failed"), (200, {}, b"ok")]) as (url, handler):
        with pytest.raises(BraintrustHTTPError) as exc_info:
            make_transport().request("GET", url, retry_mode=RetryMode.SAFE_READ)

    assert not isinstance(exc_info.value, BraintrustRetryExhaustedError)
    assert handler.request_count == 1


def test_retry_after_delay_seconds_is_used_directly():
    clock = FakeClock()
    with scripted_server([(429, {"Retry-After": "2"}, b"limited"), (200, {}, b"ok")]) as (url, handler):
        response = make_transport(clock).request("GET", url, retry_mode=RetryMode.SAFE_READ)

    assert response.status_code == 200
    assert handler.request_count == 2
    assert clock.sleeps == [2]


def test_retry_after_http_date_uses_injected_wall_clock():
    clock = FakeClock()
    retry_at = datetime.datetime.fromtimestamp(clock.wall_time + 5, tz=datetime.timezone.utc)
    with scripted_server(
        [(503, {"Retry-After": format_datetime(retry_at, usegmt=True)}, b"later"), (200, {}, b"ok")]
    ) as (url, _):
        make_transport(clock).request("GET", url, retry_mode=RetryMode.SAFE_READ)

    assert clock.sleeps == [5]


@pytest.mark.parametrize("retry_after", [None, "not-a-delay"])
def test_absent_or_malformed_retry_after_uses_backoff(retry_after):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    clock = FakeClock()
    with scripted_server([(429, headers, b"limited"), (200, {}, b"ok")]) as (url, _):
        make_transport(clock).request("GET", url, retry_mode=RetryMode.SAFE_READ)

    assert clock.sleeps == [0.5]


def test_retry_after_larger_than_budget_raises_without_sleeping():
    clock = FakeClock()
    policy = RetryPolicy(max_attempts=2, max_elapsed_time=2, timeout=1)
    with scripted_server([(429, {"Retry-After": "5"}, b"limited")]) as (url, handler):
        with pytest.raises(BraintrustRetryExhaustedError) as exc_info:
            make_transport(clock).request("GET", url, retry_mode=RetryMode.SAFE_READ, retry_policy=policy)

    error = exc_info.value
    assert error.status_code == 429
    assert error.attempts == 1
    assert error.retry_after == 5
    assert error.retry_after_header == "5"
    assert handler.request_count == 1
    assert clock.sleeps == []


def test_retry_exhaustion_preserves_structured_http_context_and_compatibility():
    policy = RetryPolicy(max_attempts=2, max_elapsed_time=2, timeout=1)
    response_headers = {
        "x-bt-internal-trace-id": "trace-id",
        "x-vercel-id": "deployment-id",
        "Set-Cookie": "secret=cookie",
        "Content-Type": "application/json",
    }
    body = b'{"error":"failed","api_key":"super-secret"}'
    with scripted_server([(500, response_headers, body)]) as (url, handler):
        with pytest.raises(BraintrustRetryExhaustedError) as exc_info:
            make_transport().request("GET", f"{url}/test", retry_mode=RetryMode.SAFE_READ, retry_policy=policy)

    error = exc_info.value
    assert isinstance(error, AugmentedHTTPError)
    assert isinstance(error.__cause__, requests.exceptions.HTTPError)
    assert error.method == "GET"
    assert error.url == f"{url}/test"
    assert error.status_code == 500
    assert error.attempts == 2
    assert error.retryable is True
    assert error.request_id == "trace-id"
    assert error.request_id_header == "x-bt-internal-trace-id"
    assert error.response_headers["content-type"] == "application/json"
    assert "set-cookie" not in error.response_headers
    assert error.response_body == body.decode()
    assert str(error).endswith(body.decode())
    assert handler.request_count == 2


def test_transport_exception_retries_and_final_error_chains_cause():
    policy = RetryPolicy(max_attempts=2, max_elapsed_time=2, timeout=1)
    with scripted_server(["close"]) as (url, handler):
        with pytest.raises(BraintrustTransportRetryExhaustedError) as exc_info:
            make_transport().request("GET", url, retry_mode=RetryMode.SAFE_READ, retry_policy=policy)

    error = exc_info.value
    assert isinstance(error, BraintrustTransportError)
    assert error.attempts == 2
    assert error.retryable is True
    assert isinstance(error.__cause__, requests.exceptions.RequestException)
    assert handler.request_count == 2


def test_none_wraps_transport_exception_without_retrying():
    with scripted_server(["close"]) as (url, handler):
        with pytest.raises(BraintrustTransportError) as exc_info:
            make_transport().request("POST", url, retry_mode=RetryMode.NONE)

    assert not isinstance(exc_info.value, BraintrustTransportRetryExhaustedError)
    assert exc_info.value.attempts == 1
    assert handler.request_count == 1


def test_deterministic_request_error_is_not_retried():
    clock = FakeClock()
    with pytest.raises(BraintrustTransportError) as exc_info:
        make_transport(clock).request("GET", "ftp://example.com", retry_mode=RetryMode.SAFE_READ)

    assert not isinstance(exc_info.value, BraintrustTransportRetryExhaustedError)
    assert exc_info.value.attempts == 1
    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, requests.exceptions.InvalidSchema)
    assert clock.sleeps == []


def test_safe_read_timeout_leaves_room_for_retry():
    policy = RetryPolicy(max_attempts=2, max_elapsed_time=1, timeout=0.1, backoff_factor=0)
    script = [
        ("sleep", 0.3, 200, {}, b"late"),
        (200, {}, b"ok"),
    ]
    with scripted_server(script) as (url, handler):
        response = Transport().request("GET", url, retry_mode=RetryMode.SAFE_READ, retry_policy=policy)

    assert response.status_code == 200
    assert handler.request_count == 2


def test_invalid_json_is_not_retried_and_preserves_response_context():
    with scripted_server([(200, {"Content-Type": "application/json"}, b"not json")]) as (url, handler):
        with pytest.raises(BraintrustResponseError) as exc_info:
            make_transport().request_json("GET", url, retry_mode=RetryMode.SAFE_READ)

    assert exc_info.value.status_code == 200
    assert exc_info.value.response_body == "not json"
    assert exc_info.value.attempts == 1
    assert handler.request_count == 1
    assert exc_info.value.__cause__ is not None


def test_file_like_request_body_disables_sdk_retries():
    body = io.BytesIO(b"payload")
    with scripted_server([(500, {}, b"failed"), (200, {}, b"ok")]) as (url, handler):
        with pytest.raises(BraintrustHTTPError) as exc_info:
            make_transport().request("POST", url, data=body, retry_mode=RetryMode.SAFE_READ)

    assert not isinstance(exc_info.value, BraintrustRetryExhaustedError)
    assert exc_info.value.retryable is False
    assert handler.request_count == 1
    assert handler.requests[0][2] == b"payload"


def test_file_upload_disables_sdk_retries():
    files = {"file": ("payload.txt", io.BytesIO(b"payload"))}
    with scripted_server([(500, {}, b"failed"), (200, {}, b"ok")]) as (url, handler):
        with pytest.raises(BraintrustHTTPError) as exc_info:
            make_transport().request("POST", url, files=files, retry_mode=RetryMode.IDEMPOTENT_WRITE)

    assert not isinstance(exc_info.value, BraintrustRetryExhaustedError)
    assert handler.request_count == 1


def retrying_adapter():
    return HTTPAdapter(
        max_retries=Retry(
            total=1,
            status=1,
            backoff_factor=0,
            status_forcelist={500},
            allowed_methods=None,
            raise_on_status=False,
        )
    )


def test_custom_adapter_disables_sdk_retry_loop():
    with scripted_server([(500, {}, b"failed")]) as (url, handler):
        with pytest.raises(BraintrustHTTPError) as exc_info:
            make_transport(adapter=retrying_adapter()).request("GET", url, retry_mode=RetryMode.SAFE_READ)

    assert not isinstance(exc_info.value, BraintrustRetryExhaustedError)
    assert exc_info.value.attempts == 1
    assert handler.request_count == 2  # The adapter's two attempts, with no outer-loop multiplication.


def test_injected_session_disables_sdk_retry_loop():
    session = requests.Session()
    session.mount("http://", retrying_adapter())
    with scripted_server([(500, {}, b"failed")]) as (url, handler):
        with pytest.raises(BraintrustHTTPError) as exc_info:
            Transport(session=session).request("GET", url, retry_mode=RetryMode.SAFE_READ)

    assert not isinstance(exc_info.value, BraintrustRetryExhaustedError)
    assert exc_info.value.attempts == 1
    assert handler.request_count == 2


def test_non_retrying_custom_adapter_can_delegate_retries_to_sdk():
    clock = FakeClock()
    transport = Transport(
        adapter=HTTPAdapter(),
        enable_sdk_retries=True,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        wall_clock=clock.time,
    )
    with scripted_server([(500, {}, b"failed"), (200, {}, b"ok")]) as (url, handler):
        response = transport.request("GET", url, retry_mode=RetryMode.SAFE_READ)

    assert response.status_code == 200
    assert handler.request_count == 2
