"""Writer regressions exercised through real HTTP requests."""

import contextlib
import json

import pytest
from braintrust.api._ingestion import LogIngestionAPI
from braintrust.api._routing import EndpointRouter
from braintrust.api._test_server import scripted_server
from braintrust.logger import _HTTPBackgroundLogger
from braintrust.util import LazyValue


def test_permanent_failure_is_reported_and_released(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    with scripted_server([(413, {}, b"Payload Too Large")]) as (url, handler):
        connection = LogIngestionAPI(EndpointRouter(app_url=url, api_url=url), "test", concurrency=4)
        writer = _HTTPBackgroundLogger(lambda: contextlib.nullcontext(connection))
        writer._max_request_size_result = {"max_request_size": 10**9, "can_use_overflow": False}
        writer.queue.put(LazyValue(lambda: {"id": "score-row", "scores": {"quality": 1}}, use_mutex=False))
        from braintrust.logger import BraintrustLogFlushError

        with pytest.raises(BraintrustLogFlushError, match="413"):
            writer.flush()
        writer.flush()
        assert handler.request_count == 1
        assert json.loads(handler.requests[0][2])["rows"][0]["id"] == "score-row"
        assert writer.pending_count == 0
        connection.close()


@pytest.fixture
def ingestion_writer(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    with contextlib.ExitStack() as stack:

        def make(script, *, concurrency=4, capacity=100, persistent=True):
            url, handler = stack.enter_context(scripted_server(script, persistent=persistent))
            service = stack.enter_context(
                contextlib.closing(
                    LogIngestionAPI(
                        EndpointRouter(app_url=url, api_url=url),
                        "test",
                        concurrency=concurrency,
                    )
                )
            )
            writer = _HTTPBackgroundLogger(lambda: contextlib.nullcontext(service))
            writer.max_concurrency = concurrency
            writer.queue.maxsize = capacity
            writer._max_request_size_result = {"max_request_size": 10**9, "can_use_overflow": False}
            return writer, service, handler

        yield make


def enqueue(writer, *rows):
    for row in rows:
        writer.queue.put(LazyValue(lambda row=row: row, use_mutex=False))


@pytest.mark.parametrize("status,retry_after", [(429, "120"), (503, "120"), (429, "date"), (429, None)])
def test_cooldown_blocks_new_batches_and_is_scoped_to_destination(ingestion_writer, status, retry_after):
    import time

    from braintrust.logger import BraintrustLogFlushError

    if retry_after == "date":
        import datetime
        from email.utils import format_datetime

        retry_after = format_datetime(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=120), usegmt=True
        )
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    writer, service, handler = ingestion_writer([(status, headers, b"later")], concurrency=1)
    enqueue(writer, {"id": "old"})
    started = time.monotonic()
    with pytest.raises(BraintrustLogFlushError):
        writer.flush(timeout=0.1)
    assert time.monotonic() - started < 0.5
    enqueue(writer, {"id": "new"})
    with pytest.raises(BraintrustLogFlushError):
        writer.flush(timeout=0.1)
    assert writer.pending_count == 2
    assert handler.request_count == 1
    assert service.destination.delay() > 0.5
    # A separate writer with the same URL/credentials must also honor the cooldown.
    other = _HTTPBackgroundLogger(lambda: contextlib.nullcontext(service))
    other._max_request_size_result = writer._max_request_size_result
    enqueue(other, {"id": "other"})
    with pytest.raises(BraintrustLogFlushError):
        other.flush(timeout=0.1)
    assert handler.request_count == 1
    independent, _, independent_handler = ingestion_writer([(200, {}, b"ok")])
    enqueue(independent, {"id": "independent"})
    independent.flush()
    assert independent_handler.request_count == 1


def test_retry_exhaustion_retains_payload_and_orders_later_updates(ingestion_writer, tmp_path):
    from braintrust.logger import BraintrustLogFlushError

    writer, _, handler = ingestion_writer([(503, {}, b"later"), (503, {}, b"later"), (200, {}, b"ok")], capacity=2)
    writer.num_tries = 2
    writer.failed_publish_payloads_dir = str(tmp_path)
    enqueue(writer, {"id": "row", "scores": {"quality": 0}})
    with pytest.raises(BraintrustLogFlushError) as failure:
        writer.flush()
    assert failure.value.pending_count == 1
    assert handler.request_count == 2
    assert len(list(tmp_path.glob("*.json"))) == 1
    prepared = writer._pending[0].payload
    enqueue(writer, {"id": "row", "_is_merge": True, "scores": {"quality": 1}})
    # Producers remain non-throwing while an earlier wave is retained.
    writer.queue.put(LazyValue(lambda: {"id": "later"}, use_mutex=False))
    writer.flush()
    assert writer.pending_count == 0
    assert handler.requests[0][2] == handler.requests[1][2] == handler.requests[2][2] == prepared
    assert json.loads(handler.requests[3][2])["rows"][0]["scores"] == {"quality": 1}


def test_bounded_concurrent_recovery_is_staggered_and_reuses_connections(ingestion_writer):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    arrivals = []
    accepted = []
    active = 0
    maximum = 0
    recover_at = time.monotonic() + 0.4
    lock = threading.Lock()

    def respond(method, path, body, headers):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            now = time.monotonic()
            arrivals.append(now)
        time.sleep(0.015)
        with lock:
            active -= 1
            if now < recover_at:
                return 429, {"Retry-After": "1"}, b"limited"
            accepted.append(now)
        return 200, {}, b"ok"

    writer, service, handler = ingestion_writer(respond, concurrency=3)
    other = _HTTPBackgroundLogger(lambda: contextlib.nullcontext(service))
    other.max_concurrency = 3
    other._max_request_size_result = writer._max_request_size_result
    enqueue(writer, *({"id": str(i)} for i in range(6)))
    enqueue(other, *({"id": str(i)} for i in range(6, 9)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(target.flush, batch_size=1) for target in (writer, other)]
        for future in futures:
            future.result(timeout=5)
    assert maximum <= 3
    assert len(accepted) == 9
    assert handler.request_count <= 12
    assert len(handler.connections) <= 3
    assert min(accepted) - max(t for t in arrivals if t < recover_at) >= 1
    assert all(b - a >= 0.04 for a, b in zip(accepted[:3], accepted[1:3]))
    assert writer.pending_count == other.pending_count == 0


@pytest.mark.parametrize("method", ["PUT", "POST"])
@pytest.mark.parametrize("failure_stage", ["ingest", "upload", "url"])
def test_overflow_retries_reuse_signed_upload_and_isolate_credentials(monkeypatch, method, failure_stage):
    from braintrust.logger import BraintrustLogFlushError

    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    logs = 0
    urls = 0
    uploads = []
    with contextlib.ExitStack() as stack:

        def storage_response(verb, path, body, headers):
            uploads.append((verb, body, headers.get("Authorization")))
            if failure_stage == "upload" and len(uploads) == 1:
                return 503, {}, b"retry"
            return 200, {}, b"ok"

        storage_url, _ = stack.enter_context(scripted_server(storage_response, persistent=True))

        def response(verb, path, body, headers):
            nonlocal logs, urls
            if path == "/version":
                return 200, {}, b'{"logs3_payload_max_bytes": 80}'
            if path == "/logs3/overflow":
                urls += 1
                if failure_stage == "url" and urls == 1:
                    return 503, {}, b"retry"
                return (
                    200,
                    {},
                    json.dumps(
                        {
                            "method": method,
                            "signedUrl": storage_url,
                            "headers": {"Content-Type": "application/json"},
                            "fields": {"key": "object"},
                            "key": "object",
                        }
                    ).encode(),
                )
            logs += 1
            return (503, {}, b"retry") if failure_stage == "ingest" and logs == 1 else (200, {}, b"ok")

        writer, service, handler = stack.enter_context(_writer_context(response))
        writer.num_tries = 1
        enqueue(writer, {"id": "large", "input": "ü" * 300})
        with pytest.raises(BraintrustLogFlushError):
            writer.flush()
        payload = writer._pending[0].payload
        writer.flush()
        assert len(uploads) == (2 if failure_stage == "upload" else 1)
        assert all(upload[0] == method and upload[2] is None and payload in upload[1] for upload in uploads)
        assert writer._overflow_upload_count == 1
        assert [r[1] for r in handler.requests].count("/logs3/overflow") == (2 if failure_stage == "url" else 1)
        log_requests = [r for r in handler.requests if r[1] == "/logs3"]
        assert len(log_requests) == (2 if failure_stage == "ingest" else 1)
        assert all(request[2] == log_requests[0][2] for request in log_requests)
        assert all(r[3] == "Bearer test" for r in handler.requests)


# This context also makes version negotiation part of the HTTP regression coverage.
@contextlib.contextmanager
def _writer_context(script):
    with scripted_server(script, persistent=True) as (url, handler):
        with contextlib.closing(
            LogIngestionAPI(EndpointRouter(app_url=url, api_url=url), "test", concurrency=4)
        ) as service:
            writer = _HTTPBackgroundLogger(lambda: contextlib.nullcontext(service))
            yield writer, service, handler


def test_relogin_and_adapter_replacement_keep_retained_batches_on_original_credentials(monkeypatch):
    from braintrust import logger
    from braintrust.logger import BraintrustLogFlushError, BraintrustState
    from requests.adapters import HTTPAdapter

    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    monkeypatch.delenv("BRAINTRUST_API_URL", raising=False)
    monkeypatch.delenv("BRAINTRUST_PROXY_URL", raising=False)
    monkeypatch.delenv("BRAINTRUST_ORG_NAME", raising=False)
    calls = 0

    def respond(method, path, body, headers):
        nonlocal calls
        if path == "/api/apikey/login":
            return 200, {}, json.dumps({"org_info": [{"id": "org", "name": "org", "api_url": url}]}).encode()
        if path == "/version":
            return 200, {}, b"{}"
        if path == "/logs3":
            calls += 1
            return (503, {}, b"later") if calls == 1 else (200, {}, b"ok")
        return 200, {}, b'{"ok": true}'

    with scripted_server(respond, persistent=True) as (url, handler):
        state = BraintrustState()
        monkeypatch.setattr(logger, "_state", state)
        monkeypatch.setattr(logger, "_http_adapter", None)
        state.login(app_url=url, api_key="original")
        writer = state.global_bg_logger()
        writer.num_tries = 1
        enqueue(writer, {"id": "retained"})
        with pytest.raises(BraintrustLogFlushError):
            writer.flush()
        original = writer._pending_service
        owned_adapter = original.transport.session.get_adapter(url)
        assert len(owned_adapter.poolmanager.pools) == 1
        assert owned_adapter._pool_block is False
        state.login(app_url=url, api_key="replacement", force_login=True)
        caller_adapter = HTTPAdapter()
        logger.set_http_adapter(caller_adapter)
        # Evicted while leased: old pools remain usable until the retained wave is delivered.
        assert len(owned_adapter.poolmanager.pools) == 1
        enqueue(writer, {"id": "new"})
        writer.flush()
        assert len(owned_adapter.poolmanager.pools) == 0
        requests = [request for request in handler.requests if request[1] == "/logs3"]
        assert [request[3] for request in requests] == ["Bearer original", "Bearer original", "Bearer replacement"]
        assert requests[0][2] == requests[1][2]
        assert state.api_conn().get_json("ping") == {"ok": True}
        # Clearing owned sessions leaves the shared, caller-owned adapter open.
        state._ingestion_cache.clear()
        assert len(caller_adapter.poolmanager.pools) > 0
        state._client.close()
        state.api_conn().close()
        state.app_conn().close()
        assert len(caller_adapter.poolmanager.pools) > 0
        caller_adapter.close()


def test_ingestion_rejects_adapter_retries_without_changing_legacy_policy(ingestion_writer):
    from braintrust.api._transport import HTTPConnection, RetryRequestExceptionsAdapter
    from requests.adapters import HTTPAdapter

    writer, service, handler = ingestion_writer([(200, {}, b"ok")])
    for adapter in (HTTPAdapter(max_retries=2), RetryRequestExceptionsAdapter(base_num_retries=2)):
        with pytest.raises(ValueError, match="single-attempt"):
            LogIngestionAPI(service.router, "test", concurrency=4, adapter=adapter)
        connection = HTTPConnection(service.router.api_url, adapter=adapter)
        assert connection.get("ping").status_code == 200
        connection.close()
        assert len(adapter.poolmanager.pools) > 0
        adapter.close()
    assert handler.request_count == 2


def test_legacy_connection_replacement_leases_retained_batches_and_closes_owned_pools(ingestion_writer):
    from braintrust.api._transport import HTTPConnection
    from braintrust.logger import BraintrustLogFlushError

    writer, service, handler = ingestion_writer([(503, {}, b"later"), (200, {}, b"ok")])
    connection = HTTPConnection(service.router.api_url)
    try:
        connection.set_token("original")
        writer.internal_replace_api_conn(connection)
        writer.num_tries = 1
        enqueue(writer, {"id": "retained"})
        with pytest.raises(BraintrustLogFlushError):
            writer.flush()
        original = writer._pending_service
        original_adapter = original.transport.session.get_adapter(connection.base_url)
        assert len(original_adapter.poolmanager.pools) == 1

        connection.set_token("replacement")
        writer.internal_replace_api_conn(connection)
        assert len(original_adapter.poolmanager.pools) == 1
        enqueue(writer, {"id": "new"})
        writer.flush()
        assert len(original_adapter.poolmanager.pools) == 0
        requests = [request for request in handler.requests if request[1] == "/logs3"]
        assert [request[3] for request in requests] == ["Bearer original", "Bearer original", "Bearer replacement"]
        assert requests[0][2] == requests[1][2]

        replacement_adapter = writer._limit_service.transport.session.get_adapter(connection.base_url)
        assert len(replacement_adapter.poolmanager.pools) == 1
        writer.internal_replace_api_conn(connection)
        assert len(replacement_adapter.poolmanager.pools) == 0
        enqueue(writer, {"id": "last"})
        writer.flush()
        final_adapter = writer._limit_service.transport.session.get_adapter(connection.base_url)
        assert len(final_adapter.poolmanager.pools) == 1
        writer._finalize()
        assert len(final_adapter.poolmanager.pools) == 0
    finally:
        connection.close()


def test_concurrent_explicit_flush_timeout_leaves_accounting_intact(ingestion_writer):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from braintrust.logger import BraintrustLogFlushError

    entered = threading.Event()
    released = threading.Event()

    def respond(method, path, body, headers):
        entered.set()
        assert released.wait(3)
        return 200, {}, b"ok"

    writer, _, handler = ingestion_writer(respond)
    enqueue(writer, {"id": "row"})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer.flush)
        assert entered.wait(3)
        try:
            with pytest.raises(BraintrustLogFlushError, match="still in progress"):
                writer.flush(timeout=0.05)
            assert writer.pending_count == 1
        finally:
            released.set()
        future.result(timeout=3)
    assert writer.pending_count == 0
    assert handler.request_count == 1


def test_background_permanent_failure_is_dropped_and_later_rows_continue(ingestion_writer):
    import time

    from braintrust.queue import LogQueue

    writer, _, handler = ingestion_writer([(413, {}, b"too large"), (200, {}, b"ok")])
    writer.queue = LogQueue(maxsize=1)
    writer.log(LazyValue(lambda: {"id": "failed"}, use_mutex=False))
    deadline = time.monotonic() + 3
    while handler.request_count < 1 or writer.pending_count:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    writer.log(LazyValue(lambda: {"id": "new"}, use_mutex=False))
    deadline = time.monotonic() + 3
    while handler.request_count < 2:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert writer.pending_count == 0
    assert handler.request_count == 2


def test_background_writer_resumes_after_retry_budget_during_outage(ingestion_writer):
    import time

    writer, _, handler = ingestion_writer([(503, {}, b"outage")] * 3 + [(200, {}, b"ok")])
    writer.num_tries = 2
    writer.log(LazyValue(lambda: {"id": "survives-outage"}, use_mutex=False))
    deadline = time.monotonic() + 8
    while writer.pending_count or handler.request_count < 4:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert writer.pending_count == 0
    assert handler.request_count == 4


@pytest.mark.parametrize("status", [400, 401, 403, 413])
def test_permanent_log_rejections_do_not_block_later_batches(ingestion_writer, status):
    from braintrust.logger import BraintrustLogFlushError

    def respond(method, path, body, headers):
        row = json.loads(body)["rows"][0]
        return (status, {}, b"rejected") if row["id"] == "bad" else (200, {}, b"ok")

    writer, _, handler = ingestion_writer(respond, concurrency=1)
    enqueue(writer, {"id": "bad"}, {"id": "good"})
    with pytest.raises(BraintrustLogFlushError):
        writer.flush(batch_size=1)
    assert writer.pending_count == 0
    assert handler.request_count == 2
    assert [json.loads(request[2])["rows"][0]["id"] for request in handler.requests] == ["bad", "good"]


def test_expired_signed_url_is_refreshed_after_403(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    uploads = 0
    urls = 0
    with contextlib.ExitStack() as stack:

        def storage_response(method, path, body, headers):
            nonlocal uploads
            uploads += 1
            return (403, {}, b"expired") if uploads == 1 else (200, {}, b"ok")

        storage_url, _ = stack.enter_context(scripted_server(storage_response, persistent=True))

        def api_response(method, path, body, headers):
            nonlocal urls
            if path == "/version":
                return 200, {}, b'{"logs3_payload_max_bytes": 80}'
            if path == "/logs3/overflow":
                urls += 1
                return (
                    200,
                    {},
                    json.dumps(
                        {"method": "PUT", "signedUrl": storage_url, "headers": {}, "key": f"key-{urls}"}
                    ).encode(),
                )
            return 200, {}, b"ok"

        writer, _, handler = stack.enter_context(_writer_context(api_response))
        enqueue(writer, {"id": "large", "input": "x" * 300})
        writer.flush()
        assert urls == 2
        assert uploads == 2
        assert handler.request_count == 4


def test_unprepared_records_are_retained_after_local_resolution_failure(ingestion_writer):
    writer, _, handler = ingestion_writer([(200, {}, b"ok")])
    writer.num_tries = 1
    calls = 0

    def resolve():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("cannot resolve yet")
        return {"id": "row"}

    writer.queue.put(LazyValue(resolve, use_mutex=False))
    with pytest.raises(Exception, match="constructing records"):
        writer.flush()
    assert writer.pending_count == 1
    assert handler.request_count == 0
    writer.flush()
    assert writer.pending_count == 0
    assert handler.request_count == 1


def test_successful_batches_release_capacity_while_failed_batches_remain(ingestion_writer):
    from braintrust.logger import BraintrustLogFlushError

    def respond(method, path, body, headers):
        row = json.loads(body)["rows"][0]
        return (413, {}, b"too large") if row["id"] == "failed" else (200, {}, b"ok")

    writer, _, handler = ingestion_writer(respond, concurrency=2, capacity=2)
    enqueue(writer, {"id": "failed"}, {"id": "delivered"})
    with pytest.raises(BraintrustLogFlushError) as failure:
        writer.flush(batch_size=1)
    assert failure.value.pending_count == 0
    enqueue(writer, {"id": "later"})
    assert writer.pending_count == 1
    writer.flush()
    assert handler.request_count == 3


def test_explicit_flush_finishes_its_wave_while_producers_keep_logging(ingestion_writer):
    def respond(method, path, body, headers):
        row = json.loads(body)["rows"][0]
        if row["id"] == "initial":
            enqueue(writer, {"id": "next-wave"})
        return 200, {}, b"ok"

    writer, _, handler = ingestion_writer(respond)
    enqueue(writer, {"id": "initial"})
    writer.flush()
    assert handler.request_count == 1
    assert writer.pending_count == 1
    writer.flush()
    assert handler.request_count == 2
    assert writer.pending_count == 0


def test_interpreter_shutdown_flushes_without_an_available_thread_pool(ingestion_writer):
    import os
    import subprocess
    import sys

    _, service, handler = ingestion_writer([(200, {}, b"ok")])
    program = f"""
import contextlib
from braintrust.api._ingestion import LogIngestionAPI
from braintrust.api._routing import EndpointRouter
from braintrust.logger import _HTTPBackgroundLogger
from braintrust.util import LazyValue
service = LogIngestionAPI(EndpointRouter(app_url={service.router.api_url!r}, api_url={service.router.api_url!r}), "test", concurrency=4)
writer = _HTTPBackgroundLogger(lambda: contextlib.nullcontext(service))
writer.sync_flush = True
writer._max_request_size_result = {{"max_request_size": 1000000, "can_use_overflow": False}}
writer.log(LazyValue(lambda: {{"id": "shutdown-row"}}, use_mutex=False))
service.transport.session.get_adapter({service.router.api_url!r}).poolmanager.clear()
"""
    process = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "BRAINTRUST_DISABLE_ATEXIT_FLUSH": "false"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    assert handler.request_count == 1, process.stderr
    assert json.loads(handler.requests[0][2])["rows"][0]["id"] == "shutdown-row"


def test_forked_child_does_not_inherit_parent_pending_rows(monkeypatch):
    import os

    if not hasattr(os, "fork"):
        pytest.skip("fork is not available")
    from braintrust import logger
    from braintrust.logger import BraintrustState

    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "1")
    state = BraintrustState()
    monkeypatch.setattr(logger, "_state", state)
    writer = state.global_bg_logger()
    enqueue(writer, {"id": "parent-only"})
    child = os.fork()
    if child == 0:
        os._exit(0 if state.global_bg_logger().pending_count == 0 else 3)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert writer.pending_count == 1


def test_new_background_writer_does_not_spin_during_shared_long_cooldown(ingestion_writer):
    import time

    from braintrust.logger import BraintrustLogFlushError

    writer, service, handler = ingestion_writer([(429, {"Retry-After": "120"}, b"later")])
    enqueue(writer, {"id": "throttled"})
    with pytest.raises(BraintrustLogFlushError):
        writer.flush(timeout=0.1)
    # This writer must negotiate /version after the same destination recovers.
    other = _HTTPBackgroundLogger(lambda: contextlib.nullcontext(service))
    observed = []

    def mask(value):
        if value == "observe":
            observed.append(value)
        return value

    other.set_masking_function(mask)
    other.log(LazyValue(lambda: {"id": "waiting", "input": "observe"}, use_mutex=False))
    deadline = time.monotonic() + 3
    while not observed:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    time.sleep(0.1)
    assert len(observed) == 1
    assert other.pending_count == 1
    assert handler.request_count == 1
