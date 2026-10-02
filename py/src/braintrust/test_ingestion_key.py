"""Tests for loggers that publish with a public ingestion key.

The fake data plane below implements the ingestion key contract over real HTTP,
so these tests exercise the actual logger, batcher, and upload flow.
"""

import hashlib
import http.server
import json
import logging
import re
import threading
import time
import uuid
from contextlib import contextmanager

import braintrust
import pytest
from braintrust import logger
from braintrust.logger import Attachment, ExternalAttachment, _parse_ingestion_url
from braintrust.test_helpers import memory_logger, simulate_login  # noqa: F401


KEY = "bt-ik-" + "a" * 48
OTHER_KEY = "bt-ik-" + "b" * 48
BASE_PATH = "/deployment/base/ingest"
UPLOAD_PATH = re.compile(rf"^{BASE_PATH}/v1/uploads/([0-9a-f-]+)/(chunks/(\d+)|complete)$")


class FakeDataPlane:
    """Records requests and answers them like a data plane serving one ingestion key."""

    def __init__(self, key: str = KEY, chunk_bytes: int = 4):
        self.key = key
        self.chunk_bytes = chunk_bytes
        self.requests: list[dict] = []
        self.rows: list[dict] = []
        self.uploads: dict[str, dict] = {}
        # Responses to return before handling requests normally, keyed by (method, route).
        self.scripted: dict[tuple[str, str], list[tuple[int, dict, bytes]]] = {}
        self.grant_lifetimes_ms: list[int] = []
        # Fields to override in the next grants, to simulate malformed responses.
        self.grant_overrides: list[dict] = []
        # Delay before answering upload creation, to expire short grants regardless of clock resolution.
        self.create_delay_s = 0.0
        self.project_id: str | None = None
        self.lock = threading.Lock()

    def script(self, method: str, route: str, *responses: tuple[int, dict, bytes]) -> None:
        self.scripted.setdefault((method, route), []).extend(responses)

    def paths(self) -> list[tuple[str, str]]:
        return [(r["method"], r["path"]) for r in self.requests]

    def handle(self, method: str, path: str, headers, body: bytes) -> tuple[int, dict, bytes]:
        with self.lock:
            self.requests.append({"method": method, "path": path, "headers": dict(headers), "body": body})
            route = self._route(path)
            scripted = self.scripted.get((method, route))
            if scripted:
                return scripted.pop(0)
            if headers.get("Authorization") != f"Bearer {self.key}":
                return 401, {}, b'{"error": "invalid ingestion key"}'
            if route == "logs" and method == "POST":
                return self._logs(json.loads(body))
            if route == "create" and method == "POST":
                return self._create(json.loads(body))
            if route == "chunk" and method == "PUT":
                return self._chunk(path, headers, body)
            if route == "complete" and method == "POST":
                return self._complete(path)
            return 404, {}, b'{"error": "not found"}'

    @staticmethod
    def _route(path: str) -> str:
        if path == f"{BASE_PATH}/v1/logs":
            return "logs"
        if path == f"{BASE_PATH}/v1/uploads":
            return "create"
        match = UPLOAD_PATH.match(path)
        if match:
            return "chunk" if match.group(3) is not None else "complete"
        return "unknown"

    def _logs(self, body: dict) -> tuple[int, dict, bytes]:
        assert body["api_version"] == 2
        rows = body["rows"]
        if isinstance(rows, dict):
            assert rows == {"type": "logs3_overflow", "key": rows["key"]}
            upload = self.uploads[rows["key"]]
            assert upload["purpose"] == "logs3_overflow" and upload["completed"]
            rows = json.loads(upload["data"])["rows"]
        for row in rows:
            if self.project_id is not None and row.get("project_id", self.project_id) != self.project_id:
                return 403, {}, b'{"error": "row project does not match the ingestion key"}'
            # External references can point at any object store content, so ingestion keys can't write them.
            if '"external_attachment"' in json.dumps(row):
                return 400, {}, b'{"error": "external_attachment references are not allowed"}'
        self.rows.extend(rows)
        return 200, {}, json.dumps({"ids": [row["id"] for row in rows]}).encode()

    def _create(self, body: dict) -> tuple[int, dict, bytes]:
        time.sleep(self.create_delay_s)
        self.create_delay_s = 0.0
        upload_id = str(uuid.uuid4())
        num_chunks = -(-body["size_bytes"] // self.chunk_bytes)
        self.uploads[upload_id] = {**body, "chunks": {}, "completed": False}
        lifetime = self.grant_lifetimes_ms.pop(0) if self.grant_lifetimes_ms else 300000
        response = {
            "upload_id": upload_id,
            "chunk_bytes": self.chunk_bytes,
            "num_chunks": num_chunks,
            "expires_in_ms": lifetime,
            **(self.grant_overrides.pop(0) if self.grant_overrides else {}),
        }
        return 201, {}, json.dumps(response).encode()

    def _chunk(self, path: str, headers, body: bytes) -> tuple[int, dict, bytes]:
        match = UPLOAD_PATH.match(path)
        assert match is not None
        upload = self.uploads[match.group(1)]
        index = int(match.group(3))
        assert headers.get("Content-Type") == "application/octet-stream"
        assert "Content-Encoding" not in headers
        upload["chunks"][index] = body
        return 200, {}, json.dumps({"index": index, "size_bytes": len(body)}).encode()

    def _complete(self, path: str) -> tuple[int, dict, bytes]:
        match = UPLOAD_PATH.match(path)
        assert match is not None
        upload_id = match.group(1)
        upload = self.uploads[upload_id]
        data = b"".join(upload["chunks"][i] for i in sorted(upload["chunks"]))
        assert len(data) == upload["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == upload["sha256"]
        upload["data"] = data
        upload["completed"] = True
        if upload["purpose"] == "attachment":
            reference = {
                "type": "braintrust_attachment",
                "key": upload_id,
                "filename": upload["filename"],
                "content_type": upload["content_type"],
            }
        else:
            reference = {"type": "logs3_overflow", "key": upload_id}
        response = {"reference": reference, "size_bytes": len(data), "sha256": upload["sha256"]}
        return 200, {}, json.dumps(response).encode()


@contextmanager
def serve(data_plane: FakeDataPlane):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _handle(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            status, headers, response_body = data_plane.handle(self.command, self.path, self.headers, body)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}{BASE_PATH}"
    finally:
        server.shutdown()
        server.server_close()


def ingestion_url(root: str, key: str = KEY) -> str:
    return f"{root}?ingestKey={key}"


@pytest.fixture(autouse=True)
def deterministic_flush(monkeypatch):
    # Rows only publish on explicit flushes, failures raise, and retries don't sleep.
    monkeypatch.setenv("BRAINTRUST_SYNC_FLUSH", "true")
    monkeypatch.setenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH", "true")
    monkeypatch.delenv("BRAINTRUST_INGESTION_KEY", raising=False)
    monkeypatch.setattr(logger, "BACKGROUND_LOGGER_BASE_SLEEP_TIME_S", 0)
    logger._state.reset_parent_state()
    yield
    logger._state.reset_parent_state()


@pytest.fixture
def data_plane():
    plane = FakeDataPlane()
    with serve(plane) as root:
        plane.root = root
        yield plane


@pytest.fixture
def no_private_login(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("ingestion key loggers must not log in")

    monkeypatch.setattr(logger, "login_to_state", fail)
    monkeypatch.setattr(logger, "_compute_logger_metadata", fail)


def flush_error(target) -> str:
    """Flush and return the error, which publishing raises inside an exception group."""
    with pytest.raises(BaseException) as exc_info:
        target.flush()
    return "; ".join(str(e) for e in exc_info.value.exceptions)


def assert_no_key_in_urls(plane: FakeDataPlane) -> None:
    for request in plane.requests:
        assert "ingestKey" not in request["path"]
        assert plane.key not in request["path"]


@pytest.mark.parametrize(
    "url,expected",
    [
        (f"https://dp.example/ingest?ingestKey={KEY}", "https://dp.example/ingest"),
        (f"https://dp.example/deployment/base/ingest/?ingestKey={KEY}", "https://dp.example/deployment/base/ingest"),
        (f"http://localhost:8000/ingest?ingestKey={KEY}\n", "http://localhost:8000/ingest"),
        (f"https://dp.example/base/ingest///?ingestKey={KEY}", "https://dp.example/base/ingest"),
        (f"https://dp.example/ingest?&ingestKey={KEY}&&", "https://dp.example/ingest"),
        (f"https://dp.example/ingest?ingest%4Bey=bt%2Dik%2D{'a' * 48}", "https://dp.example/ingest"),
    ],
)
def test_parse_ingestion_url(url, expected):
    endpoint = _parse_ingestion_url(url)

    assert endpoint.url == expected
    assert endpoint.key == KEY
    assert KEY not in repr(endpoint)


@pytest.mark.parametrize(
    "url",
    [
        KEY,
        f"ftp://dp.example/ingest?ingestKey={KEY}",
        f"https://user:pass@dp.example/ingest?ingestKey={KEY}",
        f"https://dp.example/ingest?ingestKey={KEY}#fragment",
        f"https://dp.example/ingest?ingestKey={KEY}#",
        f"https://dp.example/ingest#?ingestKey={KEY}",
        "https://dp.example/ingest?ingestKey",
        f"https://dp.example/ingest?ingestKey={KEY}&ingestKey=",
        f"https://dp.example/ingest?ingestKey={KEY}=x",
        f"https://dp.example/ingest/v1/logs?ingestKey={KEY}",
        f"https://dp.example/not-ingest?ingestKey={KEY}",
        f"https://dp.example/ingest?ingestKey={KEY}&ingestKey={KEY}",
        f"https://dp.example/ingest?ingestKey={KEY}&project=foo",
        f"https://dp.example/ingest?key={KEY}",
        f"https://dp.example/ingest?ingestKey={KEY}x",
        "https://dp.example/ingest?ingestKey=sk-" + "a" * 48,
        f"https://dp.example:notaport/ingest?ingestKey={KEY}",
    ],
)
def test_parse_ingestion_url_rejects_invalid_urls_without_leaking_the_key(url):
    with pytest.raises(ValueError, match="Invalid Braintrust ingestion key URL") as exc_info:
        _parse_ingestion_url(url)

    assert "a" * 48 not in str(exc_info.value)
    assert exc_info.value.__cause__ is None and exc_info.value.__context__ is None


def test_first_flush_only_posts_rows_with_the_key(data_plane, monkeypatch, no_private_login):
    # Ambient private credentials must be ignored.
    monkeypatch.setenv("BRAINTRUST_API_KEY", "sk-private")

    public_logger = braintrust.init_logger(project="my-project", ingestion_key=ingestion_url(data_plane.root))
    row_id = public_logger.log(input="question", output="answer", scores={"good": 1}, metadata={"a": 1})
    public_logger.flush()

    assert data_plane.paths() == [("POST", f"{BASE_PATH}/v1/logs")]
    request = data_plane.requests[0]
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    assert request["headers"]["Content-Type"] == "application/json"
    [row] = data_plane.rows
    assert row["id"] == row_id
    assert row["log_id"] == "g"
    assert "project_id" not in row
    assert {
        "input": "question",
        "output": "answer",
        "scores": {"good": 1},
        "metadata": {"a": 1},
    }.items() <= row.items()
    assert set(row) <= set(logger._INGESTION_ROW_FIELDS)
    assert public_logger.id is None
    assert not logger._state.logged_in
    assert_no_key_in_urls(data_plane)


def test_trace_context_round_trips_without_project_metadata(data_plane, no_private_login):
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))

    with public_logger.start_span(name="root") as root:
        exported = root.export()
        headers = root.inject()
        assert root.permalink()
    # A parent exported elsewhere only contributes its span ids.
    with braintrust.start_span(name="from-slug", parent=exported) as from_slug:
        pass
    with braintrust.start_span(name="from-headers", parent=braintrust.extract_trace_context(headers)) as from_headers:
        pass
    public_logger.flush()

    rows = {row["span_attributes"]["name"]: row for row in data_plane.rows}
    assert rows["from-slug"]["root_span_id"] == root.root_span_id
    assert rows["from-slug"]["span_parents"] == [root.span_id]
    assert rows["from-headers"]["root_span_id"] == root.root_span_id
    assert rows["from-headers"]["span_parents"] == [root.span_id]
    assert all("project_id" not in row for row in data_plane.rows)
    assert from_slug.public_bg_logger is public_logger._public_bg_logger
    assert from_headers.public_bg_logger is public_logger._public_bg_logger


def test_explicit_project_id_is_sent_and_mismatches_are_not_retried(data_plane, no_private_login):
    data_plane.project_id = "key-project"
    public_logger = braintrust.init_logger(project_id="other-project", ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input="x")

    assert "403" in flush_error(public_logger)

    [request] = data_plane.requests
    [row] = json.loads(request["body"])["rows"]
    assert row["project_id"] == "other-project"
    assert data_plane.rows == []


def test_retries_transient_failures_with_the_same_key(data_plane, monkeypatch, no_private_login):
    monkeypatch.setenv("BRAINTRUST_NUM_RETRIES", "2")
    data_plane.script(
        "POST",
        "logs",
        (429, {"Retry-After": "0"}, b'{"error": "rate limited"}'),
        (503, {}, b'{"error": "unavailable"}'),
    )
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input="x")
    public_logger.flush()

    assert data_plane.paths() == [("POST", f"{BASE_PATH}/v1/logs")] * 3
    assert {r["headers"]["Authorization"] for r in data_plane.requests} == {f"Bearer {KEY}"}
    assert len(data_plane.rows) == 1


def test_rejected_writes_are_not_retried_or_sent_elsewhere(data_plane, monkeypatch, no_private_login):
    monkeypatch.setenv("BRAINTRUST_NUM_RETRIES", "2")
    data_plane.script("POST", "logs", (401, {}, f'{{"error": "bad key {KEY}"}}'.encode()))
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input="x")

    error = flush_error(public_logger)

    assert "401" in error
    assert KEY not in error
    assert data_plane.paths() == [("POST", f"{BASE_PATH}/v1/logs")]


def test_attachments_upload_in_advertised_chunks_before_rows(data_plane, no_private_login):
    data_plane.chunk_bytes = 4
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    attachment = Attachment(data=b"hello world", filename="greeting.txt", content_type="text/plain")
    empty = Attachment(data=b"", filename="empty.txt", content_type="text/plain")
    # A reference committed earlier is plain data, so it passes through unchanged.
    committed = {
        "type": "braintrust_attachment",
        "key": str(uuid.uuid4()),
        "filename": "old.txt",
        "content_type": "text/plain",
    }
    public_logger.log(input={"file": attachment, "empty": empty, "committed": committed, "again": attachment})
    public_logger.flush()

    [attachment_id, empty_id] = list(data_plane.uploads)
    assert data_plane.paths() == [
        ("POST", f"{BASE_PATH}/v1/uploads"),
        ("PUT", f"{BASE_PATH}/v1/uploads/{attachment_id}/chunks/0"),
        ("PUT", f"{BASE_PATH}/v1/uploads/{attachment_id}/chunks/1"),
        ("PUT", f"{BASE_PATH}/v1/uploads/{attachment_id}/chunks/2"),
        ("POST", f"{BASE_PATH}/v1/uploads/{attachment_id}/complete"),
        ("POST", f"{BASE_PATH}/v1/uploads"),
        ("POST", f"{BASE_PATH}/v1/uploads/{empty_id}/complete"),
        ("POST", f"{BASE_PATH}/v1/logs"),
    ]
    assert [len(r["body"]) for r in data_plane.requests if r["method"] == "PUT"] == [4, 4, 3]
    assert json.loads(data_plane.requests[0]["body"]) == {
        "purpose": "attachment",
        "size_bytes": 11,
        "content_type": "text/plain",
        "sha256": hashlib.sha256(b"hello world").hexdigest(),
        "filename": "greeting.txt",
    }
    assert json.loads(data_plane.requests[4]["body"]) == {}
    assert data_plane.uploads[empty_id]["size_bytes"] == 0
    [row] = data_plane.rows
    file_reference = {
        "type": "braintrust_attachment",
        "key": attachment_id,
        "filename": "greeting.txt",
        "content_type": "text/plain",
    }
    assert row["input"] == {
        "file": file_reference,
        "empty": {
            "type": "braintrust_attachment",
            "key": empty_id,
            "filename": "empty.txt",
            "content_type": "text/plain",
        },
        "committed": committed,
        "again": file_reference,
    }


def test_external_attachments_are_rejected_before_publishing(data_plane, no_private_login):
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    external = ExternalAttachment(url="s3://private-bucket/file.txt", filename="file.txt", content_type="text/plain")
    public_logger.log(input={"file": external})
    kept_id = public_logger.log(input="no attachment")

    with pytest.raises(ValueError, match="ExternalAttachment is not supported with ingestion keys"):
        public_logger.flush()

    assert data_plane.paths() == [("POST", f"{BASE_PATH}/v1/logs")]
    assert [row["id"] for row in data_plane.rows] == [kept_id]
    assert b"private-bucket" not in data_plane.requests[0]["body"]


def test_raw_external_attachment_references_are_not_a_silent_success(data_plane, no_private_login):
    # Plain dicts aren't validated client side, so the data plane's rejection must surface.
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input={"type": "external_attachment", "url": "s3://private-bucket/file.txt"})

    assert "400" in flush_error(public_logger)
    assert data_plane.rows == []


def test_rows_with_failed_attachments_are_dropped(data_plane, no_private_login):
    data_plane.script("POST", "create", (400, {}, b'{"error": "uploads disabled"}'))
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input=Attachment(data=b"abc", filename="a.txt", content_type="text/plain"))
    kept_id = public_logger.log(input="no attachment")

    with pytest.raises(RuntimeError, match="400"):
        public_logger.flush()

    assert [row["id"] for row in data_plane.rows] == [kept_id]
    assert not any("attachment" in path for _, path in data_plane.paths())


def test_expired_upload_grant_starts_a_new_upload(data_plane, no_private_login):
    # The grant expires while its creation response is in flight.
    data_plane.grant_lifetimes_ms = [50]
    data_plane.create_delay_s = 0.2
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input=Attachment(data=b"abcdef", filename="a.txt", content_type="text/plain"))
    public_logger.flush()

    [expired_id, upload_id] = list(data_plane.uploads)
    assert not data_plane.uploads[expired_id]["chunks"]
    assert data_plane.uploads[upload_id]["completed"]
    assert data_plane.rows[0]["input"]["key"] == upload_id


def test_oversized_batches_overflow_through_uploads(data_plane, monkeypatch, no_private_login):
    monkeypatch.setenv("BRAINTRUST_MAX_REQUEST_SIZE", "2000")
    data_plane.chunk_bytes = 1024
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    row_id = public_logger.log(input="x" * 5000)
    public_logger.flush()

    [upload_id] = list(data_plane.uploads)
    upload = data_plane.uploads[upload_id]
    assert upload["purpose"] == "logs3_overflow"
    assert upload["content_type"] == "application/json"
    overflow_payload = json.loads(upload["data"])
    assert set(overflow_payload) == {"api_version", "rows"}
    assert overflow_payload["api_version"] == 2
    assert [row["id"] for row in overflow_payload["rows"]] == [row_id]
    [_, *chunk_requests, _, logs_request] = data_plane.requests
    assert [len(r["body"]) for r in chunk_requests] == [1024] * (len(chunk_requests) - 1) + [
        len(upload["data"]) - 1024 * (len(chunk_requests) - 1)
    ]
    assert json.loads(logs_request["body"]) == {
        "api_version": 2,
        "rows": {"type": "logs3_overflow", "key": upload_id},
    }
    assert [row["id"] for row in data_plane.rows] == [row_id]


def test_feedback_and_disallowed_fields(data_plane, no_private_login):
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    with public_logger.start_span(name="root") as span:
        span.log(input="x", experiment_id="someone-elses-experiment", dataset_id="d")
    public_logger.log_feedback(id=span.id, scores={"thumbs_up": 1}, tags=["a"])
    braintrust.update_span(span.export(), output="updated")
    with pytest.raises(ValueError, match="ingestion key"):
        public_logger.log_feedback(id=span.id, comment="nice")
    with pytest.raises(ValueError, match="ingestion key"):
        span.log_feedback(scores={"thumbs_up": 0}, metadata={"user": "u"})
    public_logger.flush()

    [row] = data_plane.rows
    assert row["id"] == span.id
    assert row["scores"] == {"thumbs_up": 1}
    assert row["output"] == "updated"
    assert set(row) <= set(logger._INGESTION_ROW_FIELDS)


def test_masking_and_global_flush_cover_public_loggers(data_plane, no_private_login):
    braintrust.set_masking_function(lambda value: "masked" if value == "secret" else value)
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input="secret")
    braintrust.flush()

    assert data_plane.rows[0]["input"] == "masked"


def test_env_ingestion_key_takes_precedence_over_ambient_private_credentials(
    data_plane, monkeypatch, no_private_login
):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "sk-private")
    monkeypatch.setenv("BRAINTRUST_INGESTION_KEY", ingestion_url(data_plane.root))

    public_logger = braintrust.init_logger(project="p")
    public_logger.log(input="x")
    public_logger.flush()

    assert public_logger._public_bg_logger is not None
    assert len(data_plane.rows) == 1


def test_env_ingestion_key_ignores_previous_login(data_plane, monkeypatch, memory_logger):
    simulate_login()
    monkeypatch.setenv("BRAINTRUST_INGESTION_KEY", ingestion_url(data_plane.root))

    public_logger = braintrust.init_logger(project="p")
    public_logger.log(input="x")
    public_logger.flush()

    assert len(data_plane.rows) == 1
    assert memory_logger.pop() == []


def test_explicit_ingestion_key_overrides_env(data_plane, monkeypatch, no_private_login):
    other_plane = FakeDataPlane(key=OTHER_KEY)
    with serve(other_plane) as other_root:
        monkeypatch.setenv("BRAINTRUST_INGESTION_KEY", ingestion_url(other_root, OTHER_KEY))
        public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
        public_logger.log(input="x")
        public_logger.flush()

    assert len(data_plane.rows) == 1
    assert other_plane.requests == []


def test_explicit_api_key_overrides_env_ingestion_key(data_plane, monkeypatch, memory_logger):
    monkeypatch.setenv("BRAINTRUST_INGESTION_KEY", ingestion_url(data_plane.root))

    private_logger = braintrust.init_logger(project="p", project_id="private-project", api_key=logger.TEST_API_KEY)
    private_logger.log(input="x")
    private_logger.flush()

    assert private_logger._public_bg_logger is None
    assert [row["input"] for row in memory_logger.pop()] == ["x"]
    assert data_plane.requests == []


def test_explicit_api_key_and_ingestion_key_conflict(data_plane):
    with pytest.raises(ValueError, match="not both"):
        braintrust.init_logger(api_key="sk-private", ingestion_key=ingestion_url(data_plane.root))


def test_public_and_private_loggers_do_not_mix(data_plane, memory_logger):
    simulate_login()
    other_plane = FakeDataPlane(key=OTHER_KEY)
    with serve(other_plane) as other_root:
        private_logger = braintrust.init_logger(project="private", project_id="private-project", set_current=False)
        public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root), set_current=False)
        other_logger = braintrust.init_logger(ingestion_key=ingestion_url(other_root, OTHER_KEY), set_current=False)

        def log_many(target, name):
            for i in range(20):
                target.log(input=f"{name}-{i}")

        threads = [
            threading.Thread(target=log_many, args=(target, name))
            for target, name in [(public_logger, "public"), (other_logger, "other")]
        ]
        for thread in threads:
            thread.start()
        # The memory logger override is thread local, so private rows log from this thread.
        log_many(private_logger, "private")
        for thread in threads:
            thread.join()
        braintrust.flush()

    assert sorted(row["input"] for row in data_plane.rows) == sorted(f"public-{i}" for i in range(20))
    assert sorted(row["input"] for row in other_plane.rows) == sorted(f"other-{i}" for i in range(20))
    assert {r["headers"]["Authorization"] for r in other_plane.requests} == {f"Bearer {OTHER_KEY}"}
    assert sorted(row["input"] for row in memory_logger.pop()) == sorted(f"private-{i}" for i in range(20))


def test_key_does_not_leak_through_repr(data_plane):
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))

    assert KEY not in repr(public_logger)
    assert KEY not in repr(public_logger._public_bg_logger.endpoint)
    assert KEY not in public_logger.export()


def test_explicit_empty_or_malformed_ingestion_key_never_falls_back(data_plane, monkeypatch, no_private_login):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "sk-private")
    monkeypatch.setenv("BRAINTRUST_INGESTION_KEY", ingestion_url(data_plane.root))

    for value in ["", "   ", "https://dp.example/ingest", KEY]:
        with pytest.raises(ValueError, match="Invalid Braintrust ingestion key URL"):
            braintrust.init_logger(ingestion_key=value)

    assert braintrust.current_logger() is None
    assert data_plane.requests == []


def test_malformed_env_ingestion_key_never_falls_back(monkeypatch, no_private_login):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "sk-private")
    monkeypatch.setenv("BRAINTRUST_INGESTION_KEY", "https://dp.example/ingest?ingestKey=not-a-key")

    with pytest.raises(ValueError, match="Invalid Braintrust ingestion key URL"):
        braintrust.init_logger(project="p")


def test_default_overflow_threshold_is_512_kib(data_plane, monkeypatch, no_private_login):
    monkeypatch.delenv("BRAINTRUST_MAX_REQUEST_SIZE", raising=False)
    data_plane.chunk_bytes = 512 * 1024
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))

    small_id = public_logger.log(input="x" * (200 * 1024))
    public_logger.flush()
    assert data_plane.paths() == [("POST", f"{BASE_PATH}/v1/logs")]
    assert len(data_plane.requests[0]["body"]) < 512 * 1024

    large_id = public_logger.log(input="x" * (600 * 1024))
    public_logger.flush()
    [upload_id] = list(data_plane.uploads)
    assert [method for method, _ in data_plane.paths()[1:]] == ["POST", "PUT", "PUT", "POST", "POST"]
    assert json.loads(data_plane.requests[-1]["body"]) == {
        "api_version": 2,
        "rows": {"type": "logs3_overflow", "key": upload_id},
    }
    assert [row["id"] for row in data_plane.rows] == [small_id, large_id]
    assert not any(path.endswith("/version") for _, path in data_plane.paths())


@pytest.mark.parametrize(
    "override",
    [
        {"expires_in_ms": 0},
        {"expires_in_ms": -1},
        {"expires_in_ms": 300001},
        {"expires_in_ms": "300000"},
        {"expires_in_ms": True},
        {"expires_in_ms": None},
        {"num_chunks": 3},
        {"num_chunks": 1},
        {"chunk_bytes": 0},
        {"chunk_bytes": 2.5},
        {"upload_id": "../../v1/logs"},
        {"upload_id": None},
    ],
)
def test_malformed_upload_grants_are_rejected(data_plane, override, no_private_login):
    # 6 bytes in 4 byte chunks is 2 chunks.
    data_plane.grant_overrides = [override]
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input=Attachment(data=b"abcdef", filename="a.txt", content_type="text/plain"))

    with pytest.raises(RuntimeError, match="Invalid upload grant") as exc_info:
        public_logger.flush()

    assert KEY not in str(exc_info.value)
    assert data_plane.paths() == [("POST", f"{BASE_PATH}/v1/uploads")]
    assert data_plane.rows == []


def test_maximum_grant_lifetime_is_accepted(data_plane, no_private_login):
    data_plane.grant_lifetimes_ms = [300000]
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input=Attachment(data=b"abcdef", filename="a.txt", content_type="text/plain"))
    public_logger.flush()

    assert len(data_plane.rows) == 1


def test_gone_upload_grant_starts_a_new_upload(data_plane, no_private_login):
    data_plane.script("PUT", "chunk", (410, {}, b'{"error": "upload grant expired"}'))
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input=Attachment(data=b"abcdef", filename="a.txt", content_type="text/plain"))
    public_logger.flush()

    [gone_id, upload_id] = list(data_plane.uploads)
    assert not data_plane.uploads[gone_id]["completed"]
    assert data_plane.rows[0]["input"]["key"] == upload_id


def test_gone_upload_grants_are_retried_a_bounded_number_of_times(data_plane, monkeypatch, no_private_login):
    monkeypatch.setenv("BRAINTRUST_NUM_RETRIES", "1")
    data_plane.script("PUT", "chunk", *[(410, {}, b'{"error": "upload grant expired"}')] * 5)
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input=Attachment(data=b"abcdef", filename="a.txt", content_type="text/plain"))

    with pytest.raises(RuntimeError, match="kept expiring"):
        public_logger.flush()

    assert len(data_plane.uploads) == 2
    assert data_plane.rows == []


def test_retry_waits_never_outlive_the_upload_grant(data_plane, no_private_login):
    data_plane.grant_lifetimes_ms = [2000]
    data_plane.script("PUT", "chunk", (503, {"Retry-After": "30"}, b'{"error": "unavailable"}'))
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    public_logger.log(input=Attachment(data=b"abcdef", filename="a.txt", content_type="text/plain"))

    started_at = time.monotonic()
    public_logger.flush()

    assert time.monotonic() - started_at < 5
    [expired_id, upload_id] = list(data_plane.uploads)
    assert not data_plane.uploads[expired_id]["completed"]
    assert data_plane.rows[0]["input"]["key"] == upload_id


def test_transport_failures_never_reveal_the_key(data_plane, monkeypatch, capsys, caplog, no_private_login):
    monkeypatch.setenv("BRAINTRUST_NUM_RETRIES", "1")
    # Reach the closed port directly, even if the environment configures a proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    caplog.set_level(logging.DEBUG)
    with serve(FakeDataPlane()) as closed_root:
        pass
    errors = []
    for root in [closed_root, data_plane.root]:
        data_plane.script("POST", "logs", *[(500, {}, f'{{"error": "bad {KEY}"}}'.encode())] * 2)
        public_logger = braintrust.init_logger(ingestion_key=ingestion_url(root))
        public_logger.log(input="x")
        errors.append(flush_error(public_logger))

    assert "Failed to establish a new connection" in errors[0]
    assert "500" in errors[1]
    output = capsys.readouterr()
    for text in [*errors, output.out, output.err, caplog.text]:
        assert KEY not in text
        assert "ingestKey" not in text


def test_redirects_are_not_followed(data_plane, no_private_login):
    other_plane = FakeDataPlane()
    with serve(other_plane) as other_root:
        data_plane.script("POST", "logs", (307, {"Location": f"{other_root}/v1/logs"}, b""))
        public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
        public_logger.log(input="x")
        error = flush_error(public_logger)

    assert "307" in error
    assert other_plane.requests == []


def test_background_thread_publishes_without_explicit_flush(data_plane, monkeypatch, no_private_login):
    monkeypatch.delenv("BRAINTRUST_SYNC_FLUSH")
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    row_id = public_logger.log(input="x")

    deadline = time.monotonic() + 10
    while not data_plane.rows and time.monotonic() < deadline:
        time.sleep(0.01)

    assert [row["id"] for row in data_plane.rows] == [row_id]


def test_atexit_flush_publishes_queued_rows(data_plane, monkeypatch, no_private_login):
    monkeypatch.delenv("BRAINTRUST_DISABLE_ATEXIT_FLUSH")
    registered = []
    monkeypatch.setattr(logger.atexit, "register", registered.append)
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))
    row_id = public_logger.log(input="x")

    bg_logger = public_logger._public_bg_logger
    assert registered == [bg_logger._finalize]
    bg_logger._finalize()

    assert [row["id"] for row in data_plane.rows] == [row_id]


def test_public_and_private_spans_nest_without_crossing_loggers(data_plane, memory_logger):
    simulate_login()
    private_logger = braintrust.init_logger(project="private", project_id="private-project", set_current=False)
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root), set_current=False)

    with private_logger.start_span(name="private-root") as private_root:
        with public_logger.start_span(name="public-child") as public_child:
            # Global helpers follow the current span.
            with braintrust.start_span(name="public-grandchild"):
                pass
            with private_logger.start_span(name="private-grandchild"):
                pass
    braintrust.flush()

    public_rows = {row["span_attributes"]["name"]: row for row in data_plane.rows}
    private_rows = {row["span_attributes"]["name"]: row for row in memory_logger.pop()}
    assert set(public_rows) == {"public-child", "public-grandchild"}
    assert set(private_rows) == {"private-root", "private-grandchild"}
    assert public_rows["public-child"]["span_parents"] == [private_root.span_id]
    assert public_rows["public-grandchild"]["span_parents"] == [public_child.span_id]
    assert private_rows["private-grandchild"]["span_parents"] == [public_child.span_id]
    assert {row["root_span_id"] for row in [*public_rows.values(), *private_rows.values()]} == {
        private_root.root_span_id
    }
    assert all(row["project_id"] == "private-project" for row in private_rows.values())
    assert all("project_id" not in row for row in public_rows.values())


def test_public_trace_context_continues_in_a_private_service(data_plane, memory_logger):
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root), set_current=False)
    with public_logger.start_span(name="client") as client_span:
        headers = client_span.inject()
    public_logger.flush()

    simulate_login()
    # Resolving the id up front keeps the private logger from looking up its project by name.
    braintrust.init_logger(project="server", project_id="server-project")._lazy_id.get()
    with braintrust.start_span(name="handler", parent=braintrust.extract_trace_context(headers)):
        pass
    braintrust.flush()

    [handler] = memory_logger.pop()
    assert handler["project_id"] == "server-project"
    assert handler["root_span_id"] == client_span.root_span_id
    assert handler["span_parents"] == [client_span.span_id]
    assert [row["span_attributes"]["name"] for row in data_plane.rows] == ["client"]


def test_traced_functions_under_a_public_logger_stay_public(data_plane, memory_logger):
    simulate_login()
    public_logger = braintrust.init_logger(ingestion_key=ingestion_url(data_plane.root))

    span_ids = []

    @braintrust.traced
    def answer(question):
        span_ids.append(braintrust.current_span().id)
        with braintrust.start_span(name="lookup") as lookup:
            lookup.log(metadata={"q": question})
        return "42"

    answer("why")
    public_logger.log_feedback(id=span_ids[0], scores={"ok": 1})
    braintrust.flush()

    rows = {row["span_attributes"]["name"]: row for row in data_plane.rows}
    assert set(rows) == {"answer", "lookup"}
    assert rows["answer"]["output"] == "42"
    assert rows["answer"]["scores"] == {"ok": 1}
    assert memory_logger.pop() == []
