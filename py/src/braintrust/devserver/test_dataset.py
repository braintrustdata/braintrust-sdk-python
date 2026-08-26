import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from braintrust.api import BraintrustClient
from braintrust.devserver.dataset import get_dataset
from braintrust.logger import BraintrustState


class _DatasetAPIHandler(http.server.BaseHTTPRequestHandler):
    requests: list[tuple[str, str, dict[str, list[str]], Any]] = []

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed_url = urlsplit(self.path)
        self.requests.append(("GET", parsed_url.path, parse_qs(parsed_url.query), None))

        if parsed_url.path == "/v1/dataset/dataset-reference":
            self._send_json({"id": "dataset-reference", "project_id": "project-id", "name": "dataset-name"})
        elif parsed_url.path == "/v1/project/project-id":
            self._send_json({"id": "project-id", "name": "project-name"})
        elif parsed_url.path in {
            "/environment-object/dataset/dataset-id/prod%2Fstable",
            "/environment-object/dataset/dataset-reference/prod%2Fstable",
        }:
            self._send_json({"object_version": "2"})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed_url = urlsplit(self.path)
        content_length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(content_length))
        self.requests.append(("POST", parsed_url.path, parse_qs(parsed_url.query), body))

        if parsed_url.path == "/v1/project":
            self._send_json({"id": "project-id", "name": "project-name"})
        elif parsed_url.path == "/v1/dataset":
            self._send_json({"id": "dataset-id", "project_id": "project-id", "name": "dataset-name"})
        elif parsed_url.path in {
            "/v1/dataset/dataset-id/fetch",
            "/v1/dataset/dataset-reference/fetch",
        }:
            self._send_json({"events": []})
        elif parsed_url.path == "/btql":
            self._send_json({"data": []})
        else:
            self._send_json({"error": "not found"}, status=404)


@pytest.fixture
def dataset_api_server() -> Iterator[str]:
    _DatasetAPIHandler.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _DatasetAPIHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _logged_in_state(base_url: str, *, org_name: str | None = "test org") -> BraintrustState:
    state = BraintrustState()
    state.logged_in = True
    state.app_url = base_url
    state.api_url = base_url
    state.org_id = "org-id"
    state.org_name = org_name
    state._client = BraintrustClient(api_key="test-key", app_url=base_url, api_url=base_url)
    return state


def _request_body(path: str) -> dict[str, Any]:
    return next(
        body
        for method, request_path, _query, body in _DatasetAPIHandler.requests
        if method == "POST" and request_path == path
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference",
    [
        pytest.param({"project_name": "project-name", "dataset_name": "dataset-name"}, id="name"),
        pytest.param({"dataset_id": "dataset-reference"}, id="id"),
    ],
)
@pytest.mark.parametrize(
    ("org_name", "expected_query"),
    [
        pytest.param("test org", {"org_name": ["test org"]}, id="with-org"),
        pytest.param(None, {}, id="without-org"),
    ],
)
async def test_get_dataset_resolves_environment_to_pinned_version(
    dataset_api_server: str,
    reference: dict[str, str],
    org_name: str | None,
    expected_query: dict[str, list[str]],
) -> None:
    dataset = await get_dataset(
        _logged_in_state(dataset_api_server, org_name=org_name),
        {
            **reference,
            "dataset_environment": "prod/stable",
            "_internal_btql": {"limit": 10},
        },
    )

    assert list(dataset) == []
    expected_dataset_id = reference.get("dataset_id", "dataset-id")
    assert (
        "GET",
        f"/environment-object/dataset/{expected_dataset_id}/prod%2Fstable",
        expected_query,
        None,
    ) in _DatasetAPIHandler.requests
    assert _request_body("/btql")["version"] == "2"
    assert _request_body("/btql")["query"]["limit"] == 10
    if "dataset_id" in reference:
        assert not any(
            method == "POST" and path == "/v1/dataset" for method, path, _query, _body in _DatasetAPIHandler.requests
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference",
    [
        pytest.param({"project_name": "project-name", "dataset_name": "dataset-name"}, id="name"),
        pytest.param({"dataset_id": "dataset-reference"}, id="id"),
    ],
)
async def test_get_dataset_prefers_explicit_version_over_environment(
    dataset_api_server: str, reference: dict[str, str]
) -> None:
    dataset = await get_dataset(
        _logged_in_state(dataset_api_server),
        {
            **reference,
            "dataset_version": "1",
            "dataset_environment": "prod/stable",
        },
    )

    assert list(dataset) == []
    assert not any(
        path.startswith("/environment-object/") for _method, path, _query, _body in _DatasetAPIHandler.requests
    )
    expected_dataset_id = reference.get("dataset_id", "dataset-id")
    assert _request_body(f"/v1/dataset/{expected_dataset_id}/fetch")["version"] == "1"


@pytest.mark.asyncio
async def test_get_dataset_returns_inline_data_unchanged() -> None:
    data = [{"input": "hello", "expected": "world"}]

    assert await get_dataset(BraintrustState(), {"data": data}) is data
