import http.server
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from braintrust.logger import BraintrustState
from braintrust.test_helpers import has_devserver_installed


@pytest.fixture
def function_server() -> Iterator[tuple[str, list[str | None]]]:
    org_names = []

    class FunctionHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            org_names.append(self.headers.get("x-bt-org-name"))

            body = b'{"score": 1}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FunctionHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", org_names
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_ui_scorer_forwards_org_name_to_function_invoke(function_server):
    if not has_devserver_installed():
        pytest.skip("Devserver dependencies not installed (requires .[cli])")

    from braintrust.devserver.server import make_scorer

    base_url, org_names = function_server
    state = BraintrustState()
    state.org_name = "test-org"
    state.proxy_url = base_url

    scorer = make_scorer(state, "ui-scorer", {"function_id": "function-id"}, "project-id")
    scorer("input", "output", "expected", {})

    assert org_names == ["test-org"]
