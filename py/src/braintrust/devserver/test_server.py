import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from braintrust.logger import BraintrustState, Experiment
from braintrust.test_helpers import (
    has_devserver_installed,
    init_test_exp,
    with_memory_logger,  # noqa: F401
)


@pytest.fixture
def function_server() -> Iterator[tuple[str, list[tuple[str | None, dict[str, Any]]]]]:
    requests = []

    class FunctionHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(content_length))
            requests.append((self.headers.get("x-bt-org-name"), request))

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
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("global_span_active", [False, True])
def test_ui_scorer_forwards_org_name_and_active_parent_to_function_invoke(
    function_server, with_memory_logger, global_span_active
):
    if not has_devserver_installed():
        pytest.skip("Devserver dependencies not installed (requires .[cli])")

    from braintrust.devserver.server import make_scorer

    base_url, requests = function_server
    state = BraintrustState()
    state.org_name = "test-org"
    state.proxy_url = base_url
    state._override_bg_logger.logger = with_memory_logger

    global_experiment = init_test_exp("test-project")
    request_experiment = Experiment(global_experiment._lazy_metadata, state=state)

    scorer = make_scorer(state, "ui-scorer", {"function_id": "function-id"}, "project-id")

    with global_experiment.start_span(name="unrelated", set_current=global_span_active):
        with request_experiment.start_span(name="eval") as row_span:
            with row_span.start_span(name="ui-scorer", type="score") as scorer_span:
                result = scorer("input", "output", "expected", {})
                expected_parent = scorer_span.export()

    assert result == {"score": 1}
    [(org_name, request)] = requests
    assert org_name == "test-org"
    assert request["parent"] == expected_parent
