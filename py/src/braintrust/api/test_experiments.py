import contextlib
import http.server
import json
import os
import re
import socketserver
import threading
from collections.abc import Mapping
from urllib.parse import urlsplit

import braintrust
import pytest
from braintrust.api import (
    BraintrustClient,
    BraintrustRetryExhaustedError,
    EndpointRouter,
    ExperimentComparison,
    ExperimentRecord,
)
from braintrust.api._transport import Transport
from braintrust.conftest import get_vcr_config
from braintrust.framework import EvalCase, Evaluator, run_evaluator
from braintrust.git_fields import RepoInfo
from braintrust.logger import SummarySkipped, SummarySuccess
from braintrust.test_helpers import init_test_exp, with_memory_logger, with_simulate_login  # noqa: F401


def _normalize_experiment_request(request):
    if not request.body or urlsplit(request.uri).path != "/logs3":
        return request
    try:
        payload = json.loads(request.body)
    except (TypeError, ValueError):
        return request

    for row in payload.get("rows", []):
        row.pop("context", None)
        row.pop("created", None)
        row.pop("root_span_id", None)
        row.pop("span_id", None)
        metrics = row.get("metrics", {})
        metrics.pop("start", None)
        metrics.pop("end", None)
    request.body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return request


def _scrub_experiment_response(response):
    body = response.get("body", {}).get("string")
    body_was_bytes = isinstance(body, bytes)
    if body_was_bytes:
        body = body.decode(errors="replace")
    if not isinstance(body, str):
        return response

    body = re.sub(r"user_email=[^\]]+", "user_email=redacted", body)
    try:
        payload = json.loads(body)
    except ValueError:
        pass
    else:

        def scrub_repo_info(value):
            if isinstance(value, dict):
                if "repo_info" in value:
                    value["repo_info"] = {}
                for nested in value.values():
                    scrub_repo_info(nested)
            elif isinstance(value, list):
                for nested in value:
                    scrub_repo_info(nested)

        scrub_repo_info(payload)
        body = json.dumps(payload, separators=(",", ":"))

    response["body"]["string"] = body.encode() if body_was_bytes else body
    return response


@pytest.fixture(scope="module")
def vcr_config():
    config = get_vcr_config()
    config["before_record_request"] = _normalize_experiment_request
    scrub_sensitive_headers = config["before_record_response"]

    def scrub_response(response):
        return _scrub_experiment_response(scrub_sensitive_headers(response))

    config["before_record_response"] = scrub_response
    return config


@contextlib.contextmanager
def experiment_server(routes):
    class ExperimentHandler(http.server.BaseHTTPRequestHandler):
        requests = []
        route_counts = {}

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle()

        def _handle(self):
            path = urlsplit(self.path).path
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length else b""
            type(self).requests.append((self.command, self.path, body, self.headers.get("Authorization")))
            request_number = type(self).route_counts.get(path, 0)
            type(self).route_counts[path] = request_number + 1
            actions = routes[path]
            action = actions[min(request_number, len(actions) - 1)]
            status, response = action[:2]
            response_headers = action[2] if len(action) > 2 else {}
            response_body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for name, value in response_headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ExperimentHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", ExperimentHandler
    finally:
        server.shutdown()
        server.server_close()


def _client_for_server(url):
    return BraintrustClient.from_transport(
        transport=Transport(sleep=lambda _delay: None),
        router=EndpointRouter(app_url=url, api_url=url),
        api_key="test-key",
        org_id="test-org-id",
        org_name="test-org-name",
    )


def _use_client_for_experiment(experiment, client, url):
    state = experiment.state
    state._client = client
    # Keep the legacy destinations local too, so this remains a focused red/green
    # regression test while Experiment migrates from HTTPConnection to ExperimentsAPI.
    state.app_url = url
    state.api_url = url
    state._app_conn = None
    state._api_conn = None


def _comparison_response(score=0.9):
    return {
        "scores": {
            "accuracy": {
                "name": "accuracy",
                "score": score,
                "improvements": 1,
                "regressions": 0,
                "diff": 0.4,
                "new_score_field": {"preserved": True},
            }
        },
        "metrics": {
            "duration": {
                "name": "duration",
                "metric": 1.25,
                "unit": "s",
                "improvements": 0,
                "regressions": 1,
                "diff": 0.1,
                "new_metric_field": ["preserved"],
            }
        },
        "new_backend_field": {"preserved": True},
    }


def test_experiment_service_parses_additive_responses_and_uses_expected_targets():
    routes = {
        "/v1/experiment/base-exp-id": [
            (
                200,
                {
                    "id": "base-exp-id",
                    "name": "baseline",
                    "project_id": "project-id",
                    "new_backend_field": {"preserved": True},
                },
            )
        ],
        "/api/base_experiment/get_id": [
            (
                200,
                {
                    "id": "candidate-exp-id",
                    "name": "candidate",
                    "project_id": "project-id",
                    "base_exp_id": "base-exp-id",
                    "base_exp_name": "baseline",
                    "new_backend_field": 42,
                },
            )
        ],
        "/experiment-comparison2": [(200, _comparison_response())],
    }
    with experiment_server(routes) as (url, handler):
        service = _client_for_server(url).experiments
        experiment = service.get("base-exp-id")
        base = service.get_base("candidate-exp-id")
        comparison = service.compare("candidate-exp-id", base_experiment_id="base-exp-id")

    assert isinstance(experiment, ExperimentRecord)
    assert experiment.id == "base-exp-id"
    assert experiment.name == "baseline"
    assert experiment.raw["new_backend_field"] == {"preserved": True}
    assert base is not None
    assert base.id == "base-exp-id"
    assert base.name == "baseline"
    assert base.raw["new_backend_field"] == 42
    assert isinstance(comparison, ExperimentComparison)
    assert comparison.scores["accuracy"].score == 0.9
    assert comparison.scores["accuracy"].raw["score"] == 0.9
    assert comparison.scores["accuracy"].raw["new_score_field"] == {"preserved": True}
    assert comparison.metrics["duration"].raw["new_metric_field"] == ["preserved"]
    assert comparison.raw["new_backend_field"] == {"preserved": True}

    requests = handler.requests
    assert [request[0] for request in requests] == ["GET", "POST", "GET"]
    assert requests[0][1] == "/v1/experiment/base-exp-id"
    assert json.loads(requests[1][2]) == {"id": "candidate-exp-id"}
    assert requests[2][1] == ("/experiment-comparison2?experiment_id=candidate-exp-id&base_experiment_id=base-exp-id")
    assert all(request[3] == "Bearer test-key" for request in requests)


def test_experiment_summarize_retries_transient_comparison_failure_issue_639(with_memory_logger, with_simulate_login):
    # Regression test for https://github.com/braintrustdata/braintrust-sdk-python/issues/639
    routes = {
        "/v1/experiment/base-exp-id": [(200, {"id": "base-exp-id", "name": "baseline"})],
        "/experiment-comparison2": [(503, {"error": "temporary outage"}), (200, _comparison_response())],
    }
    with experiment_server(routes) as (url, handler):
        client = _client_for_server(url)
        experiment = init_test_exp("candidate-exp-id", "project-id")
        _use_client_for_experiment(experiment, client, url)

        summary = experiment.summarize(comparison_experiment_id="base-exp-id")

    assert summary.comparison_experiment_name == "baseline"
    assert isinstance(summary.comparison, SummarySuccess)
    assert summary.comparison.scores["accuracy"].score == 0.9
    assert summary.comparison.metrics["duration"].metric == 1.25
    assert handler.route_counts["/experiment-comparison2"] == 2


def test_experiment_summarize_raises_after_retry_exhaustion(with_memory_logger, with_simulate_login):
    routes = {
        "/v1/experiment/base-exp-id": [(200, {"id": "base-exp-id", "name": "baseline"})],
        "/experiment-comparison2": [
            (
                503,
                {"error": "persistent outage"},
                {"x-bt-internal-trace-id": "summary-trace-id", "x-unrelated-header": "excluded"},
            )
        ],
    }
    with experiment_server(routes) as (url, handler):
        experiment = init_test_exp("candidate-exp-id", "project-id")
        _use_client_for_experiment(experiment, _client_for_server(url), url)

        with pytest.raises(BraintrustRetryExhaustedError) as exc_info:
            experiment.summarize(comparison_experiment_id="base-exp-id")

    error = exc_info.value
    assert error.status_code == 503
    assert error.attempts == 4
    assert error.response_body == '{"error": "persistent outage"}'
    assert error.request_id == "summary-trace-id"
    assert error.response_headers == {
        "content-type": "application/json",
        "x-bt-internal-trace-id": "summary-trace-id",
    }
    assert handler.route_counts["/experiment-comparison2"] == 4


def test_experiment_summarize_surfaces_comparison_lookup_failure(with_memory_logger, with_simulate_login):
    routes = {
        "/v1/experiment/base-exp-id": [(503, {"error": "lookup unavailable"})],
    }
    with experiment_server(routes) as (url, handler):
        experiment = init_test_exp("candidate-exp-id", "project-id")
        _use_client_for_experiment(experiment, _client_for_server(url), url)

        with pytest.raises(BraintrustRetryExhaustedError):
            experiment.summarize(comparison_experiment_id="base-exp-id")

    assert handler.route_counts["/v1/experiment/base-exp-id"] == 4


def test_experiment_summarize_genuine_empty_comparison_is_success(with_memory_logger, with_simulate_login):
    routes = {
        "/v1/experiment/base-exp-id": [(200, {"id": "base-exp-id", "name": "baseline"})],
        "/experiment-comparison2": [(200, {"scores": {}, "metrics": {}})],
    }
    with experiment_server(routes) as (url, _handler):
        experiment = init_test_exp("candidate-exp-id", "project-id")
        _use_client_for_experiment(experiment, _client_for_server(url), url)

        summary = experiment.summarize(comparison_experiment_id="base-exp-id")

    assert isinstance(summary.comparison, SummarySuccess)
    assert summary.comparison.scores == {}
    assert summary.comparison.metrics == {}
    assert summary.as_dict()["comparison"] == {
        "scores": {},
        "metrics": {},
        "status": "success",
    }


def test_experiment_summarize_without_scores_is_marked_skipped(with_memory_logger, with_simulate_login):
    experiment = init_test_exp("candidate-exp-id", "project-id")

    summary = experiment.summarize(summarize_scores=False)

    assert isinstance(summary.comparison, SummarySkipped)
    assert summary.comparison.reason == "Score summarization was disabled"
    assert summary.as_dict()["comparison"] == {
        "status": "skipped",
        "reason": "Score summarization was disabled",
    }


@pytest.mark.asyncio
async def test_run_evaluator_raises_on_summary_failure(with_memory_logger, with_simulate_login):
    routes = {
        "/v1/experiment/base-exp-id": [(200, {"id": "base-exp-id", "name": "baseline"})],
        "/experiment-comparison2": [(503, {"error": "persistent outage"})],
    }
    evaluator = Evaluator(
        project_name="project-id",
        eval_name="candidate-exp-id",
        data=[EvalCase(input="hello", expected="hello")],
        task=lambda input_value: input_value,
        scores=[lambda input_value, output, expected: 1.0],
        experiment_name="candidate-exp-id",
        metadata=None,
        base_experiment_id="base-exp-id",
    )
    with experiment_server(routes) as (url, _handler):
        experiment = init_test_exp("candidate-exp-id", "project-id")
        _use_client_for_experiment(experiment, _client_for_server(url), url)

        with pytest.raises(BraintrustRetryExhaustedError):
            await run_evaluator(experiment, evaluator, position=None, filters=[])


def test_base_experiment_400_is_none_without_retry():
    routes = {"/api/base_experiment/get_id": [(400, {"error": "No base experiment"})]}
    with experiment_server(routes) as (url, handler):
        base = _client_for_server(url).experiments.get_base("fresh-exp-id")

    assert base is None
    assert handler.route_counts["/api/base_experiment/get_id"] == 1


def _api_key():
    return os.environ.get("BRAINTRUST_API_KEY", "sk-dummy-for-vcr-replay")


def _log_score(experiment, score):
    experiment.log(
        id=f"{experiment.name}-row",
        input={"question": "What is 2 + 2?"},
        output="4",
        expected="4",
        scores={"exact_match": score},
    )
    experiment.flush()


@pytest.mark.vcr
def test_experiment_summarize_end_to_end_with_real_backend():
    project_name = "python-sdk-api-experiment-service-vcr"
    base = braintrust.init(
        project=project_name,
        experiment="experiment-service-base",
        api_key=_api_key(),
        update=True,
        set_current=False,
        repo_info=RepoInfo(),
    )
    _log_score(base, 0.5)

    candidate = braintrust.init(
        project=project_name,
        experiment="experiment-service-candidate",
        api_key=_api_key(),
        base_experiment_id=base.id,
        update=True,
        set_current=False,
        repo_info=RepoInfo(),
    )
    _log_score(candidate, 1.0)

    service = candidate.state.api_client().experiments
    record = service.get(base.id)
    assert record.id == base.id
    assert record.name == base.name

    persisted_base = candidate.fetch_base_experiment()
    assert persisted_base is not None
    assert persisted_base.id == base.id
    assert persisted_base.name == base.name

    automatic_summary = candidate.summarize()
    assert automatic_summary.comparison_experiment_name == base.name
    assert isinstance(automatic_summary.comparison, SummarySuccess)
    assert automatic_summary.comparison.status == "success"
    assert automatic_summary.comparison.scores["exact_match"].score == 1.0
    assert automatic_summary.comparison.scores["exact_match"].diff == 0.5
    assert isinstance(automatic_summary.comparison.metrics, Mapping)
    assert automatic_summary.as_dict()["comparison"]["status"] == "success"

    explicit_summary = candidate.summarize(comparison_experiment_id=base.id)
    assert explicit_summary.comparison_experiment_name == base.name
    assert isinstance(explicit_summary.comparison, SummarySuccess)
    assert explicit_summary.comparison.scores == automatic_summary.comparison.scores
    assert explicit_summary.comparison.metrics == automatic_summary.comparison.metrics


@pytest.mark.vcr
def test_fresh_experiment_has_no_base_on_real_backend():
    experiment = braintrust.init(
        project="python-sdk-api-experiment-service-fresh-vcr",
        experiment="fresh-experiment-without-base",
        api_key=_api_key(),
        update=True,
        set_current=False,
        repo_info=RepoInfo(),
    )

    assert experiment.fetch_base_experiment() is None
