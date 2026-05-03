import json
import os
from pathlib import Path
from typing import Any

import pytest
from braintrust.framework import _evals
from braintrust.test_helpers import has_devserver_installed


HAS_PYDANTIC = __import__("importlib.util").util.find_spec("pydantic") is not None


@pytest.fixture
def client():
    """Create test client using the real simple_eval.py example."""
    # Skip if devserver dependencies are not installed
    if not has_devserver_installed():
        pytest.skip("Devserver dependencies not installed (requires .[cli])")

    # Import CLI dependencies inside the fixture
    from braintrust.devserver.server import create_app
    from starlette.testclient import TestClient

    # Use the real simple_eval.py example
    eval_file = Path(__file__).parent.parent.parent.parent / "examples" / "evals" / "simple_eval.py"

    # Clear any existing evaluators
    _evals.clear()

    # Load the eval file to register evaluators (but don't run them)
    spec = __import__("importlib.util").util.spec_from_file_location("simple_eval", str(eval_file))
    module = __import__("importlib.util").util.module_from_spec(spec)

    # Get evaluators from the module without executing Eval()
    # We need to parse the file and extract the Evaluator definition
    import re

    from braintrust import Evaluator

    def task(input: str, hooks) -> str:
        """Simple math task."""
        match = re.search(r"(\d+)\+(\d+)", input)
        if match:
            return str(int(match.group(1)) + int(match.group(2)))
        return "I don't know"

    def scorer(input: str, output: str, expected: str) -> float:
        """Simple exact match scorer."""
        return 1.0 if output == expected else 0.0

    def classifier(input: str, output: str, expected: str) -> dict[str, str]:
        return {"id": "correct" if output == expected else "incorrect", "name": "answer_type"}

    evaluator = Evaluator(
        project_name="test-math-eval",
        eval_name="simple-math-eval",
        data=lambda: [
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "What is 3+3?", "expected": "6"},
            {"input": "What is 5+5?", "expected": "10"},
        ],
        task=task,
        scores=[scorer],
        classifiers=[classifier],
        experiment_name=None,
        metadata=None,
    )

    # Create app with the evaluator
    app = create_app([evaluator])
    return TestClient(app)


@pytest.fixture
def api_key():
    """Provide test API key."""
    return os.getenv("BRAINTRUST_API_KEY", "test-api-key")


@pytest.fixture
def org_name():
    """Provide test org name."""
    return os.getenv("BRAINTRUST_ORG_NAME", "matt-test-org")


def test_devserver_health_check(client):
    """Test that server responds to health check."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "Hello, world!"


def test_cors_preflight_allows_gateway_header(client):
    """Test that CORS preflight accepts x-bt-use-gateway header.

    The Braintrust Playground sends this header when gateway routing is
    enabled.  If it is missing from the devserver's allowed-headers list
    the browser blocks the actual request with a CORS error.
    """
    response = client.options(
        "/eval",
        headers={
            "origin": "https://www.braintrust.dev",
            "access-control-request-method": "POST",
            "access-control-request-headers": "x-bt-use-gateway",
        },
    )
    assert response.status_code == 200
    allowed = response.headers.get("access-control-allow-headers", "")
    assert "x-bt-use-gateway" in allowed, f"x-bt-use-gateway not found in access-control-allow-headers: {allowed}"


@pytest.mark.vcr
def test_devserver_list_evaluators(client, api_key, org_name):
    """Test listing evaluators endpoint."""
    response = client.get("/list", headers={"x-bt-auth-token": api_key, "x-bt-org-name": org_name})
    assert response.status_code == 200
    evaluators = response.json()
    assert "simple-math-eval" in evaluators
    assert evaluators["simple-math-eval"]["scores"] == [{"name": "scorer"}]
    assert evaluators["simple-math-eval"]["classifiers"] == [{"name": "classifier"}]


def parse_sse_events(response_text: str) -> list[dict[str, Any]]:
    """Parse SSE events from response text."""
    events = []
    lines = response_text.strip().split("\n")
    i = 0

    while i < len(lines):
        if lines[i].startswith("event: "):
            event_type = lines[i][7:].strip()
            i += 1

            if i < len(lines) and lines[i].startswith("data: "):
                data_str = lines[i][6:].strip()
                try:
                    data = json.loads(data_str) if data_str else None
                except json.JSONDecodeError:
                    data = data_str

                events.append({"event": event_type, "data": data})
                i += 1
            else:
                events.append({"event": event_type, "data": None})
        else:
            i += 1

    return events


@pytest.mark.skip
@pytest.mark.vcr
def test_eval_sse_streaming(client, api_key, org_name):
    """
    Comprehensive test for SSE streaming during eval execution.

    Verifies:
    1. Event order: start → progress* → summary → done
    2. Progress events are emitted
    3. Start event has metadata (experimentName, projectName)
    4. Summary event has camelCase fields (not snake_case)
    5. Response format is correct
    """
    response = client.post(
        "/eval",
        headers={
            "x-bt-auth-token": api_key,
            "x-bt-org-name": org_name,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json={
            "name": "simple-math-eval",
            "stream": True,
            "data": [
                {"input": "What is 2+2?", "expected": "4"},
                {"input": "What is 3+3?", "expected": "6"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/event-stream; charset=utf-8"

    events = parse_sse_events(response.text)
    event_types = [e["event"] for e in events]

    # Verify event order
    assert len(event_types) > 0
    assert event_types[0] == "start"
    assert event_types[-1] == "done"
    assert "summary" in event_types

    # Verify progress events exist
    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) > 0

    # Verify start event has metadata
    start_event = next(e for e in events if e["event"] == "start")
    assert "experimentName" in start_event["data"]
    assert "projectName" in start_event["data"]

    # Verify summary event has camelCase fields
    summary_event = next(e for e in events if e["event"] == "summary")
    assert summary_event is not None
    summary_data = summary_event["data"]
    assert summary_data is not None

    assert "experimentName" in summary_data
    assert "projectName" in summary_data
    assert "scores" in summary_data

    # Should NOT have snake_case fields
    assert "experiment_name" not in summary_data
    assert "project_name" not in summary_data


@pytest.mark.vcr
def test_eval_error_handling(client, api_key, org_name):
    """Test error handling for non-existent evaluator."""
    response = client.post(
        "/eval",
        headers={
            "x-bt-auth-token": api_key,
            "x-bt-org-name": org_name,
            "Content-Type": "application/json",
        },
        json={"name": "non-existent-eval", "stream": False},
    )

    assert response.status_code == 404
    error = response.json()
    assert "error" in error
    assert "not found" in error["error"].lower()


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_eval_uses_inline_request_parameters(api_key, org_name, monkeypatch):
    from braintrust import Evaluator
    from braintrust.devserver import server as devserver_module
    from braintrust.devserver.server import create_app
    from braintrust.logger import BraintrustState
    from pydantic import BaseModel
    from starlette.testclient import TestClient

    class RequiredInt(BaseModel):
        value: int

    def task(input: str, hooks) -> dict[str, Any]:
        return {"input": input, "num_samples": hooks.parameters["num_samples_without_default"]}

    evaluator = Evaluator(
        project_name="test-math-eval",
        eval_name="inline-parameter-eval",
        data=lambda: [{"input": "What is 2+2?", "expected": "4"}],
        task=task,
        scores=[],
        experiment_name=None,
        metadata=None,
        parameters={"num_samples_without_default": RequiredInt},
    )

    async def fake_cached_login(**_kwargs):
        return BraintrustState()

    class FakeSummary:
        def as_dict(self):
            return {"experiment_name": "inline-parameter-eval", "project_name": "test-math-eval", "scores": {}}

    class FakeResult:
        summary = FakeSummary()

    async def fake_eval_async(*, task, data, parameters, **_kwargs):
        assert parameters == {"num_samples_without_default": 1}
        datum = data[0]
        hooks = type("Hooks", (), {"parameters": parameters, "report_progress": lambda self, _progress: None})()
        await task(datum["input"], hooks)
        return FakeResult()

    monkeypatch.setattr(devserver_module, "cached_login", fake_cached_login)
    monkeypatch.setattr(devserver_module, "EvalAsync", fake_eval_async)

    response = TestClient(create_app([evaluator])).post(
        "/eval",
        headers={
            "x-bt-auth-token": api_key,
            "x-bt-org-name": org_name,
            "Content-Type": "application/json",
        },
        json={
            "name": "inline-parameter-eval",
            "stream": False,
            "parameters": {"num_samples_without_default": 1},
            "data": [{"input": "What is 2+2?", "expected": "4"}],
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize("request_project_id", [pytest.param("", id="empty"), pytest.param("__omit__", id="omitted")])
def test_eval_falls_back_to_evaluator_project_id_when_request_omits_or_empty_it(
    api_key, org_name, monkeypatch, request_project_id
):
    """run_eval must honor the registered evaluator's project_id when the request omits/empties it.

    Regression: ``run_eval`` builds ``EvalAsync(...)`` kwargs with
    ``{**eval_kwargs, ..., "project_id": eval_data.get("project_id")}``.
    The trailing key always wins in dict-spread merging, so a request
    that omits ``project_id`` clobbers the registered evaluator's
    ``project_id`` to ``None``. ``EvalAsync`` then falls back to using
    ``name`` as the project name (per ``framework.Eval`` docstring),
    routing experiments into a per-evaluator-name auto-created project
    instead of the project the evaluator was registered against.
    """
    from braintrust import Evaluator
    from braintrust.devserver import server as devserver_module
    from braintrust.devserver.server import create_app
    from braintrust.logger import BraintrustState
    from starlette.testclient import TestClient

    evaluator = Evaluator(
        project_name="ignored-project-name",
        eval_name="project-id-fallback-eval",
        data=lambda: [{"input": "ping", "expected": "pong"}],
        task=lambda input, _hooks: "pong",
        scores=[],
        experiment_name=None,
        metadata=None,
        project_id="evaluator-registered-project-id",
    )

    captured: dict[str, Any] = {}

    async def fake_cached_login(**_kwargs):
        return BraintrustState()

    class FakeSummary:
        def as_dict(self):
            return {"experiment_name": evaluator.eval_name, "project_name": "", "scores": {}}

    class FakeResult:
        summary = FakeSummary()

    async def fake_eval_async(*, project_id, **_kwargs):
        captured["project_id"] = project_id
        return FakeResult()

    monkeypatch.setattr(devserver_module, "cached_login", fake_cached_login)
    monkeypatch.setattr(devserver_module, "EvalAsync", fake_eval_async)

    eval_request = {
        "name": "project-id-fallback-eval",
        "stream": False,
        "data": [{"input": "ping", "expected": "pong"}],
    }
    if request_project_id != "__omit__":
        eval_request["project_id"] = request_project_id

    response = TestClient(create_app([evaluator])).post(
        "/eval",
        headers={
            "x-bt-auth-token": api_key,
            "x-bt-org-name": org_name,
            "Content-Type": "application/json",
        },
        json=eval_request,
    )

    assert response.status_code == 200
    assert captured["project_id"] == "evaluator-registered-project-id"


def test_eval_request_project_id_overrides_evaluator(api_key, org_name, monkeypatch):
    """An explicit ``project_id`` in the request body still takes precedence."""
    from braintrust import Evaluator
    from braintrust.devserver import server as devserver_module
    from braintrust.devserver.server import create_app
    from braintrust.logger import BraintrustState
    from starlette.testclient import TestClient

    evaluator = Evaluator(
        project_name="ignored-project-name",
        eval_name="project-id-override-eval",
        data=lambda: [{"input": "ping", "expected": "pong"}],
        task=lambda input, _hooks: "pong",
        scores=[],
        experiment_name=None,
        metadata=None,
        project_id="evaluator-registered-project-id",
    )

    captured: dict[str, Any] = {}

    async def fake_cached_login(**_kwargs):
        return BraintrustState()

    class FakeSummary:
        def as_dict(self):
            return {"experiment_name": evaluator.eval_name, "project_name": "", "scores": {}}

    class FakeResult:
        summary = FakeSummary()

    async def fake_eval_async(*, project_id, **_kwargs):
        captured["project_id"] = project_id
        return FakeResult()

    monkeypatch.setattr(devserver_module, "cached_login", fake_cached_login)
    monkeypatch.setattr(devserver_module, "EvalAsync", fake_eval_async)

    response = TestClient(create_app([evaluator])).post(
        "/eval",
        headers={
            "x-bt-auth-token": api_key,
            "x-bt-org-name": org_name,
            "Content-Type": "application/json",
        },
        json={
            "name": "project-id-override-eval",
            "stream": False,
            "data": [{"input": "ping", "expected": "pong"}],
            "project_id": "request-explicit-project-id",
        },
    )

    assert response.status_code == 200
    assert captured["project_id"] == "request-explicit-project-id"
