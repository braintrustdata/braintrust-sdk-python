import asyncio
import json

import pytest
from braintrust.test_helpers import has_devserver_installed


def _parse_sse_events(response_text: str) -> list[dict[str, object]]:
    events = []
    lines = response_text.strip().split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("event: "):
            event_type = lines[i][7:].strip()
            i += 1
            if i < len(lines) and lines[i].startswith("data: "):
                raw_data = lines[i][6:].strip()
                try:
                    data = json.loads(raw_data) if raw_data else None
                except json.JSONDecodeError:
                    data = raw_data
                events.append({"event": event_type, "data": data})
                i += 1
            else:
                events.append({"event": event_type, "data": None})
        else:
            i += 1
    return events


def test_dispatch_completion_webhook_retries(monkeypatch):
    from braintrust.devserver import server as devserver_module

    attempts = []
    sleep_calls = []

    async def fake_send(webhook_url, body, timeout):
        attempts.append((webhook_url, body, timeout))
        if len(attempts) < 3:
            raise RuntimeError("transient")

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(devserver_module, "_send_completion_webhook_request", fake_send)
    monkeypatch.setattr(devserver_module.asyncio, "sleep", fake_sleep)

    asyncio.run(
        devserver_module.dispatch_completion_webhook(
            "https://example.com/webhook",
            {"projectName": "my-project", "experimentName": "my-exp"},
            attempts=3,
            backoff_seconds=(1.0, 2.0, 4.0),
            timeout_seconds=10.0,
        )
    )

    assert len(attempts) == 3
    assert sleep_calls == [1.0, 2.0]


def test_parse_eval_body_accepts_on_complete_webhook():
    from braintrust.devserver.schemas import parse_eval_body

    parsed = parse_eval_body(
        {
            "name": "my-eval",
            "on_complete_webhook": "https://example.com/webhook",
        }
    )

    assert parsed["on_complete_webhook"] == "https://example.com/webhook"


@pytest.mark.skipif(not has_devserver_installed(), reason="Devserver dependencies not installed (requires .[cli])")
def test_eval_webhook_failure_non_fatal_for_stream(monkeypatch):
    from braintrust import Evaluator
    from braintrust.devserver import server as devserver_module
    from braintrust.devserver.server import create_app
    from braintrust.logger import BraintrustState
    from starlette.testclient import TestClient

    evaluator = Evaluator(
        project_name="test-project",
        eval_name="test-eval",
        data=lambda: [{"input": "x", "expected": "x"}],
        task=lambda input_value, _hooks: input_value,
        scores=[],
        experiment_name=None,
        metadata=None,
    )

    async def fake_cached_login(**_kwargs):
        return BraintrustState()

    class FakeSummary:
        def as_dict(self):
            return {
                "project_name": "test-project",
                "experiment_name": "test-eval",
                "scores": {},
            }

    class FakeResult:
        summary = FakeSummary()

    dispatch_calls = []

    async def fake_dispatch(webhook_url, summary, **_kwargs):
        dispatch_calls.append((webhook_url, summary))
        raise RuntimeError("webhook delivery failed")

    async def fake_eval_async(*, on_complete, **_kwargs):
        await on_complete(FakeSummary())
        return FakeResult()

    monkeypatch.setattr(devserver_module, "cached_login", fake_cached_login)
    monkeypatch.setattr(devserver_module, "dispatch_completion_webhook", fake_dispatch)
    monkeypatch.setattr(devserver_module, "EvalAsync", fake_eval_async)

    response = TestClient(create_app([evaluator])).post(
        "/eval",
        headers={
            "x-bt-auth-token": "test-api-key",
            "x-bt-org-name": "test-org",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json={
            "name": "test-eval",
            "stream": True,
            "on_complete_webhook": "https://example.com/webhook",
            "data": [{"input": "x", "expected": "x"}],
        },
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    event_types = [e["event"] for e in events]

    assert "summary" in event_types
    assert "done" in event_types
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0][0] == "https://example.com/webhook"
    assert dispatch_calls[0][1]["experimentName"] == "test-eval"
