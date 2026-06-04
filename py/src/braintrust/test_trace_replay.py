import json

import pytest

from braintrust.score import Score
from braintrust.trace_replay import load_trace_file, replay_traces, run_cli


def _agent_trace_rows():
    return [
        {
            "span_id": "root-1",
            "root_span_id": "root-1",
            "is_root": True,
            "input": {"messages": [{"role": "user", "content": "refund status"}]},
            "output": {"answer": "refund pending"},
            "expected": {"answer": "refund approved"},
            "metadata": {"customer_tier": "enterprise"},
            "scores": {"answer_match": 0.0},
            "metrics": {"tokens": 120, "duration": 2.5},
            "span_attributes": {"type": "task", "name": "support-agent"},
        },
        {
            "span_id": "llm-1",
            "root_span_id": "root-1",
            "input": {"prompt": "Classify refund"},
            "output": {"tool_call": "lookup_refund"},
            "metrics": {"tokens": 80, "duration": 1.2},
            "span_parents": ["root-1"],
            "span_attributes": {"type": "llm", "name": "planner"},
        },
        {
            "span_id": "tool-1",
            "root_span_id": "root-1",
            "input": {"order_id": "ord_123"},
            "output": {"status": "approved"},
            "metrics": {"duration": 0.3},
            "span_parents": ["llm-1"],
            "span_attributes": {"type": "tool", "name": "lookup_refund"},
        },
    ]


def _approved_refund_task(input, trace, metadata=None):
    assert metadata == {"customer_tier": "enterprise"}
    assert trace.get_configuration()["root_span_id"] == "root-1"
    return {"answer": "refund approved", "tool_spans": len(trace.spans)}


async def _answer_match(input, output, expected, trace):
    assert input["messages"][0]["content"] == "refund status"
    llm_spans = await trace.get_spans(["llm"])
    assert [span.span_id for span in llm_spans] == ["llm-1"]
    return Score(name="answer_match", score=float(output["answer"] == expected["answer"]))


def _boolean_answer_match(input, output, expected):
    return output["answer"] == expected["answer"]


def test_load_trace_file_accepts_wrapped_json(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"spans": _agent_trace_rows()}), encoding="utf-8")

    cases = load_trace_file(trace_path)

    assert len(cases) == 1
    assert cases[0].trace.root_span_id == "root-1"
    assert cases[0].baseline_output == {"answer": "refund pending"}
    assert cases[0].expected == {"answer": "refund approved"}


def test_load_trace_file_accepts_jsonl(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("\n".join(json.dumps(row) for row in _agent_trace_rows()), encoding="utf-8")

    cases = load_trace_file(trace_path)

    assert len(cases) == 1
    assert len(cases[0].trace.spans) == 3


@pytest.mark.asyncio
async def test_replay_traces_runs_task_and_reports_score_deltas(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_agent_trace_rows()), encoding="utf-8")
    cases = load_trace_file(trace_path)

    summary = await replay_traces(cases, task=_approved_refund_task, scorers=[_answer_match])

    result = summary.results[0]
    assert result.output == {"answer": "refund approved", "tool_spans": 3}
    assert result.baseline_output == {"answer": "refund pending"}
    assert result.scores == {"answer_match": 1.0}
    assert result.baseline_scores == {"answer_match": 0.0}
    assert result.score_deltas == {"answer_match": 1.0}
    assert result.metrics["span_count"] == 3
    assert result.metrics["tokens"] == 200
    assert result.metric_deltas["tokens"] == 80
    assert summary.score_averages == {"answer_match": 1.0}
    assert summary.score_delta_averages == {"answer_match": 1.0}


def test_cli_emits_json_report(tmp_path, capsys):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_agent_trace_rows()), encoding="utf-8")

    class Args:
        trace_file = str(trace_path)
        task = "braintrust.test_trace_replay:_approved_refund_task"
        score = [
            "braintrust.test_trace_replay:_answer_match",
            "braintrust.test_trace_replay:_boolean_answer_match",
        ]
        json = True
        min_score = []
        min_score_delta = []
        fail_on_error = False

    run_cli(Args())

    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["trace_count"] == 1
    assert report["summary"]["failed_count"] == 0
    assert report["results"][0]["score_deltas"]["answer_match"] == 1.0
    assert report["results"][0]["scores"]["_boolean_answer_match"] == 1.0
    assert report["checks"] == {"passed": True, "failures": []}


def test_cli_thresholds_fail_on_regression(tmp_path, capsys):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_agent_trace_rows()), encoding="utf-8")

    class Args:
        trace_file = str(trace_path)
        task = None
        score = ["braintrust.test_trace_replay:_answer_match"]
        json = True
        min_score = ["answer_match=0.5"]
        min_score_delta = ["answer_match=0"]
        fail_on_error = False

    with pytest.raises(SystemExit) as exc_info:
        run_cli(Args())

    assert exc_info.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["passed"] is False
    assert "score 'answer_match' averaged 0.0000" in report["checks"]["failures"][0]
