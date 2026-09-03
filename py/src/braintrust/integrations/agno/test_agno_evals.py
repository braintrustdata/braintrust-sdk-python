# pyright: reportPrivateUsage=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Tracing coverage for agno's eval framework (``agno.eval``).

Every eval type is exercised against real agno objects and recorded provider traffic.
The modules imported at the top exist in every version of the agno matrix; the two
that do not — ``agent_as_judge`` (agno >= 2.4) and ``suite`` (agno >= 2.9) — are
imported per test so the older sessions skip just those.
"""

import asyncio

import pytest
from braintrust import logger
from braintrust.integrations.agno import eval_experiments, setup_agno, wrap_reliability_eval
from braintrust.test_helpers import find_span_by_name, find_spans_by_type, init_test_exp, init_test_logger

from ._test_agno_helpers import PROJECT_NAME


agno_agent = pytest.importorskip("agno.agent")
agno_openai = pytest.importorskip("agno.models.openai")
accuracy_module = pytest.importorskip("agno.eval.accuracy")
reliability_module = pytest.importorskip("agno.eval.reliability")
performance_module = pytest.importorskip("agno.eval.performance")

MODEL = "gpt-4o-mini"


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        assert not bgl.pop(), "spans leaked in from a previous test"
        yield bgl


@pytest.fixture(scope="module", autouse=True)
def setup_wrapper():
    setup_agno(project_name=PROJECT_NAME)
    yield


@pytest.fixture(autouse=True)
def suite_experiments_off():
    """Keep suite rows in logs unless a test opts into experiment routing.

    ``eval_experiments`` defaults to on, which would have a suite run call the real
    ``braintrust.init()``.
    """
    eval_experiments.configure(False)
    try:
        yield
    finally:
        eval_experiments.configure(None)


def _names(spans):
    return [span["span_attributes"]["name"] for span in spans]


def _spans_named(spans, prefix):
    return [span for span in spans if span["span_attributes"]["name"].startswith(prefix)]


def _make_agent(*, name="Math Agent", tools=None):
    return agno_agent.Agent(
        name=name,
        model=agno_openai.OpenAIChat(id=MODEL),
        instructions="Answer with the final number only.",
        tools=tools,
    )


# ---------------------------------------------------------------------------
# AccuracyEval
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_accuracy_eval_logs_score_and_nests_agent(memory_logger):
    evaluation = accuracy_module.AccuracyEval(
        name="Multiplication Eval",
        model=agno_openai.OpenAIChat(id=MODEL),
        agent=_make_agent(),
        input="What is 10*5?",
        expected_output="50",
        num_iterations=1,
    )
    result = evaluation.run(print_summary=False, print_results=False)
    assert result is not None
    assert result.avg_score is not None

    spans = memory_logger.pop()
    eval_span = find_span_by_name(spans, "Multiplication Eval")

    assert eval_span["span_attributes"]["type"] == "eval"
    assert eval_span["context"]["span_origin"]["instrumentation"]["name"] == "agno-auto"
    assert eval_span["input"] == "What is 10*5?"
    assert eval_span["expected"] == "50"
    assert eval_span["output"]
    assert eval_span["scores"]["accuracy"] == pytest.approx(result.avg_score / 10)
    assert 0.0 <= eval_span["scores"]["accuracy"] <= 1.0

    metadata = eval_span["metadata"]
    assert metadata["eval_type"] == "accuracy"
    assert metadata["eval_name"] == "Multiplication Eval"
    assert metadata["num_iterations"] == 1
    assert metadata["agent_name"] == "Math Agent"
    assert metadata["evaluator_model"] == MODEL
    assert metadata["avg_score"] == result.avg_score
    assert metadata["iteration_scores"] == [result.results[0].score]

    # The iteration's judge call is a scorer span under the eval row.
    score_span = find_span_by_name(spans, "accuracy")
    assert score_span["span_attributes"]["type"] == "score"
    assert score_span["span_attributes"]["purpose"] == "scorer"
    assert score_span["span_parents"] == [eval_span["span_id"]]
    assert score_span["scores"]["accuracy"] == pytest.approx(result.results[0].score / 10)
    assert score_span["output"]["reason"]

    # The agent under test runs inside the eval row, not as its own trace.
    agent_span = find_span_by_name(spans, "Math Agent.run")
    assert agent_span["span_parents"] == [eval_span["span_id"]]


@pytest.mark.vcr
def test_accuracy_eval_arun_logs_score(memory_logger):
    evaluation = accuracy_module.AccuracyEval(
        name="Async Multiplication Eval",
        model=agno_openai.OpenAIChat(id=MODEL),
        agent=_make_agent(),
        input="What is 10*5?",
        expected_output="50",
        num_iterations=1,
    )
    result = asyncio.run(evaluation.arun(print_summary=False, print_results=False))
    assert result is not None

    spans = memory_logger.pop()
    eval_span = find_span_by_name(spans, "Async Multiplication Eval")
    assert eval_span["span_attributes"]["type"] == "eval"
    assert eval_span["scores"]["accuracy"] == pytest.approx(result.avg_score / 10)

    score_span = find_span_by_name(spans, "accuracy")
    assert score_span["span_parents"] == [eval_span["span_id"]]


@pytest.mark.vcr
def test_accuracy_eval_run_with_output_skips_agent(memory_logger):
    evaluation = accuracy_module.AccuracyEval(
        name="Given Output Eval",
        model=agno_openai.OpenAIChat(id=MODEL),
        agent=_make_agent(),
        input="What is 10*5?",
        expected_output="50",
    )
    result = evaluation.run_with_output(output="50", print_summary=False, print_results=False)
    assert result is not None

    spans = memory_logger.pop()
    eval_span = find_span_by_name(spans, "Given Output Eval")
    assert eval_span["scores"]["accuracy"] == pytest.approx(result.avg_score / 10)
    assert eval_span["output"] == "50"

    # No agent ran: the judge is the only model traffic under the row.
    assert not _spans_named(spans, "Math Agent.")


# ---------------------------------------------------------------------------
# AgentAsJudgeEval (agno >= 2.4)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_agent_as_judge_eval_numeric_score(memory_logger):
    judge_module = pytest.importorskip("agno.eval.agent_as_judge")

    judge = judge_module.AgentAsJudgeEval(
        name="Tone Judge",
        model=agno_openai.OpenAIChat(id=MODEL),
        criteria="The response is polite and mentions renewable energy.",
        scoring_strategy="numeric",
        threshold=7,
    )
    result = judge.run(
        input="Tell me about renewable energy.",
        output="Certainly! Renewable energy comes from sources like wind and solar power.",
    )
    assert result is not None
    assert result.results

    spans = memory_logger.pop()
    eval_span = find_span_by_name(spans, "Tone Judge")
    assert eval_span["span_attributes"]["type"] == "eval"
    assert eval_span["input"] == "Tell me about renewable energy."
    assert eval_span["scores"]["judge"] == pytest.approx(result.results[0].score / 10)
    assert eval_span["output"]["passed"] == result.results[0].passed

    metadata = eval_span["metadata"]
    assert metadata["eval_type"] == "agent_as_judge"
    assert metadata["scoring_strategy"] == "numeric"
    assert metadata["threshold"] == 7
    assert metadata["criteria"] == "The response is polite and mentions renewable energy."

    # A single-pair judge run grades exactly what the row already describes, so it gets
    # no redundant scorer span -- only the judge's own model call nests under it.
    assert "judge" not in _names(spans)
    assert find_span_by_name(spans, "Agent.run")["span_parents"] == [eval_span["span_id"]]


@pytest.mark.vcr
def test_agent_as_judge_eval_batch_scores_each_case(memory_logger):
    judge_module = pytest.importorskip("agno.eval.agent_as_judge")

    judge = judge_module.AgentAsJudgeEval(
        name="Politeness Judge",
        model=agno_openai.OpenAIChat(id=MODEL),
        criteria="The response is polite.",
        scoring_strategy="binary",
    )
    result = judge.run(
        cases=[
            {"input": "Say hello politely.", "output": "Hello! How may I help you today?"},
            {"input": "Say hello politely.", "output": "what do you want"},
        ]
    )
    assert result is not None
    assert len(result.results) == 2

    spans = memory_logger.pop()
    eval_span = find_span_by_name(spans, "Politeness Judge")
    assert eval_span["metadata"]["num_cases"] == 2
    assert eval_span["metadata"]["pass_rate"] == result.pass_rate
    assert eval_span["scores"]["judge"] == pytest.approx(result.pass_rate / 100)

    # Batch mode keeps a scorer span per graded pair.
    pair_spans = _spans_named(spans, "judge")
    assert len(pair_spans) == 2
    for span, evaluation in zip(pair_spans, result.results):
        assert span["span_attributes"]["type"] == "score"
        assert span["span_parents"] == [eval_span["span_id"]]
        assert span["scores"]["judge"] == (1.0 if evaluation.passed else 0.0)
        assert span["output"]["reason"]


@pytest.mark.vcr
def test_agent_as_judge_post_hook_scores_the_agent_row(memory_logger):
    judge_module = pytest.importorskip("agno.eval.agent_as_judge")

    judge = judge_module.AgentAsJudgeEval(
        name="Quality Check",
        model=agno_openai.OpenAIChat(id=MODEL),
        criteria="The response answers the question directly.",
        scoring_strategy="binary",
    )
    agent = agno_agent.Agent(
        name="Post Hook Agent",
        model=agno_openai.OpenAIChat(id=MODEL),
        instructions="Answer in one short sentence.",
        post_hooks=[judge],
    )
    response = agent.run("What is the capital of France?")
    assert response.content

    spans = memory_logger.pop()
    agent_span = find_span_by_name(spans, "Post Hook Agent.run")
    judge_span = find_span_by_name(spans, "Quality Check")

    # The verdict is mirrored onto the agent's own row, so the trace is scored where a
    # reader (and an experiment summary) looks for it.
    assert agent_span["scores"]["judge"] in (0.0, 1.0)
    assert judge_span["scores"]["judge"] == agent_span["scores"]["judge"]
    assert judge_span["span_parents"] == [agent_span["span_id"]]


# ---------------------------------------------------------------------------
# ReliabilityEval
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_reliability_eval_scores_tool_calls(memory_logger):
    calculator = pytest.importorskip("agno.tools.calculator")

    agent = _make_agent(name="Calculator Agent", tools=[calculator.CalculatorTools()])
    response = agent.run("What is 10*5? Use your tools.")
    memory_logger.pop()  # the agent run is its own trace; assert on the eval below

    evaluation = reliability_module.ReliabilityEval(
        name="Calculator Reliability",
        agent_response=response,
        expected_tool_calls=["multiply"],
    )
    result = evaluation.run(print_results=False)
    assert result is not None

    spans = memory_logger.pop()
    eval_span = find_span_by_name(spans, "Calculator Reliability")
    assert eval_span["span_attributes"]["type"] == "eval"
    assert eval_span["input"]["expected_tool_calls"] == ["multiply"]
    assert eval_span["output"]["eval_status"] == result.eval_status
    assert eval_span["scores"]["reliability"] == (1.0 if result.eval_status == "PASSED" else 0.0)
    assert eval_span["metadata"]["eval_type"] == "reliability"


# ---------------------------------------------------------------------------
# PerformanceEval
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_performance_eval_logs_metrics_and_suppresses_child_spans(memory_logger):
    agent = _make_agent(name="Perf Agent")

    def run_agent():
        return agent.run("What is 2+2?")

    evaluation = performance_module.PerformanceEval(
        name="Agent Latency",
        func=run_agent,
        warmup_runs=0,
        num_iterations=2,
        measure_memory=False,
    )
    result = evaluation.run(print_summary=False, print_results=False)
    assert result is not None
    assert len(result.run_times) == 2

    spans = memory_logger.pop()
    # One row for the eval; the measured iterations do not each produce a trace.
    assert _names(spans) == ["Agent Latency"]

    eval_span = spans[0]
    assert eval_span["span_attributes"]["type"] == "eval"
    assert eval_span["metrics"]["avg_run_time"] == pytest.approx(result.avg_run_time)
    assert eval_span["metrics"]["p95_run_time"] == pytest.approx(result.p95_run_time)
    assert eval_span["output"]["median_run_time"] == pytest.approx(result.median_run_time)

    metadata = eval_span["metadata"]
    assert metadata["eval_type"] == "performance"
    assert metadata["func"] == "run_agent"
    assert metadata["num_iterations"] == 2
    assert metadata["measure_memory"] is False


# ---------------------------------------------------------------------------
# Eval suites (agno >= 2.9)
# ---------------------------------------------------------------------------


async def multiply(a: int, b: int) -> str:
    """Multiply two numbers.

    Args:
        a: the first number
        b: the second number
    """
    return str(a * b)


def _suite_cases(suite):
    # An async tool is awaited on the event loop. A sync tool would run on agno's worker
    # thread, where the memory background logger (a threading.local override) does not
    # apply, so its span would escape to the real API.
    agent = _make_agent(name="Suite Calculator Agent", tools=[multiply])
    return (
        suite.Case(
            name="multiplies_with_tool",
            agent=agent,
            input="What is 10*5? Use your tools.",
            tags=("smoke",),
            criteria="States that the answer is 50.",
            expected_tool_calls=("multiply",),
        ),
        suite.Case(
            name="answers_without_tool",
            agent=agent,
            input="Say the word 'hello' and nothing else.",
            criteria="The response is the word hello.",
        ),
    )


@pytest.mark.vcr
def test_eval_suite_logs_one_row_per_case(memory_logger):
    suite = pytest.importorskip("agno.eval.suite")

    cases = _suite_cases(suite)
    result = suite.run_cases(cases, judge_model=agno_openai.OpenAIChat(id=MODEL))
    assert result.total == 2

    spans = memory_logger.pop()
    rows = find_spans_by_type(spans, "eval")
    assert _names(rows) == ["multiplies_with_tool", "answers_without_tool"]

    tool_case, plain_case = rows
    assert tool_case["input"] == "What is 10*5? Use your tools."
    assert tool_case["tags"] == ["smoke"]
    assert set(tool_case["scores"]) == {"judge", "reliability"}
    assert tool_case["scores"]["judge"] in (0.0, 1.0)
    assert tool_case["metadata"]["eval_type"] == "suite_case"
    assert tool_case["metadata"]["case_name"] == "multiplies_with_tool"
    assert tool_case["metadata"]["expected_tool_calls"] == ["multiply"]
    assert "multiply" in tool_case["metadata"]["tools_called"]
    assert tool_case["metadata"]["judge_reason"]
    assert tool_case["output"]

    # A case without a reliability check is scored by the judge alone.
    assert set(plain_case["scores"]) == {"judge"}
    assert "tags" not in plain_case

    # The suite's own judge/reliability evals render as scorer spans inside the case.
    scorer_spans = find_spans_by_type(spans, "score")
    assert "judge" in _names(scorer_spans)
    assert "reliability" in _names(scorer_spans)
    for span in scorer_spans:
        assert span["span_attributes"]["purpose"] == "scorer"

    agent_spans = _spans_named(spans, "Suite Calculator Agent.")
    assert agent_spans, f"no agent span under the case rows. Available: {_names(spans)}"
    assert agent_spans[0]["span_parents"] == [tool_case["span_id"]]


@pytest.mark.vcr
def test_eval_suite_routes_rows_to_an_experiment(memory_logger, monkeypatch):
    suite = pytest.importorskip("agno.eval.suite")

    opened = {}

    def fake_init(**kwargs):
        opened.update(kwargs)
        return init_test_exp("agno-suite", project_name=PROJECT_NAME)

    monkeypatch.setattr(eval_experiments, "init", fake_init)
    eval_experiments.configure(True)

    cases = _suite_cases(suite)[:1]
    result = suite.run_cases(cases, tag="smoke", judge_model=agno_openai.OpenAIChat(id=MODEL))
    assert result.total == 1

    # The experiment inherits the project already in scope, and carries what ran.
    assert opened["project"] == PROJECT_NAME
    assert opened["set_current"] is False
    assert opened["metadata"] == {
        "source": "agno",
        "eval_type": "suite",
        "num_cases": 1,
        "tag": "smoke",
    }

    spans = memory_logger.pop()
    row = find_span_by_name(spans, "multiplies_with_tool")
    assert row["span_parents"] is None or row["span_parents"] == []
    assert row["scores"]["judge"] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Patcher wiring (no provider traffic)
# ---------------------------------------------------------------------------


def test_eval_patchers_are_registered_and_idempotent():
    from braintrust.integrations.agno.integration import AgnoIntegration

    available = AgnoIntegration.available_patchers()
    for expected in (
        "agno.eval.accuracy",
        "agno.eval.agent_as_judge",
        "agno.eval.reliability",
        "agno.eval.performance",
        "agno.eval.suite",
    ):
        assert expected in available

    # wrapt hands out a fresh BoundFunctionWrapper per attribute access, so identity is
    # not the idempotency signal -- a second layer of wrapping is.
    original = accuracy_module.AccuracyEval.run.__wrapped__
    assert AgnoIntegration.setup()
    assert accuracy_module.AccuracyEval.run.__wrapped__ is original
    assert not hasattr(original, "__wrapped__"), "setup() must not wrap an already patched target twice"

    # The manual wrap_*() helpers are no-ops once auto-instrumentation has run.
    assert wrap_reliability_eval(reliability_module.ReliabilityEval) is reliability_module.ReliabilityEval
    assert reliability_module.ReliabilityEval.run.__wrapped__ is not None
