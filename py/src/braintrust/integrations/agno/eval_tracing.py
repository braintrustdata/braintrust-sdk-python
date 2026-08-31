"""Braintrust tracing for Agno's eval framework (``agno.eval``).

Each eval run becomes a span shaped like a Braintrust eval row — ``input``,
``expected``, ``output``, ``scores`` — so an agno eval reads the same way whether it
lands in logs or, when the caller (or :mod:`.eval_experiments`) has an experiment in
scope, as an experiment row.

Agno grades on a 1-10 scale (accuracy, numeric judge) or a pass/fail verdict
(binary judge, reliability); Braintrust scores are 0-1, so numeric scores are divided
by their scale and verdicts become 1.0/0.0. The raw agno values stay in ``metadata``.

Sub-evals that the suite runner creates per case (``AgentAsJudgeEval``,
``ReliabilityEval``) render as scorer spans named for the check they perform, so a
case's row shows *what* graded it rather than repeating the case name three times.

The eval span carries the common metadata (which eval, which agent, which model) from
the moment it starts; the payload builders below add only what the result reveals.
"""

import contextvars
from contextlib import contextmanager
from typing import Any

from braintrust.logger import NOOP_SPAN, current_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import clean_nones, is_numeric

from .eval_experiments import active_experiment, suite_experiment
from .tracing import bound_args, extract_metadata, omit, start_span, suppress_spans


# Agno's numeric scales: 1-10 for accuracy and a numeric judge, 0-100 for a pass rate.
_AGNO_SCORE_SCALE = 10.0
_PASS_RATE_SCALE = 100.0

_EVAL_SPAN_ATTRIBUTES = {"type": SpanTypeAttribute.EVAL}
_SCORER_SPAN_ATTRIBUTES = {"type": SpanTypeAttribute.SCORE, "purpose": "scorer"}

_EVAL_SPAN_NAMES = {
    "accuracy": "AccuracyEval",
    "agent_as_judge": "AgentAsJudgeEval",
    "reliability": "ReliabilityEval",
    "performance": "PerformanceEval",
}

# Inside a suite case the eval's own name is the case name (the suite passes it
# through), which would repeat the row name on every child. Name them for the check
# they perform instead — only the judge's eval type needs spelling differently.
_IN_CASE_SPAN_NAMES = {"agent_as_judge": "judge"}

# Set while a suite case is running, so the judge/reliability evals the suite runs for
# that case render as scorer spans instead of top-level eval rows.
_IN_CASE: contextvars.ContextVar[bool] = contextvars.ContextVar("braintrust_agno_in_eval_case", default=False)

# Set while an AgentAsJudgeEval grades a single pair, so the ``_evaluate`` span — which
# would carry exactly the same input, output and score as the eval row wrapping it — is
# skipped. Batch runs keep their per-case spans.
_JUDGE_SINGLE: contextvars.ContextVar[bool] = contextvars.ContextVar("braintrust_agno_judge_single", default=False)

# The span a judge verdict should also be logged onto: set while ``post_check`` runs an
# AgentAsJudgeEval as an agent post-hook, so the verdict lands on the agent's own row
# and not only on the nested scorer span.
_SCORE_TARGET: contextvars.ContextVar[Any] = contextvars.ContextVar("braintrust_agno_score_target", default=NOOP_SPAN)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_from_scale(value: Any, scale: float = _AGNO_SCORE_SCALE) -> float | None:
    """Convert an agno score to a Braintrust 0-1 score."""
    if not is_numeric(value):
        return None
    return _clamp01(float(value) / scale)


def _score_from_flag(value: Any) -> float | None:
    """Convert a pass/fail verdict to a Braintrust 0-1 score."""
    if value is None:
        return None
    return 1.0 if value else 0.0


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def _eval_metadata(instance: Any, eval_type: str) -> dict[str, Any]:
    """Metadata common to every eval span: what ran, and against which component."""
    metadata: dict[str, Any] = {
        "component": "eval",
        "eval_type": eval_type,
        "eval_name": getattr(instance, "name", None),
    }

    agent = getattr(instance, "agent", None)
    team = getattr(instance, "team", None)
    if agent is not None:
        metadata.update(omit(extract_metadata(agent, "agent"), ["component"]))
        metadata["agent_id"] = getattr(agent, "id", None)
    elif team is not None:
        metadata.update(omit(extract_metadata(team, "team"), ["component"]))
        metadata["team_id"] = getattr(team, "id", None)

    evaluator_model = getattr(instance, "model", None)
    if evaluator_model is not None:
        evaluator = extract_metadata(evaluator_model, "model")
        metadata["evaluator_model"] = evaluator.get("model") or evaluator.get("model_class")

    return clean_nones(metadata)


def _start_scorer_span(name: str, **event: Any):
    return start_span(name=name, span_attributes=dict(_SCORER_SPAN_ATTRIBUTES), **clean_nones(event))


def _start_eval_span(instance: Any, eval_type: str, **event: Any):
    """Start the span for one eval run, as a row or as a scorer span inside a case."""
    metadata = _eval_metadata(instance, eval_type)
    if _IN_CASE.get():
        return _start_scorer_span(_IN_CASE_SPAN_NAMES.get(eval_type, eval_type), metadata=metadata, **event)
    return start_span(
        name=getattr(instance, "name", None) or _EVAL_SPAN_NAMES[eval_type],
        span_attributes=dict(_EVAL_SPAN_ATTRIBUTES),
        metadata=metadata,
        **clean_nones(event),
    )


def _log_to_score_target(scores: dict[str, float] | None) -> None:
    """Mirror a post-hook judge's verdict onto the span of the run being judged."""
    target = _SCORE_TARGET.get()
    if scores and target is not NOOP_SPAN:
        target.log(scores=scores)


@contextmanager
def _score_the_enclosing_span():
    """Aim the next judge verdict at the span we are running inside."""
    token = _SCORE_TARGET.set(current_span())
    try:
        yield
    finally:
        _SCORE_TARGET.reset(token)


# ---------------------------------------------------------------------------
# AccuracyEval
# ---------------------------------------------------------------------------


def _accuracy_payload(result: Any) -> dict[str, Any]:
    """Row fields for an ``AccuracyResult``.

    ``input``/``expected_output`` may be callables that agno invokes once per run, so
    they are read back off the evaluations rather than called a second time here.
    """
    evaluations = list(getattr(result, "results", None) or [])
    payload: dict[str, Any] = {}
    if evaluations:
        last = evaluations[-1]
        payload["input"] = getattr(last, "input", None)
        payload["expected"] = getattr(last, "expected_output", None)
        payload["output"] = getattr(last, "output", None)

    avg_score = getattr(result, "avg_score", None)
    score = _score_from_scale(avg_score)
    if score is not None:
        payload["scores"] = {"accuracy": score}

    payload["metadata"] = clean_nones(
        {
            "eval_run_id": getattr(result, "run_id", None),
            "num_iterations": len(evaluations) or None,
            "avg_score": avg_score,
            "min_score": getattr(result, "min_score", None),
            "max_score": getattr(result, "max_score", None),
            "std_dev_score": getattr(result, "std_dev_score", None),
            "iteration_scores": [getattr(item, "score", None) for item in evaluations] or None,
        }
    )
    return clean_nones(payload)


def _accuracy_run_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AccuracyEval.run`` and ``AccuracyEval.run_with_output``."""
    with _start_eval_span(instance, "accuracy") as span:
        result = wrapped(*args, **kwargs)
        span.log(**_accuracy_payload(result))
        return result


async def _accuracy_arun_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AccuracyEval.arun`` and ``AccuracyEval.arun_with_output``."""
    with _start_eval_span(instance, "accuracy") as span:
        result = await wrapped(*args, **kwargs)
        span.log(**_accuracy_payload(result))
        return result


_ACCURACY_EVALUATE_ARGS = (
    "input",
    "evaluator_agent",
    "evaluation_input",
    "evaluator_expected_output",
    "agent_output",
    "run_metrics",
)


@contextmanager
def _accuracy_iteration_span(args: Any, kwargs: Any):
    """Scorer span for one iteration's judge call, with that iteration's fields."""
    bound = bound_args(args, kwargs, _ACCURACY_EVALUATE_ARGS)
    with _start_scorer_span(
        "accuracy",
        input=clean_nones({"input": bound.get("input"), "expected": bound.get("evaluator_expected_output")}),
    ) as span:
        yield span, bound


def _accuracy_evaluation_payload(bound: dict[str, Any], evaluation: Any) -> dict[str, Any]:
    """Row fields for one iteration's ``AccuracyEvaluation``."""
    if evaluation is None:
        return clean_nones({"output": bound.get("agent_output")})

    payload: dict[str, Any] = {
        "expected": bound.get("evaluator_expected_output"),
        "output": {
            "score": getattr(evaluation, "score", None),
            "reason": getattr(evaluation, "reason", None),
            "output": getattr(evaluation, "output", None),
        },
    }
    score = _score_from_scale(getattr(evaluation, "score", None))
    if score is not None:
        payload["scores"] = {"accuracy": score}
    return clean_nones(payload)


def _accuracy_evaluate_answer_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AccuracyEval.evaluate_answer`` — one iteration's judge call."""
    with _accuracy_iteration_span(args, kwargs) as (span, bound):
        evaluation = wrapped(*args, **kwargs)
        span.log(**_accuracy_evaluation_payload(bound, evaluation))
        return evaluation


async def _accuracy_aevaluate_answer_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AccuracyEval.aevaluate_answer``."""
    with _accuracy_iteration_span(args, kwargs) as (span, bound):
        evaluation = await wrapped(*args, **kwargs)
        span.log(**_accuracy_evaluation_payload(bound, evaluation))
        return evaluation


# ---------------------------------------------------------------------------
# AgentAsJudgeEval
# ---------------------------------------------------------------------------


def _judge_evaluation_score(evaluation: Any) -> float | None:
    """Numeric score if the judge graded one, else its pass/fail verdict."""
    score = _score_from_scale(getattr(evaluation, "score", None))
    if score is not None:
        return score
    return _score_from_flag(getattr(evaluation, "passed", None))


def _judge_output(evaluation: Any) -> dict[str, Any]:
    return {
        "score": getattr(evaluation, "score", None),
        "passed": getattr(evaluation, "passed", None),
        "reason": getattr(evaluation, "reason", None),
    }


def _judge_scores(result: Any, evaluations: list) -> dict[str, float] | None:
    if not evaluations:
        return None

    if len(evaluations) > 1:
        # Batch mode: the per-case verdicts are scored on their own spans, so the batch
        # row carries the pass rate.
        pass_rate = _score_from_scale(getattr(result, "pass_rate", None), scale=_PASS_RATE_SCALE)
        return {"judge": pass_rate} if pass_rate is not None else None

    score = _judge_evaluation_score(evaluations[0])
    return {"judge": score} if score is not None else None


_JUDGE_RUN_ARGS = ("input", "output", "cases")


def _judge_payload(instance: Any, bound: dict[str, Any], result: Any) -> dict[str, Any]:
    evaluations = list(getattr(result, "results", None) or [])
    scoring_strategy = getattr(instance, "scoring_strategy", None)

    payload: dict[str, Any] = {
        "input": bound.get("input"),
        "output": _judge_output(evaluations[0]) if len(evaluations) == 1 else bound.get("output"),
        "metadata": clean_nones(
            {
                "eval_run_id": getattr(result, "run_id", None),
                "criteria": getattr(instance, "criteria", None),
                "scoring_strategy": scoring_strategy,
                "threshold": getattr(instance, "threshold", None) if scoring_strategy == "numeric" else None,
                "pass_rate": getattr(result, "pass_rate", None),
                "num_cases": len(evaluations) or None,
            }
        ),
    }
    scores = _judge_scores(result, evaluations)
    if scores:
        payload["scores"] = scores
    return clean_nones(payload)


@contextmanager
def _judge_run_span(instance: Any, args: Any, kwargs: Any):
    """Eval span for one ``AgentAsJudgeEval`` run, flagging single- vs batch-mode."""
    bound = bound_args(args, kwargs, _JUDGE_RUN_ARGS)
    token = _JUDGE_SINGLE.set(bound.get("cases") is None)
    try:
        with _start_eval_span(instance, "agent_as_judge", input=bound.get("input")) as span:
            yield span, bound
    finally:
        _JUDGE_SINGLE.reset(token)


def _finish_judge_run(span: Any, instance: Any, bound: dict[str, Any], result: Any) -> None:
    payload = _judge_payload(instance, bound, result)
    span.log(**payload)
    _log_to_score_target(payload.get("scores"))


def _judge_run_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AgentAsJudgeEval.run`` (single and batch modes)."""
    with _judge_run_span(instance, args, kwargs) as (span, bound):
        result = wrapped(*args, **kwargs)
        _finish_judge_run(span, instance, bound, result)
        return result


async def _judge_arun_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AgentAsJudgeEval.arun``."""
    with _judge_run_span(instance, args, kwargs) as (span, bound):
        result = await wrapped(*args, **kwargs)
        _finish_judge_run(span, instance, bound, result)
        return result


_JUDGE_EVALUATE_ARGS = ("input", "output", "evaluator_agent", "run_metrics")


@contextmanager
def _judge_pair_span(args: Any, kwargs: Any):
    bound = bound_args(args, kwargs, _JUDGE_EVALUATE_ARGS)
    with _start_scorer_span(
        "judge",
        input=clean_nones({"input": bound.get("input"), "output": bound.get("output")}),
    ) as span:
        yield span, bound


def _judge_evaluation_payload(evaluation: Any) -> dict[str, Any]:
    if evaluation is None:
        return {}
    payload: dict[str, Any] = {
        "output": _judge_output(evaluation),
        "metadata": clean_nones({"criteria": getattr(evaluation, "criteria", None)}),
    }
    score = _judge_evaluation_score(evaluation)
    if score is not None:
        payload["scores"] = {"judge": score}
    return payload


def _judge_evaluate_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AgentAsJudgeEval._evaluate`` — one graded input/output pair."""
    if _JUDGE_SINGLE.get():
        return wrapped(*args, **kwargs)
    with _judge_pair_span(args, kwargs) as (span, _bound):
        evaluation = wrapped(*args, **kwargs)
        span.log(**_judge_evaluation_payload(evaluation))
        return evaluation


async def _judge_aevaluate_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AgentAsJudgeEval._aevaluate``."""
    if _JUDGE_SINGLE.get():
        return await wrapped(*args, **kwargs)
    with _judge_pair_span(args, kwargs) as (span, _bound):
        evaluation = await wrapped(*args, **kwargs)
        span.log(**_judge_evaluation_payload(evaluation))
        return evaluation


def _judge_post_check_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AgentAsJudgeEval.post_check`` (agent ``post_hooks`` usage).

    ``post_check`` runs inside the agent's own span and discards the judge result, so
    rather than starting a span here, point the nested ``run()`` at the enclosing span
    and let it mirror the verdict onto the row being judged.
    """
    with _score_the_enclosing_span():
        return wrapped(*args, **kwargs)


async def _judge_async_post_check_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``AgentAsJudgeEval.async_post_check``."""
    with _score_the_enclosing_span():
        return await wrapped(*args, **kwargs)


# ---------------------------------------------------------------------------
# ReliabilityEval
# ---------------------------------------------------------------------------


def _reliability_input(instance: Any) -> dict[str, Any]:
    return clean_nones(
        {
            "expected_tool_calls": getattr(instance, "expected_tool_calls", None),
            "expected_tool_call_arguments": getattr(instance, "expected_tool_call_arguments", None),
            "allow_additional_tool_calls": getattr(instance, "allow_additional_tool_calls", None),
        }
    )


def _reliability_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"metadata": clean_nones({"eval_run_id": getattr(result, "run_id", None)})}

    eval_status = getattr(result, "eval_status", None)
    if eval_status is not None:
        payload["output"] = clean_nones(
            {
                "eval_status": eval_status,
                "passed_tool_calls": getattr(result, "passed_tool_calls", None),
                "failed_tool_calls": getattr(result, "failed_tool_calls", None),
                "missing_tool_calls": getattr(result, "missing_tool_calls", None),
                "additional_tool_calls": getattr(result, "additional_tool_calls", None),
                "passed_argument_checks": getattr(result, "passed_argument_checks", None),
                "failed_argument_checks": getattr(result, "failed_argument_checks", None),
            }
        )
        payload["scores"] = {"reliability": 1.0 if eval_status == "PASSED" else 0.0}

    return payload


def _reliability_run_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``ReliabilityEval.run``."""
    with _start_eval_span(instance, "reliability", input=_reliability_input(instance)) as span:
        result = wrapped(*args, **kwargs)
        span.log(**_reliability_payload(result))
        return result


async def _reliability_arun_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``ReliabilityEval.arun``."""
    with _start_eval_span(instance, "reliability", input=_reliability_input(instance)) as span:
        result = await wrapped(*args, **kwargs)
        span.log(**_reliability_payload(result))
        return result


# ---------------------------------------------------------------------------
# PerformanceEval
# ---------------------------------------------------------------------------

_PERFORMANCE_METRICS = (
    "avg_run_time",
    "min_run_time",
    "max_run_time",
    "median_run_time",
    "p95_run_time",
    "avg_memory_usage",
    "min_memory_usage",
    "max_memory_usage",
    "median_memory_usage",
    "p95_memory_usage",
)


def _performance_payload(instance: Any, result: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    for key in _PERFORMANCE_METRICS:
        value = getattr(result, key, None)
        if value is None:
            continue
        summary[key] = value
        if is_numeric(value):
            metrics[key] = float(value)

    return clean_nones(
        {
            "output": summary or None,
            "metrics": metrics or None,
            "metadata": clean_nones(
                {
                    "eval_run_id": getattr(result, "run_id", None),
                    "func": getattr(getattr(instance, "func", None), "__name__", None),
                    "warmup_runs": getattr(instance, "warmup_runs", None),
                    "num_iterations": getattr(instance, "num_iterations", None),
                    "measure_runtime": getattr(instance, "measure_runtime", None),
                    "measure_memory": getattr(instance, "measure_memory", None),
                }
            ),
        }
    )


def _performance_run_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``PerformanceEval.run``.

    The measured function is called ``warmup_runs + num_iterations`` times, so agno
    instrumentation stands down for the duration (see ``suppress_spans``) and the eval
    keeps one row.
    """
    with _start_eval_span(instance, "performance") as span, suppress_spans():
        result = wrapped(*args, **kwargs)
        span.log(**_performance_payload(instance, result))
        return result


async def _performance_arun_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``PerformanceEval.arun``."""
    with _start_eval_span(instance, "performance") as span, suppress_spans():
        result = await wrapped(*args, **kwargs)
        span.log(**_performance_payload(instance, result))
        return result


# ---------------------------------------------------------------------------
# Eval suites (``agno.eval.suite``)
# ---------------------------------------------------------------------------

# ``agno.eval.suite`` only exists from agno 2.9, where ``Case`` and ``CaseResult`` are
# dataclasses declaring every field read below — so these read attributes directly. A
# rename upstream should fail a test, not silently drop a field.


def _case_metadata(case: Any, result: Any) -> dict[str, Any]:
    judge_mode = case.judge_mode
    return clean_nones(
        {
            "component": "eval",
            "eval_type": "suite_case",
            "case_name": case.name,
            "criteria": case.criteria,
            "expected_tool_calls": list(case.expected_tool_calls) if case.expected_tool_calls else None,
            "judge_mode": getattr(judge_mode, "value", judge_mode),
            "agent_id": result.agent_id,
            "team_id": result.team_id,
            "session_id": result.session_id or None,
            "passed": result.passed,
            "judge_passed": result.judge_passed,
            "judge_score": result.judge_score,
            "judge_reason": result.judge_reason,
            "reliability_passed": result.reliability_passed,
            "tools_called": list(result.tools_called) if result.tools_called else None,
            "timed_out": result.timed_out or None,
            "skipped": result.skipped or None,
            "error": result.error,
            "duration_seconds": result.duration_seconds,
        }
    )


def _case_scores(result: Any) -> dict[str, float]:
    """Score a ``CaseResult`` from whichever of its checks were configured."""
    scores: dict[str, float] = {}

    if result.judge_passed is not None:
        judge_score = _score_from_scale(result.judge_score)
        scores["judge"] = judge_score if judge_score is not None else float(bool(result.judge_passed))

    if result.reliability_passed is not None:
        scores["reliability"] = float(bool(result.reliability_passed))

    # ``Case.scorer`` returns an ``agno.scorer.Score`` already in 0-1.
    value = getattr(result.score, "value", None)
    if is_numeric(value):
        scores["scorer"] = _clamp01(float(value))

    return scores


def _case_payload(case: Any, result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"output": result.output, "metadata": _case_metadata(case, result)}
    scores = _case_scores(result)
    if scores:
        payload["scores"] = scores
    return clean_nones(payload)


def _start_case_span(case: Any):
    """Start the row span for one suite case, on the suite's experiment if there is one."""
    return start_span(
        name=case.name or "case",
        span_attributes=dict(_EVAL_SPAN_ATTRIBUTES),
        parent_object=active_experiment(),
        **clean_nones(
            {
                "input": case.input,
                "expected": case.expected,
                "tags": list(case.tags) if case.tags else None,
            }
        ),
    )


async def _arun_cases_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``agno.eval.suite.arun_cases`` — the whole suite run.

    Reached from every entry point (``cli``, ``acli``, ``run_cases``, ``arun_cases``),
    because each resolves the next call through the module globals at call time.
    """
    cases = args[0] if args else kwargs.get("cases", ())
    metadata = clean_nones(
        {
            "source": "agno",
            "eval_type": "suite",
            "num_cases": len(cases),
            "tag": kwargs.get("tag"),
            "case_name_filter": kwargs.get("name"),
        }
    )
    with suite_experiment(metadata):
        return await wrapped(*args, **kwargs)


async def _arun_case_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """Wrapper for ``agno.eval.suite._arun_case`` — one case, one row."""
    case = args[0] if args else kwargs["case"]
    token = _IN_CASE.set(True)
    try:
        with _start_case_span(case) as span:
            result = await wrapped(*args, **kwargs)
            span.log(**_case_payload(case, result))
            return result
    finally:
        _IN_CASE.reset(token)
