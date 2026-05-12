"""Type-check and runtime tests for autoevals scorers in Eval."""

import pytest
from autoevals import Levenshtein  # type: ignore[import-untyped]
from braintrust.framework import Eval, EvalAsync, EvalCase, EvalScorer


def accepts_autoevals_scorer(
    scorer: EvalScorer[str, str, str],
) -> EvalScorer[str, str, str]:
    return scorer


def autoevals_data():
    return iter([EvalCase(input="query", expected="hello world")])


def autoevals_task(input: str) -> str:
    return "hello world"


async def autoevals_task_async(input: str) -> str:
    return "hello world"


autoevals_scores: list[EvalScorer[str, str, str]] = [
    accepts_autoevals_scorer(Levenshtein()),
    accepts_autoevals_scorer(Levenshtein),
    accepts_autoevals_scorer(Levenshtein.partial(foo="bar")),
]

autoevals_scores_untyped = [
    Levenshtein(),
    Levenshtein,
    Levenshtein.partial(foo="bar"),
]


def test_eval_accepts_autoevals_scorers_typed():
    result = Eval(
        "test-autoevals-scorers",
        data=autoevals_data,
        task=autoevals_task,
        scores=autoevals_scores,
        no_send_logs=True,
    )

    score = result.results[0].scores["Levenshtein"]
    assert score is not None
    assert score > 0


def test_eval_accepts_autoevals_scorers_untyped():
    result = Eval(
        "test-autoevals-scorers",
        data=autoevals_data,
        task=autoevals_task,
        scores=autoevals_scores,
        no_send_logs=True,
    )

    score = result.results[0].scores["Levenshtein"]
    assert score is not None
    assert score > 0


def test_eval_accepts_autoevals_scorers_inline():
    result = Eval(
        "test-autoevals-scorers",
        data=autoevals_data,
        task=autoevals_task,
        scores=[
            Levenshtein(),
            Levenshtein,
            Levenshtein.partial(foo="bar"),
        ],
        no_send_logs=True,
    )

    score = result.results[0].scores["Levenshtein"]
    assert score is not None
    assert score > 0


@pytest.mark.asyncio
async def test_eval_async_accepts_autoevals_scorers_typed():
    result = await EvalAsync(
        "test-autoevals-scorers",
        data=autoevals_data,
        task=autoevals_task_async,
        scores=autoevals_scores,
        no_send_logs=True,
    )

    score = result.results[0].scores["Levenshtein"]
    assert score is not None
    assert score > 0


@pytest.mark.asyncio
async def test_eval_async_accepts_autoevals_scorers_untyped():
    result = await EvalAsync(
        "test-autoevals-scorers",
        data=autoevals_data,
        task=autoevals_task_async,
        scores=autoevals_scores,
        no_send_logs=True,
    )

    score = result.results[0].scores["Levenshtein"]
    assert score is not None
    assert score > 0


@pytest.mark.asyncio
async def test_eval_async_accepts_autoevals_scorers_inline():
    result = await EvalAsync(
        "test-autoevals-scorers",
        data=autoevals_data,
        task=autoevals_task_async,
        scores=[
            Levenshtein(),
            Levenshtein,
            Levenshtein.partial(foo="bar"),
        ],
        no_send_logs=True,
    )

    score = result.results[0].scores["Levenshtein"]
    assert score is not None
    assert score > 0
