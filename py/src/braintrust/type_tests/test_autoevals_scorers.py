"""Type-check and runtime tests for autoevals scorers in Eval."""

import pytest
from autoevals import Levenshtein  # type: ignore[import-untyped]
from braintrust.framework import EvalAsync, EvalCase, EvalScorer


def accepts_autoevals_scorer(
    scorer: EvalScorer[str, str, str],
) -> EvalScorer[str, str, str]:
    return scorer


def autoevals_data():
    return iter([EvalCase(input="query", expected="hello world")])


async def autoevals_task(input: str) -> str:
    return "hello world"


autoevals_scores: list[EvalScorer[str, str, str]] = [
    accepts_autoevals_scorer(Levenshtein()),
    accepts_autoevals_scorer(Levenshtein),
    accepts_autoevals_scorer(Levenshtein.partial(hehe="hoho")),
]


@pytest.mark.asyncio
async def test_eval_accepts_autoevals_scorers():
    result = await EvalAsync(
        "test-autoevals-scorers",
        data=autoevals_data,
        task=autoevals_task,
        scores=autoevals_scores,
        no_send_logs=True,
    )

    score = result.results[0].scores["Levenshtein"]
    assert score is not None
    assert score > 0
