"""Type-check and runtime coverage for dictionary scorer returns."""

from collections.abc import Sequence

import pytest
from braintrust import NamedScoreDict, ScoreDict
from braintrust.framework import Eval, EvalAsync, EvalCase, EvalScorer, OneOrMoreScores
from braintrust.score import Score, ScoreLike


LegacyScores = float | int | bool | None | ScoreLike | Sequence[ScoreLike]


@pytest.mark.parametrize(
    "value,expected_scores",
    [
        (1.0, {"scorer": 1.0}),
        (0, {"scorer": 0}),
        (True, {"scorer": True}),
        (False, {"scorer": False}),
        (None, {"scorer": None}),
        (Score(name="named", score=0.5), {"named": 0.5}),
        ([Score(name="named", score=0.5)], {"named": 0.5}),
        ((Score(name="named", score=0.5),), {"named": 0.5}),
    ],
)
def test_eval_accepts_legacy_return_annotation(value: LegacyScores, expected_scores: dict[str, float | None]) -> None:
    def scorer(input: str, output: str, expected: str | None = None) -> LegacyScores:
        return value

    typed_scorer: EvalScorer[str, str, str] = scorer
    result = Eval(
        "test-legacy-scorer",
        data=[EvalCase(input="hello", expected="hello")],
        task=lambda input: input,
        scores=[typed_scorer],
        no_send_logs=True,
    )

    assert result.results[0].scores == expected_scores


def test_eval_accepts_single_dict_scorer() -> None:
    def scorer(input: str, output: str, expected: str | None = None) -> ScoreDict:
        return {"score": 1.0, "metadata": {"reason": "Matches"}}

    typed_scorer: EvalScorer[str, str, str] = scorer
    result = Eval(
        "test-dict-scorers",
        data=[EvalCase(input="hello", expected="hello")],
        task=lambda input: input,
        scores=[typed_scorer],
        no_send_logs=True,
    )

    assert result.results[0].scores == {"scorer": 1.0}


@pytest.mark.asyncio
async def test_eval_async_accepts_single_dict_scorer() -> None:
    async def scorer(input: str, output: str, expected: str | None = None) -> OneOrMoreScores:
        return {"score": None}

    typed_scorer: EvalScorer[str, str, str] = scorer
    result = await EvalAsync(
        "test-dict-scorers",
        data=[EvalCase(input="hello", expected="hello")],
        task=lambda input: input,
        scores=[typed_scorer],
        no_send_logs=True,
    )

    assert result.results[0].scores == {"scorer": None}


def test_eval_accepts_named_dict_scores() -> None:
    def scorer(input: str, output: str, expected: str | None = None) -> list[NamedScoreDict]:
        return [{"name": "match", "score": 1.0}, {"name": "quality", "score": 0.5}]

    result = Eval(
        "test-dict-scorers",
        data=[EvalCase(input="hello", expected="hello")],
        task=lambda input: input,
        scores=[scorer],
        no_send_logs=True,
    )

    assert result.results[0].scores == {"match": 1.0, "quality": 0.5}


def test_eval_accepts_inline_dict_scorer() -> None:
    result = Eval(
        "test-dict-scorers",
        data=[EvalCase(input="hello", expected="hello")],
        task=lambda input: input,
        scores=[lambda input, output, expected: ScoreDict(score=1.0)],
        no_send_logs=True,
    )

    assert result.results[0].scores == {"scorer_0": 1.0}
