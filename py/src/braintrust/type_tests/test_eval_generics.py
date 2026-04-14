"""Type-check tests for the Eval framework generic parameters.

These tests verify that pyright/mypy accept valid usage patterns
and that the runtime behavior is correct.

Run as type checks:
    nox -s test_types

Run as pytest:
    pytest src/braintrust/type_tests/test_eval_generics.py
"""

from typing import TypedDict

import pytest
from braintrust.framework import EvalAsync, EvalCase, EvalResultWithSummary
from braintrust.score import Score


# --- Domain types for testing ---
class ModelOutput(TypedDict):
    answer: str
    confidence: float


class AssertionSpec:
    """Assertion specification — not the same type as the model output."""

    def __init__(self, field: str, expected_value: str):
        self.field = field
        self.expected_value = expected_value


# ============================================================
# Case 1: Same-type Output and Expected (should always work)
# ============================================================


def same_type_data():
    return iter([EvalCase(input="query", expected="golden answer")])


async def same_type_task(input: str) -> str:
    return "model answer"


async def same_type_scorer(input: str, output: str, expected: str | None = None) -> Score:
    return Score(name="match", score=1.0 if output == expected else 0.0)


# ============================================================
# Case 2: Divergent Output and Expected (the bug from #240)
# ============================================================


def divergent_data():
    return iter(
        [
            EvalCase(
                input="What is 2+2?",
                expected=frozenset({AssertionSpec("answer", "4")}),
            ),
        ]
    )


async def divergent_task(input: str) -> ModelOutput:
    return ModelOutput(answer="4", confidence=0.99)


async def divergent_scorer(
    input: str,
    output: ModelOutput,
    expected: frozenset[AssertionSpec] | None = None,
) -> Score:
    if expected is None:
        return Score(name="match", score=0)
    for spec in expected:
        if output.get(spec.field) != spec.expected_value:
            return Score(name="match", score=0)
    return Score(name="match", score=1)


# ============================================================
# Runtime tests — confirm the eval framework works correctly
# with divergent types at runtime.
# ============================================================


@pytest.mark.asyncio
async def test_eval_same_type_output_and_expected():
    """Output and Expected are the same type — classic pattern."""
    result = await EvalAsync(
        "test-same-type",
        data=same_type_data,
        task=same_type_task,
        scores=[same_type_scorer],
        no_send_logs=True,
    )
    assert isinstance(result, EvalResultWithSummary)
    assert len(result.results) == 1
    assert result.results[0].output == "model answer"
    assert result.results[0].expected == "golden answer"


@pytest.mark.asyncio
async def test_eval_divergent_output_and_expected():
    """Output and Expected differ — the pattern reported in #240."""
    result = await EvalAsync(
        "test-divergent",
        data=divergent_data,
        task=divergent_task,
        scores=[divergent_scorer],
        no_send_logs=True,
    )
    assert isinstance(result, EvalResultWithSummary)
    assert len(result.results) == 1
    assert result.results[0].output == ModelOutput(answer="4", confidence=0.99)
    assert isinstance(result.results[0].expected, frozenset)
    assert result.results[0].scores.get("match") == 1.0
