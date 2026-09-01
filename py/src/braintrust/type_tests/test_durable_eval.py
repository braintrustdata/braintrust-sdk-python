"""Static and runtime type coverage for the experimental durable eval API."""

from typing import TypedDict

import pytest
from braintrust import (
    BatchCompletionPoll,
    BatchContext,
    BatchPollResult,
    BatchScorer,
    BatchScorerItem,
    BatchScorerResult,
    BatchTask,
    BatchTaskItem,
    BatchTaskResult,
    DurableEval,
    DurableEvalFailedResult,
    DurableEvalMemoryStore,
    EvalCase,
    define_durable_eval,
)


class Submission(TypedDict):
    id: str


async def submit_task(items: list[BatchTaskItem[str, str]], context: BatchContext) -> Submission:
    assert items
    return {"id": context.batch_id}


async def collect_task(submission: Submission, context: BatchContext) -> list[BatchTaskResult[int]]:
    return [BatchTaskResult(id="case:trial:0", output=len(submission["id"] + context.run_id))]


async def submit_score(items: list[BatchScorerItem[str, int, str]], context: BatchContext) -> Submission:
    assert items
    return {"id": context.batch_id}


async def collect_score(submission: Submission, context: BatchContext) -> list[BatchScorerResult]:
    assert submission["id"] == context.batch_id
    return [BatchScorerResult(id="case:trial:0", score=1)]


async def poll_batch(submission: Submission, context: BatchContext) -> BatchPollResult:
    assert submission["id"] == context.batch_id
    return BatchPollResult(status="pending")


task: BatchTask[str, int, str, Submission] = BatchTask(
    submit=submit_task,
    completion=BatchCompletionPoll(poll_batch),
    collect=collect_task,
)
score: BatchScorer[str, int, str, Submission] = BatchScorer(
    name="score",
    submit=submit_score,
    completion=BatchCompletionPoll(poll_batch),
    collect=collect_score,
)
durable_eval: DurableEval[str, int, str] = define_durable_eval(
    "project",
    store=DurableEvalMemoryStore(),
    data=[EvalCase(id="case", input="input", expected="expected")],
    task=task,
    scores=[score],
)


@pytest.mark.asyncio
async def test_durable_eval_types_at_runtime():
    result = await durable_eval.start(no_send_logs=True)
    assert result.status == "waiting"
    assert result.pending.poll == 1


def consume_failed_result(result: DurableEvalFailedResult) -> object:
    assert result.status == "failed"
    return result.error
