"""Static and runtime type coverage for the experimental workflow eval API."""

from typing import TypedDict

import pytest
from braintrust import EvalCase
from braintrust.workflow_eval import (
    WorkflowEval,
    WorkflowEvalMemoryStore,
    WorkflowScorer,
    WorkflowScorerItem,
    WorkflowScorerResult,
    WorkflowSubmissionCompletionPoll,
    WorkflowSubmissionContext,
    WorkflowSubmissionPoll,
    WorkflowTask,
    WorkflowTaskItem,
    WorkflowTaskResult,
    define_workflow_eval,
)


class Submission(TypedDict):
    id: str


async def submit_task(item: WorkflowTaskItem[str, str], context: WorkflowSubmissionContext) -> Submission:
    assert item.id
    return {"id": context.submission_id}


async def collect_task(submission: Submission, context: WorkflowSubmissionContext) -> WorkflowTaskResult[int]:
    return WorkflowTaskResult(output=len(submission["id"] + context.run_id))


async def submit_score(item: WorkflowScorerItem[str, int, str], context: WorkflowSubmissionContext) -> Submission:
    assert item.id
    return {"id": context.submission_id}


async def collect_score(submission: Submission, context: WorkflowSubmissionContext) -> WorkflowScorerResult:
    assert submission["id"] == context.submission_id
    return WorkflowScorerResult(score=1)


async def poll_submission(submission: Submission, context: WorkflowSubmissionContext) -> WorkflowSubmissionPoll:
    assert submission["id"] == context.submission_id
    return WorkflowSubmissionPoll(status="pending")


task: WorkflowTask[str, int, str, Submission] = WorkflowTask(
    submit=submit_task,
    completion=WorkflowSubmissionCompletionPoll(poll_submission),
    collect=collect_task,
)
score: WorkflowScorer[str, int, str, Submission] = WorkflowScorer(
    name="score",
    submit=submit_score,
    completion=WorkflowSubmissionCompletionPoll(poll_submission),
    collect=collect_score,
)
workflow_eval: WorkflowEval[str, int, str] = define_workflow_eval(
    "project",
    store=WorkflowEvalMemoryStore(),
    data=[EvalCase(id="case", input="input", expected="expected")],
    task=task,
    scores=[score],
)


@pytest.mark.asyncio
async def test_workflow_eval_types_at_runtime():
    result = await workflow_eval.start(no_send_logs=True)
    assert result.status == "waiting"
    assert result.pending.poll == 1
