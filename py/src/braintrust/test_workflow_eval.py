"""Tests for the experimental workflow eval API."""

import asyncio
from unittest.mock import patch

import pytest

from . import workflow_eval as workflow_eval_module
from .logger import BraintrustState, Dataset, ObjectMetadata, ProjectDatasetMetadata
from .test_helpers import init_test_exp, with_memory_logger, with_simulate_login  # noqa: F401
from .util import LazyValue
from .workflow_eval import (
    WorkflowEvalCompletedResult,
    WorkflowEvalMemoryStore,
    WorkflowEvalRedisStore,
    WorkflowEvalWaitingResult,
    WorkflowScorer,
    WorkflowScorerResult,
    WorkflowSubmissionCompletionPoll,
    WorkflowSubmissionCompletionWebhook,
    WorkflowSubmissionPoll,
    WorkflowTask,
    WorkflowTaskResult,
    define_workflow_eval,
)


@pytest.mark.asyncio
async def test_memory_store_is_atomic_and_copies_values():
    store = WorkflowEvalMemoryStore()
    value = bytearray(b"first")

    first = await store.get_or_set("key", value)
    value[:] = b"other"
    second = await store.get_or_set("key", b"second")

    assert first.created is True
    assert first.value == b"first"
    assert second.created is False
    assert second.value == b"first"
    assert await store.read("key") == b"first"
    assert await store.get_set_size("key") == 0
    await asyncio.gather(*(store.add_to_set("key", str(i % 3)) for i in range(30)))
    assert await store.get_set_size("key") == 3
    assert await store.read("key") == b"first"


class _SyncRedis:
    def __init__(self):
        self.values = {}
        self.calls = []
        self.sets = {}
        self.expirations = {}

    def eval(self, script, numkeys, key, member, ttl_ms):
        assert "SADD" in script and "PEXPIRE" in script
        assert numkeys == 1
        self.sets.setdefault(key, set()).add(member)
        self.expirations[key] = ttl_ms
        return 1

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, **kwargs):
        self.calls.append((key, kwargs))
        old = self.values.get(key)
        if kwargs.get("nx") and old is not None:
            return old if kwargs.get("get") else None
        self.values[key] = value
        return old if kwargs.get("get") else True


class _AsyncRedis(_SyncRedis):
    async def eval(self, *args):
        return super().eval(*args)

    async def scard(self, key):
        return super().scard(key)

    async def get(self, key):
        return super().get(key)

    async def set(self, key, value, **kwargs):
        return super().set(key, value, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [_SyncRedis, _AsyncRedis])
async def test_redis_store_supports_sync_and_async_redis_py(client_type):
    client = client_type()
    store = WorkflowEvalRedisStore(client, key_prefix="test:", ttl_ms=1234)

    await store.write("one", b"value")
    assert await store.read("one") == b"value"
    assert (await store.get_or_set("two", b"first")).created is True
    existing = await store.get_or_set("two", b"second")

    assert existing.created is False
    assert existing.value == b"first"
    assert all(key.startswith("test:") for key, _ in client.calls)
    assert all(options["px"] == 1234 for _, options in client.calls)
    assert await store.get_set_size("progress") == 0
    await store.add_to_set("progress", "a")
    await store.add_to_set("progress", "a")
    await store.add_to_set("progress", "b")
    assert await store.get_set_size("progress") == 2
    assert client.sets == {"test:progress": {"a", "b"}}
    assert client.expirations == {"test:progress": 1234}


@pytest.mark.asyncio
async def test_ordinary_workflow_eval_completes_locally_and_is_idempotent():
    task_calls = []
    score_calls = []

    def task(input, hooks):
        task_calls.append((input, hooks.trial_index))
        hooks.metadata["task"] = True
        hooks.tags = ["updated"]
        return input * 2

    def scorer(output, expected, metadata, tags):
        score_calls.append((output, metadata, tags))
        return output == expected

    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": 2, "expected": 4, "metadata": {"source": "test"}}],
        task=task,
        scores=[scorer],
        trial_count=2,
    )

    result = await workflow_eval.start(no_send_logs=True)
    repeated = await workflow_eval.status(result.run_id)

    assert isinstance(result, WorkflowEvalCompletedResult)
    assert isinstance(repeated, WorkflowEvalCompletedResult)
    assert result.summary.scores["scorer"].score == 1
    assert sorted(task_calls) == [(2, 0), (2, 1)]
    assert score_calls == [
        (4, {"source": "test", "task": True}, ["updated"]),
        (4, {"source": "test", "task": True}, ["updated"]),
    ]


@pytest.mark.asyncio
async def test_ordinary_only_runs_do_not_require_case_ids_and_get_new_run_ids():
    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"input": 1}],
        task=lambda value: value,
        scores=[lambda: 1],
    )

    first = await workflow_eval.start(no_send_logs=True)
    second = await workflow_eval.start(no_send_logs=True)

    assert first.status == "completed"
    assert second.status == "completed"
    assert first.run_id != second.run_id


@pytest.mark.asyncio
async def test_poll_advances_task_then_mixed_scorers_without_polling_new_submissions():
    submitted = {}
    ready = set()
    calls = []

    async def task_submit(item, context):
        calls.append(("submit-task", context.submission_id))
        submitted[context.submission_id] = item
        return {"provider_id": context.submission_id}

    async def task_collect(_submission, context):
        calls.append(("collect-task", context.submission_id))
        return WorkflowTaskResult(output=submitted[context.submission_id].input * 2)

    async def score_submit(item, context):
        calls.append(("submit-score", context.submission_id))
        submitted[context.submission_id] = item
        return {"provider_id": context.submission_id}

    async def score_collect(_submission, context):
        calls.append(("collect-score", context.submission_id))
        item = submitted[context.submission_id]
        return WorkflowScorerResult(score=item.output == item.expected)

    async def poll(_submission, context):
        calls.append(("poll", context.submission_id))
        return WorkflowSubmissionPoll("complete" if context.submission_id in ready else "pending")

    task = WorkflowTask(
        submit=task_submit,
        completion=WorkflowSubmissionCompletionPoll(poll),
        collect=task_collect,
    )
    submission_score = WorkflowScorer(
        name="submission",
        submit=score_submit,
        completion=WorkflowSubmissionCompletionPoll(poll),
        collect=score_collect,
    )
    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": str(i), "input": i, "expected": i * 2} for i in range(3)],
        task=task,
        scores=[lambda output, expected: output == expected, submission_score],
    )

    started = await workflow_eval.start(no_send_logs=True)
    assert isinstance(started, WorkflowEvalWaitingResult)
    assert started.pending.poll == 3
    task_submission_ids = set(submitted)
    ready.update(task_submission_ids)

    after_tasks = await workflow_eval.poll(started.run_id)
    assert isinstance(after_tasks, WorkflowEvalWaitingResult)
    assert after_tasks.pending.poll == 3
    score_submission_ids = set(submitted) - task_submission_ids
    assert score_submission_ids
    assert not any(call == ("poll", submission_id) for submission_id in score_submission_ids for call in calls)

    ready.update(score_submission_ids)
    completed = await workflow_eval.poll(started.run_id)
    assert isinstance(completed, WorkflowEvalCompletedResult)
    assert completed.summary.scores["scorer_0"].score == 1
    assert completed.summary.scores["submission"].score == 1
    calls_before_status = list(calls)
    assert (await workflow_eval.status(started.run_id)).status == "completed"
    assert calls == calls_before_status


@pytest.mark.asyncio
async def test_failed_poll_raises_and_can_be_retried():
    failure = RuntimeError("provider failed")
    ready = False

    async def poll(_submission, _context):
        return WorkflowSubmissionPoll("complete") if ready else WorkflowSubmissionPoll("failed", error=failure)

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": 1}],
        task=WorkflowTask(
            submit=lambda item, context: {"id": context.submission_id},
            completion=WorkflowSubmissionCompletionPoll(poll),
            collect=lambda _submission, _context: WorkflowTaskResult(output=1),
        ),
    )
    started = await workflow.start(no_send_logs=True)
    with pytest.raises(RuntimeError, match="provider failed") as raised:
        await workflow.poll(started.run_id)
    assert raised.value is failure
    assert (await workflow.status(started.run_id)).pending.poll == 1
    ready = True
    assert (await workflow.poll(started.run_id)).status == "completed"


@pytest.mark.asyncio
async def test_webhook_result_can_be_matched_by_external_id():
    submitted = {}

    async def submit(item, context):
        submitted[context.submission_id] = item
        return {"id": f"external-{context.submission_id}"}

    async def collect(_submission, context):
        return WorkflowTaskResult(output=submitted[context.submission_id].input)

    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": "ok"}],
        task=WorkflowTask(
            submit=submit,
            completion=WorkflowSubmissionCompletionWebhook(lambda submission, _context: submission["id"]),
            collect=collect,
        ),
    )
    started = await workflow_eval.start(no_send_logs=True)
    submission_id = next(iter(submitted))

    completed = await workflow_eval.process_submission_result(started.run_id, external_id=f"external-{submission_id}")

    assert isinstance(completed, WorkflowEvalCompletedResult)
    assert (
        await workflow_eval.process_submission_result(started.run_id, submission_id=submission_id)
    ).status == "completed"


@pytest.mark.asyncio
async def test_collect_rejects_array_results():
    submission_ids = []

    async def submit(_item, context):
        submission_ids.append(context.submission_id)
        return {"id": context.submission_id}

    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "a", "input": 1}, {"id": "b", "input": 2}],
        task=WorkflowTask(
            submit=submit,
            completion=WorkflowSubmissionCompletionWebhook(lambda submission, _context: submission["id"]),
            collect=lambda _submission, _context: [WorkflowTaskResult(output=1)],
        ),
    )
    started = await workflow_eval.start(no_send_logs=True)

    with pytest.raises(TypeError, match="single WorkflowTaskResult"):
        await workflow_eval.process_submission_result(started.run_id, submission_id=submission_ids[0])


@pytest.mark.asyncio
async def test_case_ids_must_be_stable_and_unique():
    submission_task = WorkflowTask(
        submit=lambda _item, _context: {"id": "unused"},
        completion=WorkflowSubmissionCompletionPoll(lambda _submission, _context: WorkflowSubmissionPoll("pending")),
        collect=lambda _submission, _context: [],
    )
    missing = define_workflow_eval(
        "project", store=WorkflowEvalMemoryStore(), data=[{"input": 1}], task=submission_task
    )
    with pytest.raises(ValueError, match="non-empty id"):
        await missing.start(no_send_logs=True)

    duplicate = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "same", "input": 1}, {"id": "same", "input": 2}],
        task=submission_task,
    )
    with pytest.raises(ValueError, match="duplicate"):
        await duplicate.start(no_send_logs=True)


@pytest.mark.asyncio
async def test_case_persistence_does_not_deepcopy_inputs():
    class SerializableWithoutDeepcopy:
        def __deepcopy__(self, _memo):
            raise AssertionError("input was deep-copied")

        def model_dump(self, **_kwargs):
            return {"value": "serialized"}

    submitted = []

    async def submit(item, _context):
        submitted.append(item)
        return {"id": "pending"}

    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": SerializableWithoutDeepcopy()}],
        task=WorkflowTask(
            submit=submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("pending")
            ),
            collect=lambda _submission, _context: [],
        ),
    )

    result = await workflow_eval.start(no_send_logs=True)

    assert result.status == "waiting"
    assert submitted[0].input == {"value": "serialized"}


@pytest.mark.asyncio
async def test_concurrent_webhooks_claim_downstream_submission_once():
    task_items = []
    collects_started = 0
    both_collecting = asyncio.Event()
    score_submissions = 0

    async def submit_task(item, _context):
        task_items.append(item)
        return {"id": "task-provider"}

    async def collect_task(_submission, _context):
        nonlocal collects_started
        collects_started += 1
        if collects_started == 2:
            both_collecting.set()
        await both_collecting.wait()
        return WorkflowTaskResult(output=task_items[0].input * 2)

    async def submit_score(_item, _context):
        nonlocal score_submissions
        score_submissions += 1
        return {"id": "score-provider"}

    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": 2, "expected": 4}],
        task=WorkflowTask(
            submit=submit_task,
            completion=WorkflowSubmissionCompletionWebhook(lambda submission, _context: submission["id"]),
            collect=collect_task,
        ),
        scores=[
            WorkflowScorer(
                name="exact",
                submit=submit_score,
                completion=WorkflowSubmissionCompletionWebhook(lambda submission, _context: submission["id"]),
                collect=lambda _submission, _context: [],
            )
        ],
    )
    started = await workflow_eval.start(no_send_logs=True)

    await asyncio.gather(
        workflow_eval.process_submission_result(started.run_id, external_id="task-provider"),
        workflow_eval.process_submission_result(started.run_id, external_id="task-provider"),
    )

    assert score_submissions == 1
    current = await workflow_eval.status(started.run_id)
    assert current.status == "waiting"
    assert current.pending.webhook == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_ids", [False, True])
async def test_workflow_logging_uses_stable_spans_and_resume_metadata(
    monkeypatch, with_memory_logger, with_simulate_login, legacy_ids
):
    if legacy_ids:
        monkeypatch.setenv("BRAINTRUST_LEGACY_IDS", "true")
    else:
        monkeypatch.delenv("BRAINTRUST_LEGACY_IDS", raising=False)
    local_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": 1, "expected": 2}],
        task=lambda value: value + 1,
        scores=[lambda output, expected: output == expected],
    )
    local_summary = (await local_eval.start(no_send_logs=True)).summary
    experiment = init_test_exp("workflow", "project")
    monkeypatch.setattr(workflow_eval_module, "init_experiment", lambda **_kwargs: experiment)
    monkeypatch.setattr(experiment, "summarize", lambda **_kwargs: local_summary)
    submitted = []
    trace_configurations = []
    classifier_trace_configurations = []

    async def submit(item, _context):
        submitted.append(item)
        return {"id": "task"}

    def scorer(output, expected, trace):
        trace_configurations.append(trace.get_configuration())
        return output == expected

    def classifier(output, trace):
        classifier_trace_configurations.append(trace.get_configuration())
        return {"id": "positive"}

    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": 1, "expected": 2}],
        task=WorkflowTask(
            submit=submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("complete")
            ),
            collect=lambda _submission, _context: WorkflowTaskResult(output=submitted[0].input + 1),
        ),
        scores=[scorer],
        classifiers=[classifier],
        experiment_name="workflow",
    )

    waiting = await workflow_eval.start()
    result = await workflow_eval.poll(waiting.run_id)
    logs = with_memory_logger.pop()

    assert result.status == "completed"
    assert len(logs) == 4
    roots = [row for row in logs if not row["span_parents"]]
    assert len(roots) == 1
    assert roots[0]["metadata"]["workflow_eval"] == {
        "run_id": result.run_id,
        "case_id": "case",
        "trial_index": 0,
    }
    assert trace_configurations == [
        {
            "object_type": "experiment",
            "object_id": experiment.id,
            "root_span_id": roots[0]["root_span_id"],
        }
    ]
    assert classifier_trace_configurations == trace_configurations
    assert len({row["span_id"] for row in logs}) == 4
    await workflow_eval.status(result.run_id)
    assert with_memory_logger.pop() == []


@pytest.mark.asyncio
async def test_workflow_logging_flushes_before_persisting_log_markers(
    monkeypatch, with_memory_logger, with_simulate_login
):
    local_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": 1}],
        task=lambda value: value,
    )
    local_summary = (await local_eval.start(no_send_logs=True)).summary
    experiment = init_test_exp("workflow", "project")
    monkeypatch.setattr(workflow_eval_module, "init_experiment", lambda **_kwargs: experiment)
    monkeypatch.setattr(experiment, "summarize", lambda **_kwargs: local_summary)

    flush_count = 0
    marker_flush_counts = []
    original_flush = with_memory_logger.flush

    def flush(*args, **kwargs):
        nonlocal flush_count
        flush_count += 1
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(with_memory_logger, "flush", flush)

    class RecordingStore(WorkflowEvalMemoryStore):
        async def write(self, key, value):
            if "/task-log/" in key or "/score-log-" in key or "/classification-log-" in key:
                marker_flush_counts.append(flush_count)
            await super().write(key, value)

    submitted = []

    async def submit(item, _context):
        submitted.append(item)
        return {"id": "task"}

    workflow_eval = define_workflow_eval(
        "project",
        store=RecordingStore(),
        data=[{"id": "case", "input": 1, "expected": 1}],
        task=WorkflowTask(
            submit=submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("complete")
            ),
            collect=lambda _submission, _context: WorkflowTaskResult(output=submitted[0].input),
        ),
        scores=[lambda output, expected: output == expected],
        classifiers=[lambda output: {"id": "positive"}],
        experiment_name="workflow",
    )

    waiting = await workflow_eval.start()
    completed = await workflow_eval.poll(waiting.run_id)

    assert completed.status == "completed"
    assert len(marker_flush_counts) == 3
    assert all(count > 0 for count in marker_flush_counts)


@pytest.mark.asyncio
async def test_workflow_dataset_rows_preserve_dataset_origin(monkeypatch, with_memory_logger, with_simulate_login):
    project_metadata = ObjectMetadata(id="test-project", name="test-project", full_info={})
    dataset_metadata = ObjectMetadata(id="active-dataset", name="test-dataset", full_info={})
    dataset = Dataset(
        lazy_metadata=LazyValue(
            lambda: ProjectDatasetMetadata(project=project_metadata, dataset=dataset_metadata),
            use_mutex=False,
        ),
        state=BraintrustState(),
    )
    row = {
        "id": "dataset-row",
        "_xact_id": "dataset-xact",
        "created": "2026-06-02T00:00:00.000Z",
        "input": 1,
    }
    local_summary = (
        await define_workflow_eval(
            "project", store=WorkflowEvalMemoryStore(), data=[{"input": 1}], task=lambda value: value
        ).start(no_send_logs=True)
    ).summary
    experiment = init_test_exp("workflow", "project")
    monkeypatch.setattr(workflow_eval_module, "init_experiment", lambda **_kwargs: experiment)
    monkeypatch.setattr(experiment, "summarize", lambda **_kwargs: local_summary)

    submitted = []

    async def submit(item, _context):
        submitted.append(item)
        return {"id": "task"}

    workflow_eval = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=dataset,
        task=WorkflowTask(
            submit=submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("complete")
            ),
            collect=lambda _submission, _context: WorkflowTaskResult(output=submitted[0].input),
        ),
        experiment_name="workflow",
    )

    with patch.object(dataset, "_refetch", return_value=[row]):
        waiting = await workflow_eval.start()
    await workflow_eval.poll(waiting.run_id)

    root = next(log for log in with_memory_logger.pop() if not log["span_parents"])
    assert root["origin"] == {
        "object_type": "dataset",
        "object_id": "active-dataset",
        "id": "dataset-row",
        "_xact_id": "dataset-xact",
        "created": "2026-06-02T00:00:00.000Z",
    }


@pytest.mark.asyncio
async def test_only_completed_cases_start_scorers_and_classifiers():
    ready = set()
    submitted = {}
    local_scores = []
    classifications = []

    async def submit(item, context):
        submitted[context.submission_id] = item
        return {"id": context.submission_id}

    async def poll(submission, _context):
        return WorkflowSubmissionPoll("complete" if submission["id"] in ready else "pending")

    async def score(output):
        local_scores.append(output)
        return 1

    async def classify(output):
        classifications.append(output)
        return {"id": "positive"}

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": str(i), "input": i} for i in range(3)],
        task=WorkflowTask(
            submit=submit,
            completion=WorkflowSubmissionCompletionPoll(poll),
            collect=lambda submission, _context: WorkflowTaskResult(output=submitted[submission["id"]].input),
        ),
        scores=[
            score,
            WorkflowScorer(
                name="remote",
                submit=submit,
                completion=WorkflowSubmissionCompletionPoll(poll),
                collect=lambda _submission, _context: WorkflowScorerResult(score=1),
            ),
        ],
        classifiers=[classify],
    )
    started = await workflow.start(no_send_logs=True)
    task_ids = set(submitted)
    ready.add(next(key for key, item in submitted.items() if item.input == 1))
    partial = await workflow.poll(started.run_id)
    assert partial.pending.poll == 3  # two tasks and the ready case's scorer
    assert local_scores == [1]
    assert classifications == [1]
    score_ids = set(submitted) - task_ids
    assert len(score_ids) == 1
    assert submitted[next(iter(score_ids))].output == 1
    ready.update(score_ids)
    partial = await workflow.poll(started.run_id)
    assert partial.status == "waiting"
    assert partial.pending.poll == 2
    assert local_scores == [1]
    ready.update(task_ids)
    await workflow.poll(started.run_id)
    ready.update(submitted)
    completed = await workflow.poll(started.run_id)
    assert completed.status == "completed"
    assert sorted(local_scores) == [0, 1, 2]
    assert sorted(classifications) == [0, 1, 2]
    assert completed.summary.scores["remote"].score == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("max_concurrency", [None, 1, 3])
async def test_provider_callback_concurrency_is_bounded(max_concurrency):
    active = {"submit": 0, "poll": 0, "collect": 0}
    peak = active.copy()

    async def callback(stage):
        active[stage] += 1
        peak[stage] = max(peak[stage], active[stage])
        await asyncio.sleep(0.001)
        active[stage] -= 1

    async def submit(item, context):
        await callback("submit")
        return {"id": context.submission_id, "input": item.input}

    async def poll(_submission, _context):
        await callback("poll")
        return WorkflowSubmissionPoll("complete")

    async def collect(submission, _context):
        await callback("collect")
        return WorkflowTaskResult(output=submission["input"])

    options = {} if max_concurrency is None else {"max_concurrency": max_concurrency}
    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": str(i), "input": i} for i in range(25)],
        task=WorkflowTask(submit=submit, completion=WorkflowSubmissionCompletionPoll(poll), collect=collect),
        **options,
    )
    started = await workflow.start(no_send_logs=True)
    assert started.pending.poll == 25
    assert (await workflow.poll(started.run_id)).status == "completed"
    limit = max_concurrency or 10
    assert peak["submit"] == limit
    assert peak["poll"] == limit
    assert 0 < peak["collect"] <= limit
    assert all(value == 0 for value in active.values())


@pytest.mark.parametrize("max_concurrency", [0, -1, 1.5, True, "2", None])
def test_invalid_concurrency_is_rejected(max_concurrency):
    with pytest.raises(ValueError, match="max_concurrency must be a positive integer"):
        define_workflow_eval(
            "project",
            store=WorkflowEvalMemoryStore(),
            data=[],
            task=lambda value: value,
            max_concurrency=max_concurrency,
        )


@pytest.mark.asyncio
async def test_poll_error_does_not_prevent_independent_case_progress():
    scores = []

    async def poll(submission, _context):
        if submission["input"] == 0:
            raise RuntimeError("provider unavailable")
        return WorkflowSubmissionPoll("complete")

    async def score(output):
        scores.append(output)
        return 1

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": str(i), "input": i} for i in range(3)],
        task=WorkflowTask(
            submit=lambda item, _context: {"input": item.input},
            completion=WorkflowSubmissionCompletionPoll(poll),
            collect=lambda submission, _context: WorkflowTaskResult(output=submission["input"]),
        ),
        scores=[score],
        max_concurrency=1,
    )
    started = await workflow.start(no_send_logs=True)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await workflow.poll(started.run_id)
    assert scores == [1, 2]
    assert (await workflow.status(started.run_id)).pending.poll == 1


@pytest.mark.asyncio
async def test_submission_error_waits_for_independent_submissions():
    calls = []
    run_ids = []

    async def submit(item, context):
        run_ids.append(context.run_id)
        if item.input == 0:
            raise RuntimeError("submission failed")
        await asyncio.sleep(0)
        calls.append(item.input)
        return {"input": item.input}

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": str(i), "input": i} for i in range(3)],
        task=WorkflowTask(
            submit=submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("pending")
            ),
            collect=lambda submission, _context: WorkflowTaskResult(output=submission["input"]),
        ),
        max_concurrency=1,
    )
    with pytest.raises(RuntimeError, match="submission failed"):
        await workflow.start(no_send_logs=True)
    assert calls == [1, 2]
    assert (await workflow.status(run_ids[0])).pending.poll == 2


@pytest.mark.asyncio
async def test_webhooks_resume_fresh_definitions_and_validate_both_locators():
    store = WorkflowEvalMemoryStore()
    submitted = {}
    collected = []

    async def submit(item, context):
        submitted[item.input] = context.submission_id
        return {"id": item.input}

    async def collect(submission, _context):
        collected.append(submission["id"])
        await asyncio.sleep(0)
        return WorkflowTaskResult(output=submission["id"])

    def definition():
        return define_workflow_eval(
            "project",
            store=store,
            data=[{"id": value, "input": value} for value in ("a", "b")],
            task=WorkflowTask(
                submit=submit,
                collect=collect,
                completion=WorkflowSubmissionCompletionWebhook(lambda submission, _context: submission["id"]),
            ),
        )

    started = await definition().start(no_send_logs=True)
    workflow = definition()
    for locators in [
        {"submission_id": submitted["a"], "external_id": "b"},
        {"submission_id": submitted["a"], "external_id": "unknown"},
        {"submission_id": "unknown", "external_id": "a"},
    ]:
        with pytest.raises(ValueError, match="different submissions"):
            await workflow.process_submission_result(started.run_id, **locators)
    with pytest.raises(ValueError, match="requires submission_id or external_id"):
        await workflow.process_submission_result(started.run_id)
    with pytest.raises(ValueError, match="No submission matches"):
        await workflow.process_submission_result(started.run_id, external_id="unknown")
    await asyncio.gather(
        workflow.process_submission_result(started.run_id, submission_id=submitted["a"]),
        workflow.process_submission_result(started.run_id, external_id="b"),
    )
    assert (await workflow.status(started.run_id)).status == "completed"
    assert sorted(collected) == ["a", "b"]
    await workflow.process_submission_result(started.run_id, external_id="a")
    assert sorted(collected) == ["a", "b"]


@pytest.mark.asyncio
async def test_workflow_trials_preserve_parameters_metadata_and_tags():
    task_items = []
    score_items = []

    async def submit_task(item, context):
        task_items.append(item)
        return {"id": context.submission_id, "trial": item.trial_index}

    async def submit_score(item, _context):
        score_items.append(item)
        return {"id": item.id}

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        case_id=lambda datum: "case",
        data=[{"input": 1, "expected": 2, "metadata": {"original": True}, "tags": ["old"], "trial_count": 2}],
        task=WorkflowTask(
            submit=submit_task,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("complete")
            ),
            collect=lambda submission, _context: WorkflowTaskResult(
                output=submission["trial"], metadata={"new": True}, tags=["updated"]
            ),
        ),
        scores=[
            WorkflowScorer(
                name="remote",
                submit=submit_score,
                completion=WorkflowSubmissionCompletionWebhook(lambda submission, _context: submission["id"]),
                collect=lambda _submission, _context: WorkflowScorerResult(score=1),
            )
        ],
    )
    started = await workflow.start({"temperature": 0.5}, no_send_logs=True)
    assert {item.id for item in task_items} == {"case:trial:0", "case:trial:1"}
    assert all(item.parameters == {"temperature": 0.5} for item in task_items)
    assert sorted(item.trial_index for item in task_items) == [0, 1]
    assert (await workflow.poll(started.run_id)).pending.webhook == 2
    assert all(item.metadata == {"original": True, "new": True} for item in score_items)
    assert all(item.tags == ["updated"] and item.expected == 2 for item in score_items)
    for item in score_items:
        result = await workflow.process_submission_result(started.run_id, external_id=item.id)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_ordinary_tasks_with_workflow_scorers_and_empty_data():
    submitted = []

    async def submit(item, _context):
        submitted.append(item)
        return {"id": item.id}

    scorer = WorkflowScorer(
        name="remote",
        submit=submit,
        completion=WorkflowSubmissionCompletionPoll(lambda _submission, _context: WorkflowSubmissionPoll("complete")),
        collect=lambda _submission, _context: WorkflowScorerResult(score=1),
    )
    for data in [[], [{"id": "case", "input": 1}]]:
        workflow = define_workflow_eval(
            "project", store=WorkflowEvalMemoryStore(), data=data, task=lambda value: value + 1, scores=[scorer]
        )
        started = await workflow.start(no_send_logs=True)
        if data:
            assert started.pending.poll == 1
            assert submitted[0].output == 2
        else:
            assert started.status == "completed"
        assert (await workflow.poll(started.run_id)).status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["poll", "webhook"])
async def test_replay_repairs_interrupted_completion_progress(mode):
    class InterruptedStore(WorkflowEvalMemoryStore):
        interrupted = False

        async def add_to_set(self, key, member):
            if key.endswith("/complete") and not self.interrupted:
                self.interrupted = True
                raise RuntimeError("interrupted progress")
            await super().add_to_set(key, member)

    collects = []

    async def collect(submission, _context):
        collects.append(submission["id"])
        return WorkflowTaskResult(output=1)

    workflow = define_workflow_eval(
        "project",
        store=InterruptedStore(),
        data=[{"id": value, "input": value} for value in ("a", "b")],
        task=WorkflowTask(
            submit=lambda item, _context: {"id": item.input},
            collect=collect,
            completion=WorkflowSubmissionCompletionWebhook(lambda submission, _context: submission["id"])
            if mode == "webhook"
            else WorkflowSubmissionCompletionPoll(
                lambda submission, _context: WorkflowSubmissionPoll(
                    "complete" if submission["id"] == "a" else "pending"
                )
            ),
        ),
    )
    started = await workflow.start(no_send_logs=True)
    with pytest.raises(RuntimeError, match="interrupted progress"):
        if mode == "webhook":
            await workflow.process_submission_result(started.run_id, external_id="a")
        else:
            await workflow.poll(started.run_id)
    if mode == "webhook":
        result = await workflow.process_submission_result(started.run_id, external_id="a")
    else:
        result = await workflow.poll(started.run_id)
    assert result.pending.poll + result.pending.webhook == 1
    assert collects == ["a"]
