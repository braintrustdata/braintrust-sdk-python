"""Tests for the experimental workflow eval API."""

import asyncio
from unittest.mock import patch

import pytest

from . import workflow_eval as workflow_eval_module
from .logger import BraintrustState, Dataset, ObjectMetadata, ProjectDatasetMetadata
from .test_helpers import init_test_exp, with_memory_logger, with_simulate_login  # noqa: F401
from .util import LazyValue
from .workflow_eval import (
    WorkflowBatchingOptions,
    WorkflowBatchItemResult,
    WorkflowBatchScorer,
    WorkflowBatchTask,
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


@pytest.mark.asyncio
async def test_memory_store_reserve_batch_is_atomic_for_overlapping_batches():
    store = WorkflowEvalMemoryStore()

    claims = await asyncio.gather(
        store.reserve_batch(
            ["workflow/{run-1}/links/a", "workflow/{run-1}/links/b"],
            b"batch-1",
            "workflow/{run-1}/submissions/batch-1",
            b"record-1",
            "workflow/{run-1}/leases/batch-1",
            b"lease-1",
            1000,
        ),
        store.reserve_batch(
            ["workflow/{run-1}/links/b", "workflow/{run-1}/links/c"],
            b"batch-2",
            "workflow/{run-1}/submissions/batch-2",
            b"record-2",
            "workflow/{run-1}/leases/batch-2",
            b"lease-2",
            1000,
        ),
    )

    assert sum(claims) == 1
    if claims[0]:
        assert await store.read("workflow/{run-1}/links/a") == b"batch-1"
        assert await store.read("workflow/{run-1}/links/b") == b"batch-1"
        assert await store.read("workflow/{run-1}/links/c") is None
        assert await store.read("workflow/{run-1}/submissions/batch-1") == b"record-1"
    else:
        assert await store.read("workflow/{run-1}/links/a") is None
        assert await store.read("workflow/{run-1}/links/b") == b"batch-2"
        assert await store.read("workflow/{run-1}/links/c") == b"batch-2"
        assert await store.read("workflow/{run-1}/submissions/batch-2") == b"record-2"


class _SyncRedis:
    def __init__(self):
        self.values = {}
        self.calls = []
        self.sets = {}
        self.expirations = {}

    def eval(self, script, numkeys, *args):
        if "SADD" in script and "PEXPIRE" in script:
            assert numkeys == 1
            key, member, ttl_ms = args
            self.sets.setdefault(key, set()).add(member)
            self.expirations[key] = ttl_ms
            return 1
        if "KEYS[#KEYS - 1]" in script:
            keys = args[:numkeys]
            hash_tags = {key.split("{", 1)[1].split("}", 1)[0] for key in keys}
            assert len(hash_tags) == 1
            claim_value, submission_value, lease_value, ttl_ms, lease_ttl_ms = args[numkeys:]
            if any(key in self.values for key in keys):
                return 0
            for key in keys[:-2]:
                self.values[key] = claim_value
                self.expirations[key] = ttl_ms
            self.values[keys[-2]] = submission_value
            self.expirations[keys[-2]] = ttl_ms
            self.values[keys[-1]] = lease_value
            self.expirations[keys[-1]] = lease_ttl_ms
            return 1
        if "redis.call('GET', KEYS[1]) == ARGV[1]" in script:
            key, expected = args[:2]
            if self.values.get(key) == expected:
                self.values.pop(key, None)
                self.expirations.pop(key, None)
                return 1
            return 0
        assert "EXISTS" in script and numkeys == len(args) - 2
        keys, member, ttl_ms = args[:numkeys], args[-2], args[-1]
        if any(key in self.values for key in keys):
            return 0
        for key in keys:
            self.values[key] = member
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
@pytest.mark.parametrize("client_type", [_SyncRedis, _AsyncRedis])
async def test_redis_store_atomically_reserves_batch_in_one_cluster_slot(client_type):
    client = client_type()
    store = WorkflowEvalRedisStore(client, key_prefix="test:", ttl_ms=1234)

    reserved = await store.reserve_batch(
        ["workflow/{run-1}/links/a", "workflow/{run-1}/links/b"],
        b"submission-1",
        "workflow/{run-1}/submissions/submission-1",
        b'{"status":"failed"}',
        "workflow/{run-1}/leases/submission-1",
        b"lease-1",
        5000,
    )

    assert reserved is True
    assert await store.read("workflow/{run-1}/submissions/submission-1") == b'{"status":"failed"}'
    assert await store.acquire_lease("workflow/{run-1}/leases/submission-1", b"lease-2", 5000) is False
    await store.release_lease("workflow/{run-1}/leases/submission-1", b"lease-1")
    assert await store.acquire_lease("workflow/{run-1}/leases/submission-1", b"lease-2", 5000) is True


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

    completed = await workflow_eval.process_submission_result(external_id=f"external-{submission_id}")

    assert isinstance(completed, WorkflowEvalCompletedResult)
    assert (
        await workflow_eval.process_submission_result(started.run_id, submission_id=submission_id)
    ).status == "completed"


@pytest.mark.asyncio
async def test_batch_tasks_and_scorers_submit_and_collect_multiple_items():
    task_batch_sizes = []
    score_batch_sizes = []

    async def submit_task(items, _context):
        task_batch_sizes.append(len(items))
        return {"inputs": [item.input for item in items]}

    async def collect_task(submission, _context):
        return [
            WorkflowBatchItemResult(custom_id=f"item-{index}", result=WorkflowTaskResult(output=value * 2))
            for index, value in enumerate(submission["inputs"])
        ]

    async def submit_score(items, _context):
        score_batch_sizes.append(len(items))
        return {"values": [(item.output, item.expected) for item in items]}

    async def collect_score(submission, _context):
        for index, (output, expected) in enumerate(submission["values"]):
            yield WorkflowBatchItemResult(
                custom_id=f"item-{index}", result=WorkflowScorerResult(score=output == expected)
            )

    completion = WorkflowSubmissionCompletionPoll(lambda _submission, _context: WorkflowSubmissionPoll("complete"))
    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": str(value), "input": value, "expected": value * 2} for value in range(3)],
        task=WorkflowBatchTask(
            batching=WorkflowBatchingOptions(max_size=2),
            submit=submit_task,
            completion=completion,
            collect=collect_task,
        ),
        scores=[
            WorkflowBatchScorer(
                name="batch-score",
                batching=WorkflowBatchingOptions(max_size=2),
                submit=submit_score,
                completion=completion,
                collect=collect_score,
            )
        ],
    )

    started = await workflow.start(no_send_logs=True)
    after_tasks = await workflow.poll(started.run_id)
    completed = await workflow.poll(started.run_id)

    assert isinstance(after_tasks, WorkflowEvalWaitingResult)
    assert after_tasks.pending.poll == 2
    assert isinstance(completed, WorkflowEvalCompletedResult)
    assert completed.summary.scores["batch-score"].score == 1
    assert task_batch_sizes == [2, 1]
    assert score_batch_sizes == [2, 1]


@pytest.mark.asyncio
async def test_batch_item_failures_skip_scoring():
    score_inputs = []

    async def submit(items, _context):
        return {"inputs": [item.input for item in items]}

    async def collect(submission, _context):
        return [
            WorkflowBatchItemResult(custom_id="item-0", result=WorkflowTaskResult(output="good")),
            WorkflowBatchItemResult(custom_id="item-1", error=ValueError("provider rejected item")),
            # item-2 is intentionally missing and should also fail.
        ]

    def score(output):
        score_inputs.append(output)
        return 1

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": str(value), "input": value} for value in range(3)],
        task=WorkflowBatchTask(
            batching=WorkflowBatchingOptions(max_size=3),
            submit=submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("complete")
            ),
            collect=collect,
        ),
        scores=[score],
    )

    started = await workflow.start(no_send_logs=True)
    completed = await workflow.poll(started.run_id)

    assert isinstance(started, WorkflowEvalWaitingResult)
    assert isinstance(completed, WorkflowEvalCompletedResult)
    assert score_inputs == ["good"]


@pytest.mark.asyncio
async def test_batch_scorer_wait_window_uses_persisted_wall_clock(monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(workflow_eval_module.time, "time", lambda: now[0])
    ready = {"a"}
    score_batches = []

    async def task_submit(item, _context):
        return {"input": item.input}

    async def task_collect(submission, _context):
        return WorkflowTaskResult(output=submission["input"])

    def task_poll(submission, _context):
        return WorkflowSubmissionPoll("complete" if submission["input"] in ready else "pending")

    async def score_submit(items, _context):
        score_batches.append([item.output for item in items])
        return {"size": len(items)}

    async def score_collect(submission, _context):
        return [
            WorkflowBatchItemResult(custom_id=f"item-{index}", result=WorkflowScorerResult(score=1))
            for index in range(submission["size"])
        ]

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": value, "input": value} for value in ("a", "b")],
        task=WorkflowTask(
            submit=task_submit,
            completion=WorkflowSubmissionCompletionPoll(task_poll),
            collect=task_collect,
        ),
        scores=[
            WorkflowBatchScorer(
                name="batch-score",
                batching=WorkflowBatchingOptions(max_size=2, max_wait_ms=100),
                submit=score_submit,
                completion=WorkflowSubmissionCompletionPoll(
                    lambda _submission, _context: WorkflowSubmissionPoll("pending")
                ),
                collect=score_collect,
            )
        ],
    )

    started = await workflow.start(no_send_logs=True)
    await workflow.poll(started.run_id)
    assert score_batches == []

    now[0] += 0.101
    await workflow.poll(started.run_id)

    assert score_batches == [["a"]]


@pytest.mark.asyncio
async def test_batch_scorer_wait_window_starts_for_remainder_while_full_batches_submit(monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(workflow_eval_module.time, "time", lambda: now[0])
    ready = {"a", "b", "c"}
    score_batches = []

    async def task_submit(item, _context):
        return {"input": item.input}

    async def task_collect(submission, _context):
        return WorkflowTaskResult(output=submission["input"])

    def task_poll(submission, _context):
        return WorkflowSubmissionPoll("complete" if submission["input"] in ready else "pending")

    async def score_submit(items, _context):
        score_batches.append([item.output for item in items])
        return {"size": len(items)}

    async def score_collect(submission, _context):
        return [
            WorkflowBatchItemResult(custom_id=f"item-{index}", result=WorkflowScorerResult(score=1))
            for index in range(submission["size"])
        ]

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": value, "input": value} for value in ("a", "b", "c", "d")],
        task=WorkflowTask(
            submit=task_submit,
            completion=WorkflowSubmissionCompletionPoll(task_poll),
            collect=task_collect,
        ),
        scores=[
            WorkflowBatchScorer(
                name="batch-score",
                batching=WorkflowBatchingOptions(max_size=2, max_wait_ms=100),
                submit=score_submit,
                completion=WorkflowSubmissionCompletionPoll(
                    lambda _submission, _context: WorkflowSubmissionPoll("pending")
                ),
                collect=score_collect,
            )
        ],
    )

    started = await workflow.start(no_send_logs=True)
    await workflow.poll(started.run_id)
    assert score_batches == [["a", "b"]]

    now[0] += 0.101
    await workflow.poll(started.run_id)

    assert score_batches == [["a", "b"], ["c"]]


@pytest.mark.asyncio
async def test_batch_webhook_id_failure_keeps_submission_recoverable():
    get_id_calls = 0
    score_submit_calls = 0

    async def task_submit(items, _context):
        return {"inputs": [item.input for item in items]}

    async def task_collect(submission, _context):
        return [WorkflowBatchItemResult(custom_id="item-0", result=WorkflowTaskResult(output=submission["inputs"][0]))]

    async def score_submit(items, _context):
        nonlocal score_submit_calls
        score_submit_calls += 1
        return {"external_id": "score-batch"}

    def get_score_external_id(submission, _context):
        nonlocal get_id_calls
        get_id_calls += 1
        if get_id_calls == 1:
            raise RuntimeError("temporary id lookup failure")
        return submission["external_id"]

    async def score_collect(_submission, _context):
        return [WorkflowBatchItemResult(custom_id="item-0", result=WorkflowScorerResult(score=1))]

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": "value"}],
        task=WorkflowBatchTask(
            batching=WorkflowBatchingOptions(max_size=1),
            submit=task_submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("complete")
            ),
            collect=task_collect,
        ),
        scores=[
            WorkflowBatchScorer(
                name="batch-score",
                batching=WorkflowBatchingOptions(max_size=1),
                submit=score_submit,
                completion=WorkflowSubmissionCompletionWebhook(get_score_external_id),
                collect=score_collect,
            )
        ],
    )

    started = await workflow.start(no_send_logs=True)
    with pytest.raises(RuntimeError, match="temporary id lookup failure"):
        await workflow.poll(started.run_id)

    waiting = await workflow.poll(started.run_id)
    assert isinstance(waiting, WorkflowEvalWaitingResult)
    assert waiting.pending.webhook == 1
    assert score_submit_calls == 1
    assert get_id_calls == 2

    completed = await workflow.process_submission_result(external_id="score-batch")
    assert isinstance(completed, WorkflowEvalCompletedResult)


@pytest.mark.asyncio
async def test_failed_batch_retry_is_leased_and_webhook_lookup_is_recoverable():
    score_submit_calls = 0
    external_id_calls = 0
    retry_started = asyncio.Event()
    allow_retry_to_finish = asyncio.Event()

    async def task_submit(items, _context):
        return {"inputs": [item.input for item in items]}

    async def task_collect(submission, _context):
        return [WorkflowBatchItemResult(custom_id="item-0", result=WorkflowTaskResult(output=submission["inputs"][0]))]

    async def score_submit(_items, _context):
        nonlocal score_submit_calls
        score_submit_calls += 1
        if score_submit_calls == 1:
            raise RuntimeError("initial provider submit failed")
        retry_started.set()
        await allow_retry_to_finish.wait()
        return {"external_id": "retried-score-job"}

    def get_external_id(submission, _context):
        nonlocal external_id_calls
        external_id_calls += 1
        if external_id_calls == 1:
            raise RuntimeError("temporary webhook ID lookup failure")
        return submission["external_id"]

    async def score_collect(_submission, _context):
        return [WorkflowBatchItemResult(custom_id="item-0", result=WorkflowScorerResult(score=1))]

    workflow = define_workflow_eval(
        "project",
        store=WorkflowEvalMemoryStore(),
        data=[{"id": "case", "input": "value"}],
        task=WorkflowBatchTask(
            batching=WorkflowBatchingOptions(max_size=1),
            submit=task_submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("complete")
            ),
            collect=task_collect,
        ),
        scores=[
            WorkflowBatchScorer(
                name="batch-score",
                batching=WorkflowBatchingOptions(max_size=1),
                submit=score_submit,
                completion=WorkflowSubmissionCompletionWebhook(get_external_id),
                collect=score_collect,
            )
        ],
    )

    started = await workflow.start(no_send_logs=True)
    with pytest.raises(RuntimeError, match="initial provider submit failed"):
        await workflow.poll(started.run_id)

    retry = asyncio.create_task(workflow.poll(started.run_id))
    await retry_started.wait()
    concurrent = await workflow.poll(started.run_id)
    assert isinstance(concurrent, WorkflowEvalWaitingResult)
    allow_retry_to_finish.set()
    with pytest.raises(RuntimeError, match="temporary webhook ID lookup failure"):
        await retry

    waiting = await workflow.poll(started.run_id)
    assert isinstance(waiting, WorkflowEvalWaitingResult)
    assert waiting.pending.webhook == 1
    assert score_submit_calls == 2
    assert external_id_calls == 2
    completed = await workflow.process_submission_result(external_id="retried-score-job")
    assert isinstance(completed, WorkflowEvalCompletedResult)


@pytest.mark.asyncio
async def test_batch_reservation_placeholder_recovers_after_pre_submit_store_failure():
    class _FailOnceStore(WorkflowEvalMemoryStore):
        def __init__(self):
            super().__init__()
            self.fail_case_read = False
            self.run_id = None

        async def reserve_batch(self, keys, value, submission_key, submission_value, lease_key, lease_value, ttl_ms):
            self.run_id = keys[0].split("{")[1].split("}")[0]
            self.fail_case_read = True
            return await super().reserve_batch(
                keys, value, submission_key, submission_value, lease_key, lease_value, ttl_ms
            )

        async def read(self, key):
            if self.fail_case_read and "/case/" in key:
                self.fail_case_read = False
                raise RuntimeError("transient case read failure")
            return await super().read(key)

    store = _FailOnceStore()
    submit_calls = 0

    async def submit(items, _context):
        nonlocal submit_calls
        submit_calls += 1
        return {"inputs": [item.input for item in items]}

    async def collect(submission, _context):
        return [WorkflowBatchItemResult(custom_id="item-0", result=WorkflowTaskResult(output=submission["inputs"][0]))]

    workflow = define_workflow_eval(
        "project",
        store=store,
        data=[{"id": "case", "input": "value"}],
        task=WorkflowBatchTask(
            batching=WorkflowBatchingOptions(max_size=1),
            submit=submit,
            completion=WorkflowSubmissionCompletionPoll(
                lambda _submission, _context: WorkflowSubmissionPoll("complete")
            ),
            collect=collect,
        ),
    )

    with pytest.raises(RuntimeError, match="transient case read failure"):
        await workflow.start(no_send_logs=True)
    assert store.run_id is not None
    stored_submissions = [value for key, value in store._values.items() if "/submission/" in key]
    assert len(stored_submissions) == 1

    await workflow.poll(store.run_id)
    completed = await workflow.poll(store.run_id)
    assert isinstance(completed, WorkflowEvalCompletedResult)
    assert submit_calls == 1


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
