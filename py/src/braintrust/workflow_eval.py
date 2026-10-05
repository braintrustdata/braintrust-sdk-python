"""Experimental workflow evaluations with asynchronous provider submissions."""

import asyncio
import base64
import dataclasses
import functools
import hashlib
import inspect
import threading
import time
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

from .bt_json import bt_dumps, bt_loads
from .env import BraintrustEnv
from .framework import (
    BaseExperiment,
    DictEvalHooks,
    EvalCase,
    EvalClassifier,
    EvalData,
    EvalResult,
    EvalScorer,
    EvalTask,
    Evaluator,
    OneOrMoreScores,
    _build_classification_span_output,
    _build_span_metadata,
    _classifier_name,
    _get_persisted_base_experiment_id,
    _normalize_score,
    _scorer_name,
    _validate_classification_result,
    _validated_object_reference,
    await_or_run,
    build_local_summary,
    call_user_fn,
    init_experiment,
    run_evaluator,
)
from .git_fields import GitMetadataSettings, RepoInfo
from .logger import (
    NOOP_SPAN,
    BraintrustState,
    Dataset,
    Experiment,
    ExperimentSummary,
    Metadata,
    Span,
    _internal_get_global_state,
    _internal_resume_span,
    _internal_start_span_with_initial_merge,
    span_components_to_object_id,
)
from .logger import init as _init_experiment
from .parameters import EvalParameters, RemoteEvalParameters, ValidatedParameters, validate_parameters
from .score import Classification, Score, ScoreLike, is_score, is_scorer
from .span_identifier_v3 import span_object_type_v3_to_typed_string
from .span_identifier_v4 import SpanComponentsV4
from .span_types import SpanTypeAttribute
from .trace import LocalTrace
from .util import get_signature, merge_dicts


Input = TypeVar("Input")
Output = TypeVar("Output")
Expected = TypeVar("Expected")
SubmissionData = TypeVar("SubmissionData")

DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_REDIS_TTL_MS = 1_000 * 60 * 60 * 24 * 7
_BATCH_LEASE_TTL_MS = 5 * 60 * 1000
_SCHEMA_PREFIX = "workflow-eval/python/v1"


@dataclasses.dataclass(frozen=True)
class WorkflowSubmissionContext:
    """Identifiers supplied to a workflow submission processor callback."""

    run_id: str
    submission_id: str


@dataclasses.dataclass(frozen=True)
class WorkflowSubmissionPoll:
    """The current state returned by a polling completion callback."""

    status: Literal["pending", "complete", "failed"]
    error: Any = None


@dataclasses.dataclass(frozen=True)
class WorkflowSubmissionCompletionPoll(Generic[SubmissionData]):
    """Configures a submission processor whose provider is checked by polling."""

    poll: Callable[
        [SubmissionData, WorkflowSubmissionContext], WorkflowSubmissionPoll | Awaitable[WorkflowSubmissionPoll]
    ]
    mode: Literal["poll"] = dataclasses.field(default="poll", init=False)


@dataclasses.dataclass(frozen=True)
class WorkflowSubmissionCompletionWebhook(Generic[SubmissionData]):
    """Configures a submission processor completed by an incoming webhook."""

    get_external_id: Callable[[SubmissionData, WorkflowSubmissionContext], str | Awaitable[str]]
    mode: Literal["webhook"] = dataclasses.field(default="webhook", init=False)


WorkflowSubmissionCompletion = (
    WorkflowSubmissionCompletionPoll[SubmissionData] | WorkflowSubmissionCompletionWebhook[SubmissionData]
)


@dataclasses.dataclass(frozen=True)
class WorkflowTaskItem(Generic[Input, Expected]):
    """One case/trial passed to a workflow task submission callback."""

    id: str
    input: Input
    expected: Expected | None
    metadata: Metadata
    tags: list[str] | None
    parameters: ValidatedParameters | None
    trial_index: int


@dataclasses.dataclass(frozen=True)
class WorkflowTaskResult(Generic[Output]):
    """A collected result for one task item."""

    output: Output
    metadata: Metadata | None = None
    tags: list[str] | None = None


@dataclasses.dataclass(frozen=True)
class WorkflowBatchingOptions:
    """Controls how workflow items are grouped into provider batches."""

    max_size: int
    max_wait_ms: int | float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_size, int) or isinstance(self.max_size, bool) or self.max_size < 1:
            raise ValueError("batching.max_size must be a positive integer")
        if self.max_wait_ms is not None and (
            isinstance(self.max_wait_ms, bool)
            or not isinstance(self.max_wait_ms, (int, float))
            or not self.max_wait_ms >= 0
        ):
            raise ValueError("batching.max_wait_ms must be a non-negative number")


@dataclasses.dataclass(frozen=True)
class WorkflowBatchItem(Generic[Input]):
    """One provider-safe item in a batch. Return its custom_id from collect."""

    custom_id: str
    item: Input

    def __getattr__(self, name: str) -> Any:
        # Expose the task/scorer item fields directly alongside custom_id.
        return getattr(self.item, name)


@dataclasses.dataclass(frozen=True)
class WorkflowBatchItemResult(Generic[Output]):
    """A result or failure for one item returned by a batch collector."""

    custom_id: str
    result: Output | None = None
    error: Any = None


@dataclasses.dataclass(frozen=True)
class WorkflowScorerItem(Generic[Input, Output, Expected]):
    """One completed case/trial passed to a workflow scorer submission callback."""

    id: str
    input: Input
    output: Output
    expected: Expected | None
    metadata: Metadata
    tags: list[str] | None
    trial_index: int


@dataclasses.dataclass(frozen=True)
class WorkflowScorerResult:
    """A collected score for one scorer item."""

    score: OneOrMoreScores


@dataclasses.dataclass(frozen=True)
class WorkflowBatchTask(Generic[Input, Output, Expected, SubmissionData]):
    """Submits multiple cases/trials in one asynchronous provider operation."""

    batching: WorkflowBatchingOptions
    submit: Callable[
        [Sequence[WorkflowBatchItem[WorkflowTaskItem[Input, Expected]]], WorkflowSubmissionContext],
        SubmissionData | Awaitable[SubmissionData],
    ]
    completion: WorkflowSubmissionCompletion[SubmissionData]
    collect: Callable[
        [SubmissionData, WorkflowSubmissionContext],
        Iterable[WorkflowBatchItemResult[WorkflowTaskResult[Output]]]
        | AsyncIterable[WorkflowBatchItemResult[WorkflowTaskResult[Output]]]
        | Awaitable[
            Iterable[WorkflowBatchItemResult[WorkflowTaskResult[Output]]]
            | AsyncIterable[WorkflowBatchItemResult[WorkflowTaskResult[Output]]]
        ],
    ]


@dataclasses.dataclass(frozen=True)
class WorkflowBatchScorer(Generic[Input, Output, Expected, SubmissionData]):
    """Submits multiple completed cases/trials in one asynchronous scorer operation."""

    name: str
    batching: WorkflowBatchingOptions
    submit: Callable[
        [Sequence[WorkflowBatchItem[WorkflowScorerItem[Input, Output, Expected]]], WorkflowSubmissionContext],
        SubmissionData | Awaitable[SubmissionData],
    ]
    completion: WorkflowSubmissionCompletion[SubmissionData]
    collect: Callable[
        [SubmissionData, WorkflowSubmissionContext],
        Iterable[WorkflowBatchItemResult[WorkflowScorerResult]]
        | AsyncIterable[WorkflowBatchItemResult[WorkflowScorerResult]]
        | Awaitable[
            Iterable[WorkflowBatchItemResult[WorkflowScorerResult]]
            | AsyncIterable[WorkflowBatchItemResult[WorkflowScorerResult]]
        ],
    ]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("WorkflowBatchScorer name must be a non-empty string")


@dataclasses.dataclass(frozen=True)
class WorkflowTask(Generic[Input, Output, Expected, SubmissionData]):
    """Submits one asynchronous provider operation per case/trial and collects its task result."""

    submit: Callable[
        [WorkflowTaskItem[Input, Expected], WorkflowSubmissionContext], SubmissionData | Awaitable[SubmissionData]
    ]
    completion: WorkflowSubmissionCompletion[SubmissionData]
    collect: Callable[
        [SubmissionData, WorkflowSubmissionContext], WorkflowTaskResult[Output] | Awaitable[WorkflowTaskResult[Output]]
    ]


@dataclasses.dataclass(frozen=True)
class WorkflowScorer(Generic[Input, Output, Expected, SubmissionData]):
    """Submits one asynchronous provider operation per case/trial and collects its score."""

    name: str
    submit: Callable[
        [WorkflowScorerItem[Input, Output, Expected], WorkflowSubmissionContext],
        SubmissionData | Awaitable[SubmissionData],
    ]
    completion: WorkflowSubmissionCompletion[SubmissionData]
    collect: Callable[
        [SubmissionData, WorkflowSubmissionContext], WorkflowScorerResult | Awaitable[WorkflowScorerResult]
    ]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("WorkflowScorer name must be a non-empty string")


@dataclasses.dataclass(frozen=True)
class WorkflowEvalStoreEntry:
    """Result of an atomic workflow-store get-or-set operation."""

    value: bytes
    created: bool


class WorkflowEvalStore(Protocol):
    """Minimal persistence interface used by workflow evaluations."""

    async def read(self, key: str) -> bytes | None: ...

    async def write(self, key: str, value: bytes) -> None: ...

    async def get_or_set(self, key: str, value: bytes) -> WorkflowEvalStoreEntry: ...

    async def reserve_batch(
        self,
        keys: Sequence[str],
        value: bytes,
        submission_key: str,
        submission_value: bytes,
        lease_key: str,
        lease_value: bytes,
        lease_ttl_ms: int,
    ) -> bool: ...

    async def acquire_lease(self, key: str, value: bytes, ttl_ms: int) -> bool: ...

    async def release_lease(self, key: str, value: bytes) -> None: ...

    async def add_to_set(self, key: str, member: str) -> None:
        """Atomically add a unique member; retain sets as long as run records."""
        ...

    async def get_set_size(self, key: str) -> int: ...


class WorkflowEvalMemoryStore:
    """Process-local workflow evaluation state, intended for tests and local runs."""

    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}
        self._sets: dict[str, set[str]] = {}
        self._expirations: dict[str, float] = {}
        self._lock = threading.Lock()

    def _expire(self, key: str) -> None:
        expires_at = self._expirations.get(key)
        if expires_at is not None and expires_at <= time.monotonic():
            self._values.pop(key, None)
            self._expirations.pop(key, None)

    async def read(self, key: str) -> bytes | None:
        with self._lock:
            self._expire(key)
            value = self._values.get(key)
            return bytes(value) if value is not None else None

    async def write(self, key: str, value: bytes) -> None:
        with self._lock:
            self._values[key] = bytes(value)
            self._expirations.pop(key, None)

    async def get_or_set(self, key: str, value: bytes) -> WorkflowEvalStoreEntry:
        with self._lock:
            self._expire(key)
            existing = self._values.get(key)
            if existing is not None:
                return WorkflowEvalStoreEntry(value=bytes(existing), created=False)
            self._values[key] = bytes(value)
            return WorkflowEvalStoreEntry(value=bytes(value), created=True)

    async def reserve_batch(
        self,
        keys: Sequence[str],
        value: bytes,
        submission_key: str,
        submission_value: bytes,
        lease_key: str,
        lease_value: bytes,
        lease_ttl_ms: int,
    ) -> bool:
        with self._lock:
            check_keys = [*keys, submission_key, lease_key]
            for key in check_keys:
                self._expire(key)
            if any(key in self._values for key in check_keys):
                return False
            for key in keys:
                self._values[key] = bytes(value)
            self._values[submission_key] = bytes(submission_value)
            self._values[lease_key] = bytes(lease_value)
            self._expirations[lease_key] = time.monotonic() + lease_ttl_ms / 1000
            return True

    async def acquire_lease(self, key: str, value: bytes, ttl_ms: int) -> bool:
        with self._lock:
            self._expire(key)
            if key in self._values:
                return False
            self._values[key] = bytes(value)
            self._expirations[key] = time.monotonic() + ttl_ms / 1000
            return True

    async def release_lease(self, key: str, value: bytes) -> None:
        with self._lock:
            self._expire(key)
            if self._values.get(key) == value:
                self._values.pop(key, None)
                self._expirations.pop(key, None)

    async def add_to_set(self, key: str, member: str) -> None:
        with self._lock:
            self._sets.setdefault(key, set()).add(member)

    async def get_set_size(self, key: str) -> int:
        with self._lock:
            return len(self._sets.get(key, set()))


class WorkflowEvalRedisStore:
    """Workflow state backed by an existing sync or async redis-py client."""

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "braintrust-eval:",
        ttl_ms: int = DEFAULT_REDIS_TTL_MS,
    ) -> None:
        if not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or ttl_ms < 1:
            raise ValueError("WorkflowEvalRedisStore ttl_ms must be a positive integer")
        self.client = client
        self.key_prefix = key_prefix
        self.ttl_ms = ttl_ms

    async def _call(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        value = await asyncio.get_running_loop().run_in_executor(None, functools.partial(method, *args, **kwargs))
        if inspect.isawaitable(value):
            return await value
        return value

    async def read(self, key: str) -> bytes | None:
        value = await self._call(self.client.get, f"{self.key_prefix}{key}")
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("ascii")
        if not isinstance(value, str):
            raise TypeError("WorkflowEvalRedisStore expected GET to return str, bytes, or None")
        return base64.b64decode(value)

    async def write(self, key: str, value: bytes) -> None:
        encoded = base64.b64encode(value).decode("ascii")
        await self._call(self.client.set, f"{self.key_prefix}{key}", encoded, px=self.ttl_ms)

    async def get_or_set(self, key: str, value: bytes) -> WorkflowEvalStoreEntry:
        redis_key = f"{self.key_prefix}{key}"
        encoded = base64.b64encode(value).decode("ascii")
        existing = await self._call(self.client.set, redis_key, encoded, px=self.ttl_ms, nx=True, get=True)
        if existing is None:
            return WorkflowEvalStoreEntry(value=bytes(value), created=True)
        if isinstance(existing, bytes):
            existing = existing.decode("ascii")
        if not isinstance(existing, str):
            raise TypeError("WorkflowEvalRedisStore expected atomic SET to return str, bytes, or None")
        return WorkflowEvalStoreEntry(value=base64.b64decode(existing), created=False)

    async def reserve_batch(
        self,
        keys: Sequence[str],
        value: bytes,
        submission_key: str,
        submission_value: bytes,
        lease_key: str,
        lease_value: bytes,
        lease_ttl_ms: int,
    ) -> bool:
        redis_keys = [f"{self.key_prefix}{key}" for key in [*keys, submission_key, lease_key]]
        encoded_claim = base64.b64encode(value).decode("ascii")
        encoded_submission = base64.b64encode(submission_value).decode("ascii")
        script = """
        for i = 1, #KEYS do
            if redis.call('EXISTS', KEYS[i]) == 1 then return 0 end
        end
        for i = 1, #KEYS - 2 do
            redis.call('SET', KEYS[i], ARGV[1], 'PX', ARGV[4])
        end
        redis.call('SET', KEYS[#KEYS - 1], ARGV[2], 'PX', ARGV[4])
        redis.call('SET', KEYS[#KEYS], ARGV[3], 'PX', ARGV[5])
        return 1
        """
        claimed = await self._call(
            self.client.eval,
            script,
            len(redis_keys),
            *redis_keys,
            encoded_claim,
            encoded_submission,
            base64.b64encode(lease_value).decode("ascii"),
            self.ttl_ms,
            lease_ttl_ms,
        )
        if claimed not in (0, 1, b"0", b"1"):
            raise TypeError("WorkflowEvalRedisStore expected atomic batch reservation to return 0 or 1")
        return claimed in (1, b"1")

    async def acquire_lease(self, key: str, value: bytes, ttl_ms: int) -> bool:
        result = await self._call(
            self.client.set, f"{self.key_prefix}{key}", base64.b64encode(value).decode("ascii"), px=ttl_ms, nx=True
        )
        return result is True or result == b"OK" or result == "OK"

    async def release_lease(self, key: str, value: bytes) -> None:
        await self._call(
            self.client.eval,
            "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end",
            1,
            f"{self.key_prefix}{key}",
            base64.b64encode(value).decode("ascii"),
        )

    async def add_to_set(self, key: str, member: str) -> None:
        await self._call(
            self.client.eval,
            "redis.call('SADD', KEYS[1], ARGV[1]); redis.call('PEXPIRE', KEYS[1], ARGV[2]); return 1",
            1,
            f"{self.key_prefix}{key}",
            member,
            self.ttl_ms,
        )

    async def get_set_size(self, key: str) -> int:
        return int(await self._call(self.client.scard, f"{self.key_prefix}{key}"))


@dataclasses.dataclass(frozen=True)
class WorkflowEvalPending:
    """Counts of submitted provider submissions awaiting completion."""

    poll: int
    webhook: int


@dataclasses.dataclass(frozen=True)
class WorkflowEvalWaitingResult:
    run_id: str
    pending: WorkflowEvalPending
    status: Literal["waiting"] = dataclasses.field(default="waiting", init=False)


@dataclasses.dataclass(frozen=True)
class WorkflowEvalCompletedResult:
    run_id: str
    pending: WorkflowEvalPending
    summary: ExperimentSummary
    status: Literal["completed"] = dataclasses.field(default="completed", init=False)


WorkflowEvalResult = WorkflowEvalWaitingResult | WorkflowEvalCompletedResult


@dataclasses.dataclass(frozen=True)
class _WorkflowEvalConfig(Generic[Input, Output, Expected]):
    project_name: str
    store: WorkflowEvalStore
    data: EvalData[Input, Expected]
    task: (
        EvalTask[Input, Output, Expected]
        | WorkflowTask[Input, Output, Expected, Any]
        | WorkflowBatchTask[Input, Output, Expected, Any]
    )
    scores: Sequence[
        EvalScorer[Input, Output, Expected]
        | WorkflowScorer[Input, Output, Expected, Any]
        | WorkflowBatchScorer[Input, Output, Expected, Any]
    ]
    classifiers: Sequence[EvalClassifier[Input, Output, Expected]]
    case_id: Callable[[EvalCase[Input, Expected]], str | Awaitable[str]] | None
    experiment_name: str | None
    trial_count: int
    max_concurrency: int
    metadata: Metadata | None
    tags: Sequence[str] | None
    is_public: bool
    project_id: str | None
    base_experiment_name: str | None
    base_experiment_id: str | None
    git_metadata_settings: GitMetadataSettings | None
    repo_info: RepoInfo | None
    description: str | None
    summarize_scores: bool
    parameters: EvalParameters | RemoteEvalParameters | None
    state: BraintrustState | None


class WorkflowEval(Generic[Input, Output, Expected]):
    """A workflow evaluation definition that can be started and resumed."""

    def __init__(self, config: _WorkflowEvalConfig[Input, Output, Expected]) -> None:
        self._config = config

    async def start(
        self, parameters: Mapping[str, Any] | None = None, *, no_send_logs: bool = False
    ) -> WorkflowEvalResult:
        return await _WorkflowEvalRunner(self._config).start(parameters, no_send_logs=no_send_logs)

    async def status(self, run_id: str) -> WorkflowEvalResult:
        return await _WorkflowEvalRunner(self._config).status(run_id)

    async def poll(self, run_id: str) -> WorkflowEvalResult:
        return await _WorkflowEvalRunner(self._config).poll(run_id)

    async def process_submission_result(
        self, run_id: str | None = None, *, submission_id: str | None = None, external_id: str | None = None
    ) -> WorkflowEvalResult:
        return await _WorkflowEvalRunner(self._config).process_submission_result(
            run_id, submission_id=submission_id, external_id=external_id
        )


def define_workflow_eval(
    project_name: str,
    *,
    store: WorkflowEvalStore,
    data: EvalData[Input, Expected],
    task: EvalTask[Input, Output, Expected]
    | WorkflowTask[Input, Output, Expected, Any]
    | WorkflowBatchTask[Input, Output, Expected, Any],
    scores: Sequence[
        EvalScorer[Input, Output, Expected]
        | WorkflowScorer[Input, Output, Expected, Any]
        | WorkflowBatchScorer[Input, Output, Expected, Any]
    ]
    | None = None,
    classifiers: Sequence[EvalClassifier[Input, Output, Expected]] | None = None,
    case_id: Callable[[EvalCase[Input, Expected]], str | Awaitable[str]] | None = None,
    experiment_name: str | None = None,
    trial_count: int = 1,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    metadata: Metadata | None = None,
    tags: Sequence[str] | None = None,
    is_public: bool = False,
    project_id: str | None = None,
    base_experiment_name: str | None = None,
    base_experiment_id: str | None = None,
    git_metadata_settings: GitMetadataSettings | None = None,
    repo_info: RepoInfo | None = None,
    description: str | None = None,
    summarize_scores: bool = True,
    parameters: EvalParameters | RemoteEvalParameters | None = None,
    state: BraintrustState | None = None,
) -> WorkflowEval[Input, Output, Expected]:
    """Define an experimental evaluation that can pause across provider submissions.

    WorkflowTask and WorkflowScorer submit one case/trial; WorkflowBatchTask and
    WorkflowBatchScorer submit multiple cases/trials in one provider operation.
    Submission data and collected values must be JSON serializable.
    Scorers and classifiers start once their case's task is persisted and logged.

    Each poll checks existing submissions once; newly submitted work waits for a
    later invocation. Provider callbacks use max_concurrency (default 10).
    Callback failures are raised after independent work advances.
    Webhooks resume via process_submission_result with submission_id or external_id.
    Collection callbacks must tolerate replay, including concurrent delivery.
    """
    if not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count < 1:
        raise ValueError("trial_count must be a positive integer")
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    return WorkflowEval(
        _WorkflowEvalConfig(
            project_name=project_name,
            store=store,
            data=data,
            task=task,
            scores=list(scores or []),
            classifiers=list(classifiers or []),
            case_id=case_id,
            experiment_name=experiment_name,
            trial_count=trial_count,
            max_concurrency=max_concurrency,
            metadata=metadata,
            tags=tags,
            is_public=is_public,
            project_id=project_id,
            base_experiment_name=base_experiment_name,
            base_experiment_id=base_experiment_id,
            git_metadata_settings=git_metadata_settings,
            repo_info=repo_info,
            description=description,
            summarize_scores=summarize_scores,
            parameters=parameters,
            state=state,
        )
    )


def _json_bytes(value: Any) -> bytes:
    return bt_dumps(value).encode("utf-8")


def _decode(value: bytes) -> Any:
    return bt_loads(value.decode("utf-8"))


def _json_value(value: Any) -> Any:
    """Validate and normalize a value at the workflow JSON boundary."""
    return _decode(_json_bytes(value))


def _stable_hex(*parts: str, length: int) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:length]


def _stable_uuid(*parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16], version=5))


def _stage_kind(prefix: str, name: str) -> str:
    return f"{prefix}-{_stable_hex(name, length=32)}"


def _scorer_args(case: Mapping[str, Any], task_result: Mapping[str, Any]) -> dict[str, Any]:
    datum = case["datum"]
    return {
        "input": datum["input"],
        "output": task_result["output"],
        "expected": datum.get("expected"),
        "metadata": task_result["metadata"],
        "id": case["case_id"],
        "tags": task_result.get("tags"),
    }


def _score_fields(result: ScoreLike) -> dict[str, Any]:
    return {key: value for key, value in result.as_dict().items() if key not in ("metadata", "name")}


def _raise_callback_errors(errors: Sequence[BaseException]) -> None:
    if len(errors) == 1:
        raise errors[0]
    raise RuntimeError(
        "Workflow submission callbacks failed: " + "; ".join(str(error) for error in errors)
    ) from errors[0]


class _WorkflowEvalRunner(Generic[Input, Output, Expected]):
    def __init__(self, config: _WorkflowEvalConfig[Input, Output, Expected]) -> None:
        self.config = config
        self.store = config.store
        self._callback_semaphore = asyncio.Semaphore(config.max_concurrency)
        self.eval_name = config.experiment_name or config.project_name
        self.definition_key = _stable_hex(config.project_name, self.eval_name, length=32)

    def _key(self, run_id: str, kind: str, identifier: str | None = None) -> str:
        suffix = f"/{identifier}" if identifier is not None else ""
        return f"{_SCHEMA_PREFIX}/{self.definition_key}/{run_id}/{kind}{suffix}"

    def _batch_key(self, run_id: str, kind: str, identifier: str) -> str:
        # Batch reservation scripts touch several keys, which must share a Redis Cluster slot.
        return f"{_SCHEMA_PREFIX}/{self.definition_key}/{{{run_id}}}/{kind}/{identifier}"

    async def _read_submission(self, run_id: str, submission_id: str) -> Any | None:
        batch_record = await self.store.read(self._batch_key(run_id, "submission", submission_id))
        if batch_record is not None:
            return _decode(batch_record)
        return await self._read(run_id, "submission", submission_id)

    async def _read_batch_link(self, run_id: str, stage: str, item_id: str) -> str | None:
        value = await self.store.read(self._batch_key(run_id, "batch-link", f"{stage}/{item_id}"))
        if value is None:
            value = await self.store.read(self._key(run_id, "batch-link", f"{stage}/{item_id}"))
        return _decode(value) if value is not None else None

    async def _write_submission(self, run_id: str, submission: Mapping[str, Any]) -> None:
        if submission.get("batch"):
            await self.store.write(self._batch_key(run_id, "submission", submission["id"]), _json_bytes(submission))
        else:
            await self._write(run_id, "submission", submission, submission["id"])

    def _external_key(self, external_id: str) -> str:
        return f"{_SCHEMA_PREFIX}/{self.definition_key}/external/{_stable_hex(external_id, length=32)}"

    async def _read(self, run_id: str, kind: str, identifier: str | None = None) -> Any | None:
        value = await self.store.read(self._key(run_id, kind, identifier))
        return _decode(value) if value is not None else None

    async def _required(self, run_id: str, kind: str, identifier: str | None = None) -> Any:
        value = await self._read(run_id, kind, identifier)
        if value is None:
            target = f" {identifier!r}" if identifier is not None else ""
            raise ValueError(f"Unknown workflow evaluation {kind}{target} for run {run_id!r}")
        return value

    async def _write(self, run_id: str, kind: str, value: Any, identifier: str | None = None) -> None:
        await self.store.write(self._key(run_id, kind, identifier), _json_bytes(value))

    async def _record_external_locator(self, run_id: str, submission_id: str, external_id: str | None) -> None:
        if external_id is not None:
            key = self._external_key(external_id)
            locator = {"run_id": run_id, "submission_id": submission_id}
            entry = await self.store.get_or_set(key, _json_bytes(locator))
            if not entry.created:
                existing = _decode(entry.value)
                if existing is not None and existing != locator:
                    # Ambiguous provider IDs require the caller to supply run_id.
                    await self.store.write(key, _json_bytes(None))

    async def _claim(self, run_id: str, action: str) -> bool:
        result = await self.store.get_or_set(self._key(run_id, "claim", action), b"1")
        return result.created

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        async with self._callback_semaphore:
            return await await_or_run(asyncio.get_running_loop(), fn, *args, **kwargs)

    async def _flush_logs(self) -> None:
        state = self.config.state or _internal_get_global_state()
        background_logger = state.global_bg_logger()
        await asyncio.get_running_loop().run_in_executor(None, background_logger.flush)

    def _parameters(self, run: Mapping[str, Any]) -> ValidatedParameters | None:
        raw = run.get("parameters")
        if self.config.parameters is None:
            return cast(ValidatedParameters | None, raw)
        return validate_parameters(raw or {}, self.config.parameters)

    def _experiment(self, run: Mapping[str, Any]) -> Experiment | None:
        if run["no_send_logs"]:
            return None
        experiment_parameters = None
        if isinstance(self.config.parameters, RemoteEvalParameters) and self.config.parameters.id is not None:
            experiment_parameters = {"id": self.config.parameters.id}
            if self.config.parameters.version is not None:
                experiment_parameters["version"] = self.config.parameters.version
        dataset = self.config.data if isinstance(self.config.data, Dataset) else None
        return init_experiment(
            project_name=self.config.project_name if self.config.project_id is None else None,
            project_id=self.config.project_id,
            experiment_name=run["experiment_name"],
            description=self.config.description,
            metadata=self.config.metadata,
            tags=self.config.tags,
            is_public=self.config.is_public,
            update=True,
            base_experiment=self.config.base_experiment_name,
            base_experiment_id=self.config.base_experiment_id,
            git_metadata_settings=self.config.git_metadata_settings,
            repo_info=self.config.repo_info,
            dataset=dataset,
            parameters=experiment_parameters,
            state=self.config.state,
        )

    async def _resolve_data(self, experiment: Experiment | None) -> list[EvalCase[Input, Expected]]:
        data: Any = self.config.data
        if inspect.isclass(data):
            data = data()
        if isinstance(data, BaseExperiment):
            if experiment is None:
                raise ValueError("Cannot use BaseExperiment without sending logs")
            base_name = data.name
            if base_name is None:
                base = experiment.fetch_base_experiment()
                if base is None:
                    raise ValueError("BaseExperiment failed to resolve a base experiment")
                base_name = base.name
            data = _init_experiment(
                project=self.config.project_name if self.config.project_id is None else None,
                project_id=self.config.project_id,
                experiment=base_name,
                open=True,
                set_current=False,
                state=self.config.state,
            ).as_dataset()
        elif callable(data) and not isinstance(data, Dataset):
            data = await self._call(data)
        if inspect.isawaitable(data):
            data = await data

        values: list[Any] = []
        if isinstance(data, AsyncIterable):
            async for value in data:
                values.append(value)
        else:
            values.extend(data)
        return [value if isinstance(value, EvalCase) else EvalCase.from_dict(value) for value in values]

    def _uses_submission_processor(self) -> bool:
        return isinstance(self.config.task, (WorkflowTask, WorkflowBatchTask)) or any(
            isinstance(score, (WorkflowScorer, WorkflowBatchScorer)) for score in self.config.scores
        )

    async def _save_completed(self, run: Mapping[str, Any], summary: ExperimentSummary) -> WorkflowEvalCompletedResult:
        completed = {**run, "status": "completed", "summary": _json_value(summary.as_dict())}
        await self._write(run["run_id"], "run", completed)
        return WorkflowEvalCompletedResult(
            run_id=run["run_id"], pending=WorkflowEvalPending(poll=0, webhook=0), summary=summary
        )

    async def _run_ordinary_evaluator(
        self,
        run: Mapping[str, Any],
        data: list[EvalCase[Input, Expected]],
        experiment: Experiment | None,
        parameters: ValidatedParameters,
    ) -> WorkflowEvalCompletedResult:
        evaluator = Evaluator(
            project_name=self.config.project_name,
            eval_name=self.eval_name,
            data=data,
            task=cast(EvalTask[Input, Output, Expected], self.config.task),
            scores=cast(Sequence[EvalScorer[Input, Output, Expected]], self.config.scores),
            classifiers=list(self.config.classifiers),
            experiment_name=run["experiment_name"],
            metadata=self.config.metadata,
            tags=self.config.tags,
            trial_count=self.config.trial_count,
            max_concurrency=self.config.max_concurrency,
            is_public=self.config.is_public,
            update=True,
            project_id=self.config.project_id,
            base_experiment_name=self.config.base_experiment_name,
            base_experiment_id=self.config.base_experiment_id,
            git_metadata_settings=self.config.git_metadata_settings,
            repo_info=self.config.repo_info,
            description=self.config.description,
            summarize_scores=self.config.summarize_scores,
            parameters=self.config.parameters,
            parameter_values=cast(dict[str, Any], parameters),
        )
        result = await run_evaluator(
            experiment,
            evaluator,
            position=None,
            filters=[],
            state=self.config.state,
        )
        return await self._save_completed(run, result.summary)

    async def _persist_cases(self, run: dict[str, Any], data: Sequence[EvalCase[Input, Expected]]) -> None:
        seen_case_ids: set[str] = set()
        item_ids: list[str] = []
        for datum in data:
            case_id = datum.id
            if not case_id and self.config.case_id is not None:
                case_id = await self._call(self.config.case_id, datum)
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("Every workflow evaluation case must have a non-empty id or be assigned by case_id")
            if case_id in seen_case_ids:
                raise ValueError(f"Workflow evaluation case IDs must be unique; found duplicate {case_id!r}")
            seen_case_ids.add(case_id)

            trial_count = datum.trial_count if datum.trial_count is not None else self.config.trial_count
            if not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count < 1:
                raise ValueError(f"trial_count for case {case_id!r} must be a positive integer")
            for trial_index in range(trial_count):
                item_id = f"{case_id}:trial:{trial_index}"
                item_ids.append(item_id)
                await self._write(
                    run["run_id"],
                    "case",
                    {
                        "id": item_id,
                        "case_id": case_id,
                        "trial_index": trial_index,
                        "datum": _json_value(
                            {field.name: getattr(datum, field.name) for field in dataclasses.fields(datum)}
                        ),
                        "metadata": _json_value(dict(datum.metadata or {})),
                        "tags": list(datum.tags) if datum.tags is not None else None,
                    },
                    item_id,
                )
        run["case_ids"] = item_ids
        await self._write(run["run_id"], "run", run)

    async def start(self, parameters: Mapping[str, Any] | None, *, no_send_logs: bool) -> WorkflowEvalResult:
        validated_parameters = validate_parameters(parameters or {}, self.config.parameters)
        run_id = str(uuid.uuid4())
        run = {
            "run_id": run_id,
            "experiment_name": self.config.experiment_name or f"{self.eval_name}-{run_id}",
            "no_send_logs": no_send_logs,
            "parameters": _json_value(validated_parameters),
            "status": "running",
            "case_ids": [],
        }
        created = await self.store.get_or_set(self._key(run_id, "run"), _json_bytes(run))
        if not created.created:
            raise RuntimeError(f"Workflow evaluation run ID collision: {run_id}")

        experiment = self._experiment(run)
        data = await self._resolve_data(experiment)
        if not self._uses_submission_processor():
            return await self._run_ordinary_evaluator(run, data, experiment, validated_parameters)

        await self._persist_cases(run, data)
        return await self._advance(run, experiment=experiment)

    async def status(self, run_id: str) -> WorkflowEvalResult:
        run = await self._required(run_id, "run")
        if run["status"] == "completed":
            return WorkflowEvalCompletedResult(
                run_id=run_id,
                pending=WorkflowEvalPending(poll=0, webhook=0),
                summary=ExperimentSummary.from_dict_deep(run["summary"]),
            )
        return await self._waiting_status(run)

    async def poll(self, run_id: str) -> WorkflowEvalResult:
        run = await self._required(run_id, "run")
        if run["status"] == "completed":
            return await self.status(run_id)
        # Snapshot first: newly submitted downstream work waits for the next poll call.
        submissions = await self._submission_records(run)

        async def poll_submission(submission: dict[str, Any], processor: Any) -> None:
            if submission["status"] == "external_id_pending":
                completion = processor.completion
                assert isinstance(completion, WorkflowSubmissionCompletionWebhook)
                context = WorkflowSubmissionContext(run_id=run_id, submission_id=submission["id"])
                external_id = await self._call(completion.get_external_id, submission["submission_data"], context)
                if not isinstance(external_id, str) or not external_id.strip():
                    raise ValueError("Submission webhook get_external_id must return a non-empty string")
                submission["external_id"] = external_id
                submission["status"] = "submitted"
                await self._write_submission(run_id, submission)
                await self._record_external_locator(run_id, submission["id"], external_id)
                await self._persist_submission_progress(run_id, submission)
                return
            if submission["status"] == "failed":
                if not submission.get("batch"):
                    return
                lease_key = self._batch_key(run_id, "retry-lease", submission["id"])
                lease_value = uuid.uuid4().hex.encode()
                if not await self.store.acquire_lease(lease_key, lease_value, _BATCH_LEASE_TTL_MS):
                    return
                try:
                    current = await self._read_submission(run_id, submission["id"])
                    if current is None or current["status"] != "failed":
                        return
                    submission = current
                    context = WorkflowSubmissionContext(run_id=run_id, submission_id=submission["id"])
                    if submission["kind"] == "task":
                        values = [
                            await self._task_item(run, item_id, self._parameters(run))
                            for item_id in submission["item_ids"]
                        ]
                    else:
                        values = [await self._scorer_item(run, item_id) for item_id in submission["item_ids"]]
                    items = [WorkflowBatchItem(f"item-{index}", value) for index, value in enumerate(values)]
                    completion = processor.completion
                    try:
                        submission_data = _json_value(await self._call(processor.submit, items, context))
                    except Exception:
                        submission["attempts"] = submission.get("attempts", 0) + 1
                        await self._write_submission(run_id, submission)
                        raise
                    submission.update(
                        submission_data=submission_data,
                        external_id=None,
                        mode=completion.mode,
                        status="external_id_pending"
                        if isinstance(completion, WorkflowSubmissionCompletionWebhook)
                        else "submitted",
                    )
                    await self._write_submission(run_id, submission)
                    await self._persist_submission_progress(run_id, submission)
                    if isinstance(completion, WorkflowSubmissionCompletionWebhook):
                        external_id = await self._call(completion.get_external_id, submission_data, context)
                        if not isinstance(external_id, str) or not external_id.strip():
                            raise ValueError("Submission webhook get_external_id must return a non-empty string")
                        submission["external_id"] = external_id
                        submission["status"] = "submitted"
                        await self._write_submission(run_id, submission)
                        await self._record_external_locator(run_id, submission["id"], external_id)
                        await self._persist_submission_progress(run_id, submission)
                    return
                finally:
                    await self.store.release_lease(lease_key, lease_value)
            completion = processor.completion
            assert isinstance(completion, WorkflowSubmissionCompletionPoll)
            context = WorkflowSubmissionContext(run_id=run_id, submission_id=submission["id"])
            result = await self._call(completion.poll, submission["submission_data"], context)
            if isinstance(result, Mapping):
                result = WorkflowSubmissionPoll(**result)
            if not isinstance(result, WorkflowSubmissionPoll):
                raise TypeError("Submission poll callback must return WorkflowSubmissionPoll")
            if result.status not in ("pending", "complete", "failed"):
                raise ValueError(f"Submission poll callback returned unsupported status {result.status!r}")
            if result.status == "failed":
                if isinstance(result.error, BaseException):
                    raise result.error
                raise RuntimeError(str(result.error))
            if result.status == "complete":
                await self._collect_submission(run, submission, processor)

        results = await asyncio.gather(
            *(
                poll_submission(submission, processor)
                for submission, processor in submissions
                if submission["status"] == "failed"
                or submission["status"] == "external_id_pending"
                or (submission["status"] == "submitted" and submission["mode"] == "poll")
            ),
            return_exceptions=True,
        )
        # Persist independent progress before reporting provider callback failures.
        errors = [result for result in results if isinstance(result, BaseException)]
        try:
            status = await self._advance(run)
        except Exception as error:
            errors.append(error)
        if errors:
            _raise_callback_errors(errors)
        return status

    async def process_submission_result(
        self, run_id: str | None, *, submission_id: str | None, external_id: str | None
    ) -> WorkflowEvalResult:
        if not submission_id and not external_id:
            raise ValueError("process_submission_result requires submission_id or external_id")
        if run_id is None:
            if external_id is None:
                raise ValueError("run_id is required when submission_id is used")
            locator = await self.store.read(self._external_key(external_id))
            resolved = _decode(locator) if locator is not None else None
            if not isinstance(resolved, Mapping):
                raise ValueError("No submission matches this external_id")
            run_id = resolved["run_id"]
            submission_id = submission_id or resolved["submission_id"]
        run = await self._required(run_id, "run")
        external_submission_id = await self._read(run_id, "external", external_id) if external_id is not None else None
        if submission_id and external_submission_id and submission_id != external_submission_id:
            raise ValueError("submission_id and external_id identify different submissions")
        submission_id = submission_id or external_submission_id
        submission = await self._read_submission(run_id, submission_id) if submission_id else None
        if submission is None:
            raise ValueError("No submission matches this result")
        if external_id is not None and submission.get("external_id") != external_id:
            raise ValueError("submission_id and external_id identify different submissions")
        if run["status"] == "completed":
            return await self.status(run_id)
        processor = (
            self.config.task
            if submission["kind"] == "task"
            else next(
                score
                for score in self.config.scores
                if isinstance(score, (WorkflowScorer, WorkflowBatchScorer)) and score.name == submission["scorer_name"]
            )
        )
        await self._collect_submission(run, submission, processor)
        # Only the completed case needs advancing; other cases can still be pending.
        return await self._advance(run, item_ids=submission.get("item_ids", [submission.get("item_id")]))

    def _submission_specs(self, run: Mapping[str, Any]) -> list[tuple[dict[str, Any], Any]]:
        specs: list[tuple[dict[str, Any], Any]] = []
        for item_id in run["case_ids"]:
            if isinstance(self.config.task, WorkflowTask):
                specs.append(
                    (
                        {
                            "id": f"submission-{_stable_hex(run['run_id'], 'task', item_id, length=32)}",
                            "kind": "task",
                            "item_id": item_id,
                        },
                        self.config.task,
                    )
                )
            for scorer in self.config.scores:
                if isinstance(scorer, WorkflowScorer):
                    specs.append(
                        (
                            {
                                "id": f"submission-{_stable_hex(run['run_id'], 'score', scorer.name, item_id, length=32)}",
                                "kind": "score",
                                "scorer_name": scorer.name,
                                "item_id": item_id,
                            },
                            scorer,
                        )
                    )
        return specs

    async def _submission_records(self, run: Mapping[str, Any]) -> list[tuple[dict[str, Any], Any]]:
        records: list[tuple[dict[str, Any], Any]] = []
        seen: set[str] = set()
        for spec, processor in self._submission_specs(run):
            record = await self._read_submission(run["run_id"], spec["id"])
            if record is not None:
                await self._persist_submission_progress(run["run_id"], record)
                records.append((record, processor))
                seen.add(record["id"])
        batch_processors: dict[str, Any] = {}
        if isinstance(self.config.task, WorkflowBatchTask):
            batch_processors["task"] = self.config.task
        for scorer in self.config.scores:
            if isinstance(scorer, WorkflowBatchScorer):
                batch_processors[f"score:{scorer.name}"] = scorer
        for item_id in run["case_ids"]:
            for stage, processor in batch_processors.items():
                submission_id = await self._read_batch_link(run["run_id"], stage, item_id)
                if not submission_id or submission_id in seen:
                    continue
                record = await self._read_submission(run["run_id"], submission_id)
                if record is not None:
                    await self._persist_submission_progress(run["run_id"], record)
                    records.append((record, processor))
                    seen.add(submission_id)
        return records

    async def _case(self, run_id: str, item_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], await self._required(run_id, "case", item_id))

    async def _task_item(
        self, run: Mapping[str, Any], item_id: str, parameters: ValidatedParameters | None
    ) -> WorkflowTaskItem[Any, Any]:
        case = await self._case(run["run_id"], item_id)
        datum = case["datum"]
        return WorkflowTaskItem(
            id=item_id,
            input=datum["input"],
            expected=datum.get("expected"),
            metadata=dict(case["metadata"]),
            tags=case.get("tags"),
            parameters=parameters,
            trial_index=case["trial_index"],
        )

    async def _scorer_item(self, run: Mapping[str, Any], item_id: str) -> WorkflowScorerItem[Any, Any, Any]:
        case = await self._case(run["run_id"], item_id)
        task = await self._required(run["run_id"], "task-result", item_id)
        datum = case["datum"]
        return WorkflowScorerItem(
            id=item_id,
            input=datum["input"],
            output=task["output"],
            expected=datum.get("expected"),
            metadata=dict(task["metadata"]),
            tags=task.get("tags"),
            trial_index=case["trial_index"],
        )

    async def _submit_submission(self, run: Mapping[str, Any], spec: dict[str, Any], processor: Any) -> None:
        run_id = run["run_id"]
        existing = await self._read(run_id, "submission", spec["id"])
        if existing is not None:
            await self._persist_submission_progress(run_id, existing)
            return
        if not await self._claim(run_id, f"submit:{spec['id']}"):
            return
        parameters = self._parameters(run)
        if spec["kind"] == "task":
            item = await self._task_item(run, spec["item_id"], parameters)
        else:
            item = await self._scorer_item(run, spec["item_id"])
        context = WorkflowSubmissionContext(run_id=run_id, submission_id=spec["id"])
        submission_data = await self._call(processor.submit, item, context)
        submission_data = _json_value(submission_data)
        completion = processor.completion
        external_id = None
        if isinstance(completion, WorkflowSubmissionCompletionWebhook):
            external_id = await self._call(completion.get_external_id, submission_data, context)
            if not isinstance(external_id, str) or not external_id.strip():
                raise ValueError("Submission webhook get_external_id must return a non-empty string")
        submission = {
            **spec,
            "submission_data": submission_data,
            "external_id": external_id,
            "mode": completion.mode,
            "status": "submitted",
        }
        await self._write(run_id, "submission", submission, spec["id"])
        await self._record_external_locator(run_id, spec["id"], external_id)
        await self._persist_submission_progress(run_id, submission)

    async def _schedule_batch_stage(
        self,
        run: Mapping[str, Any],
        processor: WorkflowBatchTask[Any, Any, Any, Any] | WorkflowBatchScorer[Any, Any, Any, Any],
        *,
        kind: Literal["task", "score"],
    ) -> None:
        run_id = run["run_id"]
        scorer_name = processor.name if kind == "score" else None
        stage = f"score:{scorer_name}" if scorer_name else "task"

        async def find_candidates() -> list[str]:
            result: list[str] = []
            for item_id in run["case_ids"]:
                if await self._read_batch_link(run_id, stage, item_id) is not None:
                    continue
                if kind == "task":
                    if await self._read(run_id, "task-result", item_id) is None:
                        result.append(item_id)
                else:
                    task_result = await self._read(run_id, "task-result", item_id)
                    score_result = await self._read(
                        run_id, _stage_kind("score-result", cast(str, scorer_name)), item_id
                    )
                    if task_result is not None and task_result.get("error") is None and score_result is None:
                        result.append(item_id)
            return result

        candidates = await find_candidates()
        if not candidates:
            return

        max_size = processor.batching.max_size
        submit_partial = kind == "task"
        if kind == "score":
            completed_tasks = sum(
                [await self._read(run_id, "task-result", item_id) is not None for item_id in run["case_ids"]]
            )
            if completed_tasks == len(run["case_ids"]):
                submit_partial = True
            elif processor.batching.max_wait_ms is not None:
                full_item_count = (len(candidates) // max_size) * max_size
                remainder = candidates[full_item_count:]
                if remainder:
                    window_key = f"{stage}/{remainder[0]}"
                    first_wait = await self.store.get_or_set(
                        self._key(run_id, "batch-wait", window_key), str(time.time()).encode()
                    )
                    started = float(first_wait.value.decode())
                    submit_partial = time.time() - started >= processor.batching.max_wait_ms / 1000
            if len(candidates) < max_size and not submit_partial:
                return

        def make_batches(item_ids: list[str]) -> list[list[str]]:
            batches = [item_ids[index : index + max_size] for index in range(0, len(item_ids), max_size)]
            if not submit_partial and batches and len(batches[-1]) < max_size:
                batches.pop()
            return batches

        async def submit_batch(item_ids: list[str]) -> bool:
            submission_id = f"submission-{_stable_hex(run_id, stage, *item_ids, length=32)}"
            spec: dict[str, Any] = {
                "id": submission_id,
                "kind": kind,
                "item_ids": item_ids,
                "batch": True,
            }
            if scorer_name is not None:
                spec["scorer_name"] = scorer_name
            claim_keys = [self._batch_key(run_id, "batch-link", f"{stage}/{item_id}") for item_id in item_ids]
            submission_key = self._batch_key(run_id, "submission", submission_id)
            lease_key = self._batch_key(run_id, "retry-lease", submission_id)
            lease_value = uuid.uuid4().hex.encode()
            completion = processor.completion
            reservation = {
                **spec,
                "submission_data": None,
                "external_id": None,
                "mode": completion.mode,
                "status": "failed",
                "attempts": 0,
            }
            if not await self.store.reserve_batch(
                claim_keys,
                _json_bytes(submission_id),
                submission_key,
                _json_bytes(reservation),
                lease_key,
                lease_value,
                _BATCH_LEASE_TTL_MS,
            ):
                return False
            try:
                context = WorkflowSubmissionContext(run_id=run_id, submission_id=submission_id)
                if kind == "task":
                    values = [await self._task_item(run, item_id, self._parameters(run)) for item_id in item_ids]
                else:
                    values = [await self._scorer_item(run, item_id) for item_id in item_ids]
                submitted_items = [WorkflowBatchItem(f"item-{index}", value) for index, value in enumerate(values)]
                try:
                    submission_data = _json_value(await self._call(processor.submit, submitted_items, context))
                except Exception:
                    reservation["attempts"] = 1
                    await self._write_submission(run_id, reservation)
                    await self._persist_submission_progress(run_id, reservation)
                    raise
                reservation.update(
                    submission_data=submission_data,
                    status="external_id_pending"
                    if isinstance(completion, WorkflowSubmissionCompletionWebhook)
                    else "submitted",
                )
                await self._write_submission(run_id, reservation)
                await self._persist_submission_progress(run_id, reservation)
                external_id = None
                if isinstance(completion, WorkflowSubmissionCompletionWebhook):
                    external_id = await self._call(completion.get_external_id, submission_data, context)
                    if not isinstance(external_id, str) or not external_id.strip():
                        raise ValueError("Submission webhook get_external_id must return a non-empty string")
                    reservation["external_id"] = external_id
                    reservation["status"] = "submitted"
                    await self._write_submission(run_id, reservation)
                await self._record_external_locator(run_id, submission_id, external_id)
                await self._persist_submission_progress(run_id, reservation)
                return True
            finally:
                await self.store.release_lease(lease_key, lease_value)

        batches = make_batches(candidates)
        while batches:
            results = await asyncio.gather(*(submit_batch(item_ids) for item_ids in batches), return_exceptions=True)
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                _raise_callback_errors(errors)
            if all(result is True for result in results):
                break
            candidates = await find_candidates()
            batches = make_batches(candidates)

    async def _persist_submission_progress(self, run_id: str, submission: Mapping[str, Any]) -> None:
        if submission.get("external_id") is not None:
            await self._write(run_id, "external", submission["id"], submission["external_id"])
        await self.store.add_to_set(self._key(run_id, "progress", f"{submission['mode']}/submitted"), submission["id"])
        if submission["status"] == "complete":
            await self.store.add_to_set(
                self._key(run_id, "progress", f"{submission['mode']}/complete"), submission["id"]
            )

    async def _collect_submission(self, run: Mapping[str, Any], submission: dict[str, Any], processor: Any) -> None:
        run_id = run["run_id"]
        if submission["status"] != "complete":
            context = WorkflowSubmissionContext(run_id=run_id, submission_id=submission["id"])
            if submission.get("batch"):
                await self._collect_batch_submission(run, submission, processor, context)
                submission["status"] = "complete"
                await self._write_submission(run_id, submission)
                await self._persist_submission_progress(run_id, submission)
                return
            result = await self._call(processor.collect, submission["submission_data"], context)
            result_type = WorkflowTaskResult if submission["kind"] == "task" else WorkflowScorerResult
            if isinstance(result, Mapping):
                result = result_type(**result)
            if not isinstance(result, result_type):
                raise TypeError(f"Submission collect callback must return a single {result_type.__name__}")
            item_id = submission["item_id"]
            if submission["kind"] == "task":
                case = await self._case(run_id, item_id)
                metadata = {**case["metadata"], **(result.metadata or {})}
                tags = result.tags if result.tags is not None else case.get("tags")
                await self._write(
                    run_id,
                    "task-result",
                    {
                        "output": _json_value(result.output),
                        "metadata": metadata,
                        "tags": tags,
                    },
                    item_id,
                )
            else:
                await self._write(
                    run_id, _stage_kind("score-result", submission["scorer_name"]), _json_value(result.score), item_id
                )
            submission["status"] = "complete"
            await self._write_submission(run_id, submission)
        # Replay repairs an interrupted progress update without collecting again.
        await self._persist_submission_progress(run_id, submission)

    async def _collect_batch_submission(
        self,
        run: Mapping[str, Any],
        submission: dict[str, Any],
        processor: WorkflowBatchTask[Any, Any, Any, Any] | WorkflowBatchScorer[Any, Any, Any, Any],
        context: WorkflowSubmissionContext,
    ) -> None:
        run_id = run["run_id"]
        async with self._callback_semaphore:
            if inspect.isasyncgenfunction(processor.collect):
                collected = processor.collect(submission["submission_data"], context)
            else:
                collected = await await_or_run(
                    asyncio.get_running_loop(), processor.collect, submission["submission_data"], context
                )
            if inspect.isawaitable(collected):
                collected = await collected
            if isinstance(collected, AsyncIterable):
                entries = [entry async for entry in collected]
            else:
                entries = list(collected)
        item_ids = submission["item_ids"]
        expected = {f"item-{index}": item_id for index, item_id in enumerate(item_ids)}
        outcomes: dict[str, WorkflowBatchItemResult[Any]] = {}
        for entry in entries:
            if isinstance(entry, Mapping):
                entry = WorkflowBatchItemResult(**entry)
            if not isinstance(entry, WorkflowBatchItemResult):
                raise TypeError("Batch collect callback entries must be WorkflowBatchItemResult values")
            if entry.custom_id not in expected:
                raise ValueError(f"Batch collect returned unknown custom_id {entry.custom_id!r}")
            if entry.custom_id in outcomes:
                raise ValueError(f"Batch collect returned custom_id {entry.custom_id!r} more than once")
            outcomes[entry.custom_id] = entry
        for custom_id, item_id in expected.items():
            entry = outcomes.get(custom_id)
            if entry is None:
                error = f"Batch submission {submission['id']} returned no result for {custom_id}"
                result = None
            elif entry.error is not None:
                error = str(entry.error)
                result = None
            else:
                error = None
                result = entry.result
            if submission["kind"] == "task":
                if error is None:
                    if isinstance(result, Mapping):
                        result = WorkflowTaskResult(**result)
                    if not isinstance(result, WorkflowTaskResult):
                        raise TypeError("Batch task result must be a WorkflowTaskResult")
                    case = await self._case(run_id, item_id)
                    metadata = {**case["metadata"], **(result.metadata or {})}
                    tags = result.tags if result.tags is not None else case.get("tags")
                    await self._write(
                        run_id,
                        "task-result",
                        {"output": _json_value(result.output), "metadata": metadata, "tags": tags},
                        item_id,
                    )
                else:
                    await self._write(
                        run_id,
                        "task-result",
                        {"output": None, "metadata": {}, "tags": None, "error": error},
                        item_id,
                    )
            elif error is None:
                if isinstance(result, Mapping):
                    result = WorkflowScorerResult(**result)
                if not isinstance(result, WorkflowScorerResult):
                    raise TypeError("Batch scorer result must be a WorkflowScorerResult")
                await self._write(
                    run_id,
                    _stage_kind("score-result", submission["scorer_name"]),
                    _json_value(result.score),
                    item_id,
                )
            else:
                await self._write(run_id, _stage_kind("score-error", submission["scorer_name"]), error, item_id)

    async def _waiting_status(self, run: Mapping[str, Any]) -> WorkflowEvalWaitingResult:
        pending = {}
        for mode in ("poll", "webhook"):
            completed = await self.store.get_set_size(self._key(run["run_id"], "progress", f"{mode}/complete"))
            submitted = await self.store.get_set_size(self._key(run["run_id"], "progress", f"{mode}/submitted"))
            pending[mode] = submitted - completed
        return WorkflowEvalWaitingResult(
            run_id=run["run_id"], pending=WorkflowEvalPending(poll=pending["poll"], webhook=pending["webhook"])
        )

    def _span_ids(self, run_id: str, item_id: str, stage: str) -> tuple[str, str, str]:
        if BraintrustEnv.LEGACY_IDS:
            root_span_id = _stable_uuid(run_id, item_id, "root")
            span_id = root_span_id if stage == "root" else _stable_uuid(run_id, item_id, stage)
            return _stable_uuid(run_id, item_id, f"row:{stage}"), span_id, root_span_id
        root_span_id = _stable_hex(run_id, item_id, "root", length=32)
        span_id = root_span_id[:16] if stage == "root" else _stable_hex(run_id, item_id, stage, length=16)
        row_id = _stable_uuid(run_id, item_id, stage)
        return row_id, span_id, root_span_id

    def _start_root(self, experiment: Experiment | None, run: Mapping[str, Any], case: Mapping[str, Any]) -> Span:
        if experiment is None:
            return NOOP_SPAN
        datum = case["datum"]
        event_dataset = experiment.dataset or (self.config.data if isinstance(self.config.data, Dataset) else None)
        if (
            event_dataset is not None
            and isinstance(datum.get("id"), str)
            and datum["id"]
            and isinstance(datum.get("_xact_id"), str)
            and datum["_xact_id"]
        ):
            origin = {
                "object_type": "dataset",
                "object_id": event_dataset.id,
                "id": datum["id"],
                "_xact_id": datum["_xact_id"],
                **({"created": datum["created"]} if isinstance(datum.get("created"), str) else {}),
            }
        else:
            origin = _validated_object_reference(datum.get("origin"))
        row_id, span_id, root_span_id = self._span_ids(run["run_id"], case["id"], "root")
        return _internal_start_span_with_initial_merge(
            "eval",
            parent=experiment.export(),
            span_id=span_id,
            root_span_id=root_span_id,
            state=self.config.state,
            type=SpanTypeAttribute.EVAL,
            id=row_id,
            input=datum["input"],
            expected=datum.get("expected"),
            metadata={
                **case["metadata"],
                "workflow_eval": {
                    "run_id": run["run_id"],
                    "case_id": case["case_id"],
                    "trial_index": case["trial_index"],
                },
            },
            tags=case.get("tags"),
            **({"origin": origin} if origin is not None else {}),
        )

    def _start_child(
        self,
        root: Span,
        run_id: str,
        item_id: str,
        stage: str,
        name: str,
        span_type: SpanTypeAttribute,
        **event: Any,
    ) -> Span:
        row_id, span_id, _ = self._span_ids(run_id, item_id, stage)
        span_attributes: dict[str, Any] = {"type": span_type}
        if span_type != SpanTypeAttribute.TASK:
            span_attributes["purpose"] = "scorer"
        return root.start_span(
            name,
            span_attributes=span_attributes,
            internal={"initial_span_write_as_merge": True, "span_id": span_id},
            id=row_id,
            **event,
        )

    async def _run_ordinary_task(
        self,
        run: Mapping[str, Any],
        case: dict[str, Any],
        experiment: Experiment | None,
        parameters: ValidatedParameters | None,
    ) -> None:
        run_id = run["run_id"]
        item_id = case["id"]
        if await self._read(run_id, "task-result", item_id) is not None:
            await self._log_task(run, case, experiment)
            return
        if not await self._claim(run_id, f"task:{item_id}"):
            return
        datum = case["datum"]
        metadata = dict(case["metadata"])
        hooks = DictEvalHooks(
            metadata,
            expected=datum.get("expected"),
            trial_index=case["trial_index"],
            tags=case.get("tags"),
            parameters=parameters,
        )
        root = self._start_root(experiment, run, case)
        task = cast(Callable[..., Any], self.config.task)
        task_args: list[Any] = [datum["input"]]
        try:
            if len(get_signature(task).parameters) == 2:
                task_args.append(hooks)
        except Exception:
            pass
        with root:
            with self._start_child(
                root,
                run_id,
                item_id,
                "task",
                "task",
                SpanTypeAttribute.TASK,
                input=datum["input"],
            ) as span:
                hooks.set_span(span)
                output = await self._call(task, *task_args)
                span.log(output=output)
            tags = list(hooks.tags) if hooks.tags else None
            task_result = {
                "output": _json_value(output),
                "metadata": _json_value(metadata),
                "tags": tags,
            }
            await self._write(run_id, "task-result", task_result, item_id)
            root.log(output=output, metadata=metadata, tags=tags)
        if root is not NOOP_SPAN:
            await self._flush_logs()
        await self._write(
            run_id,
            "task-log",
            {"root_span": root.export() if experiment is not None else None},
            item_id,
        )

    async def _log_task(self, run: Mapping[str, Any], case: dict[str, Any], experiment: Experiment | None) -> None:
        run_id = run["run_id"]
        item_id = case["id"]
        if await self._read(run_id, "task-log", item_id) is not None:
            return
        task_result = await self._required(run_id, "task-result", item_id)
        root = self._start_root(experiment, run, case)
        with root:
            with self._start_child(
                root,
                run_id,
                item_id,
                "task",
                "task",
                SpanTypeAttribute.TASK,
                input=case["datum"]["input"],
            ) as span:
                if task_result.get("error") is not None:
                    span.log(error=task_result["error"])
                else:
                    span.log(output=task_result["output"])
            if task_result.get("error") is not None:
                root.log(error=task_result["error"])
            else:
                root.log(output=task_result["output"], metadata=task_result["metadata"], tags=task_result.get("tags"))
        if root is not NOOP_SPAN:
            await self._flush_logs()
        await self._write(
            run_id,
            "task-log",
            {"root_span": root.export() if experiment is not None else None},
            item_id,
        )

    async def _root_for_case(self, run: Mapping[str, Any], item_id: str) -> Span:
        task_log = await self._required(run["run_id"], "task-log", item_id)
        exported = task_log.get("root_span")
        return _internal_resume_span(exported, self.config.state) if exported else NOOP_SPAN

    async def _trace_for_case(self, run: Mapping[str, Any], item_id: str) -> LocalTrace | None:
        task_log = await self._required(run["run_id"], "task-log", item_id)
        exported = task_log.get("root_span")
        if not exported:
            return None
        components = SpanComponentsV4.from_str(exported)
        if not components.root_span_id:
            raise ValueError("Persisted workflow evaluation root span is missing its root span ID")
        trace_state = self.config.state or _internal_get_global_state()

        async def ensure_spans_flushed() -> None:
            await asyncio.get_running_loop().run_in_executor(None, trace_state.flush)
            await trace_state.flush_otel()

        return LocalTrace(
            object_type=span_object_type_v3_to_typed_string(components.object_type),
            object_id=span_components_to_object_id(components),
            root_span_id=components.root_span_id,
            ensure_spans_flushed=ensure_spans_flushed,
            state=trace_state,
        )

    def _prepare_scores(self, raw: Any, name: str) -> list[ScoreLike]:
        if isinstance(raw, dict):
            raw = _normalize_score(raw, "When returning a dict, it must be a valid Score object.")
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, Mapping)):
            return [
                _normalize_score(value, "When returning an array of scores, each score must be a valid Score object.")
                for value in raw
            ]
        if is_score(raw):
            return [raw]
        return [Score(name=name, score=raw)]

    async def _run_ordinary_score(self, run: Mapping[str, Any], case: dict[str, Any], scorer: Any, name: str) -> None:
        run_id = run["run_id"]
        item_id = case["id"]
        result_kind = _stage_kind("score-result", name)
        if await self._read(run_id, result_kind, item_id) is None:
            if not await self._claim(run_id, f"score:{name}:{item_id}"):
                return
            task_result = await self._required(run_id, "task-result", item_id)
            fn = scorer.eval_async if hasattr(scorer, "eval_async") else scorer
            trace = await self._trace_for_case(run, item_id)
            raw = await call_user_fn(
                asyncio.get_running_loop(),
                fn,
                **_scorer_args(case, task_result),
                trace=trace,
            )
            await self._write(run_id, result_kind, _json_value(raw), item_id)
        await self._log_score(run, case, name)

    async def _log_score(self, run: Mapping[str, Any], case: dict[str, Any], name: str) -> None:
        run_id = run["run_id"]
        item_id = case["id"]
        log_kind = _stage_kind("score-log", name)
        if await self._read(run_id, log_kind, item_id) is not None:
            return
        raw = await self._required(run_id, _stage_kind("score-result", name), item_id)
        results = self._prepare_scores(raw, name)
        task_result = await self._required(run_id, "task-result", item_id)
        root = await self._root_for_case(run, item_id)
        propagated = merge_dicts({**(root.propagated_event or {})}, {"span_attributes": {"purpose": "scorer"}})
        with root:
            with self._start_child(
                root,
                run_id,
                item_id,
                f"score:{name}",
                name,
                SpanTypeAttribute.SCORE,
                input=_scorer_args(case, task_result),
                propagated_event=propagated,
            ) as span:
                output = (
                    {result.name: _score_fields(result) for result in results}
                    if len(results) != 1
                    else _score_fields(results[0])
                )
                scores = {result.name: result.score for result in results}
                span.log(output=output, metadata=_build_span_metadata(results), scores=scores)
                root.log(scores=scores)
        if root is not NOOP_SPAN:
            await self._flush_logs()
        await self._write(run_id, log_kind, True, item_id)

    async def _log_score_error(self, run: Mapping[str, Any], case: dict[str, Any], name: str, error: str) -> None:
        run_id = run["run_id"]
        item_id = case["id"]
        log_kind = _stage_kind("score-log", name)
        if await self._read(run_id, log_kind, item_id) is not None:
            return
        task_result = await self._required(run_id, "task-result", item_id)
        root = await self._root_for_case(run, item_id)
        propagated = merge_dicts({**(root.propagated_event or {})}, {"span_attributes": {"purpose": "scorer"}})
        with root:
            with self._start_child(
                root,
                run_id,
                item_id,
                f"score:{name}",
                name,
                SpanTypeAttribute.SCORE,
                input=_scorer_args(case, task_result),
                propagated_event=propagated,
            ) as span:
                span.log(error=error)
        if root is not NOOP_SPAN:
            await self._flush_logs()
        await self._write(run_id, log_kind, True, item_id)

    async def _run_classifier(self, run: Mapping[str, Any], case: dict[str, Any], classifier: Any, name: str) -> None:
        run_id = run["run_id"]
        item_id = case["id"]
        result_kind = _stage_kind("classification-result", name)
        if await self._read(run_id, result_kind, item_id) is None:
            if not await self._claim(run_id, f"classification:{name}:{item_id}"):
                return
            task_result = await self._required(run_id, "task-result", item_id)
            trace = await self._trace_for_case(run, item_id)
            raw = await call_user_fn(
                asyncio.get_running_loop(),
                classifier,
                **_scorer_args(case, task_result),
                trace=trace,
            )
            if raw is None:
                values: list[Any] = []
            elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, Mapping)):
                values = list(raw)
            else:
                values = [raw]
            classifications = [_validate_classification_result(value, name) for value in values]
            await self._write(run_id, result_kind, [value.as_dict() for value in classifications], item_id)
        await self._log_classifier(run, case, name)

    async def _log_classifier(self, run: Mapping[str, Any], case: dict[str, Any], name: str) -> None:
        run_id = run["run_id"]
        item_id = case["id"]
        log_kind = _stage_kind("classification-log", name)
        if await self._read(run_id, log_kind, item_id) is not None:
            return
        raw = await self._required(run_id, _stage_kind("classification-result", name), item_id)
        classifications = [Classification.from_dict(value) for value in raw]
        task_result = await self._required(run_id, "task-result", item_id)
        root = await self._root_for_case(run, item_id)
        with root:
            with self._start_child(
                root,
                run_id,
                item_id,
                f"classification:{name}",
                name,
                SpanTypeAttribute.CLASSIFIER,
                input=_scorer_args(case, task_result),
            ) as span:
                if classifications:
                    span.log(
                        output=_build_classification_span_output(classifications),
                        metadata=_build_span_metadata(classifications),
                    )
                    grouped: dict[str, list[Any]] = {}
                    for result in classifications:
                        grouped.setdefault(cast(str, result.name), []).append(result.as_item())
                    root.log(classifications=grouped)
                else:
                    span.log(output={}, metadata=None)
        if root is not NOOP_SPAN:
            await self._flush_logs()
        await self._write(run_id, log_kind, True, item_id)

    async def _advance_case(
        self,
        run: Mapping[str, Any],
        item_id: str,
        experiment: Experiment | None,
        scorers: Sequence[Any],
        scorer_names: Sequence[str],
        classifier_names: Sequence[str],
    ) -> bool:
        run_id = run["run_id"]
        case = await self._case(run_id, item_id)
        case_run = {**run, "case_ids": [item_id]}
        if isinstance(self.config.task, WorkflowTask):
            for spec, processor in self._submission_specs(case_run):
                if spec["kind"] == "task":
                    await self._submit_submission(run, spec, processor)
            if await self._read(run_id, "task-result", item_id) is None:
                return False
            await self._log_task(run, case, experiment)
        else:
            if not isinstance(self.config.task, WorkflowBatchTask):
                await self._run_ordinary_task(run, case, experiment, self._parameters(run))
            elif await self._read(run_id, "task-result", item_id) is None:
                return False
            else:
                await self._log_task(run, case, experiment)
        if await self._read(run_id, "task-log", item_id) is None:
            return False
        task_result = await self._required(run_id, "task-result", item_id)
        if task_result.get("error") is not None:
            await self.store.add_to_set(self._key(run_id, "progress", "cases"), item_id)
            return True

        for spec, processor in self._submission_specs(case_run):
            if spec["kind"] == "score":
                await self._submit_submission(run, spec, processor)
        complete = True
        for scorer, name in zip(scorers, scorer_names):
            if isinstance(scorer, WorkflowScorer):
                if await self._read(run_id, _stage_kind("score-result", name), item_id) is None:
                    complete = False
                    continue
                await self._log_score(run, case, name)
            elif isinstance(scorer, WorkflowBatchScorer):
                if await self._read(run_id, _stage_kind("score-result", name), item_id) is None:
                    error = await self._read(run_id, _stage_kind("score-error", name), item_id)
                    if error is None:
                        complete = False
                        continue
                    await self._log_score_error(run, case, name, error)
                else:
                    await self._log_score(run, case, name)
            else:
                await self._run_ordinary_score(run, case, scorer, name)
            if await self._read(run_id, _stage_kind("score-log", name), item_id) is None:
                complete = False
        for classifier, name in zip(self.config.classifiers, classifier_names):
            await self._run_classifier(run, case, classifier, name)
            if await self._read(run_id, _stage_kind("classification-log", name), item_id) is None:
                complete = False
        if complete:
            await self.store.add_to_set(self._key(run_id, "progress", "cases"), item_id)
        return complete

    async def _advance(
        self, run: Mapping[str, Any], *, experiment: Experiment | None = None, item_ids: Sequence[str] | None = None
    ) -> WorkflowEvalResult:
        run = await self._required(run["run_id"], "run")
        run_id = run["run_id"]
        if run["status"] == "completed":
            return await self.status(run_id)
        if experiment is None and not run["no_send_logs"]:
            experiment = self._experiment(run)
        resolved_scores = [
            score() if inspect.isclass(score) and is_scorer(score) else score for score in self.config.scores
        ]
        scorer_names = [
            score.name if isinstance(score, (WorkflowScorer, WorkflowBatchScorer)) else _scorer_name(score, index)
            for index, score in enumerate(resolved_scores)
        ]
        if len(scorer_names) != len(set(scorer_names)):
            raise ValueError("Workflow evaluation scorer names must be unique")
        classifier_names = [
            _classifier_name(classifier, index) for index, classifier in enumerate(self.config.classifiers)
        ]
        if len(classifier_names) != len(set(classifier_names)):
            raise ValueError("Workflow evaluation classifier names must be unique")

        if isinstance(self.config.task, WorkflowBatchTask):
            await self._schedule_batch_stage(run, self.config.task, kind="task")

        async def advance_cases() -> list[BaseException]:
            results = await asyncio.gather(
                *(
                    self._advance_case(run, item_id, experiment, resolved_scores, scorer_names, classifier_names)
                    for item_id in (item_ids if item_ids is not None else run["case_ids"])
                ),
                return_exceptions=True,
            )
            return [result for result in results if isinstance(result, BaseException)]

        errors = await advance_cases()
        if errors:
            _raise_callback_errors(errors)
        for scorer in self.config.scores:
            if isinstance(scorer, WorkflowBatchScorer):
                await self._schedule_batch_stage(run, scorer, kind="score")
        # Newly submitted batch scorer work is now represented in the waiting
        # count; this pass also logs results collected by webhook/poll calls.
        errors = await advance_cases()
        if errors:
            _raise_callback_errors(errors)
        if await self.store.get_set_size(self._key(run_id, "progress", "cases")) != len(run["case_ids"]):
            return await self._waiting_status(run)
        if not await self._claim(run_id, "finalize"):
            return await self.status(run_id)
        summary = await self._summary(run, scorer_names, classifier_names, experiment)
        return await self._save_completed(run, summary)

    async def _summary(
        self,
        run: Mapping[str, Any],
        scorer_names: Sequence[str],
        classifier_names: Sequence[str],
        experiment: Experiment | None,
    ) -> ExperimentSummary:
        if experiment is not None:
            comparison_experiment_id = self.config.base_experiment_id
            if comparison_experiment_id is None:
                comparison_experiment_id = _get_persisted_base_experiment_id(experiment)
            return experiment.summarize(
                summarize_scores=self.config.summarize_scores,
                comparison_experiment_id=comparison_experiment_id,
            )

        results: list[EvalResult[Any, Any, Any]] = []
        for item_id in run["case_ids"]:
            case = await self._case(run["run_id"], item_id)
            task_result = await self._required(run["run_id"], "task-result", item_id)
            scores: dict[str, float | None] = {}
            for name in scorer_names:
                raw = await self._read(run["run_id"], _stage_kind("score-result", name), item_id)
                if raw is None:
                    continue
                for score in self._prepare_scores(raw, name):
                    scores[score.name] = score.score
            classifications: dict[str, list[Any]] = {}
            for name in classifier_names:
                if task_result.get("error") is not None:
                    continue
                raw = await self._required(run["run_id"], _stage_kind("classification-result", name), item_id)
                for value in raw:
                    classification = Classification.from_dict(value)
                    classifications.setdefault(cast(str, classification.name), []).append(classification.as_item())
            datum = case["datum"]
            results.append(
                EvalResult(
                    input=datum["input"],
                    output=task_result["output"],
                    scores=scores,
                    classifications=classifications or None,
                    expected=datum.get("expected"),
                    metadata=task_result["metadata"],
                    tags=task_result.get("tags"),
                    error=RuntimeError(task_result["error"]) if task_result.get("error") is not None else None,
                )
            )
        evaluator = Evaluator(
            project_name=self.config.project_name,
            eval_name=self.eval_name,
            data=[],
            task=cast(Any, lambda value: value),
            scores=[],
            experiment_name=run["experiment_name"],
            metadata=self.config.metadata,
        )
        return build_local_summary(evaluator, cast(Any, results))


__all__ = [
    "WorkflowBatchItem",
    "WorkflowBatchItemResult",
    "WorkflowBatchScorer",
    "WorkflowBatchTask",
    "WorkflowBatchingOptions",
    "WorkflowSubmissionCompletionPoll",
    "WorkflowSubmissionCompletionWebhook",
    "WorkflowSubmissionContext",
    "WorkflowSubmissionPoll",
    "WorkflowScorer",
    "WorkflowScorerItem",
    "WorkflowScorerResult",
    "WorkflowTask",
    "WorkflowTaskItem",
    "WorkflowTaskResult",
    "WorkflowEval",
    "WorkflowEvalCompletedResult",
    "WorkflowEvalMemoryStore",
    "WorkflowEvalPending",
    "WorkflowEvalRedisStore",
    "WorkflowEvalResult",
    "WorkflowEvalStore",
    "WorkflowEvalStoreEntry",
    "WorkflowEvalWaitingResult",
    "define_workflow_eval",
]
