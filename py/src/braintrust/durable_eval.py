"""Experimental durable evaluation support for asynchronous batch providers."""

import asyncio
import base64
import dataclasses
import functools
import hashlib
import inspect
import threading
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

DEFAULT_BATCH_SIZE = 1_000
DEFAULT_REDIS_TTL_MS = 1_000 * 60 * 60 * 24 * 7
_SCHEMA_PREFIX = "durable-eval/python/v1"


@dataclasses.dataclass(frozen=True)
class BatchContext:
    """Identifiers supplied to a durable batch processor callback."""

    run_id: str
    batch_id: str


@dataclasses.dataclass(frozen=True)
class BatchPollResult:
    """The current state returned by a polling completion callback."""

    status: Literal["pending", "complete", "failed"]
    error: Any = None


@dataclasses.dataclass(frozen=True)
class BatchCompletionPoll(Generic[SubmissionData]):
    """Configures a batch processor whose provider is checked by polling."""

    poll: Callable[[SubmissionData, BatchContext], BatchPollResult | Awaitable[BatchPollResult]]
    mode: Literal["poll"] = dataclasses.field(default="poll", init=False)


@dataclasses.dataclass(frozen=True)
class BatchCompletionWebhook(Generic[SubmissionData]):
    """Configures a batch processor completed by an incoming webhook."""

    get_external_id: Callable[[SubmissionData, BatchContext], str | Awaitable[str]]
    mode: Literal["webhook"] = dataclasses.field(default="webhook", init=False)


BatchCompletion = BatchCompletionPoll[SubmissionData] | BatchCompletionWebhook[SubmissionData]


@dataclasses.dataclass(frozen=True)
class BatchTaskItem(Generic[Input, Expected]):
    """A stable task item submitted to a provider batch."""

    id: str
    input: Input
    expected: Expected | None
    metadata: Metadata
    tags: list[str] | None
    parameters: ValidatedParameters | None
    trial_index: int


@dataclasses.dataclass(frozen=True)
class BatchTaskResult(Generic[Output]):
    """A collected result for one task item."""

    id: str
    output: Output
    metadata: Metadata | None = None
    tags: list[str] | None = None


@dataclasses.dataclass(frozen=True)
class BatchScorerItem(Generic[Input, Output, Expected]):
    """A stable scorer item submitted to a provider batch."""

    id: str
    input: Input
    output: Output
    expected: Expected | None
    metadata: Metadata
    tags: list[str] | None
    trial_index: int


@dataclasses.dataclass(frozen=True)
class BatchScorerResult:
    """A collected score for one scorer item."""

    id: str
    score: OneOrMoreScores


@dataclasses.dataclass(frozen=True)
class BatchTask(Generic[Input, Output, Expected, SubmissionData]):
    """Runs an evaluation task through asynchronous provider batch operations."""

    submit: Callable[[list[BatchTaskItem[Input, Expected]], BatchContext], SubmissionData | Awaitable[SubmissionData]]
    completion: BatchCompletion[SubmissionData]
    collect: Callable[
        [SubmissionData, BatchContext], list[BatchTaskResult[Output]] | Awaitable[list[BatchTaskResult[Output]]]
    ]
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("BatchTask batch_size must be a positive integer")


@dataclasses.dataclass(frozen=True)
class BatchScorer(Generic[Input, Output, Expected, SubmissionData]):
    """Runs an evaluation scorer through asynchronous provider batch operations."""

    name: str
    submit: Callable[
        [list[BatchScorerItem[Input, Output, Expected]], BatchContext], SubmissionData | Awaitable[SubmissionData]
    ]
    completion: BatchCompletion[SubmissionData]
    collect: Callable[[SubmissionData, BatchContext], list[BatchScorerResult] | Awaitable[list[BatchScorerResult]]]
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BatchScorer name must be a non-empty string")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("BatchScorer batch_size must be a positive integer")


@dataclasses.dataclass(frozen=True)
class DurableEvalStoreEntry:
    """Result of an atomic durable-store get-or-set operation."""

    value: bytes
    created: bool


class DurableEvalStore(Protocol):
    """Minimal persistence interface used by durable evaluations."""

    async def read(self, key: str) -> bytes | None: ...

    async def write(self, key: str, value: bytes) -> None: ...

    async def get_or_set(self, key: str, value: bytes) -> DurableEvalStoreEntry: ...


class DurableEvalMemoryStore:
    """Process-local durable evaluation state, intended for tests and local runs."""

    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}
        self._lock = threading.Lock()

    async def read(self, key: str) -> bytes | None:
        with self._lock:
            value = self._values.get(key)
            return bytes(value) if value is not None else None

    async def write(self, key: str, value: bytes) -> None:
        with self._lock:
            self._values[key] = bytes(value)

    async def get_or_set(self, key: str, value: bytes) -> DurableEvalStoreEntry:
        with self._lock:
            existing = self._values.get(key)
            if existing is not None:
                return DurableEvalStoreEntry(value=bytes(existing), created=False)
            self._values[key] = bytes(value)
            return DurableEvalStoreEntry(value=bytes(value), created=True)


class DurableEvalRedisStore:
    """Durable state backed by an existing sync or async redis-py client."""

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "braintrust-eval:",
        ttl_ms: int = DEFAULT_REDIS_TTL_MS,
    ) -> None:
        if not isinstance(ttl_ms, int) or isinstance(ttl_ms, bool) or ttl_ms < 1:
            raise ValueError("DurableEvalRedisStore ttl_ms must be a positive integer")
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
            raise TypeError("DurableEvalRedisStore expected GET to return str, bytes, or None")
        return base64.b64decode(value)

    async def write(self, key: str, value: bytes) -> None:
        encoded = base64.b64encode(value).decode("ascii")
        await self._call(self.client.set, f"{self.key_prefix}{key}", encoded, px=self.ttl_ms)

    async def get_or_set(self, key: str, value: bytes) -> DurableEvalStoreEntry:
        redis_key = f"{self.key_prefix}{key}"
        encoded = base64.b64encode(value).decode("ascii")
        existing = await self._call(self.client.set, redis_key, encoded, px=self.ttl_ms, nx=True, get=True)
        if existing is None:
            return DurableEvalStoreEntry(value=bytes(value), created=True)
        if isinstance(existing, bytes):
            existing = existing.decode("ascii")
        if not isinstance(existing, str):
            raise TypeError("DurableEvalRedisStore expected atomic SET to return str, bytes, or None")
        return DurableEvalStoreEntry(value=base64.b64decode(existing), created=False)


@dataclasses.dataclass(frozen=True)
class DurableEvalPending:
    """Counts of submitted provider batches awaiting completion."""

    poll: int
    webhook: int


@dataclasses.dataclass(frozen=True)
class DurableEvalWaitingResult:
    run_id: str
    pending: DurableEvalPending
    status: Literal["waiting"] = dataclasses.field(default="waiting", init=False)


@dataclasses.dataclass(frozen=True)
class DurableEvalCompletedResult:
    run_id: str
    pending: DurableEvalPending
    summary: ExperimentSummary
    status: Literal["completed"] = dataclasses.field(default="completed", init=False)


@dataclasses.dataclass(frozen=True)
class DurableEvalFailedResult:
    run_id: str
    batch_id: str
    error: Any
    pending: DurableEvalPending
    status: Literal["failed"] = dataclasses.field(default="failed", init=False)


DurableEvalResult = DurableEvalWaitingResult | DurableEvalCompletedResult | DurableEvalFailedResult


@dataclasses.dataclass(frozen=True)
class _DurableEvalConfig(Generic[Input, Output, Expected]):
    project_name: str
    store: DurableEvalStore
    data: EvalData[Input, Expected]
    task: EvalTask[Input, Output, Expected] | BatchTask[Input, Output, Expected, Any]
    scores: Sequence[EvalScorer[Input, Output, Expected] | BatchScorer[Input, Output, Expected, Any]]
    classifiers: Sequence[EvalClassifier[Input, Output, Expected]]
    case_id: Callable[[EvalCase[Input, Expected]], str | Awaitable[str]] | None
    experiment_name: str | None
    trial_count: int
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


class DurableEval(Generic[Input, Output, Expected]):
    """A durable evaluation definition that can be started and resumed."""

    def __init__(self, config: _DurableEvalConfig[Input, Output, Expected]) -> None:
        self._config = config

    async def start(
        self, parameters: Mapping[str, Any] | None = None, *, no_send_logs: bool = False
    ) -> DurableEvalResult:
        return await _DurableEvalRunner(self._config).start(parameters, no_send_logs=no_send_logs)

    async def status(self, run_id: str) -> DurableEvalResult:
        return await _DurableEvalRunner(self._config).status(run_id)

    async def poll(self, run_id: str) -> DurableEvalResult:
        return await _DurableEvalRunner(self._config).poll(run_id)

    async def process_batch_result(
        self, run_id: str, *, batch_id: str | None = None, external_id: str | None = None
    ) -> DurableEvalResult:
        return await _DurableEvalRunner(self._config).process_batch_result(
            run_id, batch_id=batch_id, external_id=external_id
        )


def define_durable_eval(
    project_name: str,
    *,
    store: DurableEvalStore,
    data: EvalData[Input, Expected],
    task: EvalTask[Input, Output, Expected] | BatchTask[Input, Output, Expected, Any],
    scores: Sequence[EvalScorer[Input, Output, Expected] | BatchScorer[Input, Output, Expected, Any]] | None = None,
    classifiers: Sequence[EvalClassifier[Input, Output, Expected]] | None = None,
    case_id: Callable[[EvalCase[Input, Expected]], str | Awaitable[str]] | None = None,
    experiment_name: str | None = None,
    trial_count: int = 1,
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
) -> DurableEval[Input, Output, Expected]:
    """Define an experimental evaluation that can pause across provider batches."""
    if not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count < 1:
        raise ValueError("trial_count must be a positive integer")
    return DurableEval(
        _DurableEvalConfig(
            project_name=project_name,
            store=store,
            data=data,
            task=task,
            scores=list(scores or []),
            classifiers=list(classifiers or []),
            case_id=case_id,
            experiment_name=experiment_name,
            trial_count=trial_count,
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
    """Validate and normalize a value at the durable JSON boundary."""
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


class _DurableEvalRunner(Generic[Input, Output, Expected]):
    def __init__(self, config: _DurableEvalConfig[Input, Output, Expected]) -> None:
        self.config = config
        self.store = config.store
        self.eval_name = config.experiment_name or config.project_name
        self.definition_key = _stable_hex(config.project_name, self.eval_name, length=32)

    def _key(self, run_id: str, kind: str, identifier: str | None = None) -> str:
        suffix = f"/{identifier}" if identifier is not None else ""
        return f"{_SCHEMA_PREFIX}/{self.definition_key}/{run_id}/{kind}{suffix}"

    async def _read(self, run_id: str, kind: str, identifier: str | None = None) -> Any | None:
        value = await self.store.read(self._key(run_id, kind, identifier))
        return _decode(value) if value is not None else None

    async def _required(self, run_id: str, kind: str, identifier: str | None = None) -> Any:
        value = await self._read(run_id, kind, identifier)
        if value is None:
            target = f" {identifier!r}" if identifier is not None else ""
            raise ValueError(f"Unknown durable evaluation {kind}{target} for run {run_id!r}")
        return value

    async def _write(self, run_id: str, kind: str, value: Any, identifier: str | None = None) -> None:
        await self.store.write(self._key(run_id, kind, identifier), _json_bytes(value))

    async def _claim(self, run_id: str, action: str) -> bool:
        result = await self.store.get_or_set(self._key(run_id, "claim", action), b"1")
        return result.created

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
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

    def _uses_batch_processor(self) -> bool:
        return isinstance(self.config.task, BatchTask) or any(
            isinstance(score, BatchScorer) for score in self.config.scores
        )

    async def _save_completed(self, run: Mapping[str, Any], summary: ExperimentSummary) -> DurableEvalCompletedResult:
        completed = {**run, "status": "completed", "summary": _json_value(summary.as_dict())}
        await self._write(run["run_id"], "run", completed)
        return DurableEvalCompletedResult(
            run_id=run["run_id"], pending=DurableEvalPending(poll=0, webhook=0), summary=summary
        )

    async def _save_failed(
        self, run: Mapping[str, Any], batch: Mapping[str, Any], error: Any
    ) -> DurableEvalFailedResult:
        normalized_error = _json_value(error)
        failure = {"batch_id": batch["id"], "error": normalized_error}
        failed_run = {**run, "status": "failed", "failure": failure}
        await self._write(run["run_id"], "run", failed_run)
        await self._write(
            run["run_id"],
            "batch",
            {**batch, "status": "failed", "error": normalized_error},
            batch["id"],
        )
        return DurableEvalFailedResult(
            run_id=run["run_id"],
            batch_id=batch["id"],
            error=normalized_error,
            pending=DurableEvalPending(poll=0, webhook=0),
        )

    async def _run_ordinary_evaluator(
        self,
        run: Mapping[str, Any],
        data: list[EvalCase[Input, Expected]],
        experiment: Experiment | None,
        parameters: ValidatedParameters,
    ) -> DurableEvalCompletedResult:
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
                raise ValueError("Every durable evaluation case must have a non-empty id or be assigned by case_id")
            if case_id in seen_case_ids:
                raise ValueError(f"Durable evaluation case IDs must be unique; found duplicate {case_id!r}")
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

    async def start(self, parameters: Mapping[str, Any] | None, *, no_send_logs: bool) -> DurableEvalResult:
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
            raise RuntimeError(f"Durable evaluation run ID collision: {run_id}")

        experiment = self._experiment(run)
        data = await self._resolve_data(experiment)
        if not self._uses_batch_processor():
            return await self._run_ordinary_evaluator(run, data, experiment, validated_parameters)

        await self._persist_cases(run, data)
        return await self._advance(run, experiment=experiment)

    async def status(self, run_id: str) -> DurableEvalResult:
        run = await self._required(run_id, "run")
        if run["status"] == "completed":
            return DurableEvalCompletedResult(
                run_id=run_id,
                pending=DurableEvalPending(poll=0, webhook=0),
                summary=ExperimentSummary.from_dict_deep(run["summary"]),
            )
        if run["status"] == "failed":
            failure = run["failure"]
            return DurableEvalFailedResult(
                run_id=run_id,
                batch_id=failure["batch_id"],
                error=failure.get("error"),
                pending=DurableEvalPending(poll=0, webhook=0),
            )
        return await self._waiting_status(run)

    async def poll(self, run_id: str) -> DurableEvalResult:
        run = await self._required(run_id, "run")
        if run["status"] in ("completed", "failed"):
            return await self.status(run_id)
        # Snapshot first: newly submitted downstream batches wait until the next poll call.
        batches = await self._batch_records(run)
        for batch, processor in batches:
            if batch["status"] != "submitted" or batch["mode"] != "poll":
                continue
            completion = processor.completion
            assert isinstance(completion, BatchCompletionPoll)
            context = BatchContext(run_id=run_id, batch_id=batch["id"])
            result = await self._call(completion.poll, batch["submission_data"], context)
            if not isinstance(result, BatchPollResult):
                if isinstance(result, Mapping):
                    result = BatchPollResult(**result)
                else:
                    raise TypeError("Batch poll callback must return BatchPollResult")
            if result.status not in ("pending", "complete", "failed"):
                raise ValueError(f"Batch poll callback returned unsupported status {result.status!r}")
            if result.status == "failed":
                return await self._save_failed(run, batch, result.error)
            if result.status == "complete":
                await self._collect_batch(run, batch, processor)
        return await self._advance(run)

    async def process_batch_result(
        self, run_id: str, *, batch_id: str | None, external_id: str | None
    ) -> DurableEvalResult:
        if batch_id is None and external_id is None:
            raise ValueError("process_batch_result requires batch_id or external_id")
        run = await self._required(run_id, "run")
        if run["status"] in ("completed", "failed"):
            return await self.status(run_id)
        batches = await self._batch_records(run)
        by_batch = None
        if batch_id is not None:
            by_batch = next(((batch, processor) for batch, processor in batches if batch["id"] == batch_id), None)

        by_external = None
        if external_id is not None:
            by_external = next(
                ((batch, processor) for batch, processor in batches if batch.get("external_id") == external_id),
                None,
            )

        if by_batch is not None and by_external is not None and by_batch[0]["id"] != by_external[0]["id"]:
            raise ValueError("batch_id and external_id identify different batches")
        match = by_batch or by_external
        if match is None:
            raise ValueError("No submitted durable batch matches this result")
        batch, processor = match
        if batch["status"] != "complete":
            await self._collect_batch(run, batch, processor)
        return await self._advance(run)

    def _batch_id(self, run_id: str, kind: str, index: int, item_ids: Sequence[str]) -> str:
        return f"batch-{_stable_hex(run_id, kind, str(index), *item_ids, length=32)}"

    async def _task_complete(self, run: Mapping[str, Any]) -> bool:
        for item_id in run["case_ids"]:
            if await self._read(run["run_id"], "task-result", item_id) is None:
                return False
        return True

    async def _batch_specs(self, run: Mapping[str, Any]) -> list[tuple[dict[str, Any], Any]]:
        run_id = run["run_id"]
        item_ids = run["case_ids"]
        specs: list[tuple[dict[str, Any], Any]] = []
        if isinstance(self.config.task, BatchTask):
            for index, offset in enumerate(range(0, len(item_ids), self.config.task.batch_size)):
                ids = item_ids[offset : offset + self.config.task.batch_size]
                specs.append(
                    (
                        {
                            "id": self._batch_id(run_id, "task", index, ids),
                            "kind": "task",
                            "item_ids": ids,
                        },
                        self.config.task,
                    )
                )
        if await self._task_complete(run):
            for scorer in self.config.scores:
                if not isinstance(scorer, BatchScorer):
                    continue
                for index, offset in enumerate(range(0, len(item_ids), scorer.batch_size)):
                    ids = item_ids[offset : offset + scorer.batch_size]
                    specs.append(
                        (
                            {
                                "id": self._batch_id(run_id, f"score:{scorer.name}", index, ids),
                                "kind": "score",
                                "scorer_name": scorer.name,
                                "item_ids": ids,
                            },
                            scorer,
                        )
                    )
        return specs

    async def _batch_records(self, run: Mapping[str, Any]) -> list[tuple[dict[str, Any], Any]]:
        records: list[tuple[dict[str, Any], Any]] = []
        for spec, processor in await self._batch_specs(run):
            record = await self._read(run["run_id"], "batch", spec["id"])
            if record is not None:
                records.append((record, processor))
        return records

    async def _case(self, run_id: str, item_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], await self._required(run_id, "case", item_id))

    async def _task_item(
        self, run: Mapping[str, Any], item_id: str, parameters: ValidatedParameters | None
    ) -> BatchTaskItem[Any, Any]:
        case = await self._case(run["run_id"], item_id)
        datum = case["datum"]
        return BatchTaskItem(
            id=item_id,
            input=datum["input"],
            expected=datum.get("expected"),
            metadata=dict(case["metadata"]),
            tags=case.get("tags"),
            parameters=parameters,
            trial_index=case["trial_index"],
        )

    async def _scorer_item(self, run: Mapping[str, Any], item_id: str) -> BatchScorerItem[Any, Any, Any]:
        case = await self._case(run["run_id"], item_id)
        task = await self._required(run["run_id"], "task-result", item_id)
        datum = case["datum"]
        return BatchScorerItem(
            id=item_id,
            input=datum["input"],
            output=task["output"],
            expected=datum.get("expected"),
            metadata=dict(task["metadata"]),
            tags=task.get("tags"),
            trial_index=case["trial_index"],
        )

    async def _submit_batch(self, run: Mapping[str, Any], spec: dict[str, Any], processor: Any) -> None:
        run_id = run["run_id"]
        if await self._read(run_id, "batch", spec["id"]) is not None:
            return
        if not await self._claim(run_id, f"submit:{spec['id']}"):
            return
        parameters = self._parameters(run)
        if spec["kind"] == "task":
            items = [await self._task_item(run, item_id, parameters) for item_id in spec["item_ids"]]
        else:
            items = [await self._scorer_item(run, item_id) for item_id in spec["item_ids"]]
        context = BatchContext(run_id=run_id, batch_id=spec["id"])
        submission_data = await self._call(processor.submit, items, context)
        submission_data = _json_value(submission_data)
        completion = processor.completion
        external_id = None
        if isinstance(completion, BatchCompletionWebhook):
            external_id = await self._call(completion.get_external_id, submission_data, context)
            if not isinstance(external_id, str) or not external_id:
                raise ValueError("Batch webhook get_external_id must return a non-empty string")
        await self._write(
            run_id,
            "batch",
            {
                **spec,
                "submission_data": submission_data,
                "external_id": external_id,
                "mode": completion.mode,
                "status": "submitted",
            },
            spec["id"],
        )

    async def _collect_batch(self, run: Mapping[str, Any], batch: dict[str, Any], processor: Any) -> None:
        run_id = run["run_id"]
        if batch["status"] == "complete":
            return
        context = BatchContext(run_id=run_id, batch_id=batch["id"])
        raw_results = await self._call(processor.collect, batch["submission_data"], context)
        if not isinstance(raw_results, list):
            raise TypeError("Batch collect callback must return a list")
        result_type = BatchTaskResult if batch["kind"] == "task" else BatchScorerResult
        results = [value if isinstance(value, result_type) else result_type(**value) for value in raw_results]
        result_ids = [value.id for value in results]
        expected_ids = batch["item_ids"]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError(f"Batch {batch['id']} returned duplicate item IDs")
        unknown = sorted(set(result_ids) - set(expected_ids))
        missing = sorted(set(expected_ids) - set(result_ids))
        if unknown or missing:
            raise ValueError(
                f"Batch {batch['id']} result IDs did not match submitted items; unknown={unknown}, missing={missing}"
            )

        if batch["kind"] == "task":
            for result in results:
                case = await self._case(run_id, result.id)
                metadata = {**case["metadata"], **(result.metadata or {})}
                tags = result.tags if result.tags is not None else case.get("tags")
                await self._write(
                    run_id,
                    "task-result",
                    {"output": _json_value(result.output), "metadata": metadata, "tags": tags},
                    result.id,
                )
        else:
            for result in results:
                await self._write(
                    run_id,
                    _stage_kind("score-result", batch["scorer_name"]),
                    _json_value(result.score),
                    result.id,
                )
        batch["status"] = "complete"
        await self._write(run_id, "batch", batch, batch["id"])

    async def _waiting_status(self, run: Mapping[str, Any]) -> DurableEvalWaitingResult:
        pending = {"poll": 0, "webhook": 0}
        for batch, _ in await self._batch_records(run):
            if batch["status"] == "submitted":
                pending[batch["mode"]] += 1
        return DurableEvalWaitingResult(
            run_id=run["run_id"], pending=DurableEvalPending(poll=pending["poll"], webhook=pending["webhook"])
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
                "durable_eval": {
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
                span.log(output=task_result["output"])
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
            raise ValueError("Persisted durable evaluation root span is missing its root span ID")
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

    async def _advance_tasks(
        self,
        run: Mapping[str, Any],
        experiment: Experiment | None,
        parameters: ValidatedParameters | None,
    ) -> tuple[Experiment | None, bool]:
        run_id = run["run_id"]
        if isinstance(self.config.task, BatchTask):
            for spec, processor in await self._batch_specs(run):
                if spec["kind"] == "task":
                    await self._submit_batch(run, spec, processor)
        else:
            if experiment is None and not run["no_send_logs"]:
                experiment = self._experiment(run)
            for item_id in run["case_ids"]:
                case = await self._case(run_id, item_id)
                await self._run_ordinary_task(run, case, experiment, parameters)

        if not await self._task_complete(run):
            return experiment, False

        if experiment is None and not run["no_send_logs"]:
            experiment = self._experiment(run)
        if isinstance(self.config.task, BatchTask):
            for item_id in run["case_ids"]:
                await self._log_task(run, await self._case(run_id, item_id), experiment)
        for item_id in run["case_ids"]:
            if await self._read(run_id, "task-log", item_id) is None:
                return experiment, False
        return experiment, True

    async def _advance_scorers(
        self,
        run: Mapping[str, Any],
        scorers: Sequence[Any],
        scorer_names: Sequence[str],
    ) -> None:
        run_id = run["run_id"]
        for spec, processor in await self._batch_specs(run):
            if spec["kind"] == "score":
                await self._submit_batch(run, spec, processor)

        for scorer, name in zip(scorers, scorer_names):
            for item_id in run["case_ids"]:
                if isinstance(scorer, BatchScorer):
                    if await self._read(run_id, _stage_kind("score-result", name), item_id) is not None:
                        await self._log_score(run, await self._case(run_id, item_id), name)
                else:
                    await self._run_ordinary_score(run, await self._case(run_id, item_id), scorer, name)

    async def _advance_classifiers(
        self,
        run: Mapping[str, Any],
        classifiers: Sequence[EvalClassifier[Input, Output, Expected]],
        classifier_names: Sequence[str],
    ) -> None:
        run_id = run["run_id"]
        for classifier, name in zip(classifiers, classifier_names):
            for item_id in run["case_ids"]:
                await self._run_classifier(run, await self._case(run_id, item_id), classifier, name)

    async def _stage_records_complete(self, run: Mapping[str, Any], prefix: str, names: Sequence[str]) -> bool:
        complete = True
        for name in names:
            result_kind = _stage_kind(f"{prefix}-result", name)
            log_kind = _stage_kind(f"{prefix}-log", name)
            for item_id in run["case_ids"]:
                if (
                    await self._read(run["run_id"], result_kind, item_id) is None
                    or await self._read(run["run_id"], log_kind, item_id) is None
                ):
                    complete = False
        return complete

    async def _advance(self, run: Mapping[str, Any], *, experiment: Experiment | None = None) -> DurableEvalResult:
        run = await self._required(run["run_id"], "run")
        run_id = run["run_id"]
        if run["status"] in ("completed", "failed"):
            return await self.status(run_id)
        experiment, tasks_complete = await self._advance_tasks(run, experiment, self._parameters(run))
        if not tasks_complete:
            return await self._waiting_status(run)

        resolved_scores = [
            score() if inspect.isclass(score) and is_scorer(score) else score for score in self.config.scores
        ]
        scorer_names = [
            score.name if isinstance(score, BatchScorer) else _scorer_name(score, index)
            for index, score in enumerate(resolved_scores)
        ]
        if len(scorer_names) != len(set(scorer_names)):
            raise ValueError("Durable evaluation scorer names must be unique")

        await self._advance_scorers(run, resolved_scores, scorer_names)

        classifiers = list(self.config.classifiers)
        classifier_names = [_classifier_name(classifier, index) for index, classifier in enumerate(classifiers)]
        if len(classifier_names) != len(set(classifier_names)):
            raise ValueError("Durable evaluation classifier names must be unique")
        await self._advance_classifiers(run, classifiers, classifier_names)

        scores_complete = await self._stage_records_complete(run, "score", scorer_names)
        classifiers_complete = await self._stage_records_complete(run, "classification", classifier_names)
        if not scores_complete or not classifiers_complete:
            return await self._waiting_status(run)

        if not await self._claim(run_id, "finalize"):
            current = await self._required(run_id, "run")
            if current["status"] == "completed":
                return await self.status(run_id)
            return await self._waiting_status(current)
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
                raw = await self._required(run["run_id"], _stage_kind("score-result", name), item_id)
                for score in self._prepare_scores(raw, name):
                    scores[score.name] = score.score
            classifications: dict[str, list[Any]] = {}
            for name in classifier_names:
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
    "BatchCompletionPoll",
    "BatchCompletionWebhook",
    "BatchContext",
    "BatchPollResult",
    "BatchScorer",
    "BatchScorerItem",
    "BatchScorerResult",
    "BatchTask",
    "BatchTaskItem",
    "BatchTaskResult",
    "DurableEval",
    "DurableEvalCompletedResult",
    "DurableEvalFailedResult",
    "DurableEvalMemoryStore",
    "DurableEvalPending",
    "DurableEvalRedisStore",
    "DurableEvalResult",
    "DurableEvalStore",
    "DurableEvalStoreEntry",
    "DurableEvalWaitingResult",
    "define_durable_eval",
]
