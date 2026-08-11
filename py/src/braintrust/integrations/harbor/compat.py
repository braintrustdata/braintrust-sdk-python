"""The isolated Harbor-version compatibility boundary."""

# Harbor is optional and only supports Python 3.12+, while pylint runs across
# Braintrust's full Python matrix without installing Harbor.
# pylint: disable=import-error

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import logical_task_key


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskData:
    logical_key: str
    source: str
    name: str
    input: dict[str, Any]
    expected: Any
    metadata: dict[str, Any]
    digest: str | None
    schema_version: str | None
    task_dir: Path | None


@dataclass(frozen=True)
class TrialPlan:
    trial_name: str
    trial_config: Any
    trial_lock: Any
    task: TaskData
    attempt_index: int


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    job_name: str
    job_dir: Path
    job_config: Any
    job_lock: Any
    plans: tuple[TrialPlan, ...]
    is_resuming: bool = False


def _task_download_path(job: Any, trial_config: Any) -> Path | None:
    downloads = getattr(job, "_task_download_results", {})
    try:
        result = downloads[trial_config.task.get_task_id()]
    except (KeyError, AttributeError):
        return None
    path = getattr(result, "path", None)
    return Path(path) if path is not None else None


def _task_data(trial_config: Any, trial_lock: Any, task_dir: Path | None) -> TaskData:
    task_obj = None
    if task_dir is not None:
        try:
            from harbor.models.task.task import Task

            # Dataset input is task-authored semantics only. Run-specific extra
            # instructions are attached to agent_execution by the converter.
            task_obj = Task(
                task_dir,
                disable_verification=bool(getattr(getattr(trial_config, "verifier", None), "disable", False)),
            )
        except Exception:
            task_obj = None

    task_lock = getattr(trial_lock, "task", None)
    source = (
        getattr(getattr(trial_config, "task", None), "source", None) or getattr(task_lock, "source", None) or "adhoc"
    )
    name = (
        getattr(task_obj, "name", None)
        or getattr(task_lock, "name", None)
        or trial_config.task.get_task_id().get_name()
    )
    key = logical_task_key(trial_config, trial_lock)
    steps = getattr(getattr(task_obj, "config", None), "steps", None) or []
    if task_obj is not None and steps:
        canonical_input = {
            "task": name,
            "steps": [{"name": step.name, "instruction": task_obj.step_instruction(step.name)} for step in steps],
        }
    else:
        canonical_input = {
            "task": name,
            "instruction": getattr(task_obj, "instruction", ""),
        }

    task_config = getattr(task_obj, "config", None)
    user_metadata = dict(getattr(task_config, "metadata", None) or {})
    resources: dict[str, Any] = {}
    environment = getattr(task_config, "environment", None)
    if environment is not None:
        for field in ("cpus", "memory_mb", "storage_mb", "gpus", "tpu", "os"):
            value = getattr(environment, field, None)
            if value is not None:
                resources[field] = getattr(value, "value", value)
    metadata = {
        "harbor": {
            "source": source,
            "logical_task_key": key,
            "task_digest": getattr(task_lock, "digest", None),
            "schema_version": getattr(task_config, "schema_version", None),
            "resources": resources,
            "custom": user_metadata,
        }
    }
    return TaskData(
        logical_key=key,
        source=source,
        name=name,
        input=canonical_input,
        expected=None,
        metadata=metadata,
        digest=getattr(task_lock, "digest", None),
        schema_version=getattr(task_config, "schema_version", None),
        task_dir=task_dir,
    )


def snapshot_job(job: Any) -> JobSnapshot:
    """Feature-detect the read-only resolved-plan fields allowed by the design."""
    trial_configs = tuple(getattr(job, "_trial_configs"))
    if not trial_configs:
        raise ValueError("Harbor job has no resolved trial configurations")

    job_lock = getattr(job, "_job_lock", None)
    if job_lock is None:
        from harbor.models.job.lock import build_job_lock

        job_lock = build_job_lock(
            config=job.config,
            trial_configs=trial_configs,
            task_download_results=getattr(job, "_task_download_results"),
        )
    locks = tuple(job_lock.trials)
    if len(locks) != len(trial_configs):
        raise ValueError("Harbor trial plan and lock have different lengths")

    attempts: dict[tuple[str, str], int] = {}
    plans: list[TrialPlan] = []
    for trial_config, trial_lock in zip(trial_configs, locks):
        task_dir = _task_download_path(job, trial_config)
        task = _task_data(trial_config, trial_lock, task_dir)
        agent = trial_config.agent
        attempt_key = (task.logical_key, json.dumps(agent.model_dump(mode="json", exclude_none=True), sort_keys=True))
        attempt_index = attempts.get(attempt_key, 0)
        attempts[attempt_key] = attempt_index + 1
        plans.append(TrialPlan(trial_config.trial_name, trial_config, trial_lock, task, attempt_index))

    return JobSnapshot(
        job_id=str(job.id),
        job_name=str(job.config.job_name),
        job_dir=Path(job.job_dir),
        job_config=job.config,
        job_lock=job_lock,
        plans=tuple(plans),
        is_resuming=bool(getattr(job, "is_resuming", False)),
    )


def trial_directory(result: Any) -> Path:
    config = result.config
    return Path(config.trials_dir) / result.trial_name


def _step_paths(result: Any, *parts: str) -> list[tuple[str | None, Path]]:
    """Resolve one per-step file, encoding Harbor's on-disk step layout once.

    The step name travels with the path so callers can attribute a file to the step
    that produced it; it is None for a single-phase trial, which has no steps/ level.
    """
    base = trial_directory(result)
    if result.step_results:
        return [(step.step_name, base.joinpath("steps", step.step_name, *parts)) for step in result.step_results]
    return [(None, base.joinpath(*parts))]


def trajectory_paths(result: Any) -> list[tuple[str | None, Path]]:
    return _step_paths(result, "agent", "trajectory.json")


def reward_details_paths(result: Any) -> list[tuple[str | None, Path]]:
    return _step_paths(result, "verifier", "reward-details.json")


def artifact_manifest_paths(result: Any) -> list[tuple[str | None, Path]]:
    return _step_paths(result, "artifacts", "manifest.json")


def load_backfill_snapshot(job_dir: str | Path) -> tuple[JobSnapshot, Any]:
    """Load persisted Harbor models for offline backfill."""
    from harbor.models.job.config import JobConfig
    from harbor.models.job.lock import JobLock, TrialLock
    from harbor.models.job.result import JobResult
    from harbor.models.trial.result import TrialResult

    directory = Path(job_dir).expanduser().resolve()
    config = JobConfig.model_validate_json((directory / "config.json").read_text())
    lock = JobLock.model_validate_json((directory / "lock.json").read_text())
    job_result = JobResult.model_validate_json((directory / "result.json").read_text())
    results: list[Any] = []
    result_paths = {
        result_path.parent: result_path
        for pattern in ("*/results.json", "*/result.json")
        for result_path in sorted(directory.glob(pattern))
    }
    for result_path in result_paths.values():
        try:
            trial_result = TrialResult.model_validate_json(result_path.read_text())
            # A job directory may be moved before backfill. Resolve trial files
            # relative to the directory being backfilled, not the old jobs_dir.
            trial_result.config.trials_dir = directory
            results.append(trial_result)
        except Exception:
            logger.warning("Skipping unreadable Harbor trial result %s", result_path, exc_info=True)
            continue
    if not results and job_result.trial_results:
        results = list(job_result.trial_results)
    job_result.trial_results = results

    # TrialResult.task_checksum is a task-directory hash while TaskLock.digest is
    # a content digest, so the two are not comparable. Correlate on task name and
    # accept no lock rather than attributing an unrelated trial's lock, whose task
    # identity and skills would silently collapse distinct tasks and partitions.
    lock_by_name: dict[str, list[Any]] = {}
    for item in lock.trials:
        lock_by_name.setdefault(item.task.name, []).append(item)
    attempts: dict[tuple[str, str], int] = {}
    plans: list[TrialPlan] = []
    for result in results:
        trial_lock_path = trial_directory(result) / "lock.json"
        if trial_lock_path.exists():
            trial_lock = TrialLock.model_validate_json(trial_lock_path.read_text())
        else:
            candidates = lock_by_name.get(result.task_name) or []
            trial_lock = candidates.pop(0) if candidates else None
            if trial_lock is None:
                logger.warning("No job lock entry matches Harbor task %r; continuing without it", result.task_name)
        task_dir = None
        try:
            candidate = result.config.task.get_task_id().get_local_path()
            task_dir = candidate if candidate.exists() else None
        except Exception:
            pass
        task = _task_data(result.config, trial_lock, task_dir)
        agent_key = json.dumps(result.config.agent.model_dump(mode="json", exclude_none=True), sort_keys=True)
        attempt_key = (task.logical_key, agent_key)
        attempt_index = attempts.get(attempt_key, 0)
        attempts[attempt_key] = attempt_index + 1
        plans.append(TrialPlan(result.trial_name, result.config, trial_lock, task, attempt_index))

    snapshot = JobSnapshot(
        job_id=str(job_result.id),
        job_name=config.job_name,
        job_dir=directory,
        job_config=config,
        job_lock=lock,
        plans=tuple(plans),
        is_resuming=True,
    )
    return snapshot, job_result
