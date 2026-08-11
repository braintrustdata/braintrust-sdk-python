"""Native Braintrust job plugin for Harbor."""

# Harbor is optional and only supports Python 3.12+, while pylint runs across
# Braintrust's full Python matrix without installing Harbor.
# pylint: disable=import-error

import asyncio
import fnmatch
import json
import logging
import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from braintrust.logger import Attachment, flush, init, init_dataset
from exceptiongroup import ExceptionGroup

from .atif import _INSTRUMENTATION, ATIFImportResult, import_trajectory, summarize_trajectory
from .compat import (
    JobSnapshot,
    TrialPlan,
    artifact_manifest_paths,
    load_backfill_snapshot,
    reward_details_paths,
    snapshot_job,
    trajectory_paths,
)
from .config import _UNSET, PluginConfig
from .identity import (
    canonical_json,
    child_span_id,
    dataset_display_name,
    dataset_record_id,
    dataset_scope,
    normalize_json,
    partition_key,
    semantic_agent_config,
)
from .rewards import classify_rewards, extract_json_path, validate_classifications
from .state import (
    JobEvent,
    JobMachine,
    TrialEvent,
    TrialEventKind,
    TrialMachine,
    TrialStatus,
    accepts_trial_events,
    can_reconcile,
    reduce_job,
    reduce_trial,
)


logger = logging.getLogger(__name__)
_PLUGIN_VERSION = "1"
_MANIFEST_VERSION = 1


@dataclass
class DatasetBinding:
    scope: str
    dataset: Any = None
    origins: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Partition:
    key: str
    name: str
    dataset_scope: str
    experiment: Any = None
    experiment_id: str | None = None


@dataclass
class RuntimeState:
    snapshot: JobSnapshot
    plan_by_trial: dict[str, TrialPlan]
    partition_by_trial: dict[str, Partition]
    datasets: dict[str, DatasetBinding]
    partitions: dict[str, Partition]


def _seconds(value: Any, fallback: float) -> float:
    if isinstance(value, datetime):
        # Harbor trial timestamps are timezone-aware, but job timestamps are
        # currently naive local datetimes. datetime.timestamp() preserves both
        # conventions; assigning UTC to a naive value shifts non-UTC jobs.
        return value.timestamp()
    return fallback


def _timing(value: Any, default_start: float, default_end: float) -> tuple[float, float]:
    start = _seconds(getattr(value, "started_at", None), default_start)
    end = _seconds(getattr(value, "finished_at", None), default_end)
    if end < start:
        end = start
    return start, end


def _exception(result: Any, include_traceback: bool) -> tuple[str | None, str | None]:
    info = getattr(result, "exception_info", None)
    if info is None:
        return None, None
    error = f"{info.exception_type}: {info.exception_message}"
    traceback_value = info.exception_traceback if include_traceback else None
    return error, traceback_value


def _answer_from_metadata(result: Any) -> Any:
    contexts = []
    if getattr(result, "agent_result", None) is not None:
        contexts.append(result.agent_result)
    for step in getattr(result, "step_results", None) or []:
        if getattr(step, "agent_result", None) is not None:
            contexts.append(step.agent_result)
    for context in reversed(contexts):
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        for key in ("standardized_answer", "final_answer", "answer", "output", "response"):
            if key in metadata:
                return metadata[key]
    return None


def _rewards(result: Any) -> dict[str, Any]:
    verifier = getattr(result, "verifier_result", None)
    raw = getattr(verifier, "rewards", None)
    return dict(raw or {})


def _by_step(items: list[tuple[str | None, Any]], default_key: str, *, keep_single_name: bool = True) -> Any:
    """Collapse per-step values into one metadata value, or None when there are none.

    ``keep_single_name`` decides what a single value from a *named* step becomes.
    Trajectory metadata keeps the label, because which step produced the totals is
    part of the answer; the eval-root output drops it, because a single-step trial's
    answer should read as the answer rather than as a one-entry map.
    """
    if not items:
        return None
    if len(items) == 1 and (items[0][0] is None or not keep_single_name):
        return items[0][1]
    return {name or default_key: value for name, value in items}


def _step_label(step_name: str | None, path: Path) -> str:
    return path.name if step_name is None else f"{step_name}/{path.name}"


def _read_json_summary(entries: list[tuple[str | None, Path]], max_bytes: int) -> tuple[Any, list[str]]:
    summaries: list[tuple[str | None, Any]] = []
    warnings: list[str] = []
    for step_name, path in entries:
        label = _step_label(step_name, path)
        try:
            size = path.stat().st_size
            with path.open("rb") as file_obj:
                data = file_obj.read(min(size, max_bytes) + 1)
            if len(data) > max_bytes:
                warnings.append(f"{label} omitted: size limit")
                continue
            summaries.append((step_name, json.loads(data)))
        except FileNotFoundError:
            continue
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"could not read {label}: {exc}")
    return _by_step(summaries, "manifest"), warnings


def _artifact_attachments(result: Any, config: PluginConfig) -> tuple[dict[str, Attachment], list[str]]:
    if config.attachments != "all" or not config.artifact_include:
        return {}, []
    attachments: dict[str, Attachment] = {}
    warnings: list[str] = []
    total = 0
    for step_name, manifest_path in artifact_manifest_paths(result):
        root = manifest_path.parent.resolve()
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json" or path.is_symlink():
                continue
            try:
                resolved = path.resolve()
                relative = resolved.relative_to(root).as_posix()
            except (OSError, ValueError):
                warnings.append(f"artifact {path.name} omitted: unsafe path")
                continue
            if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in config.artifact_include):
                continue
            # Each step has its own artifacts root, so the relative path alone
            # collides whenever two steps collect the same file name.
            key = relative if step_name is None else f"{step_name}/{relative}"
            try:
                size = resolved.stat().st_size
                if size > config.max_attachment_bytes or total + size > config.max_total_attachment_bytes:
                    warnings.append(f"artifact {key} omitted: attachment size limit")
                    continue
                data = resolved.read_bytes()
            except OSError as exc:
                warnings.append(f"artifact {key} omitted: {exc}")
                continue
            total += len(data)
            attachments[key] = Attachment(
                data=data,
                filename=resolved.name,
                content_type="application/octet-stream",
            )
    return attachments, warnings


def _attachment(
    entries: list[tuple[str | None, Path]], config: PluginConfig
) -> tuple[Attachment | None, Any, list[str]]:
    if config.attachments == "none":
        return None, None, []
    total = 0
    complete: list[tuple[str | None, Any]] = []
    warnings: list[str] = []
    filename = "details.json"
    for step_name, path in entries:
        label = _step_label(step_name, path)
        filename = path.name
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            warnings.append(f"could not read {label}: {exc}")
            continue
        if len(data) > config.max_attachment_bytes or total + len(data) > config.max_total_attachment_bytes:
            warnings.append(f"{label} omitted: attachment size limit")
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            warnings.append(f"{label} is not valid JSON")
            continue
        normalized = normalize_json(
            parsed,
            max_bytes=config.max_attachment_bytes,
            redact_patterns=config.redact_patterns,
            max_depth=20,
        )
        warnings.extend(normalized.warnings)
        complete.append((step_name, normalized.value))
        total += len(data)
    summary = _by_step(complete, "details")
    if summary is None:
        return None, None, warnings
    attachment_data = (canonical_json(summary) + "\n").encode()
    # One serialized payload is bounded by the per-file limit, not the job total.
    if len(attachment_data) > config.max_attachment_bytes:
        warnings.append(f"{filename} omitted after redaction: attachment size limit")
        return None, summary, warnings
    return (
        Attachment(data=attachment_data, filename=filename, content_type="application/json"),
        summary,
        warnings,
    )


class HarborPlugin:
    """Harbor plugin that reconciles final trials into Braintrust experiments."""

    def __init__(
        self,
        project_name: Any = _UNSET,
        project_id: Any = _UNSET,
        experiment_prefix: Any = _UNSET,
        base_experiment_name: Any = _UNSET,
        base_experiment_id: Any = _UNSET,
        dataset_mode: Any = _UNSET,
        dataset_name: Any = _UNSET,
        trajectory_mode: Any = _UNSET,
        content_mode: Any = _UNSET,
        include_custom_metadata: Any = _UNSET,
        max_custom_metadata_bytes: Any = _UNSET,
        score_keys: Any = _UNSET,
        metric_keys: Any = _UNSET,
        reward_rules: Any = _UNSET,
        classifier_rules: Any = _UNSET,
        invalid_score_policy: Any = _UNSET,
        include_tracebacks: Any = _UNSET,
        attachments: Any = _UNSET,
        artifact_include: Any = _UNSET,
        max_attachment_bytes: Any = _UNSET,
        max_total_attachment_bytes: Any = _UNSET,
        max_content_bytes: Any = _UNSET,
        log_retry_attempts: Any = _UNSET,
        strict: Any = _UNSET,
        **kwargs: Any,
    ) -> None:
        options = {
            "project_name": project_name,
            "project_id": project_id,
            "experiment_prefix": experiment_prefix,
            "base_experiment_name": base_experiment_name,
            "base_experiment_id": base_experiment_id,
            "dataset_mode": dataset_mode,
            "dataset_name": dataset_name,
            "trajectory_mode": trajectory_mode,
            "content_mode": content_mode,
            "include_custom_metadata": include_custom_metadata,
            "max_custom_metadata_bytes": max_custom_metadata_bytes,
            "score_keys": score_keys,
            "metric_keys": metric_keys,
            "reward_rules": reward_rules,
            "classifier_rules": classifier_rules,
            "invalid_score_policy": invalid_score_policy,
            "include_tracebacks": include_tracebacks,
            "attachments": attachments,
            "artifact_include": artifact_include,
            "max_attachment_bytes": max_attachment_bytes,
            "max_total_attachment_bytes": max_total_attachment_bytes,
            "max_content_bytes": max_content_bytes,
            "log_retry_attempts": log_retry_attempts,
            "strict": strict,
            **kwargs,
        }
        unknown = set(options) - {config_field.name for config_field in fields(PluginConfig)}
        if unknown:
            raise TypeError(f"Unexpected HarborPlugin options: {', '.join(sorted(unknown))}")
        self.config = PluginConfig.from_options(**options)
        self._job_machine = JobMachine()
        self._trial_machines: dict[str, TrialMachine] = {}
        self._trial_locks: dict[str, asyncio.Lock] = {}
        self._runtime: RuntimeState | None = None
        self._snapshot: JobSnapshot | None = None
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._manifest: dict[str, Any] = {}
        self._disabled_reason: str | None = None

    async def on_job_start(self, job: Any) -> None:
        self._job_machine = reduce_job(self._job_machine, JobEvent.INITIALIZE, strict=self.config.strict)
        try:
            snapshot = await asyncio.to_thread(snapshot_job, job)
            self._snapshot = snapshot
            self._runtime = await asyncio.to_thread(self._initialize, snapshot)
            for plan in snapshot.plans:
                self._trial_machines[plan.trial_name] = TrialMachine(plan.trial_name)
                self._trial_locks[plan.trial_name] = asyncio.Lock()
            self._register_hooks(job)
            self._job_machine = reduce_job(self._job_machine, JobEvent.READY, strict=self.config.strict)
            await asyncio.to_thread(self._persist_manifest, False)
        except Exception as exc:
            self._disable(f"Braintrust initialization failed: {exc}")
            if self._snapshot is not None:
                try:
                    await asyncio.to_thread(self._persist_disabled_manifest)
                except OSError as persist_exc:
                    self._errors.append(f"could not persist disabled manifest: {persist_exc}")
            if self.config.strict:
                raise

    async def on_job_end(self, job_result: Any) -> None:
        if self._runtime is None:
            return
        if not can_reconcile(self._job_machine):
            # Initialization failed after the runtime was built. Reconciling now
            # would write a full experiment while the manifest reports the sync as
            # disabled, and RECONCILE out of a terminal status is not legal.
            logger.warning(
                "Skipping Braintrust reconciliation while %s: %s",
                self._job_machine.status.value,
                self._disabled_reason or "job is not active",
            )
            return
        self._job_machine = reduce_job(self._job_machine, JobEvent.RECONCILE, strict=self.config.strict)
        final_names = {result.trial_name for result in job_result.trial_results}
        failures: list[BaseException] = []

        async def reconcile(result: Any) -> None:
            try:
                await self._dispatch(result.trial_name, TrialEvent(TrialEventKind.FINAL_RESULT, payload=result))
                await asyncio.to_thread(self._sync_final_result, result)
                await self._dispatch(result.trial_name, TrialEvent(TrialEventKind.SYNCED))
            except Exception as exc:
                failures.append(exc)
                self._errors.append(f"trial {result.trial_name}: {exc}")
                try:
                    await self._dispatch(result.trial_name, TrialEvent(TrialEventKind.SYNC_FAILED, payload=str(exc)))
                except Exception:
                    pass

        await asyncio.gather(*(reconcile(result) for result in job_result.trial_results))
        for name in set(self._trial_machines) - final_names:
            await self._dispatch(name, TrialEvent(TrialEventKind.OMIT))
        try:
            await asyncio.to_thread(flush)
        except Exception as exc:
            failures.append(exc)
            self._errors.append(f"final flush: {exc}")
        self._job_machine = reduce_job(self._job_machine, JobEvent.CLOSE, strict=self.config.strict)
        try:
            await asyncio.to_thread(self._persist_manifest, not failures)
        except Exception as exc:
            failures.append(exc)
            self._errors.append(f"manifest persistence: {exc}")
            logger.warning("Could not persist Harbor Braintrust sync manifest", exc_info=True)
        if failures and self.config.strict:
            # Harbor isolates finalizers, so raising here cannot fail the run; log
            # at error level so a strict sync failure is not invisible. Direct
            # callers such as backfill still observe the exception.
            logger.error("Braintrust Harbor synchronization failed: %s", "; ".join(self._errors))
            raise ExceptionGroup("Braintrust Harbor synchronization failed", failures)

    def _disable(self, message: str) -> None:
        self._disabled_reason = message
        self._errors.append(message)
        self._job_machine = reduce_job(self._job_machine, JobEvent.DISABLE)
        logger.warning(message, exc_info=True)

    def _register_hooks(self, job: Any) -> None:
        from harbor.trial.hooks import TrialEvent as HarborTrialEvent

        mapping = {
            HarborTrialEvent.START: TrialEventKind.START,
            HarborTrialEvent.ENVIRONMENT_START: TrialEventKind.ENVIRONMENT_START,
            HarborTrialEvent.AGENT_START: TrialEventKind.AGENT_START,
            HarborTrialEvent.AGENT_END: TrialEventKind.AGENT_END,
            HarborTrialEvent.VERIFICATION_START: TrialEventKind.VERIFICATION_START,
            HarborTrialEvent.END: TrialEventKind.END,
            HarborTrialEvent.CANCEL: TrialEventKind.CANCEL,
        }
        max_retries = int(getattr(getattr(job.config, "retry", None), "max_retries", 0) or 0)
        for harbor_event, internal_kind in mapping.items():

            async def callback(event: Any, kind: TrialEventKind = internal_kind) -> None:
                try:
                    machine = self._trial_machines.get(event.trial_name)
                    retry_predicted = bool(
                        kind == TrialEventKind.END
                        and machine is not None
                        and machine.retry_index < max_retries
                        and getattr(event.result, "exception_info", None) is not None
                    )
                    await self._dispatch(
                        event.trial_name,
                        TrialEvent(
                            kind,
                            timestamp=event.timestamp.timestamp(),
                            payload=event.result if kind == TrialEventKind.END else None,
                            retry_predicted=retry_predicted,
                        ),
                    )
                except Exception as exc:
                    self._errors.append(f"hook {kind.value} for {event.trial_name}: {exc}")
                    if self.config.strict:
                        raise

            job.add_hook(harbor_event, callback)

    async def _dispatch(self, identity: str, event: TrialEvent) -> None:
        if not accepts_trial_events(self._job_machine) and event.kind not in {
            TrialEventKind.FINAL_RESULT,
            TrialEventKind.SYNCED,
            TrialEventKind.SYNC_FAILED,
            TrialEventKind.OMIT,
        }:
            return
        if identity not in self._trial_machines:
            self._trial_machines[identity] = TrialMachine(identity)
            self._trial_locks[identity] = asyncio.Lock()
        async with self._trial_locks[identity]:
            new_state, _effects = reduce_trial(
                self._trial_machines[identity],
                event,
                strict=self.config.strict,
            )
            self._trial_machines[identity] = new_state

    def _initialize(self, snapshot: JobSnapshot) -> RuntimeState:
        previous = self._load_manifest(snapshot.job_dir)
        self._manifest = previous
        plan_by_trial = {plan.trial_name: plan for plan in snapshot.plans}
        datasets: dict[str, DatasetBinding] = {}
        source_tasks: dict[str, dict[str, Any]] = {}
        for plan in snapshot.plans:
            source_tasks.setdefault(plan.task.source, {})[plan.task.logical_key] = plan.task

        for source, task_map in source_tasks.items():
            scope = dataset_scope(source)
            binding = DatasetBinding(scope)
            datasets[scope] = binding
            if self.config.dataset_mode != "sync":
                continue
            try:
                if self.config.dataset_name and len(source_tasks) == 1:
                    name = self.config.dataset_name
                else:
                    name = dataset_display_name(source, prefix=self.config.dataset_name or "harbor")
                dataset = init_dataset(
                    project=self.config.project_name,
                    project_id=self.config.project_id,
                    name=name,
                    use_output=False,
                    metadata={"harbor": {"source": source, "scope": scope, "schema_version": _PLUGIN_VERSION}},
                )
                for task in task_map.values():
                    normalized = normalize_json(
                        task.metadata,
                        max_bytes=self.config.max_custom_metadata_bytes,
                        redact_patterns=self.config.redact_patterns,
                    )
                    self._warnings.extend(normalized.warnings)
                    dataset.insert(
                        id=dataset_record_id(scope, task.logical_key),
                        input=task.input,
                        expected=task.expected,
                        metadata=normalized.value,
                    )
                dataset.flush()
                rows = list(dataset)
                for row in rows:
                    if row.get("id") and row.get("_xact_id"):
                        binding.origins[row["id"]] = {
                            "object_type": "dataset",
                            "object_id": dataset.id,
                            "id": row["id"],
                            "created": row.get("created"),
                            "_xact_id": row["_xact_id"],
                        }
                binding.dataset = dataset
            except Exception as exc:
                binding.error = str(exc)
                self._warnings.append(f"dataset {scope} sync failed; continuing without association: {exc}")

        partitions: dict[str, Partition] = {}
        partition_by_trial: dict[str, Partition] = {}
        for plan in snapshot.plans:
            scope = dataset_scope(plan.task.source)
            semantic = semantic_agent_config(
                plan.trial_config.agent, list(getattr(plan.trial_lock, "skills", []) or [])
            )
            key = partition_key(scope, semantic)
            partition = partitions.get(key)
            if partition is None:
                agent_name = (
                    getattr(plan.trial_config.agent, "name", None)
                    or getattr(plan.trial_config.agent, "import_path", None)
                    or "agent"
                )
                model = getattr(plan.trial_config.agent, "model_name", None) or "default"
                prefix = self.config.experiment_prefix or snapshot.job_name
                name = f"{prefix}-{snapshot.job_id[:8]} · {agent_name}@{model} · {plan.task.source} · {key[:8]}"
                metadata = {
                    "harbor": {
                        "job_id": snapshot.job_id,
                        "job_name": snapshot.job_name,
                        "partition_key": key,
                        "semantic_agent_config": semantic,
                    }
                }
                dataset = datasets[scope].dataset
                experiment = init(
                    project=self.config.project_name,
                    project_id=self.config.project_id,
                    experiment=name,
                    update=True,
                    dataset=dataset,
                    metadata=metadata,
                    base_experiment=self.config.base_experiment_name,
                    base_experiment_id=self.config.base_experiment_id,
                )
                partition = Partition(key=key, name=name, dataset_scope=scope, experiment=experiment)
                # Resolve lazy metadata now so initialization/auth failures are isolated.
                partition.experiment_id = experiment.id
                partitions[key] = partition
            partition_by_trial[plan.trial_name] = partition

        return RuntimeState(snapshot, plan_by_trial, partition_by_trial, datasets, partitions)

    def _root_metadata(self, result: Any, plan: TrialPlan, machine: TrialMachine) -> dict[str, Any]:
        raw_rewards = _rewards(result)
        trial_custom = getattr(getattr(result, "agent_result", None), "metadata", None) or {}
        normalized = normalize_json(
            trial_custom if self.config.include_custom_metadata else {},
            max_bytes=self.config.max_custom_metadata_bytes,
            redact_patterns=self.config.redact_patterns,
        )
        task_custom = normalize_json(
            plan.task.metadata.get("harbor", {}).get("custom", {}) if self.config.include_custom_metadata else {},
            max_bytes=self.config.max_custom_metadata_bytes,
            redact_patterns=self.config.redact_patterns,
        )
        self._warnings.extend((*normalized.warnings, *task_custom.warnings))
        error, traceback_value = _exception(result, self.config.include_tracebacks)
        metadata: dict[str, Any] = {
            "harbor": {
                "job_id": self._runtime.snapshot.job_id if self._runtime else None,
                "trial_id": str(result.id),
                "task_name": result.task_name,
                "agent": result.agent_info.name,
                "model": result.agent_info.model_info.name if result.agent_info.model_info else None,
                "attempt_index": plan.attempt_index,
                "retry_index": machine.retry_index,
                "raw_rewards": raw_rewards,
                "custom": {"task": task_custom.value, "trial": normalized.value},
                "warnings": list(machine.warnings),
            }
        }
        if error and traceback_value:
            metadata["harbor"]["exception_traceback"] = traceback_value
        return metadata

    def _start_phase(
        self,
        task_span: Any,
        result: Any,
        name: str,
        timing_name: str,
        trial_id: str,
        root_start: float,
        root_end: float,
        **event: Any,
    ) -> Any:
        start, end = _timing(getattr(result, timing_name, None), root_start, root_end)
        span = task_span.start_span(
            name=name,
            type="task",
            id=child_span_id(trial_id, f"task/{name}"),
            start_time=start,
            set_current=False,
            internal={"instrumentation": _INSTRUMENTATION},
            **event,
        )
        span.end(end_time=end)
        return span

    def _sync_final_result(self, result: Any) -> None:
        if self._runtime is None:
            raise RuntimeError("plugin is not initialized")
        plan = self._runtime.plan_by_trial.get(result.trial_name)
        partition = self._runtime.partition_by_trial.get(result.trial_name)
        if plan is None or partition is None:
            raise ValueError(f"final result {result.trial_name!r} is absent from the resolved plan")
        machine = self._trial_machines[result.trial_name]
        trial_id = str(result.id)
        now = datetime.now().timestamp()
        root_start = _seconds(getattr(result, "started_at", None), now)
        root_end = _seconds(getattr(result, "finished_at", None), root_start)
        if root_end < root_start:
            root_end = root_start
        error, _ = _exception(result, self.config.include_tracebacks)
        metadata = self._root_metadata(result, plan, machine)
        rewards = _rewards(result)
        conversion = classify_rewards(rewards, self.config)
        metadata["harbor"]["warnings"].extend(conversion.warnings)
        if not rewards and error is None:
            metadata["harbor"]["warnings"].append("trial has no reward and is unevaluated")

        binding = self._runtime.datasets[partition.dataset_scope]
        record_id = dataset_record_id(partition.dataset_scope, plan.task.logical_key)
        origin = binding.origins.get(record_id)
        root_metrics = dict(conversion.metrics)
        if machine.completed_attempts:
            root_metrics["retries"] = max(machine.completed_attempts - 1, machine.retry_index)
        root_event: dict[str, Any] = {
            "id": trial_id,
            "name": "eval",
            "type": "eval",
            "start_time": root_start,
            "set_current": False,
            "input": plan.task.input,
            "expected": plan.task.expected,
            "metadata": metadata,
            "metrics": root_metrics,
        }
        if origin:
            root_event["origin"] = origin
        if error:
            root_event["error"] = error
        root = partition.experiment.start_span(
            internal={"instrumentation": _INSTRUMENTATION},
            **root_event,
        )
        task = root.start_span(
            name="task",
            type="task",
            id=child_span_id(trial_id, "task"),
            start_time=root_start,
            set_current=False,
            input=plan.task.input,
            expected=plan.task.expected,
            error=error,
            internal={"instrumentation": _INSTRUMENTATION},
        )

        self._start_phase(task, result, "environment_setup", "environment_setup", trial_id, root_start, root_end)
        self._start_phase(task, result, "agent_setup", "agent_setup", trial_id, root_start, root_end)
        agent_start, agent_end = _timing(getattr(result, "agent_execution", None), root_start, root_end)
        execution_input: dict[str, Any] = {"task": plan.task.input}
        extra_instructions: list[str] = []
        for path in getattr(result.config, "extra_instruction_paths", []) or []:
            try:
                extra_instructions.append(Path(path).read_text())
            except OSError:
                continue
        if extra_instructions:
            execution_input["extra_instructions"] = extra_instructions
        selected_artifacts, artifact_attachment_warnings = _artifact_attachments(result, self.config)
        metadata["harbor"]["warnings"].extend(artifact_attachment_warnings)
        agent_span = task.start_span(
            name="agent_execution",
            type="task",
            id=child_span_id(trial_id, "task/agent_execution"),
            start_time=agent_start,
            set_current=False,
            input=normalize_json(
                execution_input, max_bytes=self.config.max_content_bytes, redact_patterns=self.config.redact_patterns
            ).value,
            internal={"instrumentation": _INSTRUMENTATION},
        )

        atif_results: list[tuple[str | None, ATIFImportResult]] = []
        if self.config.trajectory_mode in {"atif", "summary"}:
            for step_name, path in trajectory_paths(result):
                if self.config.trajectory_mode == "summary":
                    imported = summarize_trajectory(path, self.config)
                else:
                    prefix = "task/agent_execution" if step_name is None else f"task/step:{step_name}/agent_execution"
                    imported = import_trajectory(
                        agent_span,
                        path,
                        trial_id=trial_id,
                        semantic_prefix=prefix,
                        phase_start=agent_start,
                        phase_end=agent_end,
                        config=self.config,
                    )
                atif_results.append((step_name, imported))
        if selected_artifacts:
            agent_span.log(output={"artifacts": selected_artifacts})
        agent_span.end(end_time=agent_end)
        self._start_phase(task, result, "verification", "verifier", trial_id, root_start, root_end)

        for step in getattr(result, "step_results", None) or []:
            step_start, step_end = _timing(getattr(step, "agent_execution", None), root_start, root_end)
            step_span = task.start_span(
                name=f"step:{step.step_name}",
                type="task",
                id=child_span_id(trial_id, f"task/step:{step.step_name}"),
                start_time=step_start,
                set_current=False,
                internal={"instrumentation": _INSTRUMENTATION},
            )
            step_error, _ = _exception(step, self.config.include_tracebacks)
            if step_error:
                step_span.log(error=step_error)
            step_span.end(end_time=step_end)

        trajectory_warnings = [warning for _, imported in atif_results for warning in imported.warnings]
        repairs = [repair for _, imported in atif_results for repair in imported.repairs]
        metadata["harbor"]["warnings"].extend(trajectory_warnings)
        metadata["harbor"]["trajectory"] = {
            # Report the mode so trajectory_mode="native", which deliberately skips
            # ATIF because the agent is instrumented elsewhere, is distinguishable
            # from a trajectory that could not be read.
            "mode": self.config.trajectory_mode,
            "present": bool(atif_results),
            "schema_version": next(
                (imported.schema_version for _, imported in atif_results if imported.schema_version), None
            ),
            "repairs": repairs,
        }
        # A multi-step trial has one trajectory per step, each with its own final
        # message and its own aggregate token and cost totals.
        raw_extra = _by_step(
            [(name, imported.root_extra) for name, imported in atif_results if imported.root_extra], "trajectory"
        )
        if raw_extra is not None and self.config.include_custom_metadata:
            normalized_extra = normalize_json(
                raw_extra,
                max_bytes=self.config.max_custom_metadata_bytes,
                redact_patterns=self.config.redact_patterns,
            )
            metadata["harbor"]["trajectory"]["custom"] = normalized_extra.value
            metadata["harbor"]["warnings"].extend(normalized_extra.warnings)

        output = _answer_from_metadata(result)
        if output is None:
            output = _by_step(
                [
                    (name, imported.final_message)
                    for name, imported in atif_results
                    if imported.final_message is not None
                ],
                "final",
                keep_single_name=False,
            )
        if output is None:
            output = {"status": "completed" if error is None else "error"}
        output = normalize_json(
            output, max_bytes=self.config.max_content_bytes, redact_patterns=self.config.redact_patterns
        ).value
        if error is None:
            task.log(output=output)
            root.log(output=output)
        task.log(metadata={"harbor": {"warnings": trajectory_warnings}})
        task.end(end_time=root_end)

        details_attachment, details_summary, detail_warnings = _attachment(reward_details_paths(result), self.config)
        metadata["harbor"]["warnings"].extend(detail_warnings)
        # The summary is the same for every score, so bound it once rather than
        # re-normalizing a payload up to max_attachment_bytes per scorer span.
        bounded_details = (
            None
            if details_summary is None
            else normalize_json(
                details_summary,
                max_bytes=self.config.max_content_bytes,
                redact_patterns=self.config.redact_patterns,
            ).value
        )
        for score in conversion.scores:
            scorer = root.start_span(
                name=score.name,
                type="score",
                span_attributes={"purpose": "scorer"},
                id=child_span_id(trial_id, f"scorer/{score.source_key}"),
                start_time=root_end,
                set_current=False,
                input={"reward": score.raw_value},
                internal={"instrumentation": _INSTRUMENTATION},
            )
            scorer_output: dict[str, Any] = {"score": score.value, "raw_reward": score.raw_value}
            if bounded_details is not None:
                scorer_output["reward_details_summary"] = bounded_details
            if details_attachment is not None:
                scorer_output["reward_details"] = details_attachment
            scorer.log(output=scorer_output, scores={score.name: score.value})
            scorer.end(end_time=root_end)

        classifications: dict[str, list[dict[str, Any]]] = {}
        for source_name, path in self.config.classifier_rules.items():
            classifier = root.start_span(
                name=source_name,
                type="classifier",
                span_attributes={"purpose": "scorer"},
                id=child_span_id(trial_id, f"classifier/{source_name}"),
                start_time=root_end,
                set_current=False,
                internal={"instrumentation": _INSTRUMENTATION},
            )
            try:
                items = validate_classifications(extract_json_path(result, path))
                if items:
                    classifications[source_name] = items
                    classifier.log(output=items[0] if len(items) == 1 else items)
            except Exception as exc:
                classifier.log(error=f"invalid classifier {source_name}: {exc}")
                metadata["harbor"]["warnings"].append(f"classifier {source_name!r} was malformed: {exc}")
            classifier.end(end_time=root_end)
        if classifications:
            root.log(classifications=classifications)

        manifests, manifest_warnings = _read_json_summary(
            artifact_manifest_paths(result), self.config.max_content_bytes
        )
        metadata["harbor"]["warnings"].extend(manifest_warnings)
        if manifests is not None:
            metadata["harbor"]["artifact_manifest"] = normalize_json(
                manifests, max_bytes=self.config.max_content_bytes
            ).value
        root.log(metadata=metadata)
        root.end(end_time=root_end)

    def _persist_disabled_manifest(self) -> None:
        if self._snapshot is None:
            return
        manifest = {
            "manifest_version": _MANIFEST_VERSION,
            "plugin_version": _PLUGIN_VERSION,
            "job_id": self._snapshot.job_id,
            "project": {"name": self.config.project_name, "id": self.config.project_id},
            "datasets": {},
            "experiments": {},
            "trials": {},
            "synced_trial_ids": [],
            "warnings": self._warnings,
            "errors": self._errors,
            "disabled_reason": self._disabled_reason,
            "completed": False,
        }
        path = self._snapshot.job_dir / "braintrust-sync.json"
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temp_path, path)
        self._manifest = manifest

    @staticmethod
    def _load_manifest(job_dir: Path) -> dict[str, Any]:
        path = job_dir / "braintrust-sync.json"
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _persist_manifest(self, completed: bool) -> None:
        if self._runtime is None:
            return
        snapshot = self._runtime.snapshot
        manifest = {
            "manifest_version": _MANIFEST_VERSION,
            "plugin_version": _PLUGIN_VERSION,
            "job_id": snapshot.job_id,
            "project": {"name": self.config.project_name, "id": self.config.project_id},
            "datasets": {
                scope: {
                    "id": binding.dataset.id if binding.dataset is not None else None,
                    "version": binding.dataset.version if binding.dataset is not None else None,
                    "error": binding.error,
                }
                for scope, binding in self._runtime.datasets.items()
            },
            "experiments": {
                key: {"id": partition.experiment_id, "name": partition.name}
                for key, partition in self._runtime.partitions.items()
            },
            "trials": {
                name: {
                    "status": machine.status.value,
                    "retry_count": machine.retry_index,
                    "completed_attempts": machine.completed_attempts,
                    "warnings": list(machine.warnings),
                }
                for name, machine in self._trial_machines.items()
            },
            "synced_trial_ids": sorted(
                str(machine.final_result.id)
                for machine in self._trial_machines.values()
                if machine.status == TrialStatus.SYNCED and machine.final_result is not None
            ),
            "warnings": self._warnings,
            "errors": self._errors,
            "disabled_reason": self._disabled_reason,
            "completed": completed,
        }
        path = snapshot.job_dir / "braintrust-sync.json"
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(temp_path, path)
        self._manifest = manifest

    async def sync_job_directory(self, job_dir: str | Path) -> None:
        """Backfill a persisted Harbor job directory with the online conversion core."""
        self._job_machine = reduce_job(self._job_machine, JobEvent.INITIALIZE, strict=self.config.strict)
        snapshot, result = await asyncio.to_thread(load_backfill_snapshot, job_dir)
        self._snapshot = snapshot
        self._runtime = await asyncio.to_thread(self._initialize, snapshot)
        for plan in snapshot.plans:
            self._trial_machines[plan.trial_name] = TrialMachine(plan.trial_name)
            self._trial_locks[plan.trial_name] = asyncio.Lock()
        self._job_machine = reduce_job(self._job_machine, JobEvent.READY, strict=self.config.strict)
        await self.on_job_end(result)


async def backfill_job(job_dir: str | Path, **plugin_options: Any) -> None:
    """Backfill a Harbor job directory into Braintrust."""
    await HarborPlugin(**plugin_options).sync_job_directory(job_dir)
