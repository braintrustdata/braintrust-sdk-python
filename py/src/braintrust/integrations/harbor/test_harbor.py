# Harbor is installed by the dedicated Python 3.12+ nox session, not by the
# cross-version pylint environment.
# pylint: disable=import-error

import asyncio
import inspect
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from braintrust import flush, init
from braintrust.conftest import get_vcr_config
from braintrust.git_fields import GitMetadataSettings
from braintrust.integrations.harbor.atif import _usage_metrics, import_trajectory, summarize_trajectory
from braintrust.integrations.harbor.compat import artifact_manifest_paths, load_backfill_snapshot
from braintrust.integrations.harbor.config import PluginConfig
from braintrust.integrations.harbor.identity import (
    child_span_id,
    dataset_record_id,
    dataset_scope,
    logical_task_key,
    normalize_json,
    partition_key,
    semantic_agent_config,
)
from braintrust.integrations.harbor.plugin import (
    HarborPlugin,
    RuntimeState,
    _artifact_attachments,
    _attachment,
    _resolve_project,
    _seconds,
    _timing,
)
from braintrust.integrations.harbor.rewards import classify_rewards, validate_classifications
from braintrust.integrations.harbor.state import (
    JobEvent,
    JobStatus,
    TrialEvent,
    TrialEventKind,
    TrialMachine,
    TrialPhase,
    TrialStatus,
    reduce_job,
    reduce_trial,
)
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.job.lock import AgentSkillLock, JobLock, TaskLock, TrialLock
from harbor.models.job.result import JobResult, JobStats
from harbor.models.task.id import LocalTaskId
from harbor.models.trajectories.trajectory import Trajectory
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, StepResult, TimingInfo, TrialResult


_ABSOLUTE_PATH_RE = re.compile(r"(?:/(?:Users|private|home)/[^\"\\\\\s]+|[A-Za-z]:\\\\[^\"\\\\\s]+)")


def _redact_cassette_body(body):
    if not isinstance(body, (str, bytes)):
        return body
    is_bytes = isinstance(body, bytes)
    text = body.decode("utf-8", errors="replace") if is_bytes else body
    redacted = _ABSOLUTE_PATH_RE.sub("[REDACTED_PATH]", text)
    return redacted.encode() if is_bytes else redacted


@pytest.fixture(scope="module")
def vcr_config():
    config = get_vcr_config()
    scrub_response = config["before_record_response"]

    def before_record_request(request):
        request.body = _redact_cassette_body(request.body)
        return request

    def before_record_response(response):
        response = scrub_response(response)
        body = response.get("body", {})
        if "string" in body:
            body["string"] = _redact_cassette_body(body["string"])
        return response

    return {
        **config,
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
    }


def test_config_environment_fallback_and_explicit_precedence():
    names = {
        "HARBOR_BRAINTRUST_PROJECT": "harbor-project",
        "HARBOR_BRAINTRUST_DATASET_MODE": "none",
        "HARBOR_BRAINTRUST_SCORE_KEYS": '["reward", "correct*"]',
        "HARBOR_BRAINTRUST_STRICT": "true",
    }
    original = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(names)
        from braintrust.integrations.harbor import HarborPlugin

        config = HarborPlugin(strict=False).config
        assert config.project_name == "harbor-project"
        assert config.dataset_mode == "none"
        assert config.score_keys == ("reward", "correct*")
        assert config.strict is False
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_plugin_defaults_project_to_harbor():
    assert _resolve_project(PluginConfig.from_options()) == ("Harbor", None)
    assert _resolve_project(PluginConfig.from_options(project_name="explicit-project")) == (
        "explicit-project",
        None,
    )
    assert _resolve_project(PluginConfig.from_options(project_id="project-id")) == (None, "project-id")


def test_harbor_resolves_the_plugin_through_its_entry_point():
    # Users select this plugin with `--plugin braintrust`, which Harbor resolves
    # through the harbor.plugins entry-point group. Nothing else in the test suite
    # exercises the packaging metadata, so a broken entry point would otherwise
    # only surface as "plugin not found" for a real user.
    from harbor.cli.plugin_registry import PLUGIN_ENTRY_POINT_GROUP, resolve_plugin_import_path

    assert PLUGIN_ENTRY_POINT_GROUP == "harbor.plugins"
    assert resolve_plugin_import_path("braintrust") == "braintrust.integrations.harbor:HarborPlugin"


def test_plugin_implements_the_harbor_lifecycle_protocol():
    # The oldest supported Harbor release is pinned in [tool.braintrust.matrix.harbor].
    # AGENT_END is the API that sets that floor: it arrived in 0.16.0, and the
    # lifecycle state machine subscribes to every event in this mapping, so a
    # missing member disables the whole plugin at registration time.
    from harbor.trial.hooks import TrialEvent as HarborTrialEvent

    for name in ("START", "ENVIRONMENT_START", "AGENT_START", "AGENT_END", "VERIFICATION_START", "END", "CANCEL"):
        assert hasattr(HarborTrialEvent, name), name

    plugin = HarborPlugin(project_name="unused")
    assert inspect.iscoroutinefunction(plugin.on_job_start)
    assert inspect.iscoroutinefunction(plugin.on_job_end)


def test_config_rejects_overlapping_reward_patterns_and_invalid_bounds():
    with pytest.raises(ValueError, match="overlap"):
        PluginConfig.from_options(score_keys=["correct*"], metric_keys=["correctness"])
    with pytest.raises(ValueError, match="min < max"):
        PluginConfig.from_options(reward_rules={"latency": {"type": "score", "min": 1, "max": 1}})


def test_reward_classification_is_semantic_not_range_based():
    config = PluginConfig.from_options(
        reward_rules={
            "error_rate": {
                "type": "score",
                "direction": "minimize",
                "min": 0,
                "max": 10,
                "score_name": "reliability",
            }
        },
        score_keys=["correctness"],
    )
    result = classify_rewards(
        {"reward": 0.8, "quality": 0.7, "correctness": 1, "error_rate": 2, "tokens": 40},
        config,
    )

    assert {score.name: score.value for score in result.scores} == {
        "reward": 0.8,
        "correctness": 1,
        "reliability": 0.8,
    }
    assert result.metrics == {
        "quality": 0.7,
        "harbor_reward.raw.error_rate": 2,
        "harbor_reward.tokens": 40,
    }


def test_invalid_configured_score_defaults_to_metric():
    config = PluginConfig.from_options(score_keys=["raw"])
    result = classify_rewards({"raw": 5}, config)
    assert result.scores == ()
    assert result.metrics == {"raw": 5}
    assert result.warnings


def test_classification_validation_is_atomic_and_preserves_duplicates():
    items = validate_classifications(
        [
            {"id": "cat", "label": "Cat", "metadata": {"confidence": "high"}},
            {"id": "cat", "label": None},
        ]
    )
    assert items[0]["id"] == items[1]["id"] == "cat"
    with pytest.raises(ValueError):
        validate_classifications([{"id": "ok"}, {"label": "missing id"}])


def test_metadata_normalization_redacts_secrets_paths_and_bounds_size(tmp_path):
    normalized = normalize_json(
        {
            "api_key": "secret",
            "nested": {"keep": None, "path": str(tmp_path / "private")},
            "message": "token=abc",
        },
        max_bytes=1_000,
        redact_patterns=(r"token=[a-z]+",),
    )
    assert normalized.value["api_key"] == "[REDACTED]"
    assert normalized.value["nested"]["keep"] is None
    assert normalized.value["nested"]["path"] == "[REDACTED PATH]"
    assert normalized.value["message"] == "[REDACTED]"
    assert any("absolute path" in warning for warning in normalized.warnings)
    assert normalized.complete is False


def test_secret_key_redaction_keeps_token_counters_and_credential_keys():
    normalized = normalize_json(
        {
            "max_tokens": 4096,
            "total_tokens": 17,
            "n_output_tokens": 3,
            "usage": {"prompt_tokens": 10, "cached_tokens": 2},
            "api_key": "sk-live-value",
            "github_token": "ghp-value",
            "authorization": "Bearer value",
        },
        max_bytes=10_000,
    )

    assert normalized.value["max_tokens"] == 4096
    assert normalized.value["total_tokens"] == 17
    assert normalized.value["n_output_tokens"] == 3
    assert normalized.value["usage"] == {"prompt_tokens": 10, "cached_tokens": 2}
    assert normalized.value["api_key"] == "[REDACTED]"
    assert normalized.value["github_token"] == "[REDACTED]"
    assert normalized.value["authorization"] == "[REDACTED]"
    assert any("sensitive key" in warning for warning in normalized.warnings)


def test_counting_keys_survive_redaction_even_when_their_value_is_not_numeric():
    # The numeric-value rule cannot cover these: AgentConfig.env is dict[str, str],
    # so a token budget set through the environment always arrives as a string.
    # Without the counter-segment exemption these template to ${MAX_TOKENS} and
    # collapse agent configurations that differ only by their budget.
    semantic = semantic_agent_config(
        AgentConfig(name="agent", env={"MAX_TOKENS": "8000", "OPENAI_API_KEY": "sk-live-value"}), []
    )
    assert semantic["env"]["MAX_TOKENS"] == "8000"
    assert semantic["env"]["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"

    budgets = [
        semantic_agent_config(AgentConfig(name="agent", env={"MAX_TOKENS": value}), []) for value in ("1000", "8000")
    ]
    assert partition_key("scope", budgets[0]) != partition_key("scope", budgets[1])

    # A container under a counting key must also be walked rather than collapsed.
    walked = normalize_json({"token_usage": {"prompt_tokens": 1}}, max_bytes=10_000)
    assert walked.value == {"token_usage": {"prompt_tokens": 1}}
    assert walked.complete is True


def test_normalization_keeps_sandbox_paths():
    content = normalize_json(
        {"path": "/app/answer.txt", "message": "wrote /app/answer.txt"},
        max_bytes=10_000,
        redact_absolute_paths=False,
    )
    assert content.value == {"path": "/app/answer.txt", "message": "wrote /app/answer.txt"}
    assert content.warnings == ()
    assert content.complete is True


def test_size_bounding_accumulates_entry_sizes_instead_of_reserializing():
    # Bounding a container must not re-serialize every prefix: the payload is
    # serialized again on its way to Braintrust, so entry sizes are accumulated.
    # The selected entries must still match a byte-exact prefix walk, including
    # escaping and multi-byte characters.
    def canonical_bytes(value):
        return len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    def reference(mapping, max_bytes):
        kept = {}
        for key in sorted(mapping):
            if canonical_bytes({**kept, key: mapping[key]}) > max_bytes:
                continue
            kept[key] = mapping[key]
        return kept

    payloads = [
        {f"key_{index}": {"body": "y" * 6, "n": index} for index in range(8)},
        {"é" * 4: "ü" * 6, "escaped": '\n\t"\\', "kept": 1, "emoji": "🙂" * 4},
        {"only": "x" * 30},
    ]
    for payload in payloads:
        # Sweep every budget so each entry boundary is crossed. A per-entry
        # accounting error of even one byte changes the selected set somewhere in
        # this range, which a handful of sampled budgets would miss.
        # Start at 2: no dict fits a smaller budget, because "{}" is already two bytes.
        for max_bytes in range(2, canonical_bytes(payload) + 2):
            bounded = normalize_json(payload, max_bytes=max_bytes).value
            assert isinstance(bounded, dict)
            assert bounded == reference(payload, max_bytes), (payload, max_bytes)
            assert canonical_bytes(bounded) <= max_bytes


def test_oversized_list_keeps_its_leading_elements_and_stays_a_list():
    items = [{"name": f"tool_{index}", "body": "x" * 100} for index in range(20)]
    oversized = normalize_json(items, max_bytes=400)

    assert isinstance(oversized.value, list)
    assert oversized.value == items[: len(oversized.value)]
    assert 0 < len(oversized.value) < 20
    assert len(json.dumps(oversized.value, separators=(",", ":"))) <= 400
    assert oversized.complete is False


def test_dataset_scope_depends_only_on_the_task_source():
    # Scope must not encode the resolved task set: a narrower rerun or a backfill
    # that cannot read one trial would otherwise fork a new dataset and experiment.
    assert dataset_scope("suite") == "suite:tasks"
    assert dataset_scope("suite") != dataset_scope("other-suite")


def test_config_rejects_unimplemented_retry_attempt_logging():
    with pytest.raises(ValueError, match="log_retry_attempts"):
        PluginConfig.from_options(log_retry_attempts=True)


def test_ids_and_partition_are_deterministic_and_do_not_include_concurrency(tmp_path):
    task_config = TrialConfig(
        task=TaskConfig(path=Path("relative/task"), source="suite"),
        agent=AgentConfig(
            name="agent",
            model_name="provider/model",
            n_concurrent=1,
            env={"API_KEY": "actual-secret", "MODE": "careful"},
        ),
    )
    task_lock = TrialLock(
        task=TaskLock(name="task", type="local", digest="sha256:" + "a" * 64, source="suite"),
        agent=task_config.agent,
        skills=[
            AgentSkillLock(
                name="skill",
                source=tmp_path / "skill",
                digest="sha256:" + "b" * 64,
            )
        ],
        environment=EnvironmentConfig(),
        verifier={},
    )
    semantic = semantic_agent_config(task_config.agent, task_lock.skills)
    key = logical_task_key(task_config, task_lock)

    assert "actual-secret" not in json.dumps(semantic)
    assert semantic["env"]["API_KEY"] == "${API_KEY}"
    assert partition_key(key, semantic) == partition_key(key, semantic)
    assert dataset_record_id("scope", key) == dataset_record_id("scope", key)
    assert child_span_id("trial", "task/verification") == child_span_id("trial", "task/verification")

    changed_concurrency = AgentConfig(
        name="agent",
        model_name="provider/model",
        n_concurrent=99,
        env={"API_KEY": "actual-secret", "MODE": "careful"},
    )
    assert semantic_agent_config(changed_concurrency, task_lock.skills) == semantic

    # A token budget is part of the agent's semantics, so it must partition.
    small = semantic_agent_config(AgentConfig(name="agent", kwargs={"max_tokens": 1_000}), [])
    large = semantic_agent_config(AgentConfig(name="agent", kwargs={"max_tokens": 8_000}), [])
    assert small["kwargs"] == {"max_tokens": 1_000}
    assert partition_key(key, small) != partition_key(key, large)


def test_harbor_naive_datetimes_use_the_host_timezone_and_timings_never_run_backward():
    started_at = datetime(2026, 7, 31, 9, 57, 7)
    finished_at = datetime(2026, 7, 31, 9, 57, 41)
    timing = TimingInfo(started_at=started_at, finished_at=finished_at)

    start, end = _timing(timing, 0, 0)

    assert start == started_at.timestamp()
    assert end == finished_at.timestamp()
    assert end >= start
    assert _seconds(started_at, 0) == started_at.timestamp()

    backwards = TimingInfo(started_at=finished_at, finished_at=started_at)
    backwards_start, backwards_end = _timing(backwards, 0, 0)
    assert backwards_end == backwards_start


def _trial_result(trials_dir, trial_name, task_name, step_names=()):
    """Build a real Harbor TrialResult rooted at a temporary trials directory."""
    return TrialResult(
        task_name=task_name,
        trial_name=trial_name,
        trial_uri=f"file://{trials_dir / trial_name}",
        task_id=LocalTaskId(path=Path("tasks") / task_name),
        task_checksum="0" * 64,
        config=TrialConfig(
            task=TaskConfig(path=Path("tasks") / task_name, source="suite"),
            agent=AgentConfig(name="agent", model_name="provider/model"),
            trials_dir=trials_dir,
        ),
        agent_info=AgentInfo(name="agent", version="1.0.0"),
        step_results=[StepResult(step_name=name) for name in step_names] or None,
    )


def test_artifact_attachments_are_scoped_per_step(tmp_path):
    result = _trial_result(tmp_path, "trial-1", "task-a", step_names=("first", "second"))
    for step_name, contents in (("first", b"first output"), ("second", b"second output")):
        artifacts = tmp_path / "trial-1" / "steps" / step_name / "artifacts"
        (artifacts / "logs").mkdir(parents=True)
        (artifacts / "manifest.json").write_text("{}")
        (artifacts / "logs" / "output.txt").write_bytes(contents)

    config = PluginConfig.from_options(attachments="all", artifact_include=["logs/*.txt"])
    attachments, warnings = _artifact_attachments(result, config)

    # Each step has its own artifacts root, so identically named files must not
    # overwrite one another on the way to Braintrust.
    assert sorted(attachments) == ["first/logs/output.txt", "second/logs/output.txt"]
    assert attachments["first/logs/output.txt"].data == b"first output"
    assert attachments["second/logs/output.txt"].data == b"second output"
    assert warnings == []
    assert [step for step, _ in artifact_manifest_paths(result)] == ["first", "second"]


def test_reward_details_attachment_uses_the_per_file_limit(tmp_path):
    # A multi-step trial merges one reward-details file per step, so the combined
    # payload can exceed the per-file limit while every source file fits.
    entries = []
    for step in ("first", "second"):
        path = tmp_path / step / "reward-details.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"criteria": "x" * 400}))
        entries.append((step, path))

    config = PluginConfig.from_options(max_attachment_bytes=600, max_total_attachment_bytes=100_000)
    attachment, summary, warnings = _attachment(entries, config)

    assert attachment is None
    # The summary keys each step's details, so a score cannot misattribute them.
    assert sorted(summary) == ["first", "second"]
    assert any("attachment size limit" in warning for warning in warnings)

    attachment, summary, warnings = _attachment([(None, entries[0][1])], config)
    assert attachment is not None
    assert attachment.reference["filename"] == "reward-details.json"
    assert summary == {"criteria": "x" * 400}
    assert warnings == []


def test_disabled_plugin_does_not_reconcile_or_write_spans():
    plugin = HarborPlugin(project_name="unused")
    # Reproduce the ordering that makes this reachable: the runtime is built, then
    # a later step of on_job_start fails and disables the plugin.
    plugin._job_machine = reduce_job(plugin._job_machine, JobEvent.INITIALIZE)
    plugin._runtime = RuntimeState(snapshot=None, plan_by_trial={}, partition_by_trial={}, datasets={}, partitions={})
    plugin._disable("Braintrust initialization failed: boom")
    job_result = JobResult(
        id="00000000-0000-4000-8000-000000000001",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        n_total_trials=1,
        stats=JobStats(),
        trial_results=[],
    )

    asyncio.run(plugin.on_job_end(job_result))

    # A disabled plugin must not write an experiment its manifest reports as
    # unsynchronized, and RECONCILE is not a legal transition out of DISABLED.
    assert plugin._job_machine.status == JobStatus.DISABLED
    assert plugin._job_machine.warnings == ()


def test_backfill_matches_trial_locks_by_task_name(tmp_path):
    trial_lock = TrialLock(
        task=TaskLock(name="task-a", type="local", digest="sha256:" + "a" * 64, source="suite"),
        agent=AgentConfig(name="agent", model_name="provider/model"),
        environment=EnvironmentConfig(),
        verifier={},
    )
    (tmp_path / "config.json").write_text(JobConfig(job_name="job").model_dump_json())
    (tmp_path / "lock.json").write_text(
        JobLock(n_concurrent_trials=1, retry=RetryConfig(), trials=[trial_lock]).model_dump_json()
    )
    (tmp_path / "result.json").write_text(
        JobResult(
            id="00000000-0000-4000-8000-000000000002",
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            n_total_trials=2,
            stats=JobStats(),
        ).model_dump_json()
    )
    for trial_name, task_name in (("trial-a", "task-a"), ("trial-b", "task-b")):
        directory = tmp_path / trial_name
        directory.mkdir()
        (directory / "results.json").write_text(_trial_result(tmp_path, trial_name, task_name).model_dump_json())

    snapshot, _ = load_backfill_snapshot(tmp_path)
    locks = {plan.trial_name: plan.trial_lock for plan in snapshot.plans}

    assert locks["trial-a"] is not None
    assert locks["trial-a"].task.name == "task-a"
    # task-b has no lock entry. Falling back to another trial's lock would give it
    # task-a's identity and skills, collapsing two tasks into one logical key.
    assert locks["trial-b"] is None
    keys = {plan.trial_name: plan.task.logical_key for plan in snapshot.plans}
    assert keys["trial-a"] != keys["trial-b"]


def test_trial_reducer_retry_duplicate_backward_and_reconcile():
    state = TrialMachine("trial")
    state, _ = reduce_trial(state, TrialEvent(TrialEventKind.START))
    assert state.status == TrialStatus.ACTIVE
    assert state.phase == TrialPhase.STARTED

    duplicate, effects = reduce_trial(state, TrialEvent(TrialEventKind.START))
    assert duplicate == state
    assert effects == ()

    state, _ = reduce_trial(state, TrialEvent(TrialEventKind.AGENT_START))
    backward, _ = reduce_trial(state, TrialEvent(TrialEventKind.ENVIRONMENT_START))
    assert backward.phase == TrialPhase.AGENT
    assert backward.warnings

    state, _ = reduce_trial(state, TrialEvent(TrialEventKind.END, retry_predicted=True))
    assert state.status == TrialStatus.WAITING_RETRY
    assert state.completed_attempts == 1
    state, _ = reduce_trial(state, TrialEvent(TrialEventKind.START))
    assert state.retry_index == 1
    state, _ = reduce_trial(state, TrialEvent(TrialEventKind.END))
    assert state.status == TrialStatus.FINAL_CANDIDATE

    final = "authoritative-final-result"
    state, effects = reduce_trial(state, TrialEvent(TrialEventKind.FINAL_RESULT, payload=final))
    assert state.status == TrialStatus.FINALIZING
    assert effects[0].payload is final
    state, _ = reduce_trial(state, TrialEvent(TrialEventKind.SYNCED))
    assert state.status == TrialStatus.SYNCED


def _import_into_experiment(trajectory_path, *, experiment_name, parent_id, trial_id, phase_end, config):
    """Import a trajectory under a real agent_execution span in a cassette-backed experiment."""
    phase_start = datetime.fromisoformat("2026-01-01T00:00:00+00:00").timestamp()
    experiment = init(
        project="python-sdk-harbor-tests",
        experiment=experiment_name,
        update=True,
        set_current=False,
        git_metadata_settings=GitMetadataSettings(collect="none"),
        api_key=os.environ.get("BRAINTRUST_API_KEY", "test-api-key-for-vcr-playback"),
    )
    parent = experiment.start_span(
        name="agent_execution",
        type="task",
        id=parent_id,
        start_time=phase_start,
        set_current=False,
    )
    imported = import_trajectory(
        parent,
        trajectory_path,
        trial_id=trial_id,
        semantic_prefix="task/agent_execution",
        phase_start=phase_start,
        phase_end=phase_end,
        config=config,
    )
    parent.end(end_time=phase_end)
    flush()
    return experiment, imported


@pytest.mark.vcr
def test_atif_import_round_trips_with_real_sdks(tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory = Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "agent": {
                "name": "terminus-2",
                "version": "2.0.0",
                "model_name": "openai/gpt-4o-mini",
                "tool_definitions": [
                    {
                        "type": "function",
                        "function": {"name": "calculator", "parameters": {"type": "object"}},
                    }
                ],
            },
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-01-01T00:00:00Z",
                    "source": "user",
                    "message": "What is 2+2?",
                },
                {
                    "step_id": 2,
                    "timestamp": "2026-01-01T00:00:01Z",
                    "source": "agent",
                    "message": "I'll calculate it.",
                    "metrics": {"prompt_tokens": 10, "completion_tokens": 4, "cost_usd": 0.001},
                    "tool_calls": [
                        {
                            "tool_call_id": "call_1",
                            "function_name": "calculator",
                            "arguments": {"expression": "2+2"},
                        }
                    ],
                    "observation": {"results": [{"source_call_id": "call_1", "content": "4"}]},
                },
                {
                    "step_id": 3,
                    "timestamp": "2026-01-01T00:00:02Z",
                    "source": "agent",
                    "message": "The answer is 4.",
                    "metrics": {"prompt_tokens": 15, "completion_tokens": 5},
                },
            ],
        }
    )
    trajectory_path.write_text(trajectory.model_dump_json())
    summary = summarize_trajectory(trajectory_path, PluginConfig.from_options())

    assert summary.schema_version == "ATIF-v1.7"
    assert summary.final_message == "The answer is 4."
    assert summary.warnings == ()
    assert trajectory.steps[1].llm_call_count is None
    assert _usage_metrics(trajectory.steps[1].metrics.model_dump(mode="python")) == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "tokens": 14,
        "estimated_cost": 0.001,
    }

    parent_id = "c7a87986-0192-5f40-9ac0-a535810f1fe7"
    experiment, imported = _import_into_experiment(
        trajectory_path,
        experiment_name="harbor-atif-import",
        parent_id=parent_id,
        trial_id="trial-1",
        phase_end=datetime.fromisoformat("2026-01-01T00:00:03+00:00").timestamp(),
        config=PluginConfig.from_options(),
    )

    expected_ids = {
        parent_id,
        child_span_id("trial-1", "task/agent_execution/turn/2/llm"),
        child_span_id("trial-1", "task/agent_execution/turn/2/tool/call_1"),
        child_span_id("trial-1", "task/agent_execution/turn/3/llm"),
    }
    spans = [span for span in experiment if span["id"] in expected_ids]
    leaves = sorted(
        (span for span in spans if span["span_attributes"]["type"] in {"llm", "tool"}),
        key=lambda span: (span["metrics"]["start"], span["span_attributes"]["exec_counter"]),
    )

    assert imported.imported_llm_spans == 2
    assert imported.imported_tool_spans == 1
    assert imported.repairs == (
        "step 2: inferred one model call from terminus-2 2.0.0 trajectory",
        "step 3: inferred one model call from terminus-2 2.0.0 trajectory",
    )
    assert [span["span_attributes"]["type"] for span in leaves] == ["llm", "tool", "llm"]
    assert leaves[0]["metadata"] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "tools": [{"type": "function", "function": {"name": "calculator", "parameters": {"type": "object"}}}],
    }
    assert leaves[0]["metrics"]["tokens"] == 14
    assert leaves[1]["input"] == {"expression": "2+2"}
    assert all(
        span["context"]["span_origin"]["instrumentation"]["name"] == "braintrust.plugin.harbor" for span in leaves
    )


@pytest.mark.vcr
def test_atif_import_scopes_tool_calls_per_turn_and_reports_bounded_content(tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory = Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "generic-agent", "version": "1.0.0", "model_name": "openai/gpt-4o-mini"},
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-01-01T00:00:00Z",
                    "source": "user",
                    "message": "Read /app/answer.txt and then /app/notes.txt.",
                },
                {
                    "step_id": 2,
                    "timestamp": "2026-01-01T00:00:01Z",
                    "source": "agent",
                    "message": "Reading the answer file.",
                    "llm_call_count": 1,
                    "metrics": {"prompt_tokens": 12, "completion_tokens": 5},
                    "tool_calls": [
                        {
                            "tool_call_id": "call_1",
                            "function_name": "read_file",
                            "arguments": {"path": "/app/answer.txt"},
                        }
                    ],
                    "observation": {"results": [{"source_call_id": "call_1", "content": "42"}]},
                },
                {
                    # ATIF only requires a tool_call_id to be unique within its
                    # step, so a producer may reuse call_1 in a later turn.
                    "step_id": 3,
                    "timestamp": "2026-01-01T00:00:02Z",
                    "source": "agent",
                    "message": "Now the notes.",
                    "llm_call_count": 1,
                    "metrics": {"prompt_tokens": 14, "completion_tokens": 6},
                    "tool_calls": [
                        {
                            "tool_call_id": "call_1",
                            "function_name": "read_file",
                            "arguments": {"path": "/app/notes.txt"},
                        }
                    ],
                    "observation": {"results": [{"source_call_id": "call_1", "content": "none"}]},
                },
                {
                    "step_id": 4,
                    "timestamp": "2026-01-01T00:00:03Z",
                    "source": "agent",
                    "message": "y" * 600,
                    "llm_call_count": 1,
                    "metrics": {"prompt_tokens": 16, "completion_tokens": 7},
                },
            ],
        }
    )
    trajectory_path.write_text(trajectory.model_dump_json())

    experiment, imported = _import_into_experiment(
        trajectory_path,
        experiment_name="harbor-atif-content-bounds",
        parent_id="b4de2f31-7c05-5a9e-8d64-1f3a6c9b2e70",
        trial_id="trial-2",
        phase_end=datetime.fromisoformat("2026-01-01T00:00:04+00:00").timestamp(),
        config=PluginConfig.from_options(max_content_bytes=300),
    )

    first_tool_id = child_span_id("trial-2", "task/agent_execution/turn/2/tool/call_1")
    second_tool_id = child_span_id("trial-2", "task/agent_execution/turn/3/tool/call_1")
    truncated_id = child_span_id("trial-2", "task/agent_execution/turn/4/summary")
    assert first_tool_id != second_tool_id

    assert imported.imported_llm_spans == 2
    assert imported.imported_tool_spans == 2
    # Truncation and redaction must never be silent, and a truncated payload must
    # not keep an llm label.
    assert "step 4 message: truncated value: exceeded 300 bytes" in imported.warnings
    assert "step 4: downgraded to task (message content was truncated or redacted)" in imported.warnings

    by_id = {span["id"]: span for span in experiment if span["id"] in {first_tool_id, second_tool_id, truncated_id}}

    # Sandbox paths are the substance of a filesystem tool call, and each turn's
    # call must pair with the observation that actually answered it.
    assert by_id[first_tool_id]["input"] == {"path": "/app/answer.txt"}
    assert by_id[first_tool_id]["output"] == "42"
    assert by_id[second_tool_id]["input"] == {"path": "/app/notes.txt"}
    assert by_id[second_tool_id]["output"] == "none"
    assert by_id[truncated_id]["span_attributes"]["type"] == "task"
    assert len(by_id[truncated_id]["output"]["message"]) == 300
