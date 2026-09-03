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
from braintrust.integrations.harbor.compat import (
    JobSnapshot,
    TaskData,
    TrialPlan,
    artifact_manifest_paths,
    load_backfill_snapshot,
)
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
    DatasetBinding,
    HarborPlugin,
    Partition,
    RuntimeState,
    _artifact_attachments,
    _attachment,
    _read_safe_file,
    _resolve_project,
    _seconds,
    _timing,
    _verifier_evidence,
    _verifier_output_attachment,
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
from braintrust.test_helpers import (  # noqa: F401
    find_span_by_name,
    find_spans_by_type,
    init_test_exp,
    with_memory_logger,
    with_simulate_login,
)
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.job.lock import AgentSkillLock, JobLock, TaskLock, TrialLock
from harbor.models.job.result import JobResult, JobStats
from harbor.models.task.id import LocalTaskId
from harbor.models.trajectories.trajectory import Trajectory
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, StepResult, TimingInfo, TrialResult, VerifierResult


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


def test_attachment_modes_default_to_all_and_name_the_structured_only_tier():
    assert PluginConfig.from_options().attachments == "all"
    assert PluginConfig.from_options(attachments="structured").attachments == "structured"
    with pytest.raises(ValueError, match="attachments must be 'none', 'structured', or 'all'"):
        PluginConfig.from_options(attachments="verifier-details")


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


def _verifier_trial(tmp_path):
    """Build a single-phase trial with an empty verifier output directory."""
    result = _trial_result(tmp_path, "trial-1", "task-a")
    verifier_dir = tmp_path / "trial-1" / "verifier"
    verifier_dir.mkdir(parents=True)
    return result, verifier_dir


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


def test_reward_details_attachment_merges_every_step(tmp_path):
    entries = []
    for step in ("first", "second"):
        path = tmp_path / step / "reward-details.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"criteria": "x" * 400}))
        entries.append((step, path))

    config = PluginConfig.from_options()
    attachment, summary, warnings = _attachment(entries, config)

    assert attachment is not None
    # The summary keys each step's details, so a score cannot misattribute them.
    assert sorted(summary) == ["first", "second"]
    assert json.loads(attachment.data) == summary
    assert warnings == []

    attachment, summary, warnings = _attachment([(None, entries[0][1])], config)
    assert attachment is not None
    assert attachment.reference["filename"] == "reward-details.json"
    assert summary == {"criteria": "x" * 400}
    assert warnings == []


def test_oversized_trajectory_is_refused_before_it_is_parsed(tmp_path):
    # trajectory.json is parsed into the host process rather than handed to
    # object storage, so it keeps a bound of its own now that attachments do not.
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps({"steps": [{"step_id": 1, "source": "agent", "message": "hello"}]}))
    oversized = PluginConfig.from_options(max_trajectory_bytes=4)

    summarized = summarize_trajectory(trajectory_path, oversized)
    assert summarized.warnings == ("trajectory omitted: size limit",)
    assert summarized.final_message is None

    imported = import_trajectory(
        parent=None,
        trajectory_path=trajectory_path,
        trial_id="trial-1",
        semantic_prefix="prefix",
        phase_start=0.0,
        phase_end=1.0,
        config=oversized,
    )
    assert imported.warnings == ("trajectory omitted: size limit",)
    assert imported.imported_llm_spans == 0

    # The same document is imported normally under the default bound.
    assert summarize_trajectory(trajectory_path, PluginConfig.from_options()).warnings == ()


def test_verifier_output_attachment_collects_standard_harbor_evidence(tmp_path):
    result, verifier_dir = _verifier_trial(tmp_path)
    # write_bytes, not write_text: on Windows text mode would rewrite "\n" as
    # "\r\n", and the plugin decodes the file's bytes exactly as written.
    (verifier_dir / "test-stdout.txt").write_bytes(b"FAILED test_answer.py::test_count - assert 27 == 28\n")
    (verifier_dir / "test-stderr.txt").write_bytes(b"token=secret-value\n")
    (verifier_dir / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {"tests": 1, "passed": 0, "failed": 1},
                    "tests": [
                        {
                            "name": "test_answer.py::test_count",
                            "status": "failed",
                            "message": "assert 27 == 28",
                            "trace": "Authorization: Bearer verifier-secret",
                        }
                    ],
                }
            }
        )
    )

    config = PluginConfig.from_options(attachments="all", redact_patterns=(r"secret-value|Bearer verifier-secret",))
    attachment, summary, warnings = _verifier_output_attachment(result, config)

    assert summary == {
        "stdout": "FAILED test_answer.py::test_count - assert 27 == 28\n",
        "stderr": "token=[REDACTED]\n",
        "ctrf": {
            "results": {
                "summary": {"tests": 1, "passed": 0, "failed": 1},
                "tests": [
                    {
                        "name": "test_answer.py::test_count",
                        "status": "failed",
                        "message": "assert 27 == 28",
                        "trace": "Authorization: [REDACTED]",
                    }
                ],
            }
        },
    }
    assert attachment is not None
    assert attachment.reference["filename"] == "verifier-output.json"
    assert json.loads(attachment.data) == summary
    assert warnings == []


def test_verifier_output_attachment_handles_invalid_utf8_and_configured_redaction(tmp_path):
    result, verifier_dir = _verifier_trial(tmp_path)
    (verifier_dir / "ctrf.json").write_bytes(b"\xff token=opaque-secret\n")

    attachment, summary, warnings = _verifier_output_attachment(
        result, PluginConfig.from_options(redact_patterns=(r"opaque-secret",))
    )

    assert attachment is not None
    assert summary == {"ctrf": "\ufffd token=[REDACTED]\n"}
    assert any("ctrf.json is not valid JSON" in warning for warning in warnings)


@pytest.mark.parametrize(
    "kind",
    [
        "symlink",
        pytest.param(
            "fifo",
            marks=pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="os.mkfifo is not available on Windows"),
        ),
    ],
)
def test_verifier_output_attachment_rejects_unsafe_file_types(tmp_path, kind):
    result, verifier_dir = _verifier_trial(tmp_path)
    stdout = verifier_dir / "test-stdout.txt"
    if kind == "symlink":
        secret = tmp_path / "host-secret"
        secret.write_text("must not escape")
        stdout.symlink_to(secret)
    else:
        os.mkfifo(stdout)

    config = PluginConfig.from_options(attachments="all")
    attachment, summary, warnings = _verifier_output_attachment(result, config)

    assert attachment is None
    assert summary is None
    assert warnings == ["test-stdout.txt omitted: unsafe file type"]


def test_safe_file_read_rejects_replacement_between_inspection_and_open(tmp_path, monkeypatch):
    expected = tmp_path / "expected"
    replacement = tmp_path / "replacement"
    expected.write_text("safe")
    replacement.write_text("must not escape")
    real_open = os.open

    def swap_after_inspection(_path, flags):
        return real_open(replacement, flags)

    monkeypatch.setattr(os, "open", swap_after_inspection)

    assert _read_safe_file(expected) == (None, "unsafe file type")


def test_verifier_output_attachment_scopes_steps_and_respects_attachment_mode(tmp_path):
    result = _trial_result(tmp_path, "trial-1", "task-a", step_names=("first", "second"))
    for step_name in ("first", "second"):
        verifier_dir = tmp_path / "trial-1" / "steps" / step_name / "verifier"
        verifier_dir.mkdir(parents=True)
        (verifier_dir / "test-stdout.txt").write_bytes(f"{step_name} output\n".encode())
        (verifier_dir / "ctrf.json").write_text(json.dumps({"step": step_name}))

    attachment, summary, warnings = _verifier_output_attachment(result, PluginConfig.from_options(attachments="all"))

    assert summary == {
        "first": {"stdout": "first output\n", "ctrf": {"step": "first"}},
        "second": {"stdout": "second output\n", "ctrf": {"step": "second"}},
    }
    assert attachment is not None
    assert warnings == []

    # Eval runs keep the full verifier evidence by default. The explicitly
    # structured tier retains the machine-readable report without raw logs.
    _, default_summary, default_warnings = _verifier_output_attachment(result, PluginConfig.from_options())
    assert default_summary == summary
    assert default_warnings == []

    _, structured_summary, structured_warnings = _verifier_output_attachment(
        result, PluginConfig.from_options(attachments="structured")
    )
    assert structured_summary == {
        "first": {"ctrf": {"step": "first"}},
        "second": {"ctrf": {"step": "second"}},
    }
    assert structured_warnings == []

    assert _verifier_output_attachment(result, PluginConfig.from_options(attachments="none")) == (None, None, [])


def test_verifier_summary_keeps_the_structure_the_attachment_kept(tmp_path):
    # The span preview and the attachment are the same value, so a document
    # nested deeper than normalize_json's default limit but within the limit the
    # attachment uses must not be truncated in one and whole in the other.
    result, verifier_dir = _verifier_trial(tmp_path)
    nested = {"leaf": "kept"}
    for _ in range(12):
        nested = {"a": nested}
    (verifier_dir / "ctrf.json").write_text(json.dumps(nested))

    output, warnings = _verifier_evidence(result, PluginConfig.from_options())

    assert warnings == []
    assert output["verifier_output_summary"] == {"ctrf": nested}
    assert json.loads(output["verifier_output"].data) == output["verifier_output_summary"]


def test_deeply_nested_verifier_json_is_reported_not_raised(tmp_path):
    # json.loads raises RecursionError rather than JSONDecodeError here, and it
    # used to escape the sync between starting a trial's spans and logging them.
    result, verifier_dir = _verifier_trial(tmp_path)
    (verifier_dir / "ctrf.json").write_bytes(b"[" * 200_000 + b"]" * 200_000)

    attachment, summary, warnings = _verifier_output_attachment(result, PluginConfig.from_options())

    assert attachment is not None
    assert summary["ctrf"].startswith("[[[")
    assert warnings == ["ctrf.json is not valid JSON"]


def test_deeply_nested_trajectory_is_reported_not_raised(tmp_path):
    # trajectory.json is task-controlled too, and both readers parse it in-process.
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_bytes(b"[" * 200_000 + b"]" * 200_000)
    config = PluginConfig.from_options()

    assert summarize_trajectory(trajectory_path, config).warnings == (
        "trajectory unavailable or malformed: not valid JSON",
    )

    imported = import_trajectory(
        parent=None,
        trajectory_path=trajectory_path,
        trial_id="trial-1",
        semantic_prefix="prefix",
        phase_start=0.0,
        phase_end=1.0,
        config=config,
    )
    assert imported.warnings == ("trajectory unavailable or malformed: not valid JSON",)


def test_deeply_nested_reward_details_are_reported_not_raised(tmp_path):
    path = tmp_path / "reward-details.json"
    path.write_bytes(b"[" * 200_000 + b"]" * 200_000)

    attachment, summary, warnings = _attachment([(None, path)], PluginConfig.from_options())

    assert (attachment, summary) == (None, None)
    assert warnings == ["reward-details.json is not valid JSON"]


def _install_runtime(plugin, result, trials_dir, experiment):
    """Point a plugin at one trial writing into a real Braintrust experiment."""
    task = TaskData(
        logical_key="task-key",
        source="suite",
        name="task-a",
        input={"instruction": "solve"},
        expected=None,
        metadata={"harbor": {"custom": {}}},
        digest=None,
        schema_version=None,
        task_dir=None,
    )
    plan = TrialPlan(result.trial_name, result.config, None, task, 0)
    snapshot = JobSnapshot("job-id", "job", trials_dir, None, None, (plan,))
    partition = Partition("partition", "experiment", "scope", experiment=experiment)
    plugin._runtime = RuntimeState(
        snapshot,
        {result.trial_name: plan},
        {result.trial_name: partition},
        {"scope": DatasetBinding("scope")},
        {"partition": partition},
    )
    plugin._trial_machines[result.trial_name] = TrialMachine(result.trial_name)


def _sync_spans(plugin, result, experiment, memory_logger):
    """Run one trial through the plugin and return its flushed spans."""
    plugin._sync_final_result(result)
    experiment.flush()
    return memory_logger.pop()


def _child_spans(spans, parent):
    # Root spans carry span_parents=None rather than an empty list.
    return [span for span in spans if parent["span_id"] in (span["span_parents"] or ())]


def _scored_verifier_trial(tmp_path):
    result, verifier_dir = _verifier_trial(tmp_path)
    result.verifier_result = VerifierResult(rewards={"reward": 0.25})
    (verifier_dir / "test-stdout.txt").write_bytes(b"assert 1 == 2\n")
    (verifier_dir / "ctrf.json").write_text(json.dumps({"failed": 1}))
    return result


@pytest.mark.parametrize(
    ("attachments", "expected_summary"),
    [
        ("structured", {"ctrf": {"failed": 1}}),
        ("all", {"ctrf": {"failed": 1}, "stdout": "assert 1 == 2\n"}),
    ],
)
def test_final_sync_logs_verifier_evidence_on_verification_and_score_spans(
    tmp_path, attachments, expected_summary, with_memory_logger, with_simulate_login
):
    result = _scored_verifier_trial(tmp_path)
    experiment = init_test_exp("harbor-verifier-evidence", "harbor")
    plugin = HarborPlugin(attachments=attachments)
    _install_runtime(plugin, result, tmp_path, experiment)

    spans = _sync_spans(plugin, result, experiment, with_memory_logger)

    verification = find_span_by_name(spans, "verification")
    scorer = find_spans_by_type(spans, "score")[0]
    assert verification["output"]["verifier_output_summary"] == expected_summary
    # The real serializer replaces the attachment with the reference that is
    # stored on the span, and queues the payload itself for upload.
    reference = verification["output"]["verifier_output"]
    assert reference["type"] == "braintrust_attachment"
    assert reference["filename"] == "verifier-output.json"
    assert scorer["output"]["verifier_output_summary"] == verification["output"]["verifier_output_summary"]
    assert scorer["output"]["verifier_output"] == reference
    uploaded = {attachment.reference["key"] for attachment in with_memory_logger.upload_attempts}
    assert reference["key"] in uploaded


def test_final_sync_logs_no_verifier_evidence_when_attachments_are_disabled(
    tmp_path, with_memory_logger, with_simulate_login
):
    result = _scored_verifier_trial(tmp_path)
    experiment = init_test_exp("harbor-verifier-evidence", "harbor")
    plugin = HarborPlugin(attachments="none")
    _install_runtime(plugin, result, tmp_path, experiment)

    spans = _sync_spans(plugin, result, experiment, with_memory_logger)

    verification = find_span_by_name(spans, "verification")
    assert "verifier_output_summary" not in (verification.get("output") or {})
    assert "verifier_output" not in find_spans_by_type(spans, "score")[0]["output"]
    assert with_memory_logger.upload_attempts == []


def test_verification_span_keeps_verifier_evidence_without_scores(tmp_path, with_memory_logger, with_simulate_login):
    # The verification span owns the attachment, so unevaluated trials do not
    # lose their complete evidence merely because no score span is created.
    result = _trial_result(tmp_path, "trial-1", "task-a")
    result.verifier_result = VerifierResult(rewards=None)
    verifier_dir = tmp_path / "trial-1" / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "ctrf.json").write_text(json.dumps({"failed": 1}))

    experiment = init_test_exp("harbor-verifier-evidence", "harbor")
    plugin = HarborPlugin()
    _install_runtime(plugin, result, tmp_path, experiment)

    spans = _sync_spans(plugin, result, experiment, with_memory_logger)

    assert not find_spans_by_type(spans, "score")
    verification = find_span_by_name(spans, "verification")
    assert verification["output"]["verifier_output"]["type"] == "braintrust_attachment"


def test_trajectory_images_and_verifier_evidence_are_both_logged(tmp_path, with_memory_logger, with_simulate_login):
    # ATIF images and verifier evidence are independent: neither competes with
    # the other for a shared budget.
    result = _trial_result(tmp_path, "trial-1", "task-a")
    result.verifier_result = VerifierResult(rewards={"reward": 0.25})
    verifier_dir = tmp_path / "trial-1" / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "ctrf.json").write_text(json.dumps({"failed": 1}))
    agent_dir = tmp_path / "trial-1" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "first.png").write_bytes(b"i" * 600)
    (agent_dir / "second.png").write_bytes(b"j" * 600)
    (agent_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "step_id": 1,
                        "source": "agent",
                        "message": [
                            {"type": "image", "source": {"path": "first.png", "media_type": "image/png"}},
                            {"type": "image", "source": {"path": "second.png", "media_type": "image/png"}},
                        ],
                    }
                ]
            }
        )
    )

    experiment = init_test_exp("harbor-verifier-evidence", "harbor")
    plugin = HarborPlugin()
    _install_runtime(plugin, result, tmp_path, experiment)

    spans = _sync_spans(plugin, result, experiment, with_memory_logger)

    agent_execution = find_span_by_name(spans, "agent_execution")
    trajectory_step = _child_spans(spans, agent_execution)[0]
    message = trajectory_step["output"]["message"]
    assert [part["type"] for part in message] == ["image_url", "image_url"]
    assert all(part["image_url"]["url"]["type"] == "braintrust_attachment" for part in message)
    assert find_span_by_name(spans, "verification")["output"]["verifier_output_summary"] == {"ctrf": {"failed": 1}}
    # Both images and the verifier payload reach the uploader.
    assert sorted(a.data for a in with_memory_logger.upload_attempts if a.data in (b"i" * 600, b"j" * 600)) == [
        b"i" * 600,
        b"j" * 600,
    ]


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
