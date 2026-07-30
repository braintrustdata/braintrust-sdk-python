# Harbor is installed by the dedicated Python 3.12+ nox session, not by the
# cross-version pylint environment.
# pylint: disable=import-error

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest
from braintrust import flush, init
from braintrust.conftest import get_vcr_config
from braintrust.git_fields import GitMetadataSettings
from braintrust.integrations.harbor.atif import _usage_metrics, import_trajectory, summarize_trajectory
from braintrust.integrations.harbor.config import PluginConfig
from braintrust.integrations.harbor.identity import (
    child_span_id,
    dataset_record_id,
    logical_task_key,
    normalize_json,
    partition_key,
    semantic_agent_config,
)
from braintrust.integrations.harbor.rewards import classify_rewards, validate_classifications
from braintrust.integrations.harbor.state import (
    TrialEvent,
    TrialEventKind,
    TrialMachine,
    TrialPhase,
    TrialStatus,
    reduce_trial,
)
from harbor.models.job.lock import AgentSkillLock, TaskLock, TrialLock
from harbor.models.trajectories.trajectory import Trajectory
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig


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
        verifier=VerifierConfig(),
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


@pytest.mark.vcr
def test_atif_import_round_trips_with_real_sdks(tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory = Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "agent": {
                "name": "test-agent",
                "version": "1",
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
                    "llm_call_count": 1,
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
                    "llm_call_count": 1,
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
    assert _usage_metrics(trajectory.steps[1].metrics.model_dump(mode="python")) == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "tokens": 14,
        "estimated_cost": 0.001,
    }

    phase_start = datetime.fromisoformat("2026-01-01T00:00:00+00:00").timestamp()
    phase_end = datetime.fromisoformat("2026-01-01T00:00:03+00:00").timestamp()
    experiment = init(
        project="python-sdk-harbor-tests",
        experiment="harbor-atif-import",
        update=True,
        set_current=False,
        git_metadata_settings=GitMetadataSettings(collect="none"),
        api_key=os.environ.get("BRAINTRUST_API_KEY", "test-api-key-for-vcr-playback"),
    )
    parent_id = "c7a87986-0192-5f40-9ac0-a535810f1fe7"
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
        trial_id="trial-1",
        semantic_prefix="task/agent_execution",
        phase_start=phase_start,
        phase_end=phase_end,
        config=PluginConfig.from_options(),
    )
    parent.end(end_time=phase_end)
    flush()

    expected_ids = {
        parent_id,
        child_span_id("trial-1", "task/agent_execution/turn/2/llm"),
        child_span_id("trial-1", "task/agent_execution/tool/call_1"),
        child_span_id("trial-1", "task/agent_execution/turn/3/llm"),
    }
    spans = [span for span in experiment if span["id"] in expected_ids]
    leaves = sorted(
        (span for span in spans if span["span_attributes"]["type"] in {"llm", "tool"}),
        key=lambda span: (span["metrics"]["start"], span["span_attributes"]["exec_counter"]),
    )

    assert imported.imported_llm_spans == 2
    assert imported.imported_tool_spans == 1
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
