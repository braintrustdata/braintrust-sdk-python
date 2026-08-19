import contextlib

import braintrust
import pytest
from braintrust.api._generated.experiments import OPERATIONS
from braintrust.api.policies import RetryMode
from braintrust.git_fields import RepoInfo
from braintrust.logger import SummarySuccess


def test_all_experiment_operations_have_complete_retry_classification():
    assert set(OPERATIONS) == {
        "postExperiment",
        "getExperiment",
        "getExperimentId",
        "patchExperimentId",
        "deleteExperimentId",
        "postExperimentIdInsert",
        "postExperimentIdFetch",
        "getExperimentIdFetch",
        "postExperimentIdFeedback",
        "getExperimentIdSummarize",
    }
    assert {name: operation.retry_mode for name, operation in OPERATIONS.items()} == {
        "postExperiment": RetryMode.NONE,
        "getExperiment": RetryMode.SAFE_READ,
        "getExperimentId": RetryMode.SAFE_READ,
        "patchExperimentId": RetryMode.NONE,
        "deleteExperimentId": RetryMode.NONE,
        "postExperimentIdInsert": RetryMode.NONE,
        "postExperimentIdFetch": RetryMode.SAFE_READ,
        "getExperimentIdFetch": RetryMode.SAFE_READ,
        "postExperimentIdFeedback": RetryMode.NONE,
        "getExperimentIdSummarize": RetryMode.SAFE_READ,
    }


@pytest.mark.vcr
@pytest.mark.parametrize("explicit_comparison", [False, True])
def test_experiment_summarize_with_real_backend(explicit_comparison, api_key):
    project_name = "python-sdk-generated-experiments-vcr"
    base = braintrust.init(
        project=project_name,
        experiment="generated-experiments-base",
        api_key=api_key,
        update=True,
        set_current=False,
        repo_info=RepoInfo(),
    )
    candidate = braintrust.init(
        project=project_name,
        experiment="generated-experiments-candidate",
        api_key=api_key,
        base_experiment_id=base.id,
        update=True,
        set_current=False,
        repo_info=RepoInfo(),
    )
    comparison_input = {"question": "What is the answer?"}
    base.log(
        id="generated-experiments-base-row",
        input=comparison_input,
        output="incorrect",
        expected="correct",
        scores={"accuracy": 0},
        metrics={"completion_tokens": 20},
    )
    candidate.log(
        id="generated-experiments-candidate-row",
        input=comparison_input,
        output="correct",
        expected="correct",
        scores={"accuracy": 1},
        metrics={"completion_tokens": 10},
    )

    summary = candidate.summarize(comparison_experiment_id=base.id if explicit_comparison else None)

    assert isinstance(summary.comparison, SummarySuccess)
    assert summary.comparison_experiment_name == base.name
    assert summary.project_name == project_name
    assert summary.experiment_name == candidate.name
    assert summary.scores["accuracy"].score == 1
    assert summary.scores["accuracy"].diff == 1
    assert summary.scores["accuracy"].improvements == 1
    assert summary.metrics["completion_tokens"].metric == 10
    assert summary.metrics["completion_tokens"].diff == -10
    assert summary.metrics["completion_tokens"].unit == "tok"
    serialized = summary.as_dict()
    assert serialized["comparison"]["status"] == "success"
    assert serialized["scores"]["accuracy"]["score"] == 1
    assert serialized["metrics"]["completion_tokens"]["metric"] == 10
    assert f"{candidate.name} compared to {base.name}" in str(summary)


@pytest.mark.vcr
def test_high_level_experiment_uses_generated_resources(api_key):
    project_name = "python-sdk-high-level-experiments-vcr"
    experiment_name = "high-level-generated-experiment"
    event_id = "high-level-generated-experiment-row"
    cleanup_project_id = None
    cleanup_experiment_id = None

    experiment = braintrust.init(
        project=project_name,
        experiment=experiment_name,
        api_key=api_key,
        update=True,
        set_current=False,
        repo_info=RepoInfo(),
    )
    try:
        cleanup_experiment_id = experiment.id
        cleanup_project_id = experiment.project.id
        api_client = experiment.state.api_client()
        api_client.experiments.post_experiment_id_insert(
            experiment.id,
            body={
                "events": [
                    {
                        "id": event_id,
                        "input": {"question": "What is the answer?"},
                        "output": "42",
                        "scores": {"accuracy": 1},
                        "span_id": event_id,
                        "root_span_id": event_id,
                    }
                ]
            },
        )

        events = list(experiment.fetch(batch_size=1))
        readonly = braintrust.init(
            project=project_name,
            experiment=experiment_name,
            api_key=api_key,
            open=True,
            set_current=False,
        )
        readonly_events = list(readonly.fetch(batch_size=1))

        assert readonly.id == experiment.id
        assert [event["id"] for event in events] == [event_id]
        assert [event["id"] for event in readonly_events] == [event_id]
        assert events[0]["input"] == {"question": "What is the answer?"}
        assert readonly_events[0]["output"] == "42"
    finally:
        if cleanup_experiment_id is not None:
            with contextlib.suppress(Exception):
                experiment.state.api_client().experiments.delete_experiment_id(cleanup_experiment_id)
        if cleanup_project_id is not None:
            with contextlib.suppress(Exception):
                experiment.state.api_client().projects.delete_project_id(cleanup_project_id)
