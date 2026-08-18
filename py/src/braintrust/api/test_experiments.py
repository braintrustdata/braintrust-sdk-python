import os

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


def _api_key():
    return os.environ.get("BRAINTRUST_API_KEY", "sk-dummy-for-vcr-replay")


@pytest.mark.vcr
@pytest.mark.parametrize("explicit_comparison", [False, True])
def test_experiment_summarize_with_real_backend(explicit_comparison):
    project_name = "python-sdk-generated-experiments-vcr"
    base = braintrust.init(
        project=project_name,
        experiment="generated-experiments-base",
        api_key=_api_key(),
        update=True,
        set_current=False,
        repo_info=RepoInfo(),
    )
    candidate = braintrust.init(
        project=project_name,
        experiment="generated-experiments-candidate",
        api_key=_api_key(),
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
