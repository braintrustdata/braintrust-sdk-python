"""Static and runtime checks for structured experiment summaries."""

from typing import TYPE_CHECKING

from braintrust import ExperimentSummary, MetricSummary, ScoreSummary, SummarySkipped, SummarySuccess


if TYPE_CHECKING:

    def check_summary_narrowing(summary: ExperimentSummary) -> None:
        comparison = summary.comparison
        if comparison.status == "success":
            scores: dict[str, ScoreSummary] = comparison.scores
            metrics: dict[str, MetricSummary] = comparison.metrics
        else:
            reason: str = comparison.reason

        legacy_scores: dict[str, ScoreSummary] = summary.scores
        legacy_metrics: dict[str, MetricSummary] = summary.metrics


def _summary(comparison: SummarySuccess | SummarySkipped) -> ExperimentSummary:
    return ExperimentSummary(
        project_name="project",
        project_id="project-id",
        experiment_id="experiment-id",
        experiment_name="experiment",
        project_url="https://example.com/project",
        experiment_url="https://example.com/experiment",
        comparison_experiment_name=None,
        comparison=comparison,
    )


def test_structured_experiment_summary_public_types() -> None:
    skipped = _summary(SummarySkipped(reason="disabled"))
    success = SummarySuccess(scores={}, metrics={})

    assert isinstance(skipped.comparison, SummarySkipped)
    assert skipped.comparison.reason == "disabled"
    assert success.status == "success"


def test_structured_experiment_summary_deep_deserialization() -> None:
    success = _summary(
        SummarySuccess(
            scores={
                "accuracy": ScoreSummary(
                    name="accuracy",
                    score=0.9,
                    improvements=2,
                    regressions=1,
                    diff=0.1,
                    _longest_score_name=len("accuracy"),
                )
            },
            metrics={},
        )
    )
    success_payload = success.as_dict()

    restored_success = ExperimentSummary.from_dict_deep(success_payload)
    restored_skipped = ExperimentSummary.from_dict_deep(_summary(SummarySkipped(reason="disabled")).as_dict())
    legacy_payload = {key: value for key, value in success_payload.items() if key != "comparison"}
    restored_legacy = ExperimentSummary.from_dict_deep(legacy_payload)

    assert isinstance(restored_success.comparison, SummarySuccess)
    assert restored_success.as_dict() == success_payload
    assert str(restored_success) == str(success)
    assert isinstance(restored_skipped.comparison, SummarySkipped)
    assert restored_skipped.comparison.reason == "disabled"
    assert isinstance(restored_legacy.comparison, SummarySuccess)
    assert restored_legacy.scores["accuracy"].score == 0.9
