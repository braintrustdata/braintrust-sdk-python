"""Static and runtime checks for structured experiment summaries."""

from typing import TYPE_CHECKING

from braintrust import ExperimentSummary, MetricSummary, ScoreSummary, SummarySkipped


if TYPE_CHECKING:

    def check_summary_narrowing(summary: ExperimentSummary) -> None:
        comparison = summary.comparison
        if comparison.status == "success":
            scores: dict[str, ScoreSummary] = comparison.scores
            metrics: dict[str, MetricSummary] = comparison.metrics
        else:
            reason: str = comparison.reason


def test_structured_experiment_summary_public_types() -> None:
    summary = ExperimentSummary(
        project_name="project",
        project_id="project-id",
        experiment_id="experiment-id",
        experiment_name="experiment",
        project_url="https://example.com/project",
        experiment_url="https://example.com/experiment",
        comparison_experiment_name=None,
        comparison=SummarySkipped(reason="disabled"),
    )

    assert isinstance(summary.comparison, SummarySkipped)
    assert summary.comparison.reason == "disabled"
