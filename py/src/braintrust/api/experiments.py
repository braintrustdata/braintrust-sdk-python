"""Experiment API service and response models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..util import encode_uri_component
from ._routing import RequestTarget
from ._service import ResourceAPI
from .errors import BraintrustHTTPError
from .policies import RetryMode


def _empty_mapping() -> Mapping[str, Any]:
    return MappingProxyType({})


def _preserve(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class ExperimentRecord:
    """An experiment returned by the resource API."""

    id: str
    name: str
    project_id: str | None
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentRecord":
        experiment_id = value.get("id")
        name = value.get("name")
        project_id = value.get("project_id")
        if not isinstance(experiment_id, str) or not isinstance(name, str):
            raise ValueError("Experiment data must include string id and name fields")
        if project_id is not None and not isinstance(project_id, str):
            raise ValueError("Experiment project_id must be a string or null")
        return cls(id=experiment_id, name=name, project_id=project_id, raw=_preserve(value))


@dataclass(frozen=True)
class BaseExperiment:
    """The persisted baseline selected for an experiment."""

    id: str
    name: str
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BaseExperiment":
        experiment_id = value.get("base_exp_id")
        name = value.get("base_exp_name")
        if not isinstance(experiment_id, str) or not isinstance(name, str):
            raise ValueError("Base experiment data must include string base_exp_id and base_exp_name fields")
        return cls(id=experiment_id, name=name, raw=_preserve(value))


@dataclass(frozen=True)
class ExperimentScore:
    """One score aggregate in an experiment comparison response."""

    name: str
    score: float
    improvements: int | None
    regressions: int | None
    diff: float | None = None
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, key: str, value: Mapping[str, Any]) -> "ExperimentScore":
        name = value.get("name", key)
        if not isinstance(name, str):
            raise ValueError(f"Experiment score {key!r} must include a string name")
        return cls(
            name=name,
            score=value.get("score"),
            improvements=value.get("improvements"),
            regressions=value.get("regressions"),
            diff=value.get("diff"),
            raw=_preserve(value),
        )


@dataclass(frozen=True)
class ExperimentMetric:
    """One metric aggregate in an experiment comparison response."""

    name: str
    metric: float | int
    unit: str
    improvements: int | None
    regressions: int | None
    diff: float | None = None
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, key: str, value: Mapping[str, Any]) -> "ExperimentMetric":
        name = value.get("name", key)
        unit = value.get("unit", "")
        if not isinstance(name, str) or not isinstance(unit, str):
            raise ValueError(f"Experiment metric {key!r} must include string name and unit fields")
        return cls(
            name=name,
            metric=value.get("metric"),
            unit=unit,
            improvements=value.get("improvements"),
            regressions=value.get("regressions"),
            diff=value.get("diff"),
            raw=_preserve(value),
        )


@dataclass(frozen=True)
class ExperimentComparison:
    """Scores and metrics returned by the experiment comparison endpoint."""

    scores: Mapping[str, ExperimentScore]
    metrics: Mapping[str, ExperimentMetric]
    raw: Mapping[str, Any] = field(default_factory=_empty_mapping)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentComparison":
        raw_scores = value.get("scores", {})
        raw_metrics = value.get("metrics", {})
        if not isinstance(raw_scores, Mapping) or not isinstance(raw_metrics, Mapping):
            raise ValueError("Experiment comparison scores and metrics must be objects")

        scores = {}
        for key, score in raw_scores.items():
            if not isinstance(key, str) or not isinstance(score, Mapping):
                raise ValueError("Experiment comparison scores must map string names to objects")
            scores[key] = ExperimentScore.from_dict(key, score)

        metrics = {}
        for key, metric in raw_metrics.items():
            if not isinstance(key, str) or not isinstance(metric, Mapping):
                raise ValueError("Experiment comparison metrics must map string names to objects")
            metrics[key] = ExperimentMetric.from_dict(key, metric)

        return cls(
            scores=MappingProxyType(scores),
            metrics=MappingProxyType(metrics),
            raw=_preserve(value),
        )


class ExperimentsAPI(ResourceAPI):
    """Synchronous experiment lookup and comparison operations."""

    def get(self, experiment_id: str) -> ExperimentRecord:
        """Fetch an experiment by ID."""

        response = self._request_json(
            RequestTarget.API,
            "GET",
            f"/v1/experiment/{encode_uri_component(experiment_id)}",
            retry_mode=RetryMode.SAFE_READ,
        )
        return ExperimentRecord.from_dict(response)

    def get_base(self, experiment_id: str) -> BaseExperiment | None:
        """Return an experiment's persisted baseline, or ``None`` when none exists."""

        try:
            response = self._request_json(
                RequestTarget.APP,
                "POST",
                "/api/base_experiment/get_id",
                json={"id": experiment_id},
                retry_mode=RetryMode.SAFE_READ,
            )
        except BraintrustHTTPError as exc:
            if exc.status_code == 400:
                return None
            raise

        if not response:
            return None
        return BaseExperiment.from_dict(response)

    def compare(
        self,
        experiment_id: str,
        *,
        base_experiment_id: str | None = None,
    ) -> ExperimentComparison:
        """Fetch score and metric aggregates for an experiment comparison."""

        response = self._request_json(
            RequestTarget.API,
            "GET",
            "/experiment-comparison2",
            params={
                "experiment_id": experiment_id,
                "base_experiment_id": base_experiment_id,
            },
            retry_mode=RetryMode.SAFE_READ,
        )
        return ExperimentComparison.from_dict(response)
