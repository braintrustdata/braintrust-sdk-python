from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar

from typing_extensions import NotRequired, TypedDict

from .generated_types import ObjectReference
from .logger import Metadata
from .trace import Trace


__all__ = [
    "DatasetPipeline",
    "DatasetPipelineDefinition",
    "DatasetPipelineRow",
    "DatasetPipelineScope",
    "DatasetPipelineSource",
    "DatasetPipelineTarget",
    "DatasetPipelineTransform",
    "DatasetPipelineTransformArgs",
    "DatasetPipelineTransformResult",
    "PipelineSource",
    "PipelineTarget",
]


DatasetPipelineScope: TypeAlias = Literal["span", "trace"]


class DatasetPipelineSource(TypedDict, total=False):
    project_id: str
    project_name: str
    org_name: str
    filter: str
    scope: DatasetPipelineScope


@dataclass(frozen=True)
class PipelineSource:
    filter: str | None = None
    scope: DatasetPipelineScope | None = None
    project_name: str | None = None
    project_id: str | None = None
    org_name: str | None = None

    def as_dict(self) -> DatasetPipelineSource:
        return _drop_none(
            {
                "project_id": self.project_id,
                "project_name": self.project_name,
                "org_name": self.org_name,
                "filter": self.filter,
                "scope": self.scope,
            }
        )


class DatasetPipelineTarget(TypedDict):
    dataset_name: str
    project_id: NotRequired[str]
    project_name: NotRequired[str]
    org_name: NotRequired[str]
    description: NotRequired[str]
    metadata: NotRequired[Metadata]


@dataclass(frozen=True)
class PipelineTarget:
    dataset_name: str
    project_name: str | None = None
    project_id: str | None = None
    org_name: str | None = None
    description: str | None = None
    metadata: Metadata | None = None

    def as_dict(self) -> DatasetPipelineTarget:
        return _drop_none(
            {
                "project_id": self.project_id,
                "project_name": self.project_name,
                "org_name": self.org_name,
                "dataset_name": self.dataset_name,
                "description": self.description,
                "metadata": self.metadata,
            }
        )


class DatasetPipelineRow(TypedDict, total=False):
    id: str
    input: Any | None
    expected: Any | None
    tags: Sequence[str] | None
    metadata: Metadata | None
    origin: ObjectReference


Row = TypeVar("Row", bound=DatasetPipelineRow)


class DatasetPipelineTransformArgs(TypedDict, total=False):
    input: Any | None
    output: Any | None
    metadata: Metadata | None
    expected: Any | None
    trace: Trace


DatasetPipelineTransformResult: TypeAlias = Row | Sequence[Row] | None
DatasetPipelineSourceLike: TypeAlias = DatasetPipelineSource | PipelineSource
DatasetPipelineTargetLike: TypeAlias = DatasetPipelineTarget | PipelineTarget


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _normalize_source(source: DatasetPipelineSourceLike) -> DatasetPipelineSource:
    if isinstance(source, PipelineSource):
        return source.as_dict()
    return dict(source)


def _normalize_target(target: DatasetPipelineTargetLike) -> DatasetPipelineTarget:
    if isinstance(target, PipelineTarget):
        return target.as_dict()
    return dict(target)


class DatasetPipelineTransform(Protocol[Row]):
    def __call__(
        self,
        input: Any | None = None,
        output: Any | None = None,
        metadata: Metadata | None = None,
        expected: Any | None = None,
        trace: Trace | None = None,
    ) -> DatasetPipelineTransformResult[Row] | Awaitable[DatasetPipelineTransformResult[Row]]: ...


@dataclass(frozen=True)
class DatasetPipelineDefinition(Generic[Row]):
    source: DatasetPipelineSource
    transform: DatasetPipelineTransform[Row]
    target: DatasetPipelineTarget
    name: str | None = None


_DATASET_PIPELINES: list[DatasetPipelineDefinition[Any]] = []


def get_registered_dataset_pipelines() -> list[DatasetPipelineDefinition[Any]]:
    return list(_DATASET_PIPELINES)


def is_dataset_pipeline_definition(value: object) -> bool:
    return isinstance(value, DatasetPipelineDefinition)


def DatasetPipeline(
    name: str | None = None,
    *,
    source: DatasetPipelineSourceLike,
    transform: DatasetPipelineTransform[DatasetPipelineRow],
    target: DatasetPipelineTargetLike,
) -> DatasetPipelineDefinition[DatasetPipelineRow]:
    definition = DatasetPipelineDefinition(
        name=name,
        source=_normalize_source(source),
        transform=transform,
        target=_normalize_target(target),
    )
    _DATASET_PIPELINES.append(definition)
    return definition
