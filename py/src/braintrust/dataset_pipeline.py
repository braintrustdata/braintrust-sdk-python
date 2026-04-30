from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from typing_extensions import NotRequired, TypedDict

from .generated_types import ObjectReference
from .logger import Metadata
from .trace import Trace


DatasetPipelineScope: TypeAlias = Literal["span", "trace"]


class DatasetPipelineSource(TypedDict, total=False):
    project_id: str
    project_name: str
    org_name: str
    filter: str
    scope: DatasetPipelineScope
    limit: int


class DatasetPipelineTarget(TypedDict, total=False):
    project_id: str
    project_name: str
    org_name: str
    dataset_name: str
    description: str
    metadata: Metadata


class DatasetPipelineRow(TypedDict, total=False):
    id: str
    input: Any | None
    expected: Any | None
    tags: Sequence[str] | None
    metadata: Metadata | None
    origin: ObjectReference


class DatasetPipelineCandidate(TypedDict):
    trace: Trace
    id: NotRequired[str]
    origin: NotRequired[ObjectReference]


Candidate = TypeVar("Candidate", bound=DatasetPipelineCandidate)
Row = TypeVar("Row", bound=DatasetPipelineRow)


class DatasetPipelineTransformContext(TypedDict):
    pipeline: "DatasetPipelineDefinition[Any, Any]"


DatasetPipelineTransformResult: TypeAlias = Row | Sequence[Row] | None
DatasetPipelineTransform: TypeAlias = Callable[
    [Candidate, DatasetPipelineTransformContext],
    DatasetPipelineTransformResult[Row] | Awaitable[DatasetPipelineTransformResult[Row]],
]


@dataclass(frozen=True)
class DatasetPipelineDefinition(Generic[Candidate, Row]):
    source: DatasetPipelineSource
    transform: DatasetPipelineTransform[Candidate, Row]
    target: DatasetPipelineTarget
    name: str | None = None


_DATASET_PIPELINES: list[DatasetPipelineDefinition[Any, Any]] = []


def get_registered_dataset_pipelines() -> list[DatasetPipelineDefinition[Any, Any]]:
    return list(_DATASET_PIPELINES)


def is_dataset_pipeline_definition(value: object) -> bool:
    return isinstance(value, DatasetPipelineDefinition)


def DatasetPipeline(
    name: str | None = None,
    *,
    source: DatasetPipelineSource,
    transform: DatasetPipelineTransform[DatasetPipelineCandidate, DatasetPipelineRow],
    target: DatasetPipelineTarget,
) -> DatasetPipelineDefinition[DatasetPipelineCandidate, DatasetPipelineRow]:
    definition = DatasetPipelineDefinition(
        name=name,
        source=source,
        transform=transform,
        target=target,
    )
    _DATASET_PIPELINES.append(definition)
    return definition
