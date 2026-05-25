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
    "get_registered_dataset_pipelines",
]


DatasetPipelineScope: TypeAlias = Literal["span", "trace"]


class DatasetPipelineSource(TypedDict, total=False):
    project_id: str
    project_name: str
    org_name: str
    filter: str
    scope: DatasetPipelineScope


class DatasetPipelineTarget(TypedDict):
    dataset_name: str
    project_id: NotRequired[str]
    project_name: NotRequired[str]
    org_name: NotRequired[str]
    description: NotRequired[str]
    metadata: NotRequired[Metadata]


class DatasetPipelineRow(TypedDict, total=False):
    id: str
    input: Any | None
    expected: Any | None
    tags: Sequence[str] | None
    metadata: Metadata | None
    origin: ObjectReference


Row = TypeVar("Row", bound=DatasetPipelineRow, covariant=True)


class DatasetPipelineTransformArgs(TypedDict, total=False):
    input: Any | None
    output: Any | None
    metadata: Metadata | None
    expected: Any | None
    trace: Trace


DatasetPipelineTransformResult: TypeAlias = Row | Sequence[Row] | None


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


def DatasetPipeline(
    name: str | None = None,
    *,
    source: DatasetPipelineSource,
    transform: DatasetPipelineTransform[DatasetPipelineRow],
    target: DatasetPipelineTarget,
) -> DatasetPipelineDefinition[DatasetPipelineRow]:
    definition = DatasetPipelineDefinition(
        name=name,
        source=source.copy(),
        transform=transform,
        target=target.copy(),
    )
    _DATASET_PIPELINES.append(definition)
    return definition
