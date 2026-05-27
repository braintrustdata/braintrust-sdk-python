from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar

from typing_extensions import NotRequired, TypedDict

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
]


DatasetPipelineScope: TypeAlias = Literal["span", "trace"]


class DatasetPipelineSource(TypedDict, total=False):
    """Information about what spans or traces should be passed into the dataset pipeline."""

    project_id: str
    """Project ID to take spans or traces from. Takes precedence over project_name."""
    project_name: str
    """Project name to take spans or traces from."""
    org_name: str
    """Organization name to take spans or traces from."""
    filter: str
    """Optional BTQL filter. When omitted, all spans or traces are eligible."""
    scope: DatasetPipelineScope
    """Whether to pass spans or entire traces to the pipeline. Defaults to "span"."""


class DatasetPipelineTarget(TypedDict):
    """Information about the target dataset."""

    dataset_name: str
    """Dataset name. This can be an existing dataset name or a name to create."""
    project_id: NotRequired[str]
    """Project ID where the dataset lives or should be created."""
    project_name: NotRequired[str]
    """Project name where the dataset lives or should be created."""
    org_name: NotRequired[str]
    """Organization name where the dataset lives or should be created."""
    description: NotRequired[str]
    """Dataset description to use when creating the dataset."""
    metadata: NotRequired[Metadata]
    """Dataset metadata to use when creating the dataset."""


class DatasetPipelineRow(TypedDict, total=False):
    """A row returned by a dataset pipeline transform."""

    id: str
    """Stable row ID for the target dataset. Defaults to the source span or trace ID."""
    input: Any | None
    """Input value for the target dataset row."""
    expected: Any | None
    """Expected value for the target dataset row."""
    tags: Sequence[str] | None
    """Tags for the target dataset row."""
    metadata: Metadata | None
    """Metadata for the target dataset row."""


Row = TypeVar("Row", bound=DatasetPipelineRow, covariant=True)


class DatasetPipelineTransformArgs(TypedDict, total=False):
    """Arguments passed to a dataset pipeline transform."""

    id: str
    """Source span row ID for span-scoped transforms."""
    input: Any | None
    """Source span input for span-scoped transforms."""
    output: Any | None
    """Source span output for span-scoped transforms."""
    metadata: Metadata | None
    """Source span metadata for span-scoped transforms."""
    expected: Any | None
    """Source span expected value for span-scoped transforms."""
    trace: Trace
    """Source trace. This is always available."""


DatasetPipelineTransformResult: TypeAlias = Row | Sequence[Row] | None


class DatasetPipelineTransform(Protocol[Row]):
    def __call__(
        self,
        id: str | None = None,
        input: Any | None = None,
        output: Any | None = None,
        metadata: Metadata | None = None,
        expected: Any | None = None,
        trace: Trace | None = None,
    ) -> DatasetPipelineTransformResult[Row] | Awaitable[DatasetPipelineTransformResult[Row]]: ...


@dataclass(frozen=True)
class DatasetPipelineDefinition(Generic[Row]):
    """A registered dataset pipeline definition consumed by the bt CLI."""

    source: DatasetPipelineSource
    transform: DatasetPipelineTransform[Row]
    target: DatasetPipelineTarget
    name: str | None = None


_DATASET_PIPELINES: list[DatasetPipelineDefinition[Any]] = []


def DatasetPipeline(
    name: str | None = None,
    *,
    source: DatasetPipelineSource,
    transform: DatasetPipelineTransform[DatasetPipelineRow],
    target: DatasetPipelineTarget,
) -> DatasetPipelineDefinition[DatasetPipelineRow]:
    """Create a runnable dataset pipeline.

    Dataset pipelines take trace data stored in Braintrust, filter and transform it,
    and feed it back into a Braintrust dataset.

    Run a dataset pipeline with the bt CLI:

        bt datasets pipeline run path/to/pipeline.py --limit 100

    The limit controls how many spans or traces, depending on source["scope"], are
    discovered for the pipeline.

    This API is experimental and may change or be removed across non-major versions.
    """
    stored_source = source.copy()
    stored_source["scope"] = stored_source.get("scope", "span")
    definition = DatasetPipelineDefinition(
        name=name,
        source=stored_source,
        transform=transform,
        target=target.copy(),
    )
    _DATASET_PIPELINES.append(definition)
    return definition
