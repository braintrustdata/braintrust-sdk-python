import dataclasses
import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any, TypedDict

from typing_extensions import NotRequired

from .serializable_data_class import SerializableDataClass
from .types import Metadata


# =========================================================================
#  !!!!!!!!!!!!!!!! READ THIS BEFORE CHANGING THIS FILE !!!!!!!!!!!!!!!!
#
#  Scores and Scorer classes can be defined in autoevals, braintrust_core
#  or this library or potentially user code. If you make changes here,
#  ensure they are backwards compatible with the existing interfaces.
# =========================================================================


@dataclasses.dataclass
class Score(SerializableDataClass):
    """A score for an evaluation. The score is a float between 0 and 1."""

    name: str
    """The name of the score. This should be a unique name for the scorer."""

    score: float | None = None
    """The score for the evaluation. This should be a float between 0 and 1. If the score is None, the evaluation is considered to be skipped."""

    metadata: Metadata = dataclasses.field(default_factory=dict)
    """Metadata for the score. This can be used to store additional information about the score."""

    # DEPRECATION_NOTICE: this field is deprecated, as errors are propagated up to the caller.
    error: Exception | None = None
    """Deprecated: The error field is deprecated, as errors are now propagated to the caller. The field will be removed in a future version of the library."""

    def as_dict(self):
        return {
            "name": self.name,
            "score": self.score,
            "metadata": self.metadata,
        }

    def __post_init__(self):
        if self.score is not None and (self.score < 0 or self.score > 1):
            raise ValueError(f"score ({self.score}) must be between 0 and 1")
        if self.error is not None:
            warnings.warn(
                "The error field is deprecated, as errors are now propagated to the caller."
                " The field will be removed in a future version of the library"
            )


class ClassificationItem(TypedDict):
    id: str
    label: NotRequired[str]
    metadata: NotRequired[Metadata]


@dataclasses.dataclass
class Classification(SerializableDataClass):
    """A classification result for an evaluation."""

    id: str
    """A machine-readable identifier for the classification outcome."""

    name: str | None = None
    """The name of this classification result. Defaults to the classifier function name when omitted."""

    label: str | None = None
    """Optional human-readable label."""

    metadata: Metadata | None = None
    """Optional metadata attached to the classification result."""

    def as_dict(self):
        result = {"id": self.id}
        if self.name is not None:
            result["name"] = self.name
        if self.label is not None:
            result["label"] = self.label
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    def as_item(self) -> ClassificationItem:
        result: ClassificationItem = {"id": self.id}
        if self.label is not None:
            result["label"] = self.label
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("classification id must be a non-empty string")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError("classification name must be a non-empty string when provided")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("classification label must be a string when provided")


def is_score(obj):
    return hasattr(obj, "name") and hasattr(obj, "score") and hasattr(obj, "metadata") and hasattr(obj, "as_dict")


class Scorer(ABC):
    async def eval_async(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        return await self._run_eval_async(output, expected, **kwargs)

    def eval(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        return self._run_eval_sync(output, expected, **kwargs)

    def __call__(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        return self.eval(output, expected, **kwargs)

    async def _run_eval_async(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        # By default we just run the sync version in a thread
        return self._run_eval_sync(output, expected, **kwargs)

    def _name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _run_eval_sync(self, output: Any, expected: Any = None, **kwargs: Any) -> Score: ...


def is_classification(obj):
    return hasattr(obj, "id") and hasattr(obj, "metadata") and hasattr(obj, "as_dict")


def is_scorer(obj):
    # For class objects, check for appropriate methods
    if inspect.isclass(obj):
        return (
            hasattr(obj, "eval")
            or hasattr(obj, "eval_async")
            or hasattr(obj, "_run_eval_sync")
            or hasattr(obj, "_run_eval_async")
        )
    # For instances, check for appropriate methods
    elif hasattr(obj, "eval") or hasattr(obj, "eval_async"):
        return True
    # For functions/callables, we rely on the type system
    return callable(obj)


__all__ = [
    "Classification",
    "ClassificationItem",
    "Score",
    "Scorer",
    "is_classification",
    "is_score",
    "is_scorer",
]
