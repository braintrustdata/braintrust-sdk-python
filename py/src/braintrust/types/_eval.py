"""Internal TypedDict types used in Eval/EvalAsync signatures.

These live in an underscore-prefixed module so they don't appear in
generated documentation, while the class names themselves are *not*
underscore-prefixed so pyright strict mode doesn't flag them as private.
"""

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from typing_extensions import NotRequired, TypedDict


Input = TypeVar("Input")
Expected = TypeVar("Expected")


class EvalCaseDictNoOutput(Generic[Input], TypedDict):
    """
    Workaround for the Pyright type checker handling of generics. Specifically,
    the type checker doesn't know that a dict which is missing the key
    "expected" can be used to satisfy ``EvalCaseDict[Input, Expected]`` for any
    ``Expected`` type.
    """

    input: Input
    metadata: NotRequired[dict[str, Any] | None]
    tags: NotRequired[Sequence[str] | None]
    trial_count: NotRequired[int | None]

    id: NotRequired[str | None]
    _xact_id: NotRequired[str | None]


class EvalCaseDict(Generic[Input, Expected], EvalCaseDictNoOutput[Input]):
    """
    Mirrors EvalCase for callers who pass a dict instead of dataclass.
    """

    expected: NotRequired[Expected | None]


class ExperimentDatasetEvent(TypedDict):
    """
    TODO: This could be unified with ``EvalCaseDict`` like we do in the
    TypeScript SDK, or generated from OpenAPI spec.
    """

    id: str
    _xact_id: str
    input: Any | None
    expected: Any | None
    tags: Sequence[str] | None
