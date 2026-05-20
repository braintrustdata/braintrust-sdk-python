from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias


class _PydanticV2Metadata(Protocol):
    def model_dump(self, *, exclude_none: bool = ...) -> Mapping[str, Any]: ...


class _PydanticV1Metadata(Protocol):
    def dict(self, *, exclude_none: bool = ...) -> Mapping[str, Any]: ...


Metadata: TypeAlias = Mapping[str, Any] | _PydanticV2Metadata | _PydanticV1Metadata
