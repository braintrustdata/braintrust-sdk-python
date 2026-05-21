from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias


class PydanticV2Metadata(Protocol):
    def model_dump(self, *, exclude_none: bool = ...) -> Mapping[str, Any]: ...


class PydanticV1Metadata(Protocol):
    def dict(self, *, exclude_none: bool = ...) -> Mapping[str, Any]: ...


Metadata: TypeAlias = Mapping[str, Any] | PydanticV2Metadata | PydanticV1Metadata
