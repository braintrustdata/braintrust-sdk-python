"""Braintrust integration for the TypeSafe Python SDK."""

from typing import Any

from .integration import TypeSafeIntegration
from .patchers import AsyncSystemOnePatcher, SystemOnePatcher


def setup_typesafe() -> bool:
    """Instrument installed TypeSafe sync and async clients."""
    return TypeSafeIntegration.setup()


def wrap_typesafe(client: Any) -> Any:
    """Instrument a TypeSafe sync or async client in place."""
    from typesafe_sdk import AsyncTypeSafeClient

    patcher = AsyncSystemOnePatcher if isinstance(client, AsyncTypeSafeClient) else SystemOnePatcher
    return patcher.wrap_target(client)


__all__ = [
    "TypeSafeIntegration",
    "setup_typesafe",
    "wrap_typesafe",
]
