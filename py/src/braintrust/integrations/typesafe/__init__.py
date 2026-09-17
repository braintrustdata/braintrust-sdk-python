"""Braintrust integration for the TypeSafe Python SDK."""

from typing import Any

from .integration import TypeSafeIntegration


def setup_typesafe() -> bool:
    """Instrument installed TypeSafe sync and async clients."""
    return TypeSafeIntegration.setup()


def wrap_typesafe(client: Any) -> Any:
    """Instrument a TypeSafe sync or async client in place."""
    TypeSafeIntegration.setup()
    return client


__all__ = [
    "TypeSafeIntegration",
    "setup_typesafe",
    "wrap_typesafe",
]
