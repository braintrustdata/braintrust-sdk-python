"""Compatibility re-exports — implementation lives in braintrust.integrations.litellm."""

from braintrust.integrations.litellm import (
    patch_litellm,  # noqa: F401
    wrap_litellm,  # noqa: F401
)


__all__ = [
    "patch_litellm",
    "wrap_litellm",
]
