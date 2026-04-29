"""Deprecated compatibility exports for Braintrust Temporal integration.

Prefer `braintrust.auto_instrument()` for automatic setup, or import explicit helpers
from `braintrust.integrations.temporal`.
"""

import warnings


warnings.warn(
    "braintrust.contrib.temporal is deprecated and will be removed in a future release. "
    "Use braintrust.auto_instrument() or braintrust.integrations.temporal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from braintrust.integrations.temporal import BraintrustInterceptor, BraintrustPlugin  # noqa: E402


__all__ = ["BraintrustInterceptor", "BraintrustPlugin"]
