"""
DEPRECATED: braintrust-adk is now part of the main braintrust package.

Install `braintrust` and use `braintrust.wrappers.adk` or `braintrust.auto_instrument()` instead.
This package now re-exports from `braintrust.wrappers.adk` for backward compatibility.
"""

import warnings

warnings.warn(
    "braintrust-adk is deprecated. The ADK integration is now included in the main "
    "'braintrust' package. Use 'from braintrust.wrappers.adk import setup_adk' or "
    "'braintrust.auto_instrument()' instead. This package will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export public API from the new location for backward compatibility
from braintrust.wrappers.adk import (  # noqa: E402, F401
    setup_adk,
    setup_braintrust,
    wrap_agent,
    wrap_flow,
    wrap_mcp_tool,
    wrap_runner,
)

__all__ = ["setup_braintrust", "setup_adk", "wrap_agent", "wrap_runner", "wrap_flow", "wrap_mcp_tool"]
