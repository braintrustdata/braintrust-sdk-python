"""
DEPRECATED: braintrust-adk is now part of the main braintrust package.

Install `braintrust` and use `braintrust.wrappers.adk` or `braintrust.auto_instrument()` instead.
This package now re-exports from `braintrust.wrappers.adk` for backward compatibility.
"""

import warnings
from contextlib import aclosing  # noqa: F401

warnings.warn(
    "braintrust-adk is deprecated. The ADK integration is now included in the main "
    "'braintrust' package. Use 'from braintrust.wrappers.adk import setup_adk' or "
    "'braintrust.auto_instrument()' instead. This package will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location for backward compatibility
from braintrust.bt_json import bt_safe_deep_copy  # noqa: E402, F401
from braintrust.logger import init_logger, start_span  # noqa: E402, F401
from braintrust.wrappers.adk import (  # noqa: E402, F401
    _determine_llm_call_type,
    _extract_metrics,
    _extract_model_name,
    _is_patched,
    _omit,
    _serialize_config,
    _serialize_content,
    _serialize_part,
    _serialize_pydantic_schema,
    setup_adk,
    setup_braintrust,
    wrap_agent,
    wrap_flow,
    wrap_mcp_tool,
    wrap_runner,
)

__all__ = ["setup_braintrust", "setup_adk", "wrap_agent", "wrap_runner", "wrap_flow", "wrap_mcp_tool"]
