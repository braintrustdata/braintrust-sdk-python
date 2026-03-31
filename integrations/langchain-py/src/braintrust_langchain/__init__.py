"""
DEPRECATED: braintrust-langchain is now part of the main braintrust package.

Install `braintrust` and use `from braintrust.integrations.langchain import BraintrustCallbackHandler` instead.
This package now re-exports from `braintrust.integrations.langchain` for backward compatibility.
"""

import warnings

warnings.warn(
    "braintrust-langchain is deprecated. The LangChain integration is now included in the main "
    "'braintrust' package. Use 'from braintrust.integrations.langchain import BraintrustCallbackHandler' "
    "instead. This package will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export public API from the new location for backward compatibility
from braintrust.integrations.langchain import (  # noqa: E402, F401
    BraintrustCallbackHandler,
    set_global_handler,
)

__all__ = ["BraintrustCallbackHandler", "set_global_handler"]
