"""
DEPRECATED: Import from braintrust.wrappers.langchain instead.
"""

import warnings

warnings.warn(
    "braintrust_langchain.context is deprecated. Import from 'braintrust.wrappers.langchain' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from braintrust.integrations.langchain import (  # noqa: F401
    clear_global_handler,
    set_global_handler,
)

__all__ = ["set_global_handler", "clear_global_handler"]
