"""
DEPRECATED: Import from braintrust.wrappers.langchain instead.
"""

import warnings

warnings.warn(
    "braintrust_langchain.callbacks is deprecated. Import from 'braintrust.wrappers.langchain' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from braintrust.integrations.langchain import BraintrustCallbackHandler  # noqa: F401

__all__ = ["BraintrustCallbackHandler"]
