"""Test that braintrust_langchain re-exports the public API from braintrust.integrations.langchain."""

import importlib
import warnings

import pytest


def test_public_api_reexported():
    """All public API symbols should be importable from braintrust_langchain."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        from braintrust_langchain import (
            BraintrustCallbackHandler,
            set_global_handler,
        )

    assert callable(BraintrustCallbackHandler)
    assert callable(set_global_handler)


def test_deprecation_warning():
    """Importing braintrust_langchain should emit a DeprecationWarning."""
    import braintrust_langchain

    with pytest.warns(DeprecationWarning, match="braintrust-langchain is deprecated"):
        importlib.reload(braintrust_langchain)
