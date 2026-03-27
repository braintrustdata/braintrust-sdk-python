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
            BraintrustTracer,
            clear_global_handler,
            set_global_handler,
        )

    assert callable(BraintrustCallbackHandler)
    assert callable(BraintrustTracer)
    assert callable(set_global_handler)
    assert callable(clear_global_handler)


def test_context_module_reexported():
    """braintrust_langchain.context should still work for users who imported from there directly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        from braintrust_langchain.context import clear_global_handler, set_global_handler

    assert callable(set_global_handler)
    assert callable(clear_global_handler)


def test_deprecation_warning():
    """Importing braintrust_langchain should emit a DeprecationWarning."""
    import braintrust_langchain

    with pytest.warns(DeprecationWarning, match="braintrust-langchain is deprecated"):
        importlib.reload(braintrust_langchain)
