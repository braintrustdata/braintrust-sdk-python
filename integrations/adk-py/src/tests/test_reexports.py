"""Test that braintrust_adk re-exports the public API from braintrust.wrappers.adk."""

import warnings

import pytest


def test_public_api_reexported():
    """All public API symbols should be importable from braintrust_adk."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        from braintrust_adk import (
            setup_adk,
            setup_braintrust,
            wrap_agent,
            wrap_flow,
            wrap_mcp_tool,
            wrap_runner,
        )

    assert callable(setup_adk)
    assert callable(setup_braintrust)
    assert callable(wrap_agent)
    assert callable(wrap_flow)
    assert callable(wrap_mcp_tool)
    assert callable(wrap_runner)


def test_deprecation_warning():
    """Importing braintrust_adk should emit a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="braintrust-adk is deprecated"):
        import importlib

        import braintrust_adk

        importlib.reload(braintrust_adk)
