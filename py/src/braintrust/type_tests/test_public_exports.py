"""Regression test: public symbols re-exported from the top-level ``braintrust`` package.

Pyright's ``reportPrivateImportUsage`` treats ``from .x import y`` inside a
``py.typed`` package as a private import — consumer code doing
``from braintrust import y`` or ``braintrust.y`` then fails type checking.
The local ``pyrightconfig.json`` in this directory enables the rule at
error severity so a regression here trips ``nox -s test_types``.

Run as type checks:
    nox -s test_types

Run as pytest:
    pytest src/braintrust/type_tests/test_public_exports.py
"""

import braintrust
from braintrust import (
    auto_instrument,
    setup_pydantic_ai,
    wrap_anthropic,
    wrap_litellm,
    wrap_openai,
    wrap_openrouter,
)


def test_top_level_public_symbols_are_importable() -> None:
    assert callable(auto_instrument)
    assert callable(wrap_anthropic)
    assert callable(wrap_litellm)
    assert callable(wrap_openai)
    assert callable(wrap_openrouter)
    assert callable(setup_pydantic_ai)


def test_top_level_public_symbols_are_attributes() -> None:
    assert callable(braintrust.auto_instrument)
    assert callable(braintrust.wrap_anthropic)
    assert callable(braintrust.wrap_litellm)
    assert callable(braintrust.wrap_openai)
    assert callable(braintrust.wrap_openrouter)
    assert callable(braintrust.setup_pydantic_ai)
