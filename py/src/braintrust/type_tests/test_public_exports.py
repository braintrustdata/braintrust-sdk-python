"""Regression test for pyright's ``reportPrivateImportUsage`` on top-level ``braintrust`` symbols.

Without PEP 484 ``as``-aliasing in ``braintrust/__init__.py``, pyright flags
``from braintrust import auto_instrument`` (and peers) as private in a
``py.typed`` consumer. The local ``pyrightconfig.json`` turns the rule into
an error so this file breaks ``nox -s test_types`` if someone regresses the
aliasing pattern.
"""

import braintrust
import pytest
from braintrust import (
    auto_instrument,
    setup_pydantic_ai,
    wrap_anthropic,
    wrap_litellm,
    wrap_openai,
    wrap_openrouter,
)


_PUBLIC_SYMBOLS = [
    ("auto_instrument", auto_instrument),
    ("wrap_anthropic", wrap_anthropic),
    ("wrap_litellm", wrap_litellm),
    ("wrap_openai", wrap_openai),
    ("wrap_openrouter", wrap_openrouter),
    ("setup_pydantic_ai", setup_pydantic_ai),
]


@pytest.mark.parametrize("name,imported", _PUBLIC_SYMBOLS)
def test_top_level_public_symbol(name: str, imported: object) -> None:
    assert callable(imported)
    assert callable(getattr(braintrust, name))
