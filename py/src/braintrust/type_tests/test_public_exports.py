"""Regression tests for top-level ``braintrust`` symbols.

The static resource check keeps mypy from resolving generated ``TypedDict``
names instead of the public runtime classes. The runtime checks cover PEP 484
aliasing for pyright's ``reportPrivateImportUsage`` rule and keep generated
exports synchronized with the package root.
"""

import subprocess
import sys

import braintrust
import pytest
from braintrust import (
    Acl,
    auto_instrument,
    setup_ai_sdk,
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
    ("setup_ai_sdk", setup_ai_sdk),
    ("setup_pydantic_ai", setup_pydantic_ai),
]


def accepts_public_resource_types(
    experiment: braintrust.Experiment,
    dataset: braintrust.Dataset,
    project: braintrust.Project,
    prompt: braintrust.Prompt,
    acl: Acl,
) -> None:
    experiment.fetch()
    dataset.fetch()
    _ = project.name
    prompt.build()
    _ = acl["id"]


@pytest.mark.parametrize("name,imported", _PUBLIC_SYMBOLS)
def test_top_level_public_symbol(name: str, imported: object) -> None:
    assert callable(imported)
    assert callable(getattr(braintrust, name))


def test_generated_exports_follow_generated_all() -> None:
    script = """
import importlib

import braintrust
from braintrust import generated_types

future_type = type("FutureGeneratedType", (), {})
generated_types.FutureGeneratedType = future_type
generated_types.__all__.append("FutureGeneratedType")
try:
    importlib.reload(braintrust)
    assert braintrust.FutureGeneratedType is future_type
finally:
    generated_types.__all__.remove("FutureGeneratedType")
    del generated_types.FutureGeneratedType
"""

    subprocess.run([sys.executable, "-c", script], check=True)
