"""Load BTX LLM-span spec YAML files.

Handles the three custom YAML tags used in the spec:
  !fn <name-or-lambda>  — named predicate or arbitrary lambda (eval'd in Python)
  !starts_with <prefix> — string prefix check
  !or [...]             — at-least-one-of validator
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Matcher types (parallel to SpecMatcher.java)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FnMatcher:
    """A named or lambda-expression validator.

    For well-known names (is_non_negative_number, etc.) the span_validator
    module dispatches them to dedicated functions.  For arbitrary Python
    expressions the expression string is stored and eval()'d at validation
    time.
    """

    expr: str  # e.g. "is_non_negative_number" or "lambda value: value > 0"


@dataclasses.dataclass
class StartsWithMatcher:
    prefix: str


@dataclasses.dataclass
class OrMatcher:
    alternatives: list[Any]


# ---------------------------------------------------------------------------
# YAML custom constructors
# ---------------------------------------------------------------------------


def _fn_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> FnMatcher:
    expr = loader.construct_scalar(node)  # type: ignore[arg-type]
    return FnMatcher(expr=expr)


def _starts_with_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> StartsWithMatcher:
    prefix = loader.construct_scalar(node)  # type: ignore[arg-type]
    return StartsWithMatcher(prefix=prefix)


def _or_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> OrMatcher:
    alternatives = loader.construct_sequence(node, deep=True)
    return OrMatcher(alternatives=alternatives)


def _make_loader() -> type:
    """Return a SafeLoader subclass with BTX custom tags registered."""

    class BtxLoader(yaml.SafeLoader):
        pass

    BtxLoader.add_constructor("!fn", _fn_constructor)
    BtxLoader.add_constructor("!starts_with", _starts_with_constructor)
    BtxLoader.add_constructor("!or", _or_constructor)
    return BtxLoader


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LlmSpanSpec:
    name: str
    type: str
    provider: str
    endpoint: str
    requests: list[dict[str, Any]]
    expected_brainstore_spans: list[dict[str, Any]]
    source_path: Path

    @property
    def display_name(self) -> str:
        """pytest ID: <provider>/<name>"""
        return f"{self.provider}/{self.name}"

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: Path) -> "LlmSpanSpec":
        return cls(
            name=data["name"],
            type=data["type"],
            provider=data["provider"],
            endpoint=data["endpoint"],
            requests=data.get("requests", []),
            expected_brainstore_spans=data.get("expected_brainstore_spans", []),
            source_path=source_path,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_BTX_DIR = Path(__file__).parent


def _spec_root(override: str | None = None) -> Path:
    """Return the llm_span spec root directory.

    Priority:
    1. ``override`` argument (used by the pytest fixture after fetching specs)
    2. ``BTX_SPEC_ROOT`` environment variable
    3. ``<btx-dir>/spec/test/llm_span`` (local dev snapshot)
    """
    if override:
        return Path(override)
    env = os.environ.get("BTX_SPEC_ROOT")
    if env:
        return Path(env)
    return _BTX_DIR / "spec" / "test" / "llm_span"


def load_specs(
    spec_root: str | Path | None = None,
    providers: list[str] | None = None,
) -> list[LlmSpanSpec]:
    """Load all YAML spec files under *spec_root*.

    Args:
        spec_root: Path to the ``test/llm_span`` directory.  Falls back to
                   :func:`_spec_root` resolution if ``None``.
        providers: Optional allow-list of provider names (e.g. ``["openai"]``).
                   If ``None``, all providers are loaded.

    Returns:
        Sorted list of :class:`LlmSpanSpec` instances.
    """
    root = Path(spec_root) if spec_root is not None else _spec_root()

    if not root.exists():
        raise FileNotFoundError(
            f"BTX spec root not found: {root}\n"
            "Run the spec-fetch fixture or set BTX_SPEC_ROOT to the llm_span directory."
        )

    loader_cls = _make_loader()
    specs: list[LlmSpanSpec] = []

    for yaml_path in sorted(root.rglob("*.yaml")):
        # Filter by provider directory if requested
        provider_dir = yaml_path.parent.name
        if providers is not None and provider_dir not in providers:
            continue

        with open(yaml_path) as f:
            data = yaml.load(f, Loader=loader_cls)  # noqa: S506 — intentional, custom loader

        spec = LlmSpanSpec.from_dict(data, source_path=yaml_path)
        specs.append(spec)

    return specs
