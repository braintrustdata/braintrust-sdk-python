"""BTX: Cross-language LLM-span spec tests for the Braintrust Python SDK.

Dual-mode:

  VCR off (--disable-vcr):
    Real provider API calls → spans sent to Braintrust backend → fetched via
    BTQL and validated against the spec.

  VCR on (default, requires cassettes):
    Provider HTTP replayed from cassettes → spans captured in memory →
    validated in-memory against the spec.

Recording cassettes for the first time:
    pytest src/braintrust/btx/ --vcr-record=all -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from .span_validator import validate_spans
from .spec_executor import execute_spec
from .spec_loader import LlmSpanSpec, load_specs


# ---------------------------------------------------------------------------
# Spec loading at collection time
# (pytest_configure in conftest.py sets BTX_SPEC_ROOT before this runs)
# ---------------------------------------------------------------------------

_BTX_DIR = Path(__file__).parent


def _resolve_providers() -> list[str]:
    provider = os.environ.get("BTX_PROVIDER")
    if not provider:
        raise RuntimeError(
            "BTX_PROVIDER environment variable is not set. Run btx tests via nox: nox -s 'test_btx_openai(latest)'"
        )
    return [provider.strip()]


def _load_specs() -> list[LlmSpanSpec]:
    spec_root_env = os.environ.get("BTX_SPEC_ROOT")
    if spec_root_env:
        spec_root = Path(spec_root_env)
    else:
        from .conftest import _SPEC_CACHE_DIR, _read_spec_ref

        ref = _read_spec_ref()
        spec_root = _SPEC_CACHE_DIR / ref / "test" / "llm_span"

    if not spec_root.exists():
        raise FileNotFoundError(
            f"BTX spec root not found: {spec_root}\n"
            "This module must be collected after conftest.py's pytest_configure hook "
            "has run (which sets BTX_SPEC_ROOT). If you are seeing this error, "
            "try running via pytest rather than importing directly."
        )
    providers = _resolve_providers()
    return load_specs(spec_root=spec_root, providers=providers)


_all_specs: list[LlmSpanSpec] = _load_specs()

if not _all_specs:
    providers = _resolve_providers()
    raise RuntimeError(
        f"No BTX specs found for provider(s) {providers} under {os.environ.get('BTX_SPEC_ROOT')}. "
        "Check that BTX_PROVIDER and the spec ref are correct."
    )

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.vcr
@pytest.mark.parametrize(
    "spec",
    _all_specs,
    ids=[f"{s.provider}/{s.name}" for s in _all_specs],
)
def test_btx_spec(
    spec: LlmSpanSpec,
    memory_logger: Any,
    btx_vcr_off: bool,
    btx_project_id: str | None,
    btx_spec_root: Path,
) -> None:
    root_span_id = execute_spec(spec, live=btx_vcr_off)

    if btx_vcr_off:
        # Live mode: fetch spans from Braintrust backend via BTQL
        from .span_fetcher import fetch_spans

        assert btx_project_id is not None

        num_expected = len(spec.expected_brainstore_spans)
        print(f"\n[btx] Fetching {num_expected} span(s) for root_span_id={root_span_id!r} ...")
        spans = fetch_spans(root_span_id, btx_project_id, num_expected)
        print(f"[btx] Got {len(spans)} span(s), validating...")
    else:
        # VCR mode: spans captured in memory by memory_logger
        assert memory_logger is not None, "memory_logger should be set in VCR mode"
        spans = memory_logger.pop()
        assert spans, f"{spec.display_name}: no spans captured in memory — check that the client is wrapped correctly"

    validate_spans(spans, spec)
