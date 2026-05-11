"""pytest configuration for BTX tests.

Dual-mode operation:
  VCR off (--disable-vcr or --vcr-record=all):
    - Real provider API calls
    - Spans sent to Braintrust backend via real logger
    - Spans fetched back via BTQL for validation

  VCR on (default, cassettes present):
    - Provider HTTP replayed from cassettes
    - Spans captured in memory via _internal_with_memory_background_logger
    - No Braintrust backend calls needed

The VCR mode is detected from the pytest-vcr options already present in the
test session (--disable-vcr / --vcr-record).
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.test_helpers import init_test_logger


_BTX_DIR = Path(__file__).parent
_SPEC_REF_FILE = _BTX_DIR / "spec-ref.txt"
_SPEC_CACHE_DIR = _BTX_DIR / ".spec-cache"

_TEST_PROJECT = "btx-test-project"

# Stash key: spec root path, set by pytest_configure before collection
_spec_root_key = pytest.StashKey[Path]()
# Stash key: whether VCR is disabled (live mode)
_vcr_off_key = pytest.StashKey[bool]()


# ---------------------------------------------------------------------------
# Spec fetching — before collection
# ---------------------------------------------------------------------------


def _read_spec_ref() -> str:
    return _SPEC_REF_FILE.read_text().strip()


def _fetch_spec_if_needed(ref: str) -> Path:
    """Download braintrust-spec@ref into the local cache; skip if already present.

    Pure Python implementation — no bash or curl required, works on all
    platforms including Windows.

    Race-condition safe: extracts into a temporary sibling directory and then
    atomically renames it into the final cache_dir.  If two processes race,
    one wins the rename and the other detects the final directory already
    exists and returns immediately.
    """
    import shutil

    cache_dir = _SPEC_CACHE_DIR / ref
    llm_span_root = cache_dir / "test" / "llm_span"

    if llm_span_root.exists():
        return llm_span_root

    _SPEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[btx] Fetching braintrust-spec@{ref} ...")

    url = f"https://github.com/braintrustdata/braintrust-spec/archive/{ref}.tar.gz"

    # Extract into a unique temp directory next to the final cache_dir so that
    # the eventual os.rename() is atomic (same filesystem, no cross-device move).
    tmp_dir = Path(tempfile.mkdtemp(dir=_SPEC_CACHE_DIR, prefix=f"{ref}.tmp."))
    # Use mkstemp (not deprecated mktemp) to atomically create the temp tarball.
    tmp_tar_fd, tmp_tar_str = tempfile.mkstemp(suffix=".tar.gz", dir=_SPEC_CACHE_DIR)
    os.close(tmp_tar_fd)
    tmp_tar = Path(tmp_tar_str)

    try:
        urllib.request.urlretrieve(url, tmp_tar)

        with tarfile.open(tmp_tar, "r:gz") as tar:
            members = tar.getmembers()
            # Strip the top-level directory (e.g. "braintrust-spec-af0e006/")
            top = members[0].name.split("/")[0] + "/"
            for member in members:
                member.name = member.name[len(top) :]
                if member.name:
                    # filter="data" was added in 3.12; fall back gracefully on older Pythons
                    if sys.version_info >= (3, 12):
                        tar.extract(member, tmp_dir, filter="data")
                    else:
                        tar.extract(member, tmp_dir)  # noqa: S202

        # Atomic rename: if another process already won the race, our tmp_dir
        # is redundant — clean it up and use the existing cache_dir.
        try:
            tmp_dir.rename(cache_dir)
        except (FileExistsError, OSError):
            # Another process beat us to it; that's fine.
            if not llm_span_root.exists():
                raise
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        Path(tmp_tar).unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not llm_span_root.exists():
        raise FileNotFoundError(f"Expected llm_span dir not found after fetch: {llm_span_root}")

    print(f"[btx] Spec cached at {llm_span_root}")
    return llm_span_root


def pytest_configure(config: pytest.Config) -> None:
    """Fetch specs before collection and detect VCR mode."""
    # --- spec fetch ---
    env_override = os.environ.get("BTX_SPEC_ROOT")
    if env_override:
        spec_root = Path(env_override)
    else:
        ref = _read_spec_ref()
        spec_root = _fetch_spec_if_needed(ref)

    config.stash[_spec_root_key] = spec_root
    os.environ["BTX_SPEC_ROOT"] = str(spec_root)

    # --- VCR mode detection ---
    # vcr_off means: bypass VCR entirely, make real API calls, validate via BTQL.
    # This is only true when --disable-vcr is passed.
    # --vcr-record=all means: make real API calls but still use VCR (to record
    # cassettes) and capture spans in-memory — so vcr_off stays False.
    vcr_off = bool(config.getoption("--disable-vcr", default=False, skip=True))
    config.stash[_vcr_off_key] = vcr_off


# ---------------------------------------------------------------------------
# VCR configuration
# ---------------------------------------------------------------------------

# Response headers to drop before writing cassettes. These carry sensitive or
# ephemeral values (session cookies, org/project IDs, per-request trace IDs)
# that should never be committed to source control.
_SCRUB_RESPONSE_HEADERS = {
    "set-cookie",
    "openai-organization",
    "openai-project",
    "x-request-id",
    "cf-ray",
    "cf-cache-status",
    "alt-svc",
}


def _scrub_response_headers(response: dict) -> dict:
    """Strip sensitive/ephemeral headers from responses before cassette write."""
    response["headers"] = {
        k: v for k, v in response.get("headers", {}).items() if k.lower() not in _SCRUB_RESPONSE_HEADERS
    }
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """In CI: record_mode=none. Locally: record_mode=once."""
    record_mode = "none" if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) else "once"
    return {
        "record_mode": record_mode,
        "decode_compressed_response": True,
        # Match on method + URI + body: the request payload (model, messages, etc.)
        # is what determines which cassette response is appropriate.
        # Volatile per-version metadata lives in headers, not the body, so we strip
        # those via filter_headers instead of dropping body from match_on.
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        "filter_headers": [
            "authorization",
            "Authorization",
            "x-api-key",
            "api-key",
            "openai-organization",
            "openai-api-key",
            "x-goog-api-key",
            "x-bt-auth-token",
            "cookie",
            "Cookie",
            # Stainless SDK telemetry headers — version-specific, not part of the
            # request semantics; strip so cassettes survive SDK version bumps.
            "user-agent",
            "User-Agent",
            "x-stainless-arch",
            "x-stainless-async",
            "x-stainless-lang",
            "x-stainless-os",
            "x-stainless-package-version",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
            "x-stainless-read-timeout",
            "x-stainless-retry-count",
        ],
        "before_record_response": _scrub_response_headers,
    }


def _btx_cassette_path(provider: str, spec_name: str) -> str:
    """Return the absolute cassette path for a given provider and spec name.

    Cassettes live in the provider's integration cassette directory so they
    share the same version matrix as the rest of that provider's tests:
        integrations/<provider>/cassettes/<version>/btx/<spec_name>.yaml

    Using an absolute path causes pytest-vcr to ignore vcr_cassette_dir
    entirely and write/read cassettes directly at this location.
    """
    from braintrust.integrations.conftest import _versioned_cassette_dir

    integration_cassettes = _BTX_DIR.parent / "integrations" / provider / "cassettes"
    versioned_dir = Path(_versioned_cassette_dir(str(integration_cassettes)))
    cassette = versioned_dir / "btx" / f"{spec_name}.yaml"
    cassette.parent.mkdir(parents=True, exist_ok=True)
    return str(cassette)


@pytest.fixture
def vcr_cassette_name(request: pytest.FixtureRequest) -> str:
    """Return the absolute cassette path for this spec.

    The parametrize ID is '<provider>/<spec_name>' (e.g. 'openai/completions').
    Cassettes are routed to the provider's own integration directory:
        integrations/<provider>/cassettes/<version>/btx/<spec_name>.yaml
    """
    node_name = request.node.name  # e.g. "test_btx_spec[openai/completions]"
    if "[" in node_name and node_name.endswith("]"):
        spec_id = node_name[node_name.index("[") + 1 : -1]
    else:
        spec_id = node_name

    if "/" in spec_id:
        provider, spec_name = spec_id.split("/", 1)
        return _btx_cassette_path(provider, spec_name)
    return spec_id


# ---------------------------------------------------------------------------
# Mode-aware fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def btx_vcr_off(request: pytest.FixtureRequest) -> bool:
    """True when running in live (VCR-off) mode."""
    return request.config.stash.get(_vcr_off_key, False)


@pytest.fixture(scope="session")
def btx_spec_root(request: pytest.FixtureRequest) -> Path:
    """The llm_span spec root (already fetched by pytest_configure)."""
    return request.config.stash[_spec_root_key]


@pytest.fixture(scope="session")
def btx_project_id(btx_vcr_off: bool) -> str | None:
    """Resolve the Braintrust project ID once per session (live mode only).

    In VCR mode this is never called.  In live mode the project name/ID is
    constant across all test cases, so we look it up once here rather than
    once per parametrized test.
    """
    if not btx_vcr_off:
        return None
    project_id = os.environ.get("BRAINTRUST_PROJECT_ID") or os.environ.get("BRAINTRUST_DEFAULT_PROJECT_ID")
    if project_id:
        return project_id
    from .span_fetcher import fetch_project_id

    project = os.environ.get("BRAINTRUST_PROJECT") or os.environ.get(
        "BRAINTRUST_DEFAULT_PROJECT_NAME", "python-unit-test"
    )
    return fetch_project_id(project)


@pytest.fixture
def memory_logger(btx_vcr_off):
    """In VCR-on mode: install in-memory span capture.
    In VCR-off mode: yield None (spans go to the real Braintrust backend).
    """
    if btx_vcr_off:
        yield None
    else:
        init_test_logger(_TEST_PROJECT)
        with logger._internal_with_memory_background_logger() as bgl:
            yield bgl
