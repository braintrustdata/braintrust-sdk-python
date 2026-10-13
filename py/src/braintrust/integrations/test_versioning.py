import sys
import types

import pytest
from braintrust.integrations.base import BaseIntegration
from braintrust.integrations.versioning import detect_module_version, version_satisfies


@pytest.fixture
def dist_info_without_metadata(tmp_path, monkeypatch):
    """A .dist-info left without METADATA, e.g. by an interrupted install."""
    (tmp_path / "bt_fake_provider-1.0.dist-info").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    module = types.ModuleType("bt_fake_provider")
    module.__version__ = "1.2.3"
    return module


_requires_metadata_not_found = pytest.mark.skipif(
    sys.version_info < (3, 15), reason="importlib.metadata raises MetadataNotFound on Python 3.15+"
)


@_requires_metadata_not_found
def test_detect_module_version_falls_back_when_metadata_file_missing(dist_info_without_metadata):
    assert detect_module_version(dist_info_without_metadata, ("bt_fake_provider",)) == "1.2.3"


@_requires_metadata_not_found
def test_integration_detect_version_falls_back_when_metadata_file_missing(dist_info_without_metadata):
    class FakeIntegration(BaseIntegration):
        name = "bt_fake_provider"
        import_names = ("bt_fake_provider",)
        distribution_names = ("bt-fake-provider",)
        patchers = ()

    assert FakeIntegration.detect_version(dist_info_without_metadata) == "1.2.3"


def test_version_satisfies_handles_prereleases():
    # 1.0rc1 is a pre-release of 1.0 — it sorts before the final 1.0 release.
    assert not version_satisfies("1.0rc1", "<1.0")
    assert not version_satisfies("1.0rc1", ">=1.0")

    # Pre-releases of a *different* version work as plain comparisons.
    assert version_satisfies("1.0rc1", "<1.1")
    assert not version_satisfies("1.0rc1", ">=1.1")

    # Explicit pre-release bounds.
    assert version_satisfies("1.0rc1", ">=1.0rc1")
    assert not version_satisfies("1.0rc1", ">1.0rc1")
    assert version_satisfies("1.0rc1", "<1.0rc2")


def test_version_satisfies_ignores_trailing_zeroes():
    assert version_satisfies("1.0.0", "==1.0")
    assert version_satisfies("1.2.0", ">=1.2")


def test_version_satisfies_none_handling():
    # No spec means anything is compatible.
    assert version_satisfies("1.0", None)
    assert version_satisfies(None, None)

    # No version with a spec — optimistically allow so patching still proceeds
    # when version detection fails.
    assert version_satisfies(None, ">=1.0")


def test_version_satisfies_invalid_version():
    assert not version_satisfies("not-a-version", ">=1.0")
