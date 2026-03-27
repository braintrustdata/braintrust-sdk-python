from braintrust.integrations.versioning import version_satisfies


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
