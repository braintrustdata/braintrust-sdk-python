import os
from pathlib import Path

import pytest


def _versioned_cassette_dir(base_dir: str) -> str:
    """Return a version-specific cassette directory when ``BRAINTRUST_TEST_PACKAGE_VERSION`` is set.

    Nox sessions pass the version under test via the ``BRAINTRUST_TEST_PACKAGE_VERSION``
    environment variable.  When present, cassettes are read from / written to a
    version-specific subdirectory (e.g. ``cassettes/latest/``, ``cassettes/0.50.0/``).

    When the variable is absent (e.g. running a test file directly), the base
    ``cassettes/`` directory is used for backward compatibility.
    """
    version = os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION")
    if not version:
        return base_dir
    return str(Path(base_dir) / version)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Version-aware cassette directory for all integration tests.

    Resolves ``<test_dir>/cassettes/`` by default, or
    ``<test_dir>/cassettes/<version>/`` when the
    ``BRAINTRUST_TEST_PACKAGE_VERSION`` env-var is set by a nox session.
    """
    test_dir = request.node.fspath.dirname
    base_dir = os.path.join(test_dir, "cassettes")
    return _versioned_cassette_dir(base_dir)
