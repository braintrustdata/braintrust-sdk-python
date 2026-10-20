"""In-tree PEP 517 backend: setuptools, minus colocated tests in the wheel.

pyproject.toml loads this module via ``backend-path``, which requires the
backend itself to live in the tree, so it re-exports ``setuptools.build_meta``.
"""

import fnmatch

from setuptools.build_meta import *  # noqa: F403
from setuptools.command.build_py import build_py as _build_py


_TEST_MODULE_PATTERNS = ("test_*", "_test_*", "conftest")
# Public and used by downstream test suites (e.g. with_memory_logger).
_SHIPPED_MODULES = ("test_helpers",)


def _is_test_module(module):
    if module in _SHIPPED_MODULES:
        return False
    return any(fnmatch.fnmatch(module, pattern) for pattern in _TEST_MODULE_PATTERNS)


class BuildPy(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [(pkg, mod, path) for pkg, mod, path in modules if not _is_test_module(mod)]
