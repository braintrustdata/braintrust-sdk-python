"""Setuptools build hooks: keep colocated tests out of the wheel."""

import fnmatch

from setuptools.command.build_py import build_py as _build_py


_TEST_MODULE_PATTERNS = ("test_*", "_test_*", "conftest", "test_helpers")


def _is_test_module(module):
    return any(fnmatch.fnmatch(module, pattern) for pattern in _TEST_MODULE_PATTERNS)


class BuildPy(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [(pkg, mod, path) for pkg, mod, path in modules if not _is_test_module(mod)]
