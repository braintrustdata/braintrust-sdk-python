"""Sanity checks for the built braintrust wheel.

Run by the ``test_wheel`` nox session against the wheel installed from
``dist/``, before it is published. These tests live outside the package
because colocated tests are not shipped in the wheel.
"""

import importlib
import importlib.util
import pkgutil
import shutil
import subprocess
import sysconfig
from importlib.metadata import distribution, entry_points

import braintrust
import pytest
from braintrust import version


DIST = distribution("braintrust")
WHEEL_FILES = [str(path) for path in DIST.files]


def test_imports_from_installed_wheel():
    assert "site-packages" in braintrust.__file__, braintrust.__file__


def test_version_is_templated():
    assert version.VERSION == DIST.version
    assert version.GIT_COMMIT and version.GIT_COMMIT != "__GIT_COMMIT__"
    assert version.RELEASE_CHANNEL


def test_ships_py_typed():
    assert "braintrust/py.typed" in WHEEL_FILES


def test_does_not_ship_tests():
    def is_test_file(path):
        name = path.rsplit("/", 1)[-1]
        return (
            (name.startswith(("test_", "_test_")) and name != "test_helpers.py")
            or name == "conftest.py"
            or any(part in path for part in ("/cassettes/", "/type_tests/", "/fixtures/"))
        )

    assert [path for path in WHEEL_FILES if is_test_file(path)] == []


def test_ships_public_test_helpers():
    # Downstream test suites import fixtures from braintrust.test_helpers.
    from braintrust.test_helpers import init_test_logger, with_memory_logger  # noqa: F401


def test_all_modules_import():
    # Optional third-party dependencies are not installed, so ImportErrors for
    # them are expected. A missing braintrust module means the wheel is broken.
    missing = []
    for module in pkgutil.walk_packages(braintrust.__path__, "braintrust."):
        try:
            importlib.import_module(module.name)
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[0] == "braintrust":
                missing.append(f"{module.name}: {e}")
        except ImportError:
            pass
    assert missing == []


def test_pytest_plugin_entry_point_loads():
    (plugin,) = entry_points(group="pytest11", name="braintrust")
    assert hasattr(plugin.load(), "pytest_addoption")


@pytest.mark.parametrize("group", ["console_scripts", "harbor.plugins"])
def test_entry_points_resolve(group):
    eps = [ep for ep in entry_points(group=group) if ep.dist and ep.dist.name == "braintrust"]
    assert eps
    for ep in eps:
        assert importlib.util.find_spec(ep.module) is not None, ep.value


def test_cli_help():
    cli = shutil.which("braintrust", path=sysconfig.get_path("scripts"))
    assert cli is not None
    result = subprocess.run([cli, "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "eval" in result.stdout


def test_local_eval():
    result = braintrust.Eval(
        "wheel-smoke",
        data=[{"input": 1, "expected": 2}, {"input": 2, "expected": 4}],
        task=lambda input: input * 2,
        scores=[lambda input, output, expected: float(output == expected)],
        no_send_logs=True,
    )
    assert [r.output for r in result.results] == [2, 4]
    assert [list(r.scores.values()) for r in result.results] == [[1.0], [1.0]]
