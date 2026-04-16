"""
Nox scripts the environment our tests run in and it used to verify our library
works with and without different dependencies. A few commands to check out:

    nox                        Run all sessions.
    nox -l                     List all sessions.
    nox -s <session>           Run a specific session.
    nox ... -- --no-vcr        Run tests without vcrpy.
    nox ... -- --wheel         Run tests against the wheel in dist.
    nox -h                     Get help.
"""

import functools
import glob
import os
import pathlib
import re
import sys
import tempfile

from packaging.version import Version


if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

import nox


# ---------------------------------------------------------------------------
# Dependency-group helpers
#
# All version pins live in pyproject.toml ``[dependency-groups]``.  The helpers
# below read them once at import time so the noxfile never hardcodes versions.
# ---------------------------------------------------------------------------

_PYPROJECT = tomllib.loads((pathlib.Path(__file__).parent / "pyproject.toml").read_text())
_MATRIX = _PYPROJECT.get("tool", {}).get("braintrust", {}).get("matrix", {})


_PROJECT_DIR = str(pathlib.Path(__file__).parent)


def _install_group_locked(session: nox.Session, *group_names: str) -> None:
    """Install deps from one or more dependency groups using the lockfile.

    Runs ``uv export --only-group <name>`` for each group, merges the output,
    and installs the pre-resolved pins into the session venv.  This gives
    reproducible installs without ad-hoc resolution at install time.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        req_file = f.name
    try:
        cmd = [
            "uv",
            "export",
            "--project",
            _PROJECT_DIR,
            "--no-hashes",
            "--no-emit-project",
            "-o",
            req_file,
        ]
        for name in group_names:
            cmd.extend(["--only-group", name])
        session.run_install(*cmd, silent=SILENT_INSTALLS)
        session.install("-r", req_file, silent=SILENT_INSTALLS)
    finally:
        os.unlink(req_file)


def _get_matrix_versions(prefix: str) -> tuple[str, ...]:
    """Read the version matrix for *prefix* from ``[tool.braintrust.matrix]``.

    Returns a tuple ordered with LATEST first, then descending version order.
    """
    matrix_entry = _MATRIX.get(prefix, {})
    latest = [LATEST] if "latest" in matrix_entry else []
    rest = sorted([v for v in matrix_entry if v != "latest"], key=Version, reverse=True)
    return tuple(latest + rest)


def _install_matrix_dep(session: nox.Session, prefix: str, version: str) -> None:
    """Install the matrix dep for a provider at a specific version."""
    matrix_entry = _MATRIX.get(prefix, {})
    key = "latest" if version == LATEST else version
    req = matrix_entry.get(key)
    if req:
        session.install(req, silent=SILENT_INSTALLS)


# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------


def _pinned_python_version():
    """Return the (major, minor) Python version pinned in ../.tool-versions, or None."""
    tool_versions = pathlib.Path(__file__).parent.parent / ".tool-versions"
    try:
        for line in tool_versions.read_text().splitlines():
            m = re.match(r"^python\s+(\d+)\.(\d+)", line)
            if m:
                return (int(m.group(1)), int(m.group(2)))
    except OSError:
        pass
    return None


_PINNED_PYTHON = _pinned_python_version()

# much faster than pip
nox.options.default_venv_backend = "uv"

SRC_DIR = "braintrust"
WRAPPER_DIR = "braintrust/wrappers"
INTEGRATION_DIR = "braintrust/integrations"
CONTRIB_DIR = "braintrust/contrib"
DEVSERVER_DIR = "braintrust/devserver"
TYPE_TESTS_DIR = "braintrust/type_tests"


SILENT_INSTALLS = True
LATEST = "latest"
ERROR_CODES = tuple(range(1, 256))
INTERNAL_TEST_FLAGS = {"--wheel", "--disable-vcr"}
GENERATED_LINT_EXCLUDES = {
    "src/braintrust/_generated_types.py",
    "src/braintrust/generated_types.py",
}


# ---------------------------------------------------------------------------
# Vendor packages — derived from [tool.braintrust.vendor-packages] in
# pyproject.toml.  Each entry maps a matrix key to a Python import name.
# ---------------------------------------------------------------------------
_VENDOR_TABLE: dict[str, str] = _PYPROJECT.get("tool", {}).get("braintrust", {}).get("vendor-packages", {})

# Import names — used by test_core to verify none are importable.
_VENDOR_IMPORT_NAMES = tuple(_VENDOR_TABLE.values())

# ---------------------------------------------------------------------------
# Version matrices — derived from dependency groups in pyproject.toml
# ---------------------------------------------------------------------------

ANTHROPIC_VERSIONS = _get_matrix_versions("anthropic")


@nox.session()
@nox.parametrize("version", ANTHROPIC_VERSIONS, ids=ANTHROPIC_VERSIONS)
def test_anthropic(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "anthropic", version)
    _run_tests(session, f"{INTEGRATION_DIR}/anthropic/test_anthropic.py", version=version)


OPENAI_VERSIONS = _get_matrix_versions("openai")


@nox.session()
@nox.parametrize("version", OPENAI_VERSIONS, ids=OPENAI_VERSIONS)
def test_openai(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "openai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_openai.py", version=version)
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_oai_attachments.py", version=version)
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_openai_openrouter_gateway.py", version=version)


@nox.session()
@nox.parametrize("version", OPENAI_VERSIONS, ids=OPENAI_VERSIONS)
def test_openai_http2_streaming(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "openai", version)
    # h2 is isolated to this session because it's only needed to force the
    # HTTP/2 LegacyAPIResponse streaming path used by the regression test.
    _install_group_locked(session, "test-openai-http2")
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_openai_http2.py", version=version)


OPENAI_AGENTS_VERSIONS = _get_matrix_versions("openai-agents")


@nox.session()
@nox.parametrize("version", OPENAI_AGENTS_VERSIONS, ids=OPENAI_AGENTS_VERSIONS)
def test_openai_agents(session, version):
    _install_test_deps(session)
    # openai is an auxiliary dep for openai-agents — locked from lockfile
    _install_group_locked(session, "test-openai-agents")
    _install_matrix_dep(session, "openai-agents", version)
    _run_tests(session, f"{INTEGRATION_DIR}/openai_agents/test_openai_agents.py", version=version)


LITELLM_VERSIONS = _get_matrix_versions("litellm")


@nox.session()
@nox.parametrize("version", LITELLM_VERSIONS, ids=LITELLM_VERSIONS)
def test_litellm(session, version):
    _install_test_deps(session)
    # Auxiliary deps (openai upper-bounded, fastapi, orjson) are locked in the lockfile.
    _install_group_locked(session, "test-litellm")
    _install_matrix_dep(session, "litellm", version)
    _run_tests(session, f"{INTEGRATION_DIR}/litellm/test_litellm.py", version=version)


# CLI bundling started in 0.1.10 - older versions require external Claude Code installation
CLAUDE_AGENT_SDK_VERSIONS = _get_matrix_versions("claude-agent-sdk")


@nox.session()
@nox.parametrize("version", CLAUDE_AGENT_SDK_VERSIONS, ids=CLAUDE_AGENT_SDK_VERSIONS)
def test_claude_agent_sdk(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "claude-agent-sdk", version)
    _run_tests(session, f"{INTEGRATION_DIR}/claude_agent_sdk/test_claude_agent_sdk.py", version=version)


# Pin 2.4.0 to cover the 2.4 -> 2.5 breaking change to internals we leverage for instrumentation.
AGNO_VERSIONS = _get_matrix_versions("agno")


@nox.session()
@nox.parametrize("version", AGNO_VERSIONS, ids=AGNO_VERSIONS)
def test_agno(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "agno", version)
    _install_group_locked(session, "test-agno")
    _run_tests(session, f"{INTEGRATION_DIR}/agno/test_agno.py", version=version)
    _run_tests(session, f"{INTEGRATION_DIR}/agno/test_workflow.py", version=version)


AGENTSCOPE_VERSIONS = _get_matrix_versions("agentscope")


@nox.session()
@nox.parametrize("version", AGENTSCOPE_VERSIONS, ids=AGENTSCOPE_VERSIONS)
def test_agentscope(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "agentscope", version)
    _install_group_locked(session, "test-agentscope")
    _run_tests(session, f"{INTEGRATION_DIR}/agentscope/test_agentscope.py", version=version)


# Two test suites with different version requirements:
# 1. wrap_openai approach: works with older versions (0.1.9+)
# 2. Direct wrapper (setup_pydantic_ai): requires 1.10.0+ for all features
PYDANTIC_AI_INTEGRATION_VERSIONS = _get_matrix_versions("pydantic-ai-integration")
PYDANTIC_AI_WRAP_OPENAI_VERSIONS = _get_matrix_versions("pydantic-ai-wrap-openai")


@nox.session()
@nox.parametrize("version", PYDANTIC_AI_INTEGRATION_VERSIONS, ids=PYDANTIC_AI_INTEGRATION_VERSIONS)
def test_pydantic_ai_integration(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "pydantic-ai-integration", version)
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_integration.py", version=version)


@nox.session()
@nox.parametrize("version", PYDANTIC_AI_INTEGRATION_VERSIONS, ids=PYDANTIC_AI_INTEGRATION_VERSIONS)
def test_pydantic_ai_logfire(session, version):
    """Test pydantic_ai + logfire coexistence (issue #1324)."""
    _install_test_deps(session)
    _install_matrix_dep(session, "pydantic-ai-integration", version)
    _install_group_locked(session, "test-pydantic-ai-logfire")
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_logfire.py", version=version)


@nox.session()
@nox.parametrize("version", PYDANTIC_AI_WRAP_OPENAI_VERSIONS, ids=PYDANTIC_AI_WRAP_OPENAI_VERSIONS)
def test_pydantic_ai_wrap_openai(session, version):
    """Test pydantic_ai with wrap_openai() approach - supports older versions."""
    _install_test_deps(session)
    _install_matrix_dep(session, "pydantic-ai-wrap-openai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_wrap_openai.py", version=version)


AUTOEVALS_VERSIONS = _get_matrix_versions("autoevals")


@nox.session()
@nox.parametrize("version", AUTOEVALS_VERSIONS, ids=AUTOEVALS_VERSIONS)
def test_autoevals(session, version):
    # Run all of our core tests with autoevals installed. Some tests
    # specifically validate scores from autoevals work properly, so
    # we need some tests with it installed.
    _install_test_deps(session)
    _install_matrix_dep(session, "autoevals", version)
    _run_core_tests(session)


# google-genai 1.29.0 has a broken async streaming path unless aiohttp is installed.
# 1.30.0 is the earliest version that passes our standard integration test session.
GENAI_VERSIONS = _get_matrix_versions("google-genai")


@nox.session()
@nox.parametrize("version", GENAI_VERSIONS, ids=GENAI_VERSIONS)
def test_google_genai(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "google-genai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/google_genai/test_google_genai.py", version=version)


DSPY_VERSIONS = _get_matrix_versions("dspy")


@nox.session()
@nox.parametrize("version", DSPY_VERSIONS, ids=DSPY_VERSIONS)
def test_dspy(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "dspy", version)
    _run_tests(session, f"{INTEGRATION_DIR}/dspy/test_dspy.py", version=version)


GOOGLE_ADK_VERSIONS = _get_matrix_versions("google-adk")


@nox.session()
@nox.parametrize("version", GOOGLE_ADK_VERSIONS, ids=GOOGLE_ADK_VERSIONS)
def test_google_adk(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "google-adk", version)
    _run_tests(session, f"{INTEGRATION_DIR}/adk/test_adk.py", version=version)
    _run_tests(session, f"{INTEGRATION_DIR}/adk/test_adk_mcp_tool.py", version=version)


LANGCHAIN_VERSIONS = _get_matrix_versions("langchain-core")


@nox.session()
@nox.parametrize("version", LANGCHAIN_VERSIONS, ids=LANGCHAIN_VERSIONS)
def test_langchain(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "langchain-core", version)
    _install_group_locked(session, "test-langchain")
    _run_tests(session, f"{INTEGRATION_DIR}/langchain/test_callbacks.py", version=version)
    _run_tests(session, f"{INTEGRATION_DIR}/langchain/test_context.py", version=version)
    _run_tests(session, f"{INTEGRATION_DIR}/langchain/test_anthropic.py", version=version)


OPENROUTER_VERSIONS = _get_matrix_versions("openrouter")


@nox.session()
@nox.parametrize("version", OPENROUTER_VERSIONS, ids=OPENROUTER_VERSIONS)
def test_openrouter(session, version):
    """Test the native OpenRouter SDK integration."""
    _install_test_deps(session)
    _install_matrix_dep(session, "openrouter", version)
    _run_tests(session, f"{INTEGRATION_DIR}/openrouter/test_openrouter.py", version=version)


MISTRAL_VERSIONS = _get_matrix_versions("mistralai")


@nox.session()
@nox.parametrize("version", MISTRAL_VERSIONS, ids=MISTRAL_VERSIONS)
def test_mistral(session, version):
    """Test the native Mistral SDK integration."""
    _install_test_deps(session)
    _install_matrix_dep(session, "mistralai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/mistral/test_mistral.py", version=version)


TEMPORAL_VERSIONS = _get_matrix_versions("temporalio")


@nox.session()
@nox.parametrize("version", TEMPORAL_VERSIONS, ids=TEMPORAL_VERSIONS)
def test_temporal(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "temporalio", version)
    _run_tests(session, "braintrust/contrib/temporal")


PYTEST_VERSIONS = _get_matrix_versions("pytest-matrix")


@nox.session()
@nox.parametrize("version", PYTEST_VERSIONS, ids=PYTEST_VERSIONS)
def test_pytest_plugin(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "pytest-matrix", version)
    _run_tests(session, f"{WRAPPER_DIR}/pytest_plugin/test_plugin.py")


@nox.session()
def test_core(session):
    _install_test_deps(session)
    # verify we haven't installed our 3p deps.
    for p in _VENDOR_IMPORT_NAMES:
        session.run("python", "-c", f"import {p}", success_codes=ERROR_CODES, silent=True)
    _run_core_tests(session)


@nox.session()
def test_braintrust_core(session):
    # Some tests do specific things if braintrust_core is installed, so run our
    # common tests with it installed. Testing the latest (aka the last ever version)
    # is enough.
    _install_test_deps(session)
    _install_matrix_dep(session, "braintrust-core", LATEST)
    _run_core_tests(session)


@nox.session()
def test_cli(session):
    """Test CLI/devserver with starlette installed."""
    _install_test_deps(session)
    session.install(".[cli]")
    _install_group_locked(session, "test-cli")
    _run_tests(session, DEVSERVER_DIR)


@nox.session()
def test_otel(session):
    """Test OtelExporter with OpenTelemetry installed."""
    _install_test_deps(session)
    session.install(".[otel]")
    _run_tests(session, "braintrust/test_otel.py")


@nox.session()
def test_otel_not_installed(session):
    _install_test_deps(session)
    otel_packages = ["opentelemetry", "opentelemetry.trace", "opentelemetry.exporter.otlp.proto.http.trace_exporter"]
    for pkg in otel_packages:
        session.run("python", "-c", f"import {pkg}", success_codes=ERROR_CODES, silent=True)
    _run_tests(session, "braintrust/test_otel.py")


@nox.session()
def test_types(session):
    """Run type-check tests with pyright, mypy, and pytest."""
    _install_test_deps(session)
    _install_group_locked(session, "test-types")

    type_tests_dir = f"src/{TYPE_TESTS_DIR}"
    test_files = glob.glob(os.path.join(type_tests_dir, "test_*.py"))
    if not test_files:
        session.skip("No type test files found")

    # Run pyright on each file. The local pyrightconfig.json opts these tests
    # into `reportPrivateImportUsage=error` so consumers catching the rule in
    # their editor/IDE stay in sync with what we publish.
    pyright_config = os.path.join(type_tests_dir, "pyrightconfig.json")
    session.run("pyright", "-p", pyright_config, *test_files)

    # Run mypy on each file (only check the test files themselves, not transitive deps)
    session.run("mypy", "--follow-imports=silent", *test_files)

    # Run pytest for the runtime assertions
    _run_tests(session, TYPE_TESTS_DIR)


@nox.session()
def pylint(session):
    # pylint needs everything so we don't trigger missing import errors
    session.install(".[all]")
    # Base test deps + lint tools + all vendor packages, all from the lockfile.
    _install_group_locked(session, "test", "lint")

    result = session.run("git", "ls-files", "**/*.py", silent=True, log=False)
    files = [path for path in result.strip().splitlines() if path not in GENERATED_LINT_EXCLUDES]
    if not files:
        return
    # scripts/ may use APIs only available in the latest pinned Python version
    # (e.g. datetime.UTC requires 3.11+); skip them on older versions.
    if _PINNED_PYTHON and sys.version_info[:2] < _PINNED_PYTHON:
        files = [f for f in files if not f.startswith("scripts/")]
    session.run("pylint", "--errors-only", *files)


def _install_test_deps(session):
    # Choose the way we'll install braintrust ... wheel or source.
    install_wheel = "--wheel" in session.posargs
    bt = _get_braintrust_wheel() if install_wheel else "."

    # Install braintrust itself (wheel or editable source).
    session.install(bt)

    # Install base test deps (pytest, pytest-asyncio, pytest-vcr) from the
    # lockfile so transitive deps are pinned and reproducible.
    _install_group_locked(session, "test")

    # Sanity check we have installed braintrust (and that it is from a wheel if needed)
    session.run("python", "-c", "import braintrust")
    if install_wheel:
        lines = [
            "import sys, braintrust as b",
            "print(f'Using braintrust from: {b.__file__}')",
            "sys.exit(0 if 'site-packages' in b.__file__ else 1)",
        ]
        session.run("python", "-c", ";".join(lines))


def _get_braintrust_wheel():
    path = "dist/braintrust-*.whl"
    wheels = glob.glob(path)
    if len(wheels) != 1:
        msg = f"There should be one wheel in {path}. Got {len(wheels)}"
        raise Exception(msg)
    return wheels[0]


@functools.cache
def _integration_subdirs_to_ignore() -> list[str]:
    """Return integration subdirectories that require dedicated sessions.

    Top-level tests in ``src/braintrust/integrations/`` (e.g. shared utils and
    versioning tests) should still run in ``test_core``.
    """
    integrations_root = pathlib.Path("src") / INTEGRATION_DIR
    return [
        f"{INTEGRATION_DIR}/{child.name}"
        for child in integrations_root.iterdir()
        if child.is_dir() and child.name != "__pycache__"
    ]


def _run_core_tests(session):
    """Run all tests which don't require optional dependencies."""
    _run_tests(
        session,
        SRC_DIR,
        ignore_paths=[
            WRAPPER_DIR,
            *_integration_subdirs_to_ignore(),
            CONTRIB_DIR,
            DEVSERVER_DIR,
            TYPE_TESTS_DIR,
        ],
    )


def _run_tests(session, test_path, ignore_path="", ignore_paths=None, env=None, version=None):
    """Run tests against a wheel or the source code. Paths should be relative and start with braintrust."""
    env = env.copy() if env else {}
    if version:
        env["BRAINTRUST_TEST_PACKAGE_VERSION"] = version
    wheel_flag = "--wheel" in session.posargs
    common_args = ["--disable-vcr"] if "--disable-vcr" in session.posargs else []
    pytest_posargs = [arg for arg in session.posargs if arg not in INTERNAL_TEST_FLAGS]

    # Support both ignore_path (for backward compatibility) and ignore_paths
    paths_to_ignore = []
    if ignore_path:
        paths_to_ignore.append(ignore_path)
    if ignore_paths:
        paths_to_ignore.extend(ignore_paths)

    if not wheel_flag:
        # Run the tests in the src directory
        test_args = [
            "pytest",
            # Disable the braintrust pytest plugin (registered via pytest11 entry
            # point) to avoid ImportPathMismatchError when the installed package
            # and the source tree both contain braintrust/conftest.py.
            "-p",
            "no:braintrust",
            f"src/{test_path}",
        ]
        for path in paths_to_ignore:
            test_args.append(f"--ignore=src/{path}")
        session.run(*test_args, *common_args, *pytest_posargs, env=env)
        return

    # Running the tests from the wheel involves a bit of gymnastics to ensure we don't import
    # local modules from the source directory.
    # First, we need to absolute paths to all the binaries and libs in our venv that we'll see.
    py = os.path.join(session.bin, "python")
    site_packages = session.run(py, "-c", "import site; print(site.getsitepackages()[0])", silent=True).strip()
    abs_test_path = os.path.abspath(os.path.join(site_packages, test_path))
    pytest_path = os.path.join(session.bin, "pytest")

    ignore_args = []
    for path in paths_to_ignore:
        abs_ignore_path = os.path.abspath(os.path.join(site_packages, path))
        ignore_args.append(f"--ignore={abs_ignore_path}")

    # Lastly, change to a different directory to ensure we don't install local stuff.
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        # This env var is used to detect if we're running from the wheel.
        # It proved very helpful because it's very easy
        # to accidentally import local modules from the source directory.
        env["BRAINTRUST_TESTING_WHEEL"] = "1"
        session.run(pytest_path, abs_test_path, *ignore_args, *common_args, *pytest_posargs, env=env)

    # And a final note ... if it's not clear from above, we include test files in our wheel, which
    # is perhaps not ideal?
