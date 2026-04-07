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

import glob
import os
import pathlib
import re
import sys
import tempfile

import nox


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


SILENT_INSTALLS = True
LATEST = "latest"
ERROR_CODES = tuple(range(1, 256))
INTERNAL_TEST_FLAGS = {"--wheel", "--disable-vcr"}
GENERATED_LINT_EXCLUDES = {
    "src/braintrust/_generated_types.py",
    "src/braintrust/generated_types.py",
}


# The minimal set of dependencies we need to run tests.
BASE_TEST_DEPS = ("pytest", "pytest-asyncio", "pytest-vcr")

# List your package here if it's not guaranteed to be installed. We'll (try to)
# validate things work with or without them.
VENDOR_PACKAGES = (
    "agno",
    "agentscope",
    "anthropic",
    "dspy",
    "openai",
    "openai-agents",
    # pydantic_ai is NOT included here - it has dedicated test sessions with version-specific handling
    "autoevals",
    "braintrust_core",
    "litellm",
    "mistralai",
    "openrouter",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
    "google.genai",
    "google.adk",
    "temporalio",
)

# Test matrix
ANTHROPIC_VERSIONS = (LATEST, "0.50.0", "0.49.0", "0.48.0")
OPENAI_VERSIONS = (LATEST, "1.77.0", "1.71", "1.91", "1.92")
OPENAI_AGENTS_VERSIONS = (LATEST, "0.0.19")
# litellm latest requires Python >= 3.10
# Pin litellm because 1.82.7-1.82.8 are compromised: https://github.com/BerriAI/litellm/issues/24512
LITELLM_VERSIONS = ("1.82.0", "1.74.0")
# CLI bundling started in 0.1.10 - older versions require external Claude Code installation
CLAUDE_AGENT_SDK_VERSIONS = (LATEST, "0.1.10")
# Keep LATEST for newest API coverage, and pin 2.4.0 to cover the 2.4 -> 2.5 breaking change
# to internals we leverage for instrumentation.
AGNO_VERSIONS = (LATEST, "2.4.0", "2.1.0")
AGENTSCOPE_VERSIONS = (LATEST, "1.0.0")
# pydantic_ai 1.x requires Python >= 3.10
# Two test suites with different version requirements:
# 1. wrap_openai approach: works with older versions (0.1.9+)
# 2. Direct wrapper (setup_pydantic_ai): requires 1.10.0+ for all features
PYDANTIC_AI_WRAP_OPENAI_VERSIONS = (LATEST, "1.0.1", "0.1.9")
PYDANTIC_AI_INTEGRATION_VERSIONS = (LATEST, "1.10.0")

AUTOEVALS_VERSIONS = (LATEST, "0.0.129")
# google-genai 1.29.0 has a broken async streaming path unless aiohttp is installed.
# 1.30.0 is the earliest version that passes our standard integration test session.
GENAI_VERSIONS = (LATEST, "1.30.0")
# dspy 2.6.0 is the earliest version that matches the callback/settings APIs our
# integration and tests rely on.
DSPY_VERSIONS = (LATEST, "2.6.0")
GOOGLE_ADK_VERSIONS = (LATEST, "1.14.1")
LANGCHAIN_VERSIONS = (LATEST, "0.3.28")
OPENROUTER_VERSIONS = (LATEST, "0.6.0")
MISTRAL_VERSIONS = (LATEST, "1.12.4")
# temporalio 1.19.0+ requires Python >= 3.10; skip Python 3.9 entirely
TEMPORAL_VERSIONS = (LATEST, "1.20.0", "1.19.0")
PYTEST_VERSIONS = (LATEST, "8.4.2")


@nox.session()
def test_core(session):
    _install_test_deps(session)
    # verify we haven't installed our 3p deps.
    for p in VENDOR_PACKAGES:
        session.run("python", "-c", f"import {p}", success_codes=ERROR_CODES, silent=True)
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", PYDANTIC_AI_WRAP_OPENAI_VERSIONS, ids=PYDANTIC_AI_WRAP_OPENAI_VERSIONS)
def test_pydantic_ai_wrap_openai(session, version):
    """Test pydantic_ai with wrap_openai() approach - supports older versions."""
    _install_test_deps(session)
    _install(session, "pydantic_ai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_wrap_openai.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", PYDANTIC_AI_INTEGRATION_VERSIONS, ids=PYDANTIC_AI_INTEGRATION_VERSIONS)
def test_pydantic_ai_integration(session, version):
    """Test pydantic_ai with setup_pydantic_ai() wrapper - requires 1.10.0+."""
    # Skip on Python 3.9 - pydantic_ai 1.10.0+ requires Python 3.10+
    if sys.version_info < (3, 10):
        session.skip("pydantic_ai integration tests require Python >= 3.10 (pydantic_ai 1.10.0+)")
    _install_test_deps(session)
    _install(session, "pydantic_ai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_integration.py")
    _run_core_tests(session)


@nox.session()
def test_pydantic_ai_logfire(session):
    """Test pydantic_ai + logfire coexistence (issue #1324)."""
    if sys.version_info < (3, 10):
        session.skip("pydantic_ai + logfire tests require Python >= 3.10")
    _install_test_deps(session)
    _install(session, "pydantic_ai")
    _install(session, "logfire")
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_logfire.py")


@nox.session()
@nox.parametrize("version", CLAUDE_AGENT_SDK_VERSIONS, ids=CLAUDE_AGENT_SDK_VERSIONS)
def test_claude_agent_sdk(session, version):
    # claude_agent_sdk requires Python >= 3.10
    # These tests use subprocess-transport cassettes, so they can replay in CI
    # while still exercising the real Claude Agent SDK control protocol.
    _install_test_deps(session)
    _install(session, "claude_agent_sdk", version)
    _run_tests(session, f"{INTEGRATION_DIR}/claude_agent_sdk/test_claude_agent_sdk.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", AGNO_VERSIONS, ids=AGNO_VERSIONS)
def test_agno(session, version):
    _install_test_deps(session)
    _install(session, "agno", version)
    _install(session, "openai")  # Required for agno.models.openai
    _install(session, "fastapi")  # Required for agno.workflow
    _run_tests(session, f"{INTEGRATION_DIR}/agno/test_agno.py")
    _run_tests(session, f"{INTEGRATION_DIR}/agno/test_workflow.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", AGENTSCOPE_VERSIONS, ids=AGENTSCOPE_VERSIONS)
def test_agentscope(session, version):
    _install_test_deps(session)
    _install(session, "agentscope", version)
    _install(session, "openai")
    _run_tests(session, f"{INTEGRATION_DIR}/agentscope/test_agentscope.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", ANTHROPIC_VERSIONS, ids=ANTHROPIC_VERSIONS)
def test_anthropic(session, version):
    _install_test_deps(session)
    _install(session, "anthropic", version)
    _run_tests(session, f"{WRAPPER_DIR}/test_anthropic.py")
    _run_tests(session, f"{INTEGRATION_DIR}/anthropic/test_anthropic.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", GENAI_VERSIONS, ids=GENAI_VERSIONS)
def test_google_genai(session, version):
    _install_test_deps(session)
    _install(session, "google-genai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/google_genai/test_google_genai.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", GOOGLE_ADK_VERSIONS, ids=GOOGLE_ADK_VERSIONS)
def test_google_adk(session, version):
    """Test Google ADK integration."""
    _install_test_deps(session)
    _install(session, "google-adk", version)
    _run_tests(session, f"{INTEGRATION_DIR}/adk/test_adk.py")
    _run_tests(session, f"{INTEGRATION_DIR}/adk/test_adk_mcp_tool.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", LANGCHAIN_VERSIONS, ids=LANGCHAIN_VERSIONS)
def test_langchain(session, version):
    """Test LangChain integration."""
    # langchain requires Python >= 3.10
    if sys.version_info < (3, 10):
        session.skip("langchain requires Python >= 3.10")
    _install_test_deps(session)
    _install(session, "langchain-core", version)
    _install(session, "langchain-openai")
    _install(session, "langchain-anthropic")
    _install(session, "langgraph")
    _run_tests(session, f"{INTEGRATION_DIR}/langchain/test_callbacks.py")
    _run_tests(session, f"{INTEGRATION_DIR}/langchain/test_context.py")
    _run_tests(session, f"{INTEGRATION_DIR}/langchain/test_anthropic.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", OPENAI_VERSIONS, ids=OPENAI_VERSIONS)
def test_openai(session, version):
    _install_test_deps(session)
    _install(session, "openai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_openai.py")
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_oai_attachments.py")
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_openai_openrouter_gateway.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", OPENAI_AGENTS_VERSIONS, ids=OPENAI_AGENTS_VERSIONS)
def test_openai_agents(session, version):
    if sys.version_info < (3, 10):
        session.skip("openai-agents requires Python >= 3.10")
    _install_test_deps(session)
    _install(session, "openai")
    _install(session, "openai-agents", version)
    _run_tests(session, f"{INTEGRATION_DIR}/openai_agents/test_openai_agents.py")
    _run_core_tests(session)


@nox.session()
def test_openai_http2_streaming(session):
    _install_test_deps(session)
    _install(session, "openai")
    # h2 is isolated to this session because it's only needed to force the
    # HTTP/2 LegacyAPIResponse streaming path used by the regression test.
    session.install("h2")
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_openai_http2.py")


@nox.session()
@nox.parametrize("version", OPENROUTER_VERSIONS, ids=OPENROUTER_VERSIONS)
def test_openrouter(session, version):
    """Test the native OpenRouter SDK integration."""
    _install_test_deps(session)
    _install(session, "openrouter", version)
    _run_tests(session, f"{INTEGRATION_DIR}/openrouter/test_openrouter.py")


@nox.session()
@nox.parametrize("version", MISTRAL_VERSIONS, ids=MISTRAL_VERSIONS)
def test_mistral(session, version):
    """Test the native Mistral SDK integration."""
    _install_test_deps(session)
    _install(session, "mistralai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/mistral/test_mistral.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", LITELLM_VERSIONS, ids=LITELLM_VERSIONS)
def test_litellm(session, version):
    # litellm latest requires Python >= 3.10
    if version == LATEST and sys.version_info < (3, 10):
        session.skip("litellm latest requires Python >= 3.10")
    _install_test_deps(session)
    # Install a compatible version of openai (1.99.9 or lower) to avoid the ResponseTextConfig removal in 1.100.0
    # https://github.com/BerriAI/litellm/issues/13711
    # Install fastapi and orjson as they're required by litellm for proxy/responses operations
    session.install("openai<=1.99.9", "--force-reinstall", "fastapi", "orjson")
    _install(session, "litellm", version)
    _run_tests(session, f"{INTEGRATION_DIR}/litellm/test_litellm.py")
    _run_core_tests(session)


@nox.session()
@nox.parametrize("version", DSPY_VERSIONS, ids=DSPY_VERSIONS)
def test_dspy(session, version):
    # dspy latest depends on litellm which requires Python >= 3.10
    if sys.version_info < (3, 10):
        session.skip("dspy latest requires Python >= 3.10 (litellm dependency)")
    _install_test_deps(session)
    _install(session, "dspy", version)
    _run_tests(session, f"{INTEGRATION_DIR}/dspy/test_dspy.py")


@nox.session()
@nox.parametrize("version", AUTOEVALS_VERSIONS, ids=AUTOEVALS_VERSIONS)
def test_autoevals(session, version):
    # Run all of our core tests with autoevals installed. Some tests
    # specifically validate scores from autoevals work properly, so
    # we need some tests with it installed.
    _install_test_deps(session)
    _install(session, "autoevals", version)
    _run_core_tests(session)


@nox.session()
def test_braintrust_core(session):
    # Some tests do specific things if braintrust_core is installed, so run our
    # common tests with it installed. Testing the latest (aka the last ever version)
    # is enough.
    _install_test_deps(session)
    _install(session, "braintrust_core")
    _run_core_tests(session)


@nox.session()
def test_cli(session):
    """Test CLI/devserver with starlette installed."""
    _install_test_deps(session)
    session.install(".[cli]")
    session.install("httpx")  # Required for starlette.testclient
    _run_tests(session, DEVSERVER_DIR)


@nox.session()
@nox.parametrize("version", PYTEST_VERSIONS, ids=PYTEST_VERSIONS)
def test_pytest_plugin(session, version):
    _install_test_deps(session)
    _install(session, "pytest", version)
    _run_tests(session, f"{WRAPPER_DIR}/pytest_plugin/test_plugin.py")


@nox.session()
def test_otel(session):
    """Test OtelExporter with OpenTelemetry installed."""
    _install_test_deps(session)
    session.install(".[otel]")
    _run_tests(session, "braintrust/test_otel.py")


@nox.session()
@nox.parametrize("version", TEMPORAL_VERSIONS, ids=TEMPORAL_VERSIONS)
def test_temporal(session, version):
    """Test Temporal integration with temporalio installed."""
    # temporalio 1.19.0+ requires Python >= 3.10
    if sys.version_info < (3, 10):
        session.skip("temporalio 1.19.0+ requires Python >= 3.10")
    _install_test_deps(session)
    _install(session, "temporalio", version)
    _run_tests(session, "braintrust/contrib/temporal")


@nox.session()
def test_otel_not_installed(session):
    _install_test_deps(session)
    otel_packages = ["opentelemetry", "opentelemetry.trace", "opentelemetry.exporter.otlp.proto.http.trace_exporter"]
    for pkg in otel_packages:
        session.run("python", "-c", f"import {pkg}", success_codes=ERROR_CODES, silent=True)
    _run_tests(session, "braintrust/test_otel.py")


@nox.session()
def pylint(session):
    # pylint needs everything so we don't trigger missing import errors
    session.install(".[all]")
    session.install("-r", "requirements-dev.txt")
    session.install(*VENDOR_PACKAGES)
    # pydantic_ai is not in VENDOR_PACKAGES (has dedicated test sessions),
    # but pylint needs it with minimum version constraint for proper API checking
    session.install("pydantic_ai>=1.10.0")
    session.install("google-adk")
    session.install("opentelemetry.instrumentation.openai")
    # langsmith is needed for the langsmith_wrapper module but not in VENDOR_PACKAGES
    # langchain-core, langchain-openai, langchain-anthropic are needed for the langchain integration
    session.install("langsmith", "langchain-core", "langchain-openai", "langchain-anthropic")

    result = session.run("git", "ls-files", "**/*.py", silent=True, log=False)
    files = [path for path in result.strip().splitlines() if path not in GENERATED_LINT_EXCLUDES]
    if not files:
        return
    # scripts/ may use APIs only available in the latest pinned Python version
    # (e.g. datetime.UTC requires 3.11+); skip them on older versions.
    if _PINNED_PYTHON and sys.version_info[:2] < _PINNED_PYTHON:
        files = [f for f in files if not f.startswith("scripts/")]
    session.run("pylint", "--errors-only", *files)


@nox.session()
def test_latest_wrappers_novcr(session):
    """Run the latest wrapper tests without vcrpy."""
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        session.skip("Skipping novcr tests in CI (no real API keys available)")
    # every test run we hit openai, anthropic,  at least once so we balance CI speed (with vcrpy)
    # with testing reality.
    args = session.posargs.copy()
    if "--disable-vcr" not in args:
        args.append("--disable-vcr")
    session.notify("test_openai(latest)", posargs=args)
    session.notify("test_anthropic(latest)", posargs=args)
    session.notify("test_pydantic_ai_wrap_openai(latest)", posargs=args)
    session.notify("test_pydantic_ai_integration(latest)", posargs=args)
    session.notify("test_claude_agent_sdk(latest)", posargs=args)


def _install_npm_in_session(session):
    """Install Node.js and npm in the nox session using nodeenv."""
    session.install("nodeenv", silent=SILENT_INSTALLS)
    # Create a node environment in the session's temporary directory
    node_dir = os.path.join(session.create_tmp(), "node_env")
    session.run("nodeenv", node_dir, silent=SILENT_INSTALLS)
    # Return the path to npm binary for direct use
    if sys.platform == "win32":
        npm_bin = os.path.join(node_dir, "Scripts", "npm.cmd")
    else:
        npm_bin = os.path.join(node_dir, "bin", "npm")
    return npm_bin


def _install_test_deps(session):
    # Choose the way we'll install braintrust ... wheel or source.
    install_wheel = "--wheel" in session.posargs
    bt = _get_braintrust_wheel() if install_wheel else "."

    # Install _only_ the dependencies we need for testing (not lint, black,
    # ipython, whatever). We want to carefully control the base
    # testing environment so it should be truly minimal.
    session.install(bt, *BASE_TEST_DEPS)

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


def _run_core_tests(session):
    """Run all tests which don't require optional dependencies."""
    _run_tests(
        session,
        SRC_DIR,
        ignore_paths=[
            WRAPPER_DIR,
            INTEGRATION_DIR,
            CONTRIB_DIR,
            DEVSERVER_DIR,
        ],
    )


def _run_tests(session, test_path, ignore_path="", ignore_paths=None, env=None):
    """Run tests against a wheel or the source code. Paths should be relative and start with braintrust."""
    env = env.copy() if env else {}
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


def _install(session, package, version=LATEST):
    pkg_version = f"{package}=={version}"
    if version == LATEST or not version:
        pkg_version = package
    session.install(pkg_version, silent=SILENT_INSTALLS)
