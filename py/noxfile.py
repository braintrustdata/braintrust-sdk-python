"""
Nox scripts the environment our tests run in and it used to verify our library
works with and without different dependencies. A few commands to check out:

    nox                        Run all sessions.
    nox -l                     List all sessions.
    nox -s <session>           Run a specific session.
    nox ... -- --disable-vcr  Run tests without vcrpy.
    nox ... -- --wheel         Run tests against the wheel in dist.
    nox -h                     Get help.
"""

import functools
import glob
import hashlib
import os
import pathlib
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request

from packaging.version import Version


sys.path.insert(0, str(pathlib.Path(__file__).parent))

from nox_session_constraints import incompatibility_reason  # noqa: E402


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
_UV_EXCLUDE_NEWER = _PYPROJECT["tool"]["uv"]["exclude-newer"]


_PROJECT_DIR = str(pathlib.Path(__file__).parent)


def _ensure_livekit_server(session: nox.Session) -> str:
    """Ensure a standalone livekit-server binary is available for LiveKit e2e tests."""
    existing = shutil.which("livekit-server")
    if existing:
        return os.path.dirname(existing)

    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if arch is None:
        session.skip(f"No pinned livekit-server binary for architecture {machine!r}")

    if system != "linux":
        session.skip(
            "No pinned standalone livekit-server release asset is available for this platform; "
            "install livekit-server on PATH to run LiveKit e2e tests locally"
        )

    cache_root = pathlib.Path(os.environ.get("BRAINTRUST_LIVEKIT_SERVER_DIR", ".nox/livekit-server"))
    install_dir = cache_root / LIVEKIT_SERVER_VERSION / f"{system}_{arch}"
    binary = install_dir / "livekit-server"
    if binary.exists():
        return str(install_dir.resolve())

    install_dir.mkdir(parents=True, exist_ok=True)
    asset = f"livekit_{LIVEKIT_SERVER_VERSION}_{system}_{arch}.tar.gz"
    url = f"https://github.com/livekit/livekit/releases/download/v{LIVEKIT_SERVER_VERSION}/{asset}"
    archive = install_dir / asset
    expected_sha256 = LIVEKIT_SERVER_SHA256[f"{system}_{arch}"]
    session.log(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive)  # noqa: S310 - pinned public release asset for test infra.
    actual_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        archive.unlink(missing_ok=True)
        session.error(
            f"SHA256 mismatch for {asset}: expected {expected_sha256}, got {actual_sha256}. "
            "Refusing to extract downloaded livekit-server archive."
        )
    with tarfile.open(archive, "r:gz") as tar:
        if sys.version_info >= (3, 12):
            tar.extract("livekit-server", path=install_dir, filter="data")
        else:
            tar.extract("livekit-server", path=install_dir)  # noqa: S202
    binary.chmod(0o755)
    archive.unlink()
    return str(install_dir.resolve())


def _install_group_locked(
    session: nox.Session,
    *group_names: str,
    indexes: tuple[str, ...] = (),
) -> None:
    """Install deps from one or more dependency groups using the lockfile.

    Runs ``uv export --only-group <name>`` for each group, merges the output,
    and installs the pre-resolved pins into the session venv. This gives
    reproducible installs without ad-hoc resolution at install time.

    ``indexes`` names explicit ``[[tool.uv.index]]`` entries that must remain
    available while installing exported requirements.
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
        install_args = ["-r", req_file]
        configured_indexes = {
            index.get("name"): index.get("url") for index in _PYPROJECT.get("tool", {}).get("uv", {}).get("index", [])
        }
        for index_name in indexes:
            index_url = configured_indexes.get(index_name)
            if not index_url:
                session.error(f"Unknown [[tool.uv.index]] name: {index_name!r}")
            install_args.extend(("--index", index_url))
        if indexes:
            # Exported requirements are fully pinned, so it is safe to search
            # all configured indexes for each exact version. Without this,
            # uv may stop at PyTorch's index for unrelated packages.
            install_args.extend(("--index-strategy", "unsafe-best-match"))
        _session_install(session, *install_args, silent=SILENT_INSTALLS)
    finally:
        os.unlink(req_file)


def _skip_if_incompatible(session: nox.Session) -> None:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    reason = incompatibility_reason(session.name, python_version, platform.system())
    if reason:
        session.skip(reason)


def _get_matrix_versions(prefix: str) -> tuple[str, ...]:
    """Read the version matrix for *prefix* from ``[tool.braintrust.matrix]``.

    Returns a tuple ordered with LATEST first, then descending version order.
    """
    matrix_entry = _MATRIX.get(prefix)
    if not matrix_entry:
        raise KeyError(f"Missing [tool.braintrust.matrix.{prefix}] in pyproject.toml")
    latest = [LATEST] if "latest" in matrix_entry else []
    rest = sorted([v for v in matrix_entry if v != "latest"], key=Version, reverse=True)
    return tuple(latest + rest)


def _install_matrix_dep(session: nox.Session, prefix: str, version: str, constraint_group: str | None = None) -> None:
    """Install a matrix dependency, optionally using pinned compatibility constraints."""
    matrix_entry = _MATRIX.get(prefix)
    if not matrix_entry:
        session.error(f"Missing [tool.braintrust.matrix.{prefix}] in pyproject.toml")
    key = "latest" if version == LATEST else version
    req = matrix_entry.get(key)
    if not req:
        session.error(f"Missing matrix key {key!r} in [tool.braintrust.matrix.{prefix}]")
    if constraint_group is None:
        _session_install(session, req, silent=SILENT_INSTALLS)
        return

    constraints = _PYPROJECT.get("dependency-groups", {}).get(constraint_group, [])
    if not constraints or not all(isinstance(constraint, str) for constraint in constraints):
        session.error(f"Constraint group {constraint_group!r} must contain only requirement strings")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as constraint_file:
        constraint_file.write("\n".join(constraints))
        constraint_file.flush()
        _session_install(session, req, "-c", constraint_file.name, silent=SILENT_INSTALLS)


def _session_install(session: nox.Session, *args, **kwargs) -> None:
    """Install packages with the project's reproducibility cutoff."""
    session.install("--exclude-newer", _UV_EXCLUDE_NEWER, *args, **kwargs)


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
BTX_DIR = "braintrust/btx"


SILENT_INSTALLS = True
LATEST = "latest"
LIVEKIT_SERVER_VERSION = "1.11.0"
LIVEKIT_SERVER_SHA256 = {
    "linux_amd64": "3e76ed51ecdfefc3005e4257095dccd1ccc8f8b77517d9f2353de7906650b68b",
    "linux_arm64": "6741466bc12e75544338292ab2c1c02c02f3c626568230b5548fffc53e5a87ff",
}
ERROR_CODES = tuple(range(1, 256))
INTERNAL_TEST_FLAGS = {"--wheel", "--disable-vcr"}
GENERATED_LINT_EXCLUDES = (
    "src/braintrust/_generated_types.py",
    "src/braintrust/generated_types.py",
    "src/braintrust/api/_generated/",
)


# ---------------------------------------------------------------------------
# Vendor packages — derived from [tool.braintrust.vendor-packages] in
# pyproject.toml.  Each entry maps a matrix key to a Python import name.
# ---------------------------------------------------------------------------
_VENDOR_TABLE: dict[str, str] = _PYPROJECT.get("tool", {}).get("braintrust", {}).get("vendor-packages", {})

# Import names — used by test_core to verify none are importable.
_VENDOR_IMPORT_NAMES = tuple(_VENDOR_TABLE.values())

# ---------------------------------------------------------------------------
# Version matrices — derived from [tool.braintrust.matrix] in pyproject.toml
# ---------------------------------------------------------------------------

AI_SDK_VERSIONS = _get_matrix_versions("ai-sdk")

ANTHROPIC_VERSIONS = _get_matrix_versions("anthropic")

COHERE_VERSIONS = _get_matrix_versions("cohere")

BOTO3_VERSIONS = _get_matrix_versions("boto3")

INSTRUCTOR_VERSIONS = _get_matrix_versions("instructor")

OPENAI_VERSIONS = _get_matrix_versions("openai")
OPENAI_ENDPOINT_VERSIONS = (OPENAI_VERSIONS[0], OPENAI_VERSIONS[-1])


def _register_matrix_sessions(specs):
    """Register the routine provider sessions from compact data records."""
    for spec in specs:

        def run(session, version, spec=spec):
            _skip_if_incompatible(session)
            _install_test_deps(session, *spec.get("groups", ()))
            for package, package_version in spec["dependencies"]:
                _install_matrix_dep(session, package, version if package_version == "$version" else package_version)
            _run_tests(
                session,
                spec["tests"],
                version=version,
                env=spec.get("env"),
            )

        run.__name__ = spec["name"]
        run.__qualname__ = spec["name"]
        run.__doc__ = spec.get("doc")
        decorated = nox.parametrize("version", spec["versions"], ids=spec["versions"])(run)
        globals()[spec["name"]] = nox.session()(decorated)


_register_matrix_sessions(
    [
        {
            "name": "test_ai_sdk",
            "versions": AI_SDK_VERSIONS,
            "dependencies": [("openai", LATEST), ("ai-sdk", "$version")],
            "tests": f"{INTEGRATION_DIR}/ai_sdk/test_ai_sdk.py",
        },
        {
            "name": "test_anthropic",
            "versions": ANTHROPIC_VERSIONS,
            "dependencies": [("anthropic", "$version")],
            "tests": f"{INTEGRATION_DIR}/anthropic/test_anthropic.py",
        },
        {
            "name": "test_cohere",
            "versions": COHERE_VERSIONS,
            "dependencies": [("cohere", "$version")],
            "tests": f"{INTEGRATION_DIR}/cohere/test_cohere.py",
        },
        {
            "name": "test_bedrock_runtime",
            "versions": BOTO3_VERSIONS,
            "dependencies": [("boto3", "$version"), ("botocore", "$version")],
            "tests": f"{INTEGRATION_DIR}/bedrock_runtime/test_bedrock_runtime.py",
        },
        {
            "name": "test_instructor",
            "versions": INSTRUCTOR_VERSIONS,
            "groups": ("test-instructor",),
            "dependencies": [("instructor", "$version")],
            "tests": f"{INTEGRATION_DIR}/instructor/test_instructor.py",
        },
        {
            "name": "test_openai",
            "versions": OPENAI_VERSIONS,
            "dependencies": [("openai", "$version")],
            "tests": [
                f"{INTEGRATION_DIR}/openai/test_openai.py",
                f"{INTEGRATION_DIR}/openai/test_oai_attachments.py",
                f"{INTEGRATION_DIR}/openai/test_openai_openrouter_gateway.py",
            ],
        },
    ]
)


@nox.session()
@nox.parametrize("version", OPENAI_ENDPOINT_VERSIONS, ids=OPENAI_ENDPOINT_VERSIONS)
def test_openai_http2_streaming(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "openai", version)
    # h2 is isolated to this session because it's only needed to force the
    # HTTP/2 LegacyAPIResponse streaming path used by the regression test.
    _install_group_locked(session, "test-openai-http2")
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_openai_http2.py", version=version)


@nox.session()
@nox.parametrize("version", OPENAI_ENDPOINT_VERSIONS, ids=OPENAI_ENDPOINT_VERSIONS)
def test_btx_openai(session, version):
    """Run the BTX cross-language LLM-span spec tests (OpenAI provider)."""
    _install_test_deps(session, "test-btx")
    _install_matrix_dep(session, "openai", version)
    _run_tests(session, "braintrust/btx", version=version, env={"BTX_PROVIDER": "openai", "BTX_CLIENT": "openai"})


@nox.session()
def test_openai_ddtrace(session):
    _install_test_deps(session)
    _install_matrix_dep(session, "openai", LATEST)
    _install_group_locked(session, "test-openai-ddtrace")
    _run_tests(session, f"{INTEGRATION_DIR}/openai/test_openai_ddtrace.py", version=LATEST)


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

# LiteLLM >= 1.102.0 fetches model_prices_and_context_window.json from GitHub on
# import. Recording that lands a ~3MB blob in the cassette, so force the bundled
# copy instead and keep the fetch off the wire.
#
# Only safe where the tests stick to chat completions: the bundled map of an
# older LiteLLM does not know newer model names, and test_litellm(1.74.0) fails
# provider resolution for gpt-image-1-mini under it. Scoped to DSPy for now.
_LITELLM_LOCAL_COST_MAP = {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}


@nox.session()
@nox.parametrize("version", LITELLM_VERSIONS, ids=LITELLM_VERSIONS)
def test_litellm(session, version):
    # LiteLLM 1.97.0 leaves Pydantic forward references unresolved on Python 3.10.
    _skip_if_incompatible(session)
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
    # Claude Agent SDK 0.1.10 calls Server.list_tools(), which MCP 2 removed.
    constraint_group = "test-mcp-v1" if version == "0.1.10" else None
    _install_matrix_dep(session, "claude-agent-sdk", version, constraint_group)
    _run_tests(session, f"{INTEGRATION_DIR}/claude_agent_sdk/test_claude_agent_sdk.py", version=version)


CURSOR_SDK_VERSIONS = _get_matrix_versions("cursor-sdk")


@nox.session()
@nox.parametrize("version", CURSOR_SDK_VERSIONS, ids=CURSOR_SDK_VERSIONS)
def test_cursor_sdk(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "cursor-sdk", version)
    # Characterization coverage enables likely downstream provider
    # instrumentation and verifies that Cursor's subprocess bridge does not
    # emit provider-owned Python spans for its internal model requests.
    _install_matrix_dep(session, "openai", LATEST)
    _install_matrix_dep(session, "anthropic", LATEST)
    _run_tests(session, f"{INTEGRATION_DIR}/cursor_sdk/test_cursor_sdk.py", version=version)


# Pin 2.4.0 to cover the 2.4 -> 2.5 breaking change to internals we leverage for instrumentation.
AGNO_VERSIONS = _get_matrix_versions("agno")


@nox.session()
@nox.parametrize("version", AGNO_VERSIONS, ids=AGNO_VERSIONS)
def test_agno(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "agno", version)
    _install_group_locked(session, "test-agno")
    _run_tests(session, f"{INTEGRATION_DIR}/agno", version=version)


LIVEKIT_AGENTS_VERSIONS = _get_matrix_versions("livekit-agents")


@nox.session()
@nox.parametrize("version", LIVEKIT_AGENTS_VERSIONS, ids=LIVEKIT_AGENTS_VERSIONS)
def test_livekit_agents(session, version):
    _skip_if_incompatible(session)
    _install_test_deps(session)
    _install_matrix_dep(session, "livekit-agents", version)
    _install_group_locked(session, "test-livekit-agents")
    if version == "1.3.1":
        # LiveKit 1.3.1 accepts a newer OTLP exporter, but its dependencies
        # require an SDK version that breaks LiveKit's LogData import. The
        # locked group downgrades the exporter, so remove its now-orphaned deps.
        session.run(
            "uv",
            "pip",
            "uninstall",
            "opentelemetry-exporter-otlp-common",
            "opentelemetry-exporter-http-transport",
        )
    livekit_server_dir = _ensure_livekit_server(session)
    env = {
        "LIVEKIT_URL": os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
        "LIVEKIT_API_KEY": os.environ.get("LIVEKIT_API_KEY", "devkey"),
        "LIVEKIT_API_SECRET": os.environ.get("LIVEKIT_API_SECRET", "secret"),
        "PATH": f"{livekit_server_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    _run_tests(session, f"{INTEGRATION_DIR}/livekit_agents/test_livekit_agents.py", version=version, env=env)


PIPECAT_VERSIONS = _get_matrix_versions("pipecat-ai")


@nox.session()
@nox.parametrize("version", PIPECAT_VERSIONS, ids=PIPECAT_VERSIONS)
def test_pipecat(session, version):
    _skip_if_incompatible(session)
    _install_test_deps(session)
    _install_group_locked(session, "test-pipecat")
    _install_matrix_dep(session, "pipecat-ai", version)
    # Pipecat imports NLTK, whose safe-import finder rejects dependencies
    # loaded from Nox's virtualenv when it is beneath the current directory.
    _run_tests(
        session,
        f"{INTEGRATION_DIR}/pipecat/test_pipecat.py",
        version=version,
        run_from_temp_dir=True,
    )


STRANDS_VERSIONS = _get_matrix_versions("strands-agents")


@nox.session()
@nox.parametrize("version", STRANDS_VERSIONS, ids=STRANDS_VERSIONS)
def test_strands(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "strands-agents", version)
    _install_group_locked(session, "test-strands")
    _run_tests(session, f"{INTEGRATION_DIR}/strands/test_strands.py", version=version)


AGENTSCOPE_VERSIONS = _get_matrix_versions("agentscope")


@nox.session()
@nox.parametrize("version", AGENTSCOPE_VERSIONS, ids=AGENTSCOPE_VERSIONS)
def test_agentscope(session, version):
    _skip_if_incompatible(session)
    _install_test_deps(session)
    # AgentScope 1.0.0 imports streamablehttp_client, which MCP 2 no longer exports.
    constraint_group = "test-mcp-v1" if version == "1.0.0" else None
    _install_matrix_dep(session, "agentscope", version, constraint_group)
    _install_group_locked(session, "test-agentscope")
    _run_tests(session, f"{INTEGRATION_DIR}/agentscope/test_agentscope.py", version=version)


AUTOGEN_VERSIONS = _get_matrix_versions("autogen-agentchat")


@nox.session()
@nox.parametrize("version", AUTOGEN_VERSIONS, ids=AUTOGEN_VERSIONS)
def test_autogen(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "autogen-agentchat", version)
    _install_matrix_dep(session, "autogen-ext", version)
    _run_tests(session, f"{INTEGRATION_DIR}/autogen/test_autogen.py", version=version)


# Two test suites with different version requirements:
# 1. wrap_openai approach: works with older versions (0.1.9+)
# 2. Direct wrapper (setup_pydantic_ai): requires 1.10.0+ for all features
PYDANTIC_AI_INTEGRATION_VERSIONS = _get_matrix_versions("pydantic-ai-integration")
PYDANTIC_AI_WRAP_OPENAI_VERSIONS = _get_matrix_versions("pydantic-ai-wrap-openai")


@nox.session()
@nox.parametrize("version", PYDANTIC_AI_INTEGRATION_VERSIONS, ids=PYDANTIC_AI_INTEGRATION_VERSIONS)
def test_pydantic_ai_integration(session, version):
    _install_test_deps(session)
    # Pydantic AI 1.10.0 imports opentelemetry._events, removed in opentelemetry-api 1.40.
    constraint_group = "test-pydantic-ai-otel-events" if version == "1.10.0" else None
    _install_matrix_dep(session, "pydantic-ai-integration", version, constraint_group)
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_integration.py", version=version)


@nox.session()
@nox.parametrize("version", PYDANTIC_AI_INTEGRATION_VERSIONS, ids=PYDANTIC_AI_INTEGRATION_VERSIONS)
def test_pydantic_ai_logfire(session, version):
    """Test pydantic_ai + logfire coexistence (issue #1324)."""
    _install_test_deps(session, "test-pydantic-ai-logfire")
    _install_matrix_dep(session, "pydantic-ai-integration", version, "test-pydantic-ai-logfire-constraints")
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_logfire.py", version=version)


@nox.session()
@nox.parametrize("version", PYDANTIC_AI_WRAP_OPENAI_VERSIONS, ids=PYDANTIC_AI_WRAP_OPENAI_VERSIONS)
def test_pydantic_ai_wrap_openai(session, version):
    """Test pydantic_ai with wrap_openai() approach - supports older versions."""
    _install_test_deps(session)
    # These versions import opentelemetry._events, removed in opentelemetry-api 1.40.
    constraint_group = "test-pydantic-ai-otel-events" if version in {"0.1.9", "1.0.1"} else None
    _install_matrix_dep(session, "pydantic-ai-wrap-openai", version, constraint_group)
    _run_tests(session, f"{INTEGRATION_DIR}/pydantic_ai/test_pydantic_ai_wrap_openai.py", version=version)


AUTOEVALS_VERSIONS = _get_matrix_versions("autoevals")


@nox.session()
@nox.parametrize("version", AUTOEVALS_VERSIONS, ids=AUTOEVALS_VERSIONS)
def test_autoevals(session, version):
    # Only this test exercises the changed scorer behavior. Running the full
    # core suite here duplicates test_core without extending package coverage.
    _install_test_deps(session)
    _install_matrix_dep(session, "autoevals", version)
    _run_tests(session, f"{SRC_DIR}/test_framework.py::test_run_evaluator_with_many_scorers")


# google-genai 1.29.0 has a broken async streaming path unless aiohttp is installed.
# 1.30.0 is the earliest version that passes our standard integration test session.
GENAI_VERSIONS = _get_matrix_versions("google-genai")


@nox.session()
@nox.parametrize("version", GENAI_VERSIONS, ids=GENAI_VERSIONS)
def test_google_genai(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "google-genai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/google_genai/test_google_genai.py", version=version)


GOOGLE_DISCOVERYENGINE_VERSIONS = _get_matrix_versions("google-cloud-discoveryengine")


@nox.session()
@nox.parametrize("version", GOOGLE_DISCOVERYENGINE_VERSIONS, ids=GOOGLE_DISCOVERYENGINE_VERSIONS)
def test_google_discoveryengine(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "google-cloud-discoveryengine", version)
    _run_tests(session, f"{INTEGRATION_DIR}/google_discoveryengine", version=version)


DSPY_VERSIONS = _get_matrix_versions("dspy")


@nox.session()
@nox.parametrize("version", DSPY_VERSIONS, ids=DSPY_VERSIONS)
def test_dspy(session, version):
    # DSPy latest preinstalls our latest LiteLLM pin, which is currently broken
    # on Python 3.10 due to unresolved Pydantic forward references.
    _skip_if_incompatible(session)
    _install_test_deps(session)
    if version == LATEST:
        # DSPy only lower-bounds LiteLLM, whose 1.92.0 release lacks Windows
        # and Python 3.14 wheels. Preinstall our portable matrix pin so DSPy's
        # dependency resolution does not select that incompatible release.
        _install_matrix_dep(session, "litellm", LATEST)
    _install_matrix_dep(session, "dspy", version, "test-sqlalchemy-2-0")
    _run_tests(session, f"{INTEGRATION_DIR}/dspy/test_dspy.py", version=version, env=_LITELLM_LOCAL_COST_MAP)


CREWAI_VERSIONS = _get_matrix_versions("crewai")


@nox.session()
@nox.parametrize("version", CREWAI_VERSIONS, ids=CREWAI_VERSIONS)
def test_crewai(session, version):
    _skip_if_incompatible(session)
    _install_test_deps(session)
    _install_group_locked(session, "test-crewai")
    _install_matrix_dep(session, "crewai", version)
    _run_tests(session, f"{INTEGRATION_DIR}/crewai/test_crewai.py", version=version)


GOOGLE_ADK_VERSIONS = _get_matrix_versions("google-adk")


@nox.session()
@nox.parametrize("version", GOOGLE_ADK_VERSIONS, ids=GOOGLE_ADK_VERSIONS)
def test_google_adk(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "google-adk", version, "test-sqlalchemy-2-0")
    _run_tests(session, f"{INTEGRATION_DIR}/adk/test_adk.py", version=version)


LANGCHAIN_VERSIONS = _get_matrix_versions("langchain-core")


@nox.session()
@nox.parametrize("version", LANGCHAIN_VERSIONS, ids=LANGCHAIN_VERSIONS)
def test_langchain(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "langchain-core", version)
    _install_group_locked(session, "test-langchain")
    _run_tests(
        session,
        [
            f"{INTEGRATION_DIR}/langchain/test_callbacks.py",
            f"{INTEGRATION_DIR}/langchain/test_context.py",
            f"{INTEGRATION_DIR}/langchain/test_anthropic.py",
        ],
        version=version,
    )


DEEPAGENTS_VERSIONS = _get_matrix_versions("deepagents")


@nox.session()
@nox.parametrize("version", DEEPAGENTS_VERSIONS, ids=DEEPAGENTS_VERSIONS)
def test_deepagents(session, version):
    _skip_if_incompatible(session)
    _install_test_deps(session)
    _install_group_locked(session, "test-deepagents")
    _install_matrix_dep(session, "deepagents", version)
    _run_tests(session, f"{INTEGRATION_DIR}/langchain/test_deepagents.py", version=version)


LLAMAINDEX_VERSIONS = _get_matrix_versions("llama-index-core")


@nox.session()
@nox.parametrize("version", LLAMAINDEX_VERSIONS, ids=LLAMAINDEX_VERSIONS)
def test_llamaindex(session, version):
    group = "test-llamaindex-0-13" if version == "0.13.0" else "test-llamaindex"
    _install_test_deps(session, group)
    _install_matrix_dep(session, "llama-index-core", version, "test-sqlalchemy-2-0")
    _run_tests(session, f"{INTEGRATION_DIR}/llamaindex/test_llamaindex.py", version=version)


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


TYPESAFE_VERSIONS = _get_matrix_versions("typesafe-sdk")


@nox.session()
@nox.parametrize("version", TYPESAFE_VERSIONS, ids=TYPESAFE_VERSIONS)
def test_typesafe(session, version):
    """Test the TypeSafe SDK integration."""
    _install_test_deps(session)
    _install_matrix_dep(session, "typesafe-sdk", version)
    _run_tests(session, f"{INTEGRATION_DIR}/typesafe/test_typesafe.py", version=version)


HUGGINGFACE_HUB_VERSIONS = _get_matrix_versions("huggingface-hub")


@nox.session()
@nox.parametrize("version", HUGGINGFACE_HUB_VERSIONS, ids=HUGGINGFACE_HUB_VERSIONS)
def test_huggingface_hub(session, version):
    """Test the native HuggingFace Hub SDK integration."""
    _install_test_deps(session)
    _install_matrix_dep(session, "huggingface-hub", version)
    # numpy is required by ``InferenceClient.feature_extraction`` but is not
    # an install_requires dep of ``huggingface_hub`` upstream.
    _install_group_locked(session, "test-huggingface-hub")
    _run_tests(session, f"{INTEGRATION_DIR}/huggingface_hub/test_huggingface_hub.py", version=version)


TRANSFORMERS_VERSIONS = _get_matrix_versions("transformers")


@nox.session()
@nox.parametrize("version", TRANSFORMERS_VERSIONS, ids=TRANSFORMERS_VERSIONS)
def test_transformers(session, version):
    """Test local Hugging Face Transformers pipeline instrumentation."""
    # The 4.42.0 floor pins tokenizers 0.19, whose wheels stop at Python 3.12.
    _skip_if_incompatible(session)
    _install_test_deps(session)
    _install_matrix_dep(session, "transformers", version)
    _install_group_locked(session, "test-transformers", indexes=("pytorch-cpu",))
    _run_tests(
        session,
        f"{INTEGRATION_DIR}/transformers/test_transformers.py",
        version=version,
        env={"HF_HUB_DISABLE_PROGRESS_BARS": "1"},
    )


TEMPORAL_VERSIONS = _get_matrix_versions("temporalio")


@nox.session()
@nox.parametrize("version", TEMPORAL_VERSIONS, ids=TEMPORAL_VERSIONS)
def test_temporal(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "temporalio", version)
    _run_tests(session, f"{INTEGRATION_DIR}/temporal", version=version)


HARBOR_VERSIONS = _get_matrix_versions("harbor")


@nox.session()
@nox.parametrize("version", HARBOR_VERSIONS, ids=HARBOR_VERSIONS)
def test_harbor(session, version):
    _skip_if_incompatible(session)
    _install_test_deps(session)
    _install_matrix_dep(session, "harbor", version)
    _run_tests(session, f"{INTEGRATION_DIR}/harbor", version=version)


PYTEST_VERSIONS = _get_matrix_versions("pytest-matrix")


@nox.session()
@nox.parametrize("version", PYTEST_VERSIONS, ids=PYTEST_VERSIONS)
def test_pytest_plugin(session, version):
    _install_test_deps(session)
    _install_matrix_dep(session, "pytest-matrix", version)
    _run_tests(session, f"{WRAPPER_DIR}/pytest_plugin/test_plugin.py", version=version)


@nox.session()
def test_core(session):
    _install_test_deps(session)
    # verify we haven't installed our 3p deps.
    script = f"""
import importlib.util
import sys

def is_importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False

sys.exit(1 if any(is_importable(name) for name in {_VENDOR_IMPORT_NAMES!r}) else 0)
"""
    session.run("python", "-c", script, silent=True)
    _run_core_tests(session)


@nox.session()
def test_api_codegen(session):
    """Test the pinned OpenAPI validator and deterministic model generator."""
    _install_test_deps(session)
    _install_group_locked(session, "api-codegen")
    _check_installed_deps(session)
    session.run("pytest", "-p", "no:braintrust", "tests/api_codegen", *session.posargs)


@nox.session()
def test_braintrust_core(session):
    # This is the core test whose scorer behavior changes with braintrust_core.
    # The rest of the core suite already runs in test_core.
    _install_test_deps(session)
    _install_matrix_dep(session, "braintrust-core", LATEST)
    _run_tests(session, f"{SRC_DIR}/test_framework.py::test_run_evaluator_with_many_scorers")


@nox.session()
def test_cli(session):
    """Test CLI/devserver with starlette installed."""
    _install_test_deps(session, "test-cli")
    _run_tests(session, DEVSERVER_DIR)


OTEL_VERSIONS = _get_matrix_versions("opentelemetry-sdk")


@nox.session()
@nox.parametrize("version", OTEL_VERSIONS, ids=OTEL_VERSIONS)
def test_otel(session, version):
    """Test OtelExporter with OpenTelemetry installed."""
    _skip_if_incompatible(session)
    _install_test_deps(session)
    _install_matrix_dep(session, "opentelemetry-api", version)
    _install_matrix_dep(session, "opentelemetry-sdk", version)
    _install_matrix_dep(session, "opentelemetry-exporter-otlp-proto-http", version)
    _run_tests(session, "braintrust/test_otel.py", version=version)


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
    _check_installed_deps(session)

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
    # Install the project and locked base/lint dependencies, then vendor SDKs
    # at their matrix latest versions for import coverage.
    _session_install(session, ".")
    _install_group_locked(session, "test", "lint")
    # Provider latest pins intentionally coexist in the lint environment even
    # when their dependency metadata conflicts. Validate the lock-resolved base
    # environment before adding those matrix versions for import coverage.
    _check_installed_deps(session)
    minimum_python = {
        "ai-sdk": (3, 12),
        "agentscope": (3, 11),
        "harbor": (3, 12),
        "pipecat-ai": (3, 11),
    }
    for package in _VENDOR_TABLE:
        if sys.version_info[:2] < minimum_python.get(package, (3, 10)):
            continue
        if package == "pipecat-ai" and sys.version_info >= (3, 14):
            continue
        if package == "crewai":
            continue
        _install_matrix_dep(session, package, LATEST)

    result = session.run("git", "ls-files", "**/*.py", silent=True, log=False)
    files = [path for path in result.strip().splitlines() if not path.startswith(GENERATED_LINT_EXCLUDES)]
    # Also lint repo-root examples/ — they live outside py/ but rely on the
    # same `lint` dependency-group, so we cover them in the same invocation.
    examples_result = session.run("git", "-C", "../examples", "ls-files", "**/*.py", silent=True, log=False)
    files += [f"../examples/{path}" for path in examples_result.strip().splitlines() if path]
    if not files:
        return
    # scripts/ may use APIs only available in the latest pinned Python version
    # (e.g. datetime.UTC requires 3.11+); skip them on older versions. tests/api_codegen/ imports
    # from scripts/, so it has to go with them -- pylint reports an unresolvable import once
    # scripts/ is out of the analyzed set.
    if _PINNED_PYTHON and sys.version_info[:2] < _PINNED_PYTHON:
        files = [f for f in files if not f.startswith(("scripts/", "tests/api_codegen/"))]
    # The lint group skips crewai to avoid vulnerable transitive chromadb
    # versions, so skip the matching example too.
    files = [f for f in files if not f.startswith("../examples/crewai/")]
    session.run("pylint", "--errors-only", *files)


def _install_test_deps(session, *groups):
    # Choose the way we'll install braintrust ... wheel or source.
    install_wheel = "--wheel" in session.posargs

    # Install braintrust itself. Source installs are editable so that
    # site-packages resolves to src/ instead of holding a copy. A copy goes
    # stale as soon as the session venv is reused (``nox -R``), and anything
    # that imports braintrust outside of pytest -- notably the subprocesses
    # spawned by ``verify_autoinstrument_script`` -- picks up site-packages
    # rather than the source tree, so it would silently exercise the code from
    # whenever the venv was last built.
    _session_install(session, *([_get_braintrust_wheel()] if install_wheel else ["-e", "."]))

    # Install base test deps (pytest, pytest-asyncio, pytest-vcr) from the
    # lockfile so transitive deps are pinned and reproducible.
    _install_group_locked(session, "test", *groups)

    # Sanity check braintrust imports from where this mode expects it:
    # site-packages for a wheel, the source tree for an editable install.
    lines = [
        "import sys, braintrust as b",
        "print(f'Using braintrust from: {b.__file__}')",
        f"sys.exit(0 if {install_wheel} == ('site-packages' in b.__file__) else 1)",
    ]
    session.run("python", "-c", ";".join(lines))


def _check_installed_deps(session):
    """Fail setup if installed packages have incompatible requirements."""
    session.run("uv", "pip", "check")


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
            BTX_DIR,
        ],
    )


def _run_tests(
    session,
    test_path,
    ignore_paths=None,
    env=None,
    version=None,
    run_from_temp_dir=False,
):
    """Run tests against a wheel or the source code. Paths should be relative and start with braintrust."""
    _check_installed_deps(session)
    env = env.copy() if env else {}
    if version:
        env["BRAINTRUST_TEST_PACKAGE_VERSION"] = version
    wheel_flag = "--wheel" in session.posargs
    common_args = ["--disable-vcr"] if "--disable-vcr" in session.posargs else []
    pytest_posargs = [arg for arg in session.posargs if arg not in INTERNAL_TEST_FLAGS]

    test_paths = [test_path] if isinstance(test_path, str) else list(test_path)
    paths_to_ignore = ignore_paths or []

    if not wheel_flag:
        # Run the tests in the src directory.
        source_test_paths = [f"src/{path}" for path in test_paths]
        source_ignore_paths = [f"src/{path}" for path in paths_to_ignore]
        if run_from_temp_dir:
            source_test_paths = [os.path.abspath(path) for path in source_test_paths]
            source_ignore_paths = [os.path.abspath(path) for path in source_ignore_paths]
        test_args = [
            "pytest",
            # Disable the braintrust pytest plugin (registered via pytest11 entry
            # point) to avoid ImportPathMismatchError when the installed package
            # and the source tree both contain braintrust/conftest.py.
            "-p",
            "no:braintrust",
            *source_test_paths,
        ]
        test_args.extend(f"--ignore={path}" for path in source_ignore_paths)
        if run_from_temp_dir:
            with tempfile.TemporaryDirectory() as tmp, session.chdir(tmp):
                session.run(*test_args, *common_args, *pytest_posargs, env=env)
        else:
            session.run(*test_args, *common_args, *pytest_posargs, env=env)
        return

    # Running the tests from the wheel involves a bit of gymnastics to ensure we don't import
    # local modules from the source directory.
    # First, we need to absolute paths to all the binaries and libs in our venv that we'll see.
    py = os.path.join(session.bin, "python")
    site_packages = session.run(py, "-c", "import site; print(site.getsitepackages()[0])", silent=True).strip()
    abs_test_paths = [os.path.abspath(os.path.join(site_packages, path)) for path in test_paths]
    pytest_path = os.path.join(session.bin, "pytest")

    ignore_args = []
    for path in paths_to_ignore:
        abs_ignore_path = os.path.abspath(os.path.join(site_packages, path))
        ignore_args.append(f"--ignore={abs_ignore_path}")

    # Lastly, change to a different directory to ensure we don't install local stuff.
    with tempfile.TemporaryDirectory() as tmp:
        with session.chdir(tmp):
            # This env var is used to detect if we're running from the wheel.
            # It proved very helpful because it's very easy
            # to accidentally import local modules from the source directory.
            env["BRAINTRUST_TESTING_WHEEL"] = "1"
            session.run(pytest_path, *abs_test_paths, *ignore_args, *common_args, *pytest_posargs, env=env)

    # And a final note ... if it's not clear from above, we include test files in our wheel, which
    # is perhaps not ideal?
