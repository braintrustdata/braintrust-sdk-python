"""Tests for cassette usage recording and scripts/check-unused-cassettes.py."""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-unused-cassettes.py"


def _make_tree(root: Path) -> Path:
    """Build a fake package tree: <root>/pkg/integrations/{alpha,beta}/cassettes/..."""
    pkg = root / "pkg"
    for rel in (
        "integrations/alpha/cassettes/latest/used.yaml",
        "integrations/alpha/cassettes/latest/unused.yaml",
        "integrations/alpha/cassettes/1.0.0/btx/spec.yaml",
        "integrations/beta/cassettes/latest/transport.json",
        "integrations/alpha/test_alpha.py",
    ):
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    return pkg


def _run_recorder(tmp_path: Path, code: str, *, enabled: bool = True) -> set[str]:
    pkg = _make_tree(tmp_path)
    usage_dir = tmp_path / "usage"
    env = {**os.environ, "BRAINTRUST_INTEGRATIONS_DIR": str(pkg / "integrations")}
    env.pop("BRAINTRUST_CASSETTE_USAGE_DIR", None)
    if enabled:
        env["BRAINTRUST_CASSETTE_USAGE_DIR"] = str(usage_dir)
    prelude = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from braintrust import _test_cassette_usage
        print(_test_cassette_usage.install())
        pkg = Path({str(pkg)!r})
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == str(enabled)
    lines = []
    for log in usage_dir.glob("usage-*.txt"):
        lines.extend(log.read_text().splitlines())
    assert len(lines) == len(set(lines)), "each file should be recorded once per process"
    return set(lines)


def test_recorder_logs_cassette_reads_relative_to_package(tmp_path):
    used = _run_recorder(
        tmp_path,
        """
        (pkg / "integrations/alpha/cassettes/latest/used.yaml").read_text()
        (pkg / "integrations/alpha/cassettes/latest/used.yaml").read_text()
        with open(pkg / "integrations/alpha/cassettes/1.0.0/btx/spec.yaml", "rb") as f:
            f.read()
        os.close(os.open(pkg / "integrations/beta/cassettes/latest/transport.json", os.O_RDONLY))
        """,
    )
    assert used == {
        "integrations/alpha/cassettes/latest/used.yaml",
        "integrations/alpha/cassettes/1.0.0/btx/spec.yaml",
        "integrations/beta/cassettes/latest/transport.json",
    }


def test_recorder_ignores_writes_non_cassettes_and_outside_paths(tmp_path):
    used = _run_recorder(
        tmp_path,
        """
        (pkg / "integrations/alpha/cassettes/latest/new.yaml").write_text("recorded")
        with open(pkg / "integrations/alpha/cassettes/latest/unused.yaml", "a") as f:
            f.write("appended")
        os.close(os.open(pkg / "integrations/beta/cassettes/latest/transport.json", os.O_WRONLY))
        (pkg / "integrations/alpha/test_alpha.py").read_text()
        outside = pkg.parent / "elsewhere" / "cassettes" / "x.yaml"
        outside.parent.mkdir(parents=True)
        outside.write_text("x")
        outside.read_text()
        """,
    )
    assert used == set()


def test_recorder_is_off_without_env_var(tmp_path):
    used = _run_recorder(
        tmp_path,
        """
        (pkg / "integrations/alpha/cassettes/latest/used.yaml").read_text()
        """,
        enabled=False,
    )
    assert used == set()
    assert not (tmp_path / "usage").exists()


@pytest.fixture
def checker(tmp_path, monkeypatch):
    if not _SCRIPT.exists():
        pytest.skip("scripts/ is not available (wheel mode)")
    spec = importlib.util.spec_from_file_location("check_unused_cassettes", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pkg = _make_tree(tmp_path)
    monkeypatch.setattr(module, "_PACKAGE_DIR", pkg)
    monkeypatch.setattr(module, "_INTEGRATIONS_DIR", pkg / "integrations")
    return module


def test_checker_reports_files_missing_from_merged_usage_logs(checker, tmp_path):
    usage = tmp_path / "usage"
    usage.mkdir()
    (usage / "usage-1-a.txt").write_text("integrations/alpha/cassettes/latest/used.yaml\n")
    (usage / "usage-2-b.txt").write_text("integrations/beta/cassettes/latest/transport.json\n\n")
    (usage / "ignored.txt").write_text("integrations/alpha/cassettes/1.0.0/btx/spec.yaml\n")

    used = checker.load_usage(usage)

    assert checker.integrations_with_cassettes() == ["alpha", "beta"]
    assert checker.find_unused(["alpha", "beta"], used) == [
        "integrations/alpha/cassettes/1.0.0/btx/spec.yaml",
        "integrations/alpha/cassettes/latest/unused.yaml",
    ]
    assert checker.find_unused(["beta"], used) == []


def test_checker_maps_integrations_to_nox_sessions(checker, tmp_path):
    noxfile = tmp_path / "noxfile.py"
    noxfile.write_text(
        textwrap.dedent(
            """
            def test_alpha(session, version):
                _run_tests(session, f"{INTEGRATION_DIR}/alpha/test_alpha.py", version=version)

            def test_alpha_extra(session):
                _run_tests(session, f"{INTEGRATION_DIR}/alpha")

            def test_spec_beta(session, version):
                _run_tests(session, "braintrust/btx", version=version, env={"PROVIDER": "beta"})

            def test_core(session):
                _run_tests(session, "braintrust", ignore_path=f"{INTEGRATION_DIR}/alpha")

            def test_gamma(session):
                _run_tests(session, f"{INTEGRATION_DIR}/gamma/test_gamma.py")
            """
        )
    )

    assert checker.sessions_by_integration(noxfile) == {
        "alpha": {"test_alpha", "test_alpha_extra"},
        "beta": {"test_spec_beta"},
    }
