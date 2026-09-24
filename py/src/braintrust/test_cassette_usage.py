"""Tests for cassette usage recording and scripts/check-unused-cassettes.py."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from braintrust import _test_cassette_usage


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
    env.pop(_test_cassette_usage.USAGE_DIR_ENV, None)
    if enabled:
        env[_test_cassette_usage.USAGE_DIR_ENV] = str(usage_dir)
    # Load the recorder by path: importing the braintrust package would add
    # ~0.7s per subprocess and the recorder needs nothing from it.
    prelude = textwrap.dedent(
        f"""
        import importlib.util
        import os
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("recorder", {_test_cassette_usage.__file__!r})
        recorder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recorder)
        print(recorder.install())
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


def test_checker_prepares_an_absolute_empty_usage_dir(checker, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "usage"
    stale.mkdir()
    (stale / "usage-1-a.txt").write_text("integrations/alpha/cassettes/latest/unused.yaml\n")
    (stale / "skips-1.jsonl").write_text("{}\n")
    (stale / "session-000").mkdir()
    (stale / "session-000" / "usage-2-b.txt").write_text("integrations/alpha/cassettes/latest/used.yaml\n")
    (stale / "keep.txt").write_text("unrelated")

    usage_dir = checker.prepare_usage_dir(Path("usage"))

    # Test processes may run from another cwd (e.g. run_from_temp_dir), so the
    # env var must be absolute; a fresh run must not inherit old reads.
    assert usage_dir == stale.resolve()
    assert checker.load_usage(usage_dir) == set()
    assert sorted(p.name for p in usage_dir.iterdir()) == ["keep.txt"]
    assert checker.prepare_usage_dir(None).is_absolute()


def test_record_skip_logs_package_relative_path_and_version(tmp_path, monkeypatch):
    pkg = _make_tree(tmp_path)
    usage_dir = tmp_path / "usage"
    monkeypatch.setattr(_test_cassette_usage, "_PACKAGE_DIR", str(pkg))
    monkeypatch.setenv("BRAINTRUST_TEST_PACKAGE_VERSION", "1.0.0")

    monkeypatch.delenv(_test_cassette_usage.USAGE_DIR_ENV, raising=False)
    _test_cassette_usage.record_skip(pkg / "integrations/alpha/test_alpha.py", "test_off", "not recorded")
    assert not usage_dir.exists()

    monkeypatch.setenv(_test_cassette_usage.USAGE_DIR_ENV, str(usage_dir))
    _test_cassette_usage.record_skip(pkg / "integrations/alpha/test_alpha.py", "test_sync", "no sync bridge")
    _test_cassette_usage.record_skip(tmp_path / "elsewhere/test_x.py", "test_x", "other")

    skips = [json.loads(line) for log in usage_dir.glob("skips-*.jsonl") for line in log.read_text().splitlines()]
    assert skips == [
        {
            "path": "integrations/alpha/test_alpha.py",
            "test": "test_sync",
            "version": "1.0.0",
            "reason": "no sync bridge",
        },
        {"path": str(tmp_path / "elsewhere/test_x.py"), "test": "test_x", "version": "1.0.0", "reason": "other"},
    ]


def test_checker_keeps_unread_files_next_to_skipped_tests(checker):
    unused = [
        "integrations/alpha/cassettes/1.0.0/btx/spec.yaml",
        "integrations/alpha/cassettes/latest/unused.yaml",
        "integrations/beta/cassettes/latest/transport.json",
    ]
    skips = [
        # A platform skip in alpha's latest session: its unread files may still be needed elsewhere.
        {"path": "integrations/alpha/test_alpha.py", "test": "t", "version": "latest", "reason": "no sync bridge"},
        # Unattributable (e.g. btx specs): marks that version in every integration.
        {"path": "btx/test_btx.py", "test": "t", "version": "1.0.0", "reason": "spec not supported"},
    ]

    skipped = checker.skipped_prefixes(skips, ["alpha", "beta"])
    deletable, kept = checker.split_skipped(unused, skipped)

    assert deletable == ["integrations/beta/cassettes/latest/transport.json"]
    assert kept == {
        "integrations/alpha/cassettes/1.0.0/": ["integrations/alpha/cassettes/1.0.0/btx/spec.yaml"],
        "integrations/alpha/cassettes/latest/": ["integrations/alpha/cassettes/latest/unused.yaml"],
    }
    assert skipped["integrations/alpha/cassettes/latest/"] == {"no sync bridge"}
