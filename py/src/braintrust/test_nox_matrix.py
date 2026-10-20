"""Regression tests for the CI shard timing collector."""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "nox-matrix.py"


@pytest.mark.parametrize("returncode", [0, 1])
def test_measured_durations_use_nox_report(tmp_path, monkeypatch, returncode):
    spec = importlib.util.spec_from_file_location("nox_matrix", _SCRIPT)
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)

    sessions = [
        {"name": "test_core", "result": "success", "duration": 71.9},
        {"name": "test_provider(latest)", "result": "success", "duration": 3601.2},
        {"name": "test_short", "result": "success", "duration": 0.3},
        {"name": "test_seconds", "result": "success", "duration": 12.4},
        {"name": "test_skipped", "result": "skipped", "duration": 0},
    ]
    if returncode:
        sessions.append({"name": "test_failed", "result": "failed", "duration": 3.5})
    output = tmp_path / "durations" / "measured.json"
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "0", "1", "--output-durations", str(output)])
    monkeypatch.setattr(matrix, "get_nox_sessions", lambda _: [s["name"] for s in sessions])

    def run(cmd, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout=str(_SCRIPT.parents[2]))
        report_path = Path(cmd[cmd.index("--report") + 1])
        report_path.write_text(json.dumps({"sessions": sessions}))
        return subprocess.CompletedProcess(cmd, returncode)

    monkeypatch.setattr(matrix.subprocess, "run", run)
    # The old collector only read this human-readable output and lost test_core.
    monkeypatch.setattr(
        matrix.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=io.StringIO("nox > Session test_core was successful in a minute.\n"),
            wait=lambda: returncode,
            returncode=returncode,
        ),
    )

    with pytest.raises(SystemExit) as exc:
        matrix.main()

    assert exc.value.code == returncode
    assert json.loads(output.read_text()) == {
        "test_core": 72,
        "test_provider(latest)": 3601,
        "test_short": 0,
        "test_seconds": 12,
    }
