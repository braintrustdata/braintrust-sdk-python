import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from openapi_codegen import CodegenError, read_and_verify_spec


UPDATE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update-openapi-spec.py"


def _write_json(path, value):
    content = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _commit(repository, message):
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=OpenAPI Test",
            "-c",
            "user.email=openapi-test@example.com",
            "commit",
            "-m",
            message,
        ],
        check=True,
        capture_output=True,
    )
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


@pytest.mark.parametrize("cached_spec", [True, False])
def test_updater_pins_local_head_and_writes_review_summary(tmp_path, codegen_config, minimal_spec, cached_spec):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "-C", str(upstream), "init", "--quiet"], check=True)
    upstream_spec = upstream / "spec.json"
    old_hash = _write_json(upstream_spec, minimal_spec)
    old_commit = _commit(upstream, "old spec")

    config = codegen_config
    config["spec"].update(
        {
            "repository": "braintrustdata/braintrust-openapi",
            "path": "spec.json",
            "commit": old_commit,
            "sha256": old_hash,
        }
    )
    config_path = tmp_path / "config.json"
    spec_path = tmp_path / "pinned-spec.json"
    summary_path = tmp_path / "summary.md"
    _write_json(config_path, config)
    if cached_spec:
        spec_path.write_bytes(upstream_spec.read_bytes())

    minimal_spec["components"]["schemas"]["WidgetDetails"] = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
    }
    minimal_spec["components"]["schemas"]["Widget"]["properties"]["details"] = {
        "$ref": "#/components/schemas/WidgetDetails"
    }
    _write_json(upstream_spec, minimal_spec)
    new_commit = _commit(upstream, "new spec")
    committed_content = upstream_spec.read_bytes()
    upstream_spec.write_text("uncommitted content must not be pinned\n")

    environment = {**os.environ, "BRAINTRUST_OPENAPI_ROOT": str(upstream)}
    result = subprocess.run(
        [
            sys.executable,
            str(UPDATE_SCRIPT),
            "--config",
            str(config_path),
            "--spec",
            str(spec_path),
            "--summary-file",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    updated_config = json.loads(config_path.read_text())
    assert result.stdout == "changed=true\n"
    assert updated_config["spec"]["commit"] == new_commit
    assert updated_config["spec"]["sha256"] == hashlib.sha256(committed_content).hexdigest()
    assert spec_path.read_bytes() == committed_content
    summary = summary_path.read_text()
    assert f"compare/{old_commit}...{new_commit}" in summary
    assert "1 selected operations, 1 reachable schemas → 1 selected operations, 2 reachable schemas" in summary
    assert "All upstream component schemas: 1 → 2" in summary
    assert "Added: `WidgetDetails`" in summary
    assert "never auto-merged" in summary

    unchanged = subprocess.run(
        [
            sys.executable,
            str(UPDATE_SCRIPT),
            "--config",
            str(config_path),
            "--spec",
            str(spec_path),
            "--summary-file",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert unchanged.stdout == "changed=false\n"


def test_missing_spec_is_fetched_from_pinned_commit_and_cached(tmp_path, monkeypatch, codegen_config, minimal_spec):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "-C", str(upstream), "init", "--quiet"], check=True)
    spec_hash = _write_json(upstream / "spec.json", minimal_spec)
    commit = _commit(upstream, "pinned spec")
    codegen_config["spec"].update({"path": "spec.json", "commit": commit, "sha256": spec_hash})
    _write_json(upstream / "spec.json", {"newer": "spec"})
    _commit(upstream, "newer spec")
    monkeypatch.setenv("BRAINTRUST_OPENAPI_ROOT", str(upstream))
    spec_path = tmp_path / "cache" / "spec.json"

    assert read_and_verify_spec(codegen_config, spec_path) == minimal_spec
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == spec_hash

    monkeypatch.setenv("BRAINTRUST_OPENAPI_ROOT", str(tmp_path / "unavailable"))
    assert read_and_verify_spec(codegen_config, spec_path) == minimal_spec

    spec_path.write_text('{"tampered": true}')
    with pytest.raises(CodegenError, match="spec hash mismatch"):
        read_and_verify_spec(codegen_config, spec_path)


def test_fetched_spec_hash_is_verified_before_caching(tmp_path, monkeypatch, codegen_config, minimal_spec):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "-C", str(upstream), "init", "--quiet"], check=True)
    _write_json(upstream / "spec.json", minimal_spec)
    commit = _commit(upstream, "pinned spec")
    codegen_config["spec"].update({"path": "spec.json", "commit": commit, "sha256": "0" * 64})
    monkeypatch.setenv("BRAINTRUST_OPENAPI_ROOT", str(upstream))
    spec_path = tmp_path / "spec.json"

    with pytest.raises(CodegenError, match="spec hash mismatch"):
        read_and_verify_spec(codegen_config, spec_path)
    assert not spec_path.exists()
