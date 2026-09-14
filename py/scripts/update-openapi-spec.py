#!/usr/bin/env python3
"""Update the pinned OpenAPI snapshot to the latest upstream spec commit."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from openapi_codegen import (
    CONFIG_PATH,
    HTTP_METHODS,
    SPEC_PATH,
    CodegenError,
    load_config,
    read_and_verify_spec,
    validate_config,
    validate_spec,
)


class Source(NamedTuple):
    commit: str
    content: bytes
    description: str


def _request(url: str, *, accept: str) -> bytes:
    headers = {"Accept": accept, "User-Agent": "braintrust-sdk-python-openapi-updater"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub endpoints.
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise CodegenError(f"Unable to fetch {url}: {exc}") from exc


def _latest_source(config: Mapping[str, Any]) -> Source:
    spec_config = config["spec"]
    local_root = os.environ.get("BRAINTRUST_OPENAPI_ROOT")
    if local_root:
        root = Path(local_root).expanduser().resolve()
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            content = subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{spec_config['path']}"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CodegenError(f"BRAINTRUST_OPENAPI_ROOT is not a readable git checkout: {root}") from exc
        return Source(commit, content, f"{root}@{commit}:{spec_config['path']}")

    repository = spec_config["repository"]
    query = urllib.parse.urlencode({"path": spec_config["path"], "per_page": 1})
    commits_url = f"https://api.github.com/repos/{repository}/commits?{query}"
    try:
        commits = json.loads(_request(commits_url, accept="application/vnd.github+json"))
    except json.JSONDecodeError as exc:
        raise CodegenError(f"GitHub returned invalid JSON for {commits_url}: {exc}") from exc
    if not isinstance(commits, list) or not commits or not isinstance(commits[0], dict):
        raise CodegenError(f"GitHub returned no commits for {repository}/{spec_config['path']}")
    commit = commits[0].get("sha")
    if not isinstance(commit, str) or len(commit) != 40:
        raise CodegenError(f"GitHub returned an invalid commit SHA for {repository}/{spec_config['path']}")

    quoted_path = urllib.parse.quote(spec_config["path"], safe="/")
    raw_url = f"https://raw.githubusercontent.com/{repository}/{commit}/{quoted_path}"
    content = _request(raw_url, accept="application/json")
    return Source(commit, content, raw_url)


def _parse_spec(content: bytes, source: str) -> dict[str, Any]:
    try:
        spec = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodegenError(f"OpenAPI spec from {source} is not valid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise CodegenError(f"OpenAPI spec from {source} must contain a JSON object")
    return spec


def _operation_ids(spec: Mapping[str, Any]) -> set[str]:
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return set()
    return {
        operation_id
        for path_item in paths.values()
        if isinstance(path_item, dict)
        for method, operation in path_item.items()
        if method in HTTP_METHODS
        and isinstance(operation, dict)
        and isinstance((operation_id := operation.get("operationId")), str)
    }


def _schema_names(spec: Mapping[str, Any]) -> set[str]:
    components = spec.get("components", {})
    schemas = components.get("schemas", {}) if isinstance(components, dict) else {}
    return set(schemas) if isinstance(schemas, dict) else set()


def _format_changes(added: set[str], removed: set[str]) -> str:
    lines = []
    if added:
        lines.append("- Added: " + ", ".join(f"`{name}`" for name in sorted(added)))
    if removed:
        lines.append("- Removed: " + ", ".join(f"`{name}`" for name in sorted(removed)))
    return "\n".join(lines) if lines else "- No names added or removed."


def _validation_summary(spec: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    try:
        return str(validate_spec(spec, config))
    except CodegenError as exc:
        return f"requires manual review (`{exc}`)"


def _build_summary(
    old_spec: Mapping[str, Any],
    new_spec: Mapping[str, Any],
    old_config: Mapping[str, Any],
    new_config: Mapping[str, Any],
) -> str:
    repository = new_config["spec"]["repository"]
    old_commit = old_config["spec"]["commit"]
    new_commit = new_config["spec"]["commit"]
    old_operations = _operation_ids(old_spec)
    new_operations = _operation_ids(new_spec)
    old_schemas = _schema_names(old_spec)
    new_schemas = _schema_names(new_spec)
    return f"""Automated update of the pinned Braintrust OpenAPI specification.

- Upstream commit: [`{old_commit[:12]}`](https://github.com/{repository}/commit/{old_commit}) → [`{new_commit[:12]}`](https://github.com/{repository}/commit/{new_commit})
- [Upstream spec diff](https://github.com/{repository}/compare/{old_commit}...{new_commit})
- Reviewed generated surface: {_validation_summary(old_spec, old_config)} → {_validation_summary(new_spec, new_config)}
- All upstream operations: {len(old_operations)} → {len(new_operations)}
- All upstream component schemas: {len(old_schemas)} → {len(new_schemas)}

### Operation changes

{_format_changes(new_operations - old_operations, old_operations - new_operations)}

### Component schema changes

{_format_changes(new_schemas - old_schemas, old_schemas - new_schemas)}

### Automated validation

The update workflow regenerates committed sources and public API reference sections, then runs:

- `make -C py test-api-codegen`
- `make -C py test-core`
- `cd py && nox -s test_types`

This pull request is never auto-merged. Review the upstream and generated diffs, retry-policy classifications, public type changes, and any intentionally unsupported tags before merging.
"""


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    if path.exists():
        os.chmod(temporary_path, path.stat().st_mode)
    os.replace(temporary_path, path)


def update(config_path: Path, spec_path: Path, summary_path: Path | None) -> bool:
    config = load_config(config_path)
    validate_config(config, check_installed_tools=False)
    old_spec = read_and_verify_spec(config, spec_path)
    source = _latest_source(config)
    new_spec = _parse_spec(source.content, source.description)
    new_hash = hashlib.sha256(source.content).hexdigest()

    old_commit = config["spec"]["commit"]
    old_hash = config["spec"]["sha256"]
    if source.commit == old_commit and new_hash == old_hash:
        return False

    new_config = json.loads(json.dumps(config))
    new_config["spec"]["commit"] = source.commit
    new_config["spec"]["sha256"] = new_hash
    validate_config(new_config, check_installed_tools=False)
    summary = _build_summary(old_spec, new_spec, config, new_config)

    _atomic_write(spec_path, source.content)
    _atomic_write(config_path, (json.dumps(new_config, indent=2) + "\n").encode())
    if summary_path:
        _atomic_write(summary_path, summary.encode())
    print(f"Updated OpenAPI pin {old_commit} -> {source.commit} from {source.description}", file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--summary-file", type=Path, help="Write the pull request body to this path.")
    args = parser.parse_args()

    changed = update(args.config, args.spec, args.summary_file)
    print(f"changed={str(changed).lower()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CodegenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
