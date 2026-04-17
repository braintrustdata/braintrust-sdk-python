#!/usr/bin/env python3
"""Determine GitHub Actions outputs for the daily dependency update workflow.

Inspects the working-tree diff of ``py/pyproject.toml`` and ``py/uv.lock`` to
decide whether anything changed and, if so, whether any provider SDK pins
shifted (which requires cassette re-recording).

Writes ``key=value`` lines to stdout in the format expected by
``$GITHUB_OUTPUT``:

    changed=true|false
    needs_rerecord=true|false   # only when changed=true

Run from ``py/`` (same working directory as the workflow step):

    python scripts/determine-dependency-update-labels.py >> "$GITHUB_OUTPUT"
"""

import subprocess
import sys

import tomllib


def main() -> int:
    diff = subprocess.check_output(["git", "diff", "--", "pyproject.toml", "uv.lock"], text=True)
    if not diff:
        print("changed=false")
        return 0

    # Read pyproject.toml to find provider SDK packages from the matrix table.
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    matrix = pyproject.get("tool", {}).get("braintrust", {}).get("matrix", {})

    # Exclude pure test/infra pins that do not affect cassette coverage.
    provider_matrix = {
        key: versions for key, versions in matrix.items() if key not in {"pytest-matrix", "braintrust-core"}
    }

    provider_pkgs = set()
    for versions in provider_matrix.values():
        for req in versions.values():
            # req looks like "openai==1.92.0" or "pydantic-ai==1.82.0"
            pkg = req.split("==")[0].split(">=")[0].split("<=")[0].strip()
            provider_pkgs.add(pkg)

    # Check if any provider package changed in the lockfile diff.
    needs_rerecord = any(pkg in diff for pkg in provider_pkgs)

    print("changed=true")
    print(f"needs_rerecord={str(needs_rerecord).lower()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
