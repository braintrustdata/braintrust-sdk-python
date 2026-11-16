#!/usr/bin/env python3
"""Reproduce the weekly dependency workflow's label decision for a PR branch.

``py/scripts/determine-dependency-update-labels.py`` reads the uncommitted
working-tree diff, because the workflow runs it before committing. On a
checked-out PR branch the changes are already committed, so it prints
``changed=false``. This script applies the same rule to the diff between the
merge base and the PR head.

Usage (from anywhere in the repo):

    python3 .agents/skills/sdk-dependency-updates/scripts/check_labels.py
    python3 .agents/skills/sdk-dependency-updates/scripts/check_labels.py --base origin/main --head HEAD
"""

import argparse
import subprocess
import sys
from pathlib import Path

import tomllib


# Keep in sync with py/scripts/determine-dependency-update-labels.py.
INFRA_KEYS = {"pytest-matrix", "braintrust-core"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="base ref (default: origin/main)")
    parser.add_argument("--head", default="HEAD", help="PR head ref (default: HEAD)")
    args = parser.parse_args()

    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    diff = subprocess.check_output(
        ["git", "diff", f"{args.base}...{args.head}", "--", "py/pyproject.toml", "py/uv.lock"],
        cwd=root,
        text=True,
    )
    if not diff:
        print("changed=false")
        return 0

    pyproject = subprocess.check_output(["git", "show", f"{args.head}:py/pyproject.toml"], cwd=root, text=True)
    matrix = tomllib.loads(pyproject).get("tool", {}).get("braintrust", {}).get("matrix", {})
    provider_pkgs = {
        req.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
        for key, versions in matrix.items()
        if key not in INFRA_KEYS
        for req in versions.values()
    }
    needs_rerecord = any(pkg in diff for pkg in provider_pkgs)

    print("changed=true")
    print(f"needs_rerecord={str(needs_rerecord).lower()}")
    print(f"label={'needs-cassette-rerecord' if needs_rerecord else 'auto-merge-candidate'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
