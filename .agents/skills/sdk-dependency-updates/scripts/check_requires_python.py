#!/usr/bin/env python3
"""Flag bumped matrix ``latest`` pins that no longer install on CI's lowest Python.

Compares the ``Requires-Python`` metadata (from PyPI) of every ``latest`` pin
that changed between the merge base and the PR head. A pin is flagged when its
new metadata excludes the lowest ``python-version`` in
``.github/workflows/checks.yaml`` and the previous pin allowed it. Pins that
already excluded it are listed as such; they should already be gated. Patch
bumps are included: they can raise the floor too.

Usage (from anywhere in the repo; ``packaging`` is used when available):

    py/.venv/bin/python .agents/skills/sdk-dependency-updates/scripts/check_requires_python.py
    py/.venv/bin/python .agents/skills/sdk-dependency-updates/scripts/check_requires_python.py --min-python 3.10

Exits 1 if any pin newly excludes the lowest CI Python.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import tomllib


def _latest_pins(pyproject_text: str) -> dict[str, tuple[str, str]]:
    """Map matrix key -> (distribution, version) for each ``latest`` pin."""
    matrix = tomllib.loads(pyproject_text).get("tool", {}).get("braintrust", {}).get("matrix", {})
    pins = {}
    for key, versions in matrix.items():
        req = versions.get("latest")
        if not req or "==" not in req:
            continue
        name, version = req.split("==", 1)
        pins[key] = (re.sub(r"\[.*\]", "", name).strip(), version.strip())
    return pins


def _lowest_ci_python(root: Path) -> str:
    text = (root / ".github/workflows/checks.yaml").read_text()
    versions = re.findall(r'"(3\.\d+)"', " ".join(re.findall(r"python-version:\s*\[([^\]]*)\]", text)))
    if not versions:
        raise SystemExit("could not find python-version lists in .github/workflows/checks.yaml; pass --min-python")
    return min(versions, key=lambda v: tuple(int(p) for p in v.split(".")))


def _requires_python(name: str, version: str) -> str | None:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{name}/{version}/json", headers={"User-Agent": "dependency-review-script"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["info"].get("requires_python")


def _allows(spec: str | None, python: str) -> bool:
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        floor = re.search(r">=\s*(\d+)\.(\d+)", spec)
        return floor is None or (int(floor.group(1)), int(floor.group(2))) <= tuple(int(p) for p in python.split("."))
    return SpecifierSet(spec).contains(f"{python}.0", prereleases=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="base ref (default: origin/main)")
    parser.add_argument("--head", default="HEAD", help="PR head ref (default: HEAD)")
    parser.add_argument("--min-python", help="lowest Python to check (default: lowest in checks.yaml)")
    args = parser.parse_args()

    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    merge_base = subprocess.check_output(["git", "merge-base", args.base, args.head], cwd=root, text=True).strip()

    def show(ref: str) -> str:
        return subprocess.check_output(["git", "show", f"{ref}:py/pyproject.toml"], cwd=root, text=True)

    old, new = _latest_pins(show(merge_base)), _latest_pins(show(args.head))
    lowest = args.min_python or _lowest_ci_python(root)

    flagged = 0
    print(f"lowest CI Python: {lowest}")
    for key in sorted(new):
        if old.get(key) == new[key]:
            continue
        name, new_version = new[key]
        old_version = old[key][1] if key in old else "-"
        old_spec = _requires_python(name, old_version) if key in old else None
        new_spec = _requires_python(name, new_version)
        flag = ""
        if not _allows(new_spec, lowest):
            if key in old and not _allows(old_spec, lowest):
                flag = f"  (already excluded {lowest})"
            else:
                flag = f"  <-- NEW: excludes {lowest}; gate the {key!r} latest session(s)"
                flagged += 1
        print(f"{key:30} {old_version:>12} {str(old_spec):>18} -> {new_version:<12} {new_spec}{flag}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
