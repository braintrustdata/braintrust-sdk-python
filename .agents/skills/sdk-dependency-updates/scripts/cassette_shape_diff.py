#!/usr/bin/env python3
"""Compare re-recorded ``latest`` cassettes with their committed copies by structure.

For each modified cassette under ``py/src/braintrust/integrations/*/cassettes/latest/``
in the working tree, compares the multiset of ``(method, URL without query,
status)`` interactions against the committed version. Text diffs on fresh
recordings are noisy; this shows only endpoint, status and count changes.
Freshly generated resource IDs in URLs show up as one added and one removed
endpoint, which is expected.

Also reports added and deleted cassette files.

Usage (needs PyYAML, which ``py/.venv`` has):

    py/.venv/bin/python .agents/skills/sdk-dependency-updates/scripts/cassette_shape_diff.py
    py/.venv/bin/python .agents/skills/sdk-dependency-updates/scripts/cassette_shape_diff.py openai anthropic
    py/.venv/bin/python .agents/skills/sdk-dependency-updates/scripts/cassette_shape_diff.py --base origin/main
"""

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

import yaml


def _shape(text: str) -> collections.Counter:
    data = yaml.safe_load(text) or {}
    return collections.Counter(
        (
            interaction.get("request", {}).get("method"),
            re.sub(r"\?.*", "", str(interaction.get("request", {}).get("uri", ""))),
            (interaction.get("response", {}).get("status") or {}).get("code"),
        )
        for interaction in data.get("interactions", []) or []
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("integrations", nargs="*", help="integration directory names (default: all)")
    parser.add_argument("--base", default="HEAD", help="ref holding the committed copies (default: HEAD)")
    args = parser.parse_args()

    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    integrations_dir = "py/src/braintrust/integrations"
    targets = [f"{integrations_dir}/{name}" for name in args.integrations] or [integrations_dir]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *targets], cwd=root, text=True
    )

    changed = 0
    for line in status.splitlines():
        code, path = line[:2], line[3:].strip('"')
        if "/cassettes/latest/" not in path or not path.endswith((".yaml", ".yml")):
            continue
        if code == "??":
            print(f"ADDED    {path}")
            changed += 1
            continue
        if "D" in code:
            print(f"DELETED  {path}")
            changed += 1
            continue
        try:
            old = _shape(subprocess.check_output(["git", "show", f"{args.base}:{path}"], cwd=root, text=True))
            new = _shape((root / path).read_text())
        except yaml.YAMLError as error:
            print(f"UNPARSED {path}: {error}")
            continue
        if old != new:
            changed += 1
            added, removed = sorted(new - old), sorted(old - new)
            print(f"CHANGED  {path}: {sum(old.values())} -> {sum(new.values())} interactions")
            for interaction in added:
                print(f"           + {interaction}")
            for interaction in removed:
                print(f"           - {interaction}")
    print(f"{changed} cassette(s) with structural changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
