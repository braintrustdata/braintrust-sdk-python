#!/usr/bin/env python3
"""Sync [dependency-groups].test pytest pin from the matrix table.

[tool.braintrust.matrix.pytest-matrix].latest is the canonical pytest pin.
The base [dependency-groups].test entry has to match it (so uv.lock anchors
the same version that test_pytest_plugin(latest) exercises), but TOML has
no variable substitution, so we rewrite it mechanically.

Run without arguments to rewrite the dep-group pin in place. Run with
``--check`` (used by pre-commit) to fail without modifying anything.
"""

import argparse
import pathlib
import re
import sys

import tomllib


PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def _compute_new_text(text: str) -> tuple[str, str]:
    data = tomllib.loads(text)
    canonical = data["tool"]["braintrust"]["matrix"]["pytest-matrix"]["latest"]
    if not isinstance(canonical, str) or not canonical.startswith("pytest=="):
        raise SystemExit(
            f"sync-pytest-pin: [tool.braintrust.matrix.pytest-matrix].latest "
            f"must be a 'pytest==X.Y.Z' string, got: {canonical!r}"
        )

    # Match a pytest pin used as a list element (leading indent + quote). This
    # uniquely identifies the [dependency-groups].test entry and does not
    # collide with `latest = "pytest==..."` or `"8.4.2" = "pytest==..."` in
    # the matrix table (those start with a key, not whitespace+quote).
    pattern = re.compile(r'(?m)^(?P<indent>[ \t]+)"pytest==[^"]+"(?P<tail>,?)\s*$')
    matches = list(pattern.finditer(text))
    if not matches:
        raise SystemExit("sync-pytest-pin: no pytest list-element pin found in pyproject.toml")
    if len(matches) > 1:
        raise SystemExit(
            "sync-pytest-pin: multiple pytest list-element pins found; refusing "
            "to guess. Update py/scripts/sync-pytest-pin.py."
        )

    m = matches[0]
    new_line = f'{m.group("indent")}"{canonical}"{m.group("tail")}'
    new_text = text[: m.start()] + new_line + text[m.end() :]
    return new_text, canonical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the dep-group pin is out of sync; do not modify",
    )
    args = parser.parse_args()

    text = PYPROJECT.read_text()
    new_text, canonical = _compute_new_text(text)

    if new_text == text:
        return 0

    if args.check:
        print(
            f"[dependency-groups].test pytest pin is out of sync with "
            f"[tool.braintrust.matrix.pytest-matrix].latest ({canonical}).\n"
            f"Run: python py/scripts/sync-pytest-pin.py && (cd py && uv lock)",
            file=sys.stderr,
        )
        return 1

    PYPROJECT.write_text(new_text)
    print(f"sync-pytest-pin: updated [dependency-groups].test -> {canonical}")
    print("note: run `cd py && uv lock` to refresh uv.lock", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
