#!/usr/bin/env python3
"""Update ``[tool.braintrust.matrix].*.latest`` pins in ``pyproject.toml``.

This script reads the current matrix table, fetches the newest published version
for each package from PyPI, and rewrites only the ``latest = ...`` lines in the
matching matrix sections.

Usage:
    python scripts/update-matrix-latest.py          # apply updates in place
    python scripts/update-matrix-latest.py --dry-run
"""

import argparse
import json
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request

import tomllib


_PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
_PYPROJECT = _PROJECT_DIR / "pyproject.toml"
_MATRIX_SECTION_PREFIX = "[tool.braintrust.matrix."
_LATEST_LINE_RE = re.compile(r'^(?P<indent>\s*)latest\s*=\s*"(?P<req>[^"]+)"(?P<suffix>\s*(?:#.*)?)$')
_EXACT_REQ_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?P<extras>\[[A-Za-z0-9_,.-]+\])?==(?P<version>[^\s]+)$")
_USER_AGENT = "braintrust-sdk-python dependency updater"
_TIMEOUT_SECS = 30


class UpdateError(RuntimeError):
    """Raised when the matrix latest pins cannot be updated safely."""


def _load_matrix() -> dict[str, str]:
    pyproject = tomllib.loads(_PYPROJECT.read_text())
    matrix = pyproject.get("tool", {}).get("braintrust", {}).get("matrix", {})

    latest_reqs: dict[str, str] = {}
    for key, versions in matrix.items():
        latest_req = versions.get("latest")
        if latest_req:
            latest_reqs[key] = latest_req
    return latest_reqs


def _parse_exact_req(req: str) -> tuple[str, str, str]:
    match = _EXACT_REQ_RE.fullmatch(req.strip())
    if not match:
        raise UpdateError(f"Expected an exact 'package[extras]==version' requirement, got: {req}")
    return match.group("name"), match.group("extras") or "", match.group("version")


def _fetch_latest_version(package_name: str) -> str:
    url = f"https://pypi.org/pypi/{urllib.parse.quote(package_name)}/json"
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise UpdateError(f"Failed to fetch {package_name} metadata from PyPI: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise UpdateError(f"Failed to fetch {package_name} metadata from PyPI: {exc.reason}") from exc

    version = str(payload.get("info", {}).get("version", "")).strip()
    if not version:
        raise UpdateError(f"PyPI did not return a version for {package_name}")
    return version


def _compute_updates() -> dict[str, tuple[str, str]]:
    latest_reqs = _load_matrix()
    package_cache: dict[str, str] = {}
    updates: dict[str, tuple[str, str]] = {}

    for matrix_key, current_req in latest_reqs.items():
        package_name, extras, current_version = _parse_exact_req(current_req)
        latest_version = package_cache.get(package_name)
        if latest_version is None:
            latest_version = _fetch_latest_version(package_name)
            package_cache[package_name] = latest_version

        if latest_version != current_version:
            updates[matrix_key] = (current_req, f"{package_name}{extras}=={latest_version}")

    return updates


def _rewrite_pyproject(text: str, replacements: dict[str, tuple[str, str]]) -> tuple[str, list[str]]:
    if not replacements:
        return text, []

    lines = text.splitlines(keepends=True)
    touched_keys: list[str] = []
    current_matrix_key: str | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_matrix_key = None
            if stripped.startswith(_MATRIX_SECTION_PREFIX):
                current_matrix_key = stripped[len(_MATRIX_SECTION_PREFIX) : -1]
            continue

        if current_matrix_key is None or current_matrix_key not in replacements:
            continue

        line_body = line[:-1] if line.endswith("\n") else line
        newline = "\n" if line.endswith("\n") else ""
        match = _LATEST_LINE_RE.fullmatch(line_body)
        if not match:
            continue

        old_req, new_req = replacements[current_matrix_key]
        if match.group("req") != old_req:
            raise UpdateError(
                f"Expected latest pin for matrix key {current_matrix_key!r} to be {old_req!r}, "
                f"found {match.group('req')!r}"
            )

        lines[index] = f'{match.group("indent")}latest = "{new_req}"{match.group("suffix")}{newline}'
        touched_keys.append(current_matrix_key)
        current_matrix_key = None

    missing = sorted(set(replacements) - set(touched_keys))
    if missing:
        raise UpdateError(f"Did not find latest lines for matrix key(s): {', '.join(missing)}")

    return "".join(lines), touched_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show updates without writing pyproject.toml")
    args = parser.parse_args()

    replacements = _compute_updates()
    if not replacements:
        print("No matrix latest pins to update.")
        return

    original_text = _PYPROJECT.read_text()
    updated_text, touched_keys = _rewrite_pyproject(original_text, replacements)

    for matrix_key in touched_keys:
        old_req, new_req = replacements[matrix_key]
        print(f"{matrix_key}: {old_req} -> {new_req}")

    if args.dry_run:
        return

    _PYPROJECT.write_text(updated_text)
    print(f"Updated {_PYPROJECT.relative_to(_PROJECT_DIR)}")


if __name__ == "__main__":
    main()
