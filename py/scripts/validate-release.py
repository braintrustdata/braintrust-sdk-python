#!/usr/bin/env python3
"""Validate whether the checked-out commit can be published as a Python SDK release."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request


STABLE_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
PRERELEASE_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(a|b|rc)[0-9]+$")
RELEASE_TYPES = ("stable", "prerelease")
VERSION_FILE = pathlib.Path("py/src/braintrust/version.py")


def run(*args: str, check: bool = True) -> str:
    result = subprocess.run(args, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def get_repo_root() -> pathlib.Path:
    return pathlib.Path(run("git", "rev-parse", "--show-toplevel"))


def version_file(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / VERSION_FILE


def get_version(repo_root: pathlib.Path) -> str:
    path = version_file(repo_root)
    match = re.search(r'^VERSION = "([^"]+)"$', path.read_text(), re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find VERSION in {path}")
    return match.group(1)


def set_version(repo_root: pathlib.Path, version: str) -> None:
    path = version_file(repo_root)
    text = path.read_text()
    updated = re.sub(r'^VERSION = ".*"$', f'VERSION = "{version}"', text, count=1, flags=re.MULTILINE)
    if updated == text:
        raise ValueError(f"Could not update VERSION in {path}")
    path.write_text(updated)


def infer_release_type(version: str) -> str:
    if STABLE_VERSION_RE.fullmatch(version):
        return "stable"
    if PRERELEASE_VERSION_RE.fullmatch(version):
        return "prerelease"
    raise ValueError(f"Version must be like X.Y.Z, X.Y.Zrc1, X.Y.Za1, or X.Y.Zb1; found '{version}'")


def validate_release_type(release_type: str, version: str) -> None:
    inferred_release_type = infer_release_type(version)
    if release_type != inferred_release_type:
        if release_type == "stable":
            raise ValueError(f"Stable releases require a version like X.Y.Z; found '{version}'")
        if release_type == "prerelease":
            raise ValueError(f"Prereleases require a version like X.Y.Zrc1, X.Y.Za1, or X.Y.Zb1; found '{version}'")


def check_tag_does_not_exist(tag: str) -> None:
    result = subprocess.run(("git", "rev-parse", tag), check=False, capture_output=True, text=True)
    if result.returncode == 0:
        raise ValueError(f"Tag '{tag}' already exists")


def check_commit_on_main() -> None:
    run("git", "fetch", "--tags", "--prune", "--force")
    run("git", "fetch", "origin", "main")
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", "HEAD", "origin/main"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("Releases must be cut from a commit on origin/main")


def check_version_not_on_pypi(version: str) -> None:
    url = f"https://pypi.org/pypi/braintrust/{version}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "braintrust-release-validator"})
    try:
        with urllib.request.urlopen(request) as response:
            if response.status == 200:
                raise ValueError(f"Version '{version}' already exists on PyPI")
            raise ValueError(
                f"Unexpected response from PyPI while checking version '{version}' (HTTP {response.status})"
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return
        raise ValueError(
            f"Unexpected response from PyPI while checking version '{version}' (HTTP {exc.code})"
        ) from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Failed to reach PyPI while checking version '{version}': {exc.reason}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_type", choices=(*RELEASE_TYPES, "auto"))
    parser.add_argument("--github-output", type=pathlib.Path)
    parser.add_argument("--allow-existing-tag", action="store_true")
    parser.add_argument("--version", help="Validate this version instead of reading py/src/braintrust/version.py")
    parser.add_argument("--set-version", action="store_true", help="Update py/src/braintrust/version.py to --version")
    parser.add_argument(
        "--validate-version-only", action="store_true", help="Only validate the version and release type"
    )
    parser.add_argument(
        "--print-version", action="store_true", help="Print py/src/braintrust/version.py VERSION and exit"
    )
    return parser.parse_args()


def write_github_output(output_path: pathlib.Path, values: dict[str, str]) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def main() -> int:
    args = parse_args()
    repo_root = get_repo_root()
    if args.print_version:
        print(get_version(repo_root))
        return 0

    if args.set_version and not args.version:
        raise ValueError("--set-version requires --version")

    version = args.version or get_version(repo_root)
    release_type = infer_release_type(version) if args.release_type == "auto" else args.release_type
    validate_release_type(release_type, version)

    if args.set_version:
        set_version(repo_root, version)
    if args.validate_version_only or args.set_version:
        return 0

    tag = f"py-sdk-v{version}"

    check_commit_on_main()
    if not args.allow_existing_tag:
        check_tag_does_not_exist(tag)
    check_version_not_on_pypi(version)

    outputs = {
        "commit_sha": run("git", "rev-parse", "HEAD"),
        "release_tag": tag,
        "release_type": release_type,
        "version": version,
    }
    if args.github_output is not None:
        write_github_output(args.github_output, outputs)
    else:
        for key, value in outputs.items():
            print(f"{key.upper()}={value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
