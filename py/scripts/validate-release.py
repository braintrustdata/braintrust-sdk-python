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


def run(*args: str, check: bool = True) -> str:
    result = subprocess.run(args, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def get_repo_root() -> pathlib.Path:
    return pathlib.Path(run("git", "rev-parse", "--show-toplevel"))


def get_version(repo_root: pathlib.Path) -> str:
    version_file = repo_root / "py" / "src" / "braintrust" / "version.py"
    version_line = next(
        (line for line in version_file.read_text().splitlines() if line.startswith("VERSION = ")),
        None,
    )
    if version_line is None:
        raise ValueError(f"Could not find VERSION in {version_file}")
    return version_line.split('"')[1]


def validate_release_type(release_type: str, version: str) -> None:
    if release_type == "stable" and not STABLE_VERSION_RE.fullmatch(version):
        raise ValueError(f"Stable releases require a version like X.Y.Z; found '{version}'")
    if release_type == "prerelease" and not PRERELEASE_VERSION_RE.fullmatch(version):
        raise ValueError(
            f"Prereleases require a version like X.Y.Zrc1, X.Y.Za1, or X.Y.Zb1; found '{version}'"
        )


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
    parser.add_argument("release_type", choices=("stable", "prerelease"))
    parser.add_argument("--github-output", type=pathlib.Path)
    return parser.parse_args()


def write_github_output(output_path: pathlib.Path, values: dict[str, str]) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def main() -> int:
    args = parse_args()
    repo_root = get_repo_root()
    version = get_version(repo_root)
    tag = f"py-sdk-v{version}"

    validate_release_type(args.release_type, version)
    check_commit_on_main()
    check_tag_does_not_exist(tag)
    check_version_not_on_pypi(version)

    outputs = {
        "commit_sha": run("git", "rev-parse", "HEAD"),
        "release_tag": tag,
        "release_type": args.release_type,
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
