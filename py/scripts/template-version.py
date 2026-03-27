#!/usr/bin/env python3
"""Template the package version module during build."""

import os
import pathlib
import re
import subprocess


VERSION_FILE = pathlib.Path("src/braintrust/version.py")
GIT_COMMIT_RE = re.compile(r"__GIT_COMMIT__")
RELEASE_CHANNEL_RE = re.compile(r'^RELEASE_CHANNEL = ".*"$', re.MULTILINE)
VERSION_RE = re.compile(r'^VERSION = ".*"$', re.MULTILINE)


def run(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> int:
    contents = VERSION_FILE.read_text(encoding="utf-8")
    git_commit = run("git", "rev-parse", "HEAD")
    contents = GIT_COMMIT_RE.sub(git_commit, contents)
    release_channel = os.environ.get("BRAINTRUST_RELEASE_CHANNEL")
    if release_channel:
        contents = RELEASE_CHANNEL_RE.sub(f'RELEASE_CHANNEL = "{release_channel}"', contents, count=1)

    version_override = os.environ.get("BRAINTRUST_VERSION_OVERRIDE")
    if version_override:
        contents = VERSION_RE.sub(f'VERSION = "{version_override}"', contents, count=1)
    elif os.environ.get("PYPI_REPO") == "testpypi" and os.environ.get("GITHUB_RUN_NUMBER"):
        current_version_match = re.search(r'^VERSION = "([^"]+)"$', contents, re.MULTILINE)
        if current_version_match is None:
            raise ValueError(f"Could not find VERSION in {VERSION_FILE}")
        new_version = f"{current_version_match.group(1)}rc{os.environ['GITHUB_RUN_NUMBER']}"
        contents = VERSION_RE.sub(f'VERSION = "{new_version}"', contents, count=1)

    VERSION_FILE.write_text(contents, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
