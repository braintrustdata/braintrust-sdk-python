#!/usr/bin/env python3
"""Support scripts for TestPyPI release workflows."""

from __future__ import annotations

import argparse
import datetime
import io
import json
import pathlib
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from typing import Any


try:
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover
    from pkg_resources import parse_version

    class InvalidVersion(Exception):
        """Compatibility shim when packaging is unavailable."""

    def Version(version: str) -> Any:  # type: ignore[misc]
        return parse_version(version)


STABLE_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
PRERELEASE_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(a|b|rc)[0-9]+$")
GIT_COMMIT_RE = re.compile(r'^GIT_COMMIT = "([0-9a-f]{40})"$', re.MULTILINE)
RELEASE_CHANNEL_RE = re.compile(r'^RELEASE_CHANNEL = "([^"]+)"$', re.MULTILINE)
HTTP_TIMEOUT_SECONDS = 30
USER_AGENT = "braintrust-test-publish"


def run(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def get_repo_root() -> pathlib.Path:
    return pathlib.Path(run("git", "rev-parse", "--show-toplevel"))


def get_version(repo_root: pathlib.Path) -> str:
    version_file = repo_root / "py" / "src" / "braintrust" / "version.py"
    version_line = next(
        (line for line in version_file.read_text(encoding="utf-8").splitlines() if line.startswith("VERSION = ")),
        None,
    )
    if version_line is None:
        raise ValueError(f"Could not find VERSION in {version_file}")
    return version_line.split('"')[1]


def write_github_output(output_path: pathlib.Path, values: dict[str, str]) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def make_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def urlopen(url: str):
    return urllib.request.urlopen(make_request(url), timeout=HTTP_TIMEOUT_SECONDS)


def sanitize_run_number(run_number: int) -> int:
    if run_number <= 0:
        raise ValueError(f"run_number must be positive; found '{run_number}'")
    return run_number


def build_prerelease_version(base_version: str, run_number: int) -> str:
    if STABLE_VERSION_RE.fullmatch(base_version):
        return f"{base_version}rc{run_number}"
    if PRERELEASE_VERSION_RE.fullmatch(base_version):
        match = re.fullmatch(r"(?P<prefix>[0-9]+\.[0-9]+\.[0-9]+(?:a|b|rc))(?P<number>[0-9]+)", base_version)
        if match is None:
            raise ValueError(f"Could not parse prerelease version '{base_version}'")
        return f"{match.group('prefix')}{run_number}"
    raise ValueError(f"TestPyPI prereleases require a base version like X.Y.Z or X.Y.Zrc1; found '{base_version}'")


def check_testpypi_version_does_not_exist(version: str) -> None:
    url = f"https://test.pypi.org/pypi/braintrust/{version}/json"
    try:
        with urlopen(url) as response:
            if response.status == 200:
                raise ValueError(f"Version '{version}' already exists on TestPyPI")
            raise ValueError(
                f"Unexpected response from TestPyPI while checking version '{version}' (HTTP {response.status})"
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return
        raise ValueError(
            f"Unexpected response from TestPyPI while checking version '{version}' (HTTP {exc.code})"
        ) from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Failed to reach TestPyPI while checking version '{version}': {exc.reason}") from exc


def prepare_release(release_type: str, target_ref: str, run_number: int) -> dict[str, str]:
    repo_root = get_repo_root()
    base_version = get_version(repo_root)
    commit_sha = run("git", "rev-parse", "HEAD")
    target_branch = ""
    run_number = sanitize_run_number(run_number)

    if release_type == "prerelease":
        version = build_prerelease_version(base_version, run_number)
    elif release_type == "canary":
        if not (STABLE_VERSION_RE.fullmatch(base_version) or PRERELEASE_VERSION_RE.fullmatch(base_version)):
            raise ValueError(f"Canaries require a base version like X.Y.Z or X.Y.Zrc1; found '{base_version}'")
        branch_check = subprocess.run(
            ("git", "ls-remote", "--exit-code", "--heads", "origin", target_ref),
            check=False,
            capture_output=True,
            text=True,
        )
        if branch_check.returncode != 0:
            raise ValueError(
                f"Canary releases must target a branch ref that exists on origin; '{target_ref}' is not a branch"
            )
        target_branch = target_ref
        date_stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
        version = f"{base_version}.dev{date_stamp}{run_number:06d}"
    else:
        raise ValueError(f"Unsupported release type '{release_type}'")

    check_testpypi_version_does_not_exist(version)
    return {
        "commit_sha": commit_sha,
        "release_type": release_type,
        "target_branch": target_branch,
        "version": version,
    }


def load_testpypi_project_json() -> dict[str, Any]:
    try:
        with urlopen("https://test.pypi.org/pypi/braintrust/json") as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"releases": {}}
        raise


def extract_version_module_from_distribution(file_url: str) -> str:
    with urlopen(file_url) as response:
        payload = response.read()

    if file_url.endswith(".whl"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            version_member = next(name for name in archive.namelist() if name.endswith("braintrust/version.py"))
            return archive.read(version_member).decode("utf-8")
    elif file_url.endswith((".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            version_member = next(
                member for member in archive.getmembers() if member.name.endswith("src/braintrust/version.py")
            )
            extracted = archive.extractfile(version_member)
            if extracted is None:
                raise ValueError(f"Could not read {version_member.name} from {file_url}")
            return extracted.read().decode("utf-8")
    else:
        raise ValueError(f"Unsupported distribution file for canary inspection: {file_url}")


def extract_release_metadata_from_distribution(file_url: str) -> tuple[str, str]:
    contents = extract_version_module_from_distribution(file_url)
    commit_match = GIT_COMMIT_RE.search(contents)
    release_channel_match = RELEASE_CHANNEL_RE.search(contents)
    if commit_match is None:
        raise ValueError(f"Could not find templated GIT_COMMIT in {file_url}")
    if release_channel_match is None:
        raise ValueError(f"Could not find templated RELEASE_CHANNEL in {file_url}")
    return commit_match.group(1), release_channel_match.group(1)


def inspect_release_metadata(version: str, files: list[dict[str, Any]]) -> tuple[str, str]:
    preferred_file = next((file for file in files if file["filename"].endswith(".whl")), files[0])
    try:
        return extract_release_metadata_from_distribution(preferred_file["url"])
    except (
        StopIteration,
        KeyError,
        LookupError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
        tarfile.TarError,
        urllib.error.URLError,
    ) as exc:
        raise ValueError(f"Could not inspect TestPyPI release {version}: {exc}") from exc


def check_canary_status() -> dict[str, str]:
    current_commit = run("git", "rev-parse", "HEAD")
    current_short = run("git", "rev-parse", "--short=7", "HEAD")
    releases = load_testpypi_project_json().get("releases", {})
    candidates: list[tuple[str, Any]] = []
    for version, files in releases.items():
        if not files:
            continue
        try:
            candidates.append((version, Version(version)))
        except InvalidVersion:
            continue
    candidates.sort(key=lambda item: item[1], reverse=True)

    latest_canary_version = ""
    latest_canary_commit = ""
    ignored_release_count = 0

    for version, _parsed_version in candidates:
        try:
            previous_commit, release_channel = inspect_release_metadata(version, releases[version])
        except ValueError:
            ignored_release_count += 1
            continue
        if release_channel != "canary":
            continue
        latest_canary_version = version
        latest_canary_commit = previous_commit
        break

    if not latest_canary_version:
        reason = "No existing canary found on TestPyPI."
        if ignored_release_count:
            reason = f"{reason} Ignored {ignored_release_count} unreadable release(s) while searching."
        return {
            "should_publish": "true",
            "reason": reason,
            "previous_version": "",
            "previous_commit_sha": "",
        }

    previous_short = latest_canary_commit[:7]
    if latest_canary_commit == current_commit:
        return {
            "should_publish": "false",
            "reason": f"Current HEAD {current_short} is already published as canary {latest_canary_version} on TestPyPI.",
            "previous_version": latest_canary_version,
            "previous_commit_sha": previous_short,
        }

    return {
        "should_publish": "true",
        "reason": f"Latest TestPyPI canary {latest_canary_version} is from {previous_short}, which does not match HEAD {current_short}.",
        "previous_version": latest_canary_version,
        "previous_commit_sha": previous_short,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_subparsers = prepare_parser.add_subparsers(dest="release_type", required=True)

    prerelease_parser = prepare_subparsers.add_parser("prerelease")
    prerelease_parser.add_argument("--run-number", type=int, required=True)
    prerelease_parser.add_argument("--github-output", type=pathlib.Path)

    canary_parser = prepare_subparsers.add_parser("canary")
    canary_parser.add_argument("target_ref")
    canary_parser.add_argument("--run-number", type=int, required=True)
    canary_parser.add_argument("--github-output", type=pathlib.Path)

    check_canary_parser = subparsers.add_parser("check-canary")
    check_canary_parser.add_argument("--github-output", type=pathlib.Path)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "prepare":
        target_ref = getattr(args, "target_ref", "")
        outputs = prepare_release(args.release_type, target_ref, args.run_number)
    elif args.command == "check-canary":
        outputs = check_canary_status()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command '{args.command}'")

    github_output = getattr(args, "github_output", None)
    if github_output is not None:
        write_github_output(github_output, outputs)
    else:
        for key, value in outputs.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
