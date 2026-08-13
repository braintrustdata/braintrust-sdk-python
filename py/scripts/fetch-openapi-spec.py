#!/usr/bin/env python3
"""Fetch the exact OpenAPI snapshot pinned in openapi/config.json."""

import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from openapi_codegen import CONFIG_PATH, SPEC_PATH, CodegenError, load_config, validate_config, verify_spec_hash


def _read_source(config):
    spec_config = config["spec"]
    local_root = os.environ.get("BRAINTRUST_OPENAPI_ROOT")
    if local_root:
        root = Path(local_root).expanduser().resolve()
        source = root / spec_config["path"]
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CodegenError(f"BRAINTRUST_OPENAPI_ROOT is not a readable git checkout: {root}") from exc
        if head != spec_config["commit"]:
            raise CodegenError(
                f"Local braintrust-openapi checkout is at {head}, but config pins {spec_config['commit']}"
            )
        try:
            return source.read_bytes(), str(source)
        except OSError as exc:
            raise CodegenError(f"Unable to read local OpenAPI spec {source}: {exc}") from exc

    url = (
        f"https://raw.githubusercontent.com/{spec_config['repository']}/{spec_config['commit']}/{spec_config['path']}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - explicitly pinned public source.
            return response.read(), url
    except (OSError, urllib.error.URLError) as exc:
        raise CodegenError(f"Unable to fetch pinned OpenAPI spec from {url}: {exc}") from exc


def main():
    config = load_config(CONFIG_PATH)
    validate_config(config, check_installed_tools=False)
    content, source = _read_source(config)
    verify_spec_hash(content, config, source)
    SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=SPEC_PATH.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    # NamedTemporaryFile creates the file 0600 and os.replace preserves that mode, so the committed
    # spec would silently lose group/other read access on every fetch.
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(temporary_path, 0o666 & ~umask)
    os.replace(temporary_path, SPEC_PATH)
    print(f"Fetched {source} -> {SPEC_PATH} ({config['spec']['sha256']})")


if __name__ == "__main__":
    try:
        main()
    except CodegenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
