#!/usr/bin/env python3
"""Update pinned livekit-server version and archive SHA256 hashes in noxfile.py."""

import argparse
import hashlib
import pathlib
import re
import urllib.request


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
NOXFILE = PROJECT_DIR / "noxfile.py"
PLATFORMS = ("linux_amd64", "linux_arm64")


def _asset_url(version: str, platform: str) -> str:
    system, arch = platform.split("_", 1)
    asset = f"livekit_{version}_{system}_{arch}.tar.gz"
    return f"https://github.com/livekit/livekit/releases/download/v{version}/{asset}"


def _sha256_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - pinned GitHub release URL.
        digest = hashlib.sha256()
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()


def _replace_constants(contents: str, version: str, hashes: dict[str, str]) -> str:
    contents = re.sub(
        r'LIVEKIT_SERVER_VERSION = "[^"]+"',
        f'LIVEKIT_SERVER_VERSION = "{version}"',
        contents,
        count=1,
    )
    sha_block = (
        "LIVEKIT_SERVER_SHA256 = {\n"
        + "".join(f'    "{platform}": "{hashes[platform]}",\n' for platform in PLATFORMS)
        + "}"
    )
    contents = re.sub(
        r'LIVEKIT_SERVER_SHA256 = \{\n(?:    "[^"]+": "[0-9a-f]+",\n)+\}',
        sha_block,
        contents,
        count=1,
    )
    return contents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="livekit-server version, e.g. 1.11.1")
    parser.add_argument("--dry-run", action="store_true", help="print the new constants without editing noxfile.py")
    args = parser.parse_args()

    hashes = {}
    for platform in PLATFORMS:
        url = _asset_url(args.version, platform)
        print(f"Downloading {url}")
        hashes[platform] = _sha256_url(url)
        print(f"{platform}: {hashes[platform]}")

    contents = NOXFILE.read_text()
    updated = _replace_constants(contents, args.version, hashes)

    if args.dry_run:
        print()
        print(f'LIVEKIT_SERVER_VERSION = "{args.version}"')
        print("LIVEKIT_SERVER_SHA256 = {")
        for platform in PLATFORMS:
            print(f'    "{platform}": "{hashes[platform]}",')
        print("}")
        return

    if updated == contents:
        raise SystemExit("noxfile.py did not change; constants may not have matched expected format")

    NOXFILE.write_text(updated)
    print(f"Updated {NOXFILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
