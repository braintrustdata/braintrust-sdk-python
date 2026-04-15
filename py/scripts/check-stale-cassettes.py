#!/usr/bin/env python3
"""Detect stale cassette version directories that no longer match the version matrix.

Cassette-backed integration tests store recordings under version-specific
subdirectories (e.g. ``cassettes/latest/``, ``cassettes/1.74.0/``).  When a
version is removed from the ``[tool.braintrust.matrix]`` table in
``pyproject.toml``, its cassette directory becomes orphaned.

This script reads the matrix and a directory→matrix-key mapping from
``pyproject.toml`` and reports (or removes) orphaned cassette directories.

Usage:
    python scripts/check-stale-cassettes.py          # report only
    python scripts/check-stale-cassettes.py --clean   # delete stale dirs
"""

import argparse
import pathlib
import shutil
import sys


if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


_PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
_INTEGRATIONS_DIR = _PROJECT_DIR / "src" / "braintrust" / "integrations"


def _load_config() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Return (matrix, cassette_dirs) from pyproject.toml."""
    pyproject = tomllib.loads((_PROJECT_DIR / "pyproject.toml").read_text())
    bt = pyproject.get("tool", {}).get("braintrust", {})
    matrix: dict[str, dict[str, str]] = bt.get("matrix", {})
    cassette_dirs: dict[str, list[str]] = bt.get("cassette-dirs", {})
    return matrix, cassette_dirs


def _valid_versions(matrix: dict[str, dict[str, str]], matrix_keys: list[str]) -> set[str]:
    """Collect all valid version keys across the given matrix entries."""
    versions: set[str] = set()
    for key in matrix_keys:
        entry = matrix.get(key, {})
        for v in entry:
            # The version key in the matrix is what becomes the directory name.
            # "latest" in the TOML maps to the LATEST sentinel, which becomes
            # the directory name "latest".
            versions.add(v)
    return versions


def check_stale_cassettes(
    matrix: dict[str, dict[str, str]],
    cassette_dirs: dict[str, list[str]],
    clean: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (stale_dirs, unmapped_dirs).

    *stale_dirs*: version subdirectories that exist on disk but are not in the
    matrix for the mapped integration.

    *unmapped_dirs*: integration cassette directories that have version
    subdirectories but no entry in ``[tool.braintrust.cassette-dirs]``.
    """
    stale: list[str] = []
    unmapped: list[str] = []

    # Discover all integration cassette base dirs on disk.
    if not _INTEGRATIONS_DIR.is_dir():
        return stale, unmapped

    for integration_dir in sorted(_INTEGRATIONS_DIR.iterdir()):
        if not integration_dir.is_dir() or integration_dir.name == "__pycache__":
            continue
        cassette_base = integration_dir / "cassettes"
        if not cassette_base.is_dir():
            continue

        # Collect on-disk version subdirectories (skip regular files).
        on_disk_versions = {
            child.name for child in cassette_base.iterdir() if child.is_dir() and child.name != "__pycache__"
        }
        if not on_disk_versions:
            continue

        dir_name = integration_dir.name
        matrix_keys = cassette_dirs.get(dir_name)
        if matrix_keys is None:
            # Integration has versioned cassettes but no mapping — warn.
            for v in sorted(on_disk_versions):
                unmapped.append(str(cassette_base / v))
            continue

        valid = _valid_versions(matrix, matrix_keys)
        for v in sorted(on_disk_versions):
            if v not in valid:
                full_path = cassette_base / v
                stale.append(str(full_path))
                if clean:
                    shutil.rmtree(full_path)

    return stale, unmapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", action="store_true", help="Delete stale cassette directories")
    args = parser.parse_args()

    matrix, cassette_dirs = _load_config()

    if not cassette_dirs:
        print(
            "WARNING: No [tool.braintrust.cassette-dirs] table in pyproject.toml.\n"
            "         Add it to enable stale cassette detection.\n"
            "         See AGENTS.md for details.",
            file=sys.stderr,
        )
        sys.exit(1)

    stale, unmapped = check_stale_cassettes(matrix, cassette_dirs, clean=args.clean)

    ok = True
    if stale:
        ok = False
        action = "Deleted" if args.clean else "Found"
        print(f"{action} {len(stale)} stale cassette director{'y' if len(stale) == 1 else 'ies'}:")
        for path in stale:
            print(f"  {pathlib.Path(path).relative_to(_PROJECT_DIR)}")
        if not args.clean:
            print("\nRun with --clean to remove them, or update [tool.braintrust.matrix] in pyproject.toml.")

    if unmapped:
        ok = False
        print(
            f"\n{len(unmapped)} cassette director{'y' if len(unmapped) == 1 else 'ies'} not mapped in [tool.braintrust.cassette-dirs]:"
        )
        for path in unmapped:
            print(f"  {pathlib.Path(path).relative_to(_PROJECT_DIR)}")
        print("\nAdd the integration to [tool.braintrust.cassette-dirs] in pyproject.toml.")

    if ok:
        print("No stale cassette directories found.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
