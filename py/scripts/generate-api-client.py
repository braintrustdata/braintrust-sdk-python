#!/usr/bin/env python3
"""Generate the committed private Braintrust REST API models.

Usage:
    python scripts/generate-api-client.py            # regenerate in place
    python scripts/generate-api-client.py --check     # report drift, leave the worktree alone
"""

import argparse
import sys
import tempfile
from pathlib import Path

from openapi_codegen import (
    CONFIG_PATH,
    GENERATED_ROOT,
    SPEC_PATH,
    CodegenError,
    atomic_replace_tree,
    compare_generated,
    generate_tree,
    load_config,
    read_and_verify_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into a temporary directory and diff it against the committed tree.",
    )
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    spec = read_and_verify_spec(config, SPEC_PATH)
    with tempfile.TemporaryDirectory(prefix="braintrust-api-codegen-") as temporary:
        generated = Path(temporary) / "_generated"
        report = generate_tree(generated, config, spec)
        if not args.check:
            atomic_replace_tree(generated, GENERATED_ROOT)
            print(f"Generated {GENERATED_ROOT} from pinned spec: {report}")
            return 0
        differences = compare_generated(generated, GENERATED_ROOT)

    if differences:
        print("Generated API client drift detected. Run `cd py && make generate-api-client`.", file=sys.stderr)
        print("".join(differences), file=sys.stderr)
        return 1
    print(f"Generated API client is current: {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CodegenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
