#!/usr/bin/env python3
"""Fetch the exact OpenAPI snapshot pinned in openapi/config.json."""

import sys

from openapi_codegen import CONFIG_PATH, CodegenError, fetch_spec, load_config


def main():
    config = load_config(CONFIG_PATH)
    fetch_spec(config)


if __name__ == "__main__":
    try:
        main()
    except CodegenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
