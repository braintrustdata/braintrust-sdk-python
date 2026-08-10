#!/usr/bin/env python
"""Backfill a persisted Harbor job into Braintrust."""

import argparse
import asyncio
from pathlib import Path

from braintrust.integrations.harbor import backfill_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path, help="Persisted Harbor job directory")
    parser.add_argument("--project", help="Braintrust project name (otherwise read from the environment)")
    args = parser.parse_args()

    options = {"project_name": args.project} if args.project else {}
    asyncio.run(backfill_job(args.job_dir, **options))


if __name__ == "__main__":
    main()
