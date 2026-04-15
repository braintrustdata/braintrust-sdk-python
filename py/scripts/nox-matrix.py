#!/usr/bin/env python3
"""Distribute nox sessions across shards using greedy bin-packing (LPT).

Session weights are read from session-weights.json (co-located with this
script). Unknown sessions get a default weight. The algorithm sorts sessions
by weight descending and greedily assigns each to the lightest shard.

Usage:
    python nox-matrix.py <shard_index> <number_of_shards> [--dry-run] [--exclude-session <name> ...]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# Sessions that run in the dedicated static_checks CI job and should be
# excluded from the sharded nox matrix.  Both checks.yaml and
# update-session-weights.yaml reference this via --exclude-static-checks.
STATIC_CHECK_SESSIONS = ["pylint", "test_types"]


def get_nox_sessions(noxfile: Path) -> list[str]:
    """List available nox sessions by running ``nox -l``."""
    result = subprocess.run(
        ["nox", "-l", "-f", str(noxfile)],
        capture_output=True,
        text=True,
        check=True,
    )
    sessions: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("* "):
            # Strip the leading "* " and any " -> description" suffix
            name = line[2:].split(" -> ")[0].strip()
            sessions.append(name)
    return sorted(sessions)


def load_weights(weights_file: Path) -> tuple[dict[str, int], int]:
    """Return (weights_map, default_weight) from the JSON file."""
    try:
        with open(weights_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    default = data.get("_default", 15)
    return data, default


def assign_shards(
    sessions: list[str],
    total_shards: int,
    weights: dict[str, int],
    default_weight: int,
) -> list[list[str]]:
    """Assign sessions to shards using greedy LPT bin-packing."""
    weighted = [(s, weights.get(s, default_weight)) for s in sessions]
    weighted.sort(key=lambda x: -x[1])

    shard_totals = [0] * total_shards
    shard_assignments: list[list[str]] = [[] for _ in range(total_shards)]

    for name, weight in weighted:
        lightest = min(range(total_shards), key=lambda i: shard_totals[i])
        shard_assignments[lightest].append(name)
        shard_totals[lightest] += weight

    for i in range(total_shards):
        count = len(shard_assignments[i])
        total = shard_totals[i]
        print(f"  shard {i}: {count} sessions, ~{total}s", file=sys.stderr)

    # Sort each shard's sessions for deterministic output
    for assignments in shard_assignments:
        assignments.sort()

    return shard_assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_index", type=int, help="Zero-based shard index")
    parser.add_argument("num_shards", type=int, help="Total number of shards")
    parser.add_argument("--dry-run", action="store_true", help="Print assignment without running nox")
    parser.add_argument(
        "--exclude-session",
        action="append",
        default=[],
        help="Exclude a nox session from shard assignment. May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-static-checks",
        action="store_true",
        default=False,
        help=f"Exclude sessions that run in the dedicated static_checks CI job ({', '.join(STATIC_CHECK_SESSIONS)}).",
    )
    parser.add_argument(
        "--output-durations",
        type=Path,
        default=None,
        help="Write measured session durations (seconds) to a JSON file",
    )
    args = parser.parse_args()

    if args.shard_index >= args.num_shards:
        print(
            f"Error: shard_index ({args.shard_index}) must be less than num_shards ({args.num_shards})",
            file=sys.stderr,
        )
        sys.exit(1)

    root_dir = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    noxfile = root_dir / "py" / "noxfile.py"
    weights_file = root_dir / "py" / "scripts" / "session-weights.json"

    all_sessions = get_nox_sessions(noxfile)
    excluded_sessions = set(args.exclude_session)
    if args.exclude_static_checks:
        excluded_sessions.update(STATIC_CHECK_SESSIONS)
    all_sessions = [session for session in all_sessions if session not in excluded_sessions]
    weights, default_weight = load_weights(weights_file)
    shard_assignments = assign_shards(all_sessions, args.num_shards, weights, default_weight)

    my_sessions = shard_assignments[args.shard_index]
    other_sessions = sorted(set(all_sessions) - set(my_sessions))

    print(
        f"nox matrix idx:{args.shard_index} shards:{args.num_shards} "
        f"running {len(my_sessions)}/{len(all_sessions)} sessions"
    )

    if args.dry_run:
        print("--------------------------------")
        print("Would run the following sessions:")
        print("\n".join(my_sessions))
        print()
        print("--------------------------------")
        print("Would skip the following sessions:")
        print("\n".join(other_sessions))
        return

    cmd = ["nox", "-f", str(noxfile), "-s", *my_sessions]

    if args.output_durations is None:
        sys.exit(subprocess.run(cmd).returncode)

    # Stream output while capturing session durations
    durations: dict[str, int] = {}
    duration_re = re.compile(r"nox > Session (\S+) was successful in (\d+) seconds\.")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = duration_re.search(line)
        if m:
            durations[m.group(1)] = int(m.group(2))
    process.wait()

    args.output_durations.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_durations, "w") as f:
        json.dump(durations, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {len(durations)} session durations to {args.output_durations}", file=sys.stderr)

    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
