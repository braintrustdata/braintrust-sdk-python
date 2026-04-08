#!/usr/bin/env python3
"""Compare measured nox session durations against session-weights.json.

Reads one or more measured-duration JSON files (produced by
``nox-matrix.py --output-durations``), merges them, and reports sessions
whose actual duration drifted significantly from the recorded weight.

Exit codes:
    0 — all weights are within tolerance (new/missing sessions are reported
        but do not cause a non-zero exit; they receive the default weight)
    1 — at least one weight drifted beyond the threshold

Usage:
    python check-session-weights.py measured-shard-0.json measured-shard-1.json ...

To update weights after downloading the measured-durations artifacts from CI:
    python check-session-weights.py --update measured-shard-0.json measured-shard-1.json ...
"""

import argparse
import json
import sys
from pathlib import Path


# A session is flagged when its measured duration differs from the recorded
# weight by more than this fraction (0.5 = 50%).
DRIFT_THRESHOLD = 0.5

# Ignore drift for sessions shorter than this (seconds).  Short sessions
# have high relative variance and aren't worth chasing.
MIN_DURATION_FOR_DRIFT = 8


def update_weights(weights_path: Path, weights_data: dict, measured: dict[str, int]) -> None:
    """Overwrite session-weights.json with measured durations."""
    meta_keys = {k for k in weights_data if k.startswith("_")}
    updated = {k: weights_data[k] for k in sorted(meta_keys)}
    # Merge: keep measured values, drop sessions that no longer exist
    all_sessions = sorted(set(weights_data.keys() - meta_keys) | set(measured.keys()))
    for session in all_sessions:
        if session in measured:
            updated[session] = measured[session]
        else:
            # Session wasn't measured — keep the old weight (may be platform-specific
            # or skipped; a full run across all shards would cover everything)
            updated[session] = weights_data[session]
    with open(weights_path, "w") as f:
        json.dump(updated, f, indent=2, sort_keys=True)
        f.write("\n")
    n_changed = sum(1 for s in measured if s not in meta_keys and weights_data.get(s) != measured[s])
    n_new = sum(1 for s in measured if s not in weights_data)
    print(
        f"✅ Updated {weights_path} ({n_changed} changed, {n_new} new, {len(updated) - len(meta_keys)} total sessions)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measured_files", nargs="+", type=Path, help="JSON files with measured durations")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).parent / "session-weights.json",
        help="Path to session-weights.json (default: co-located with this script)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update session-weights.json with measured durations and exit",
    )
    args = parser.parse_args()

    # Merge all measured durations (later files overwrite earlier for the same session)
    measured: dict[str, int] = {}
    for path in args.measured_files:
        with open(path) as f:
            measured.update(json.load(f))

    if not measured:
        print("⚠️  No measured durations found — nothing to check.")
        sys.exit(0)

    with open(args.weights) as f:
        weights_data: dict[str, int] = json.load(f)

    if args.update:
        update_weights(args.weights, weights_data, measured)
        return

    meta_keys = {k for k in weights_data if k.startswith("_")}

    drifted: list[str] = []
    new_sessions: list[str] = []

    print(f"Comparing {len(measured)} measured sessions against session-weights.json\n")
    print(f"{'Session':<50} {'Expected':>8} {'Actual':>8} {'Drift':>8}")
    print("-" * 78)

    for session in sorted(measured):
        actual = measured[session]
        expected = weights_data.get(session)

        if expected is None:
            new_sessions.append(session)
            print(f"{session:<50} {'(new)':>8} {actual:>7}s {'':>8}")
            continue

        if expected == 0:
            drift_pct = 0.0
        else:
            drift_pct = (actual - expected) / expected

        flag = ""
        if abs(drift_pct) > DRIFT_THRESHOLD and max(actual, expected) >= MIN_DURATION_FOR_DRIFT:
            flag = " ⚠️"
            drifted.append(session)

        print(f"{session:<50} {expected:>7}s {actual:>7}s {drift_pct:>+7.0%}{flag}")

    # Check for sessions in weights but not measured (may have been removed)
    known_sessions = {k for k in weights_data if k not in meta_keys}
    missing = sorted(known_sessions - set(measured))

    print()

    if new_sessions:
        print(f"🆕 {len(new_sessions)} new session(s) not in session-weights.json:")
        for s in new_sessions:
            print(f"   {s}: {measured[s]}s")
        print()

    if missing:
        print(f"❓ {len(missing)} session(s) in session-weights.json but not measured")
        print("   (may be in another shard — only a concern if missing from ALL shards):")
        for s in missing:
            print(f"   {s}")
        print()

    if drifted:
        print(f"⚠️  {len(drifted)} session(s) drifted beyond {DRIFT_THRESHOLD:.0%} threshold.")
        print("   Consider updating py/scripts/session-weights.json")
        sys.exit(1)
    else:
        print("✅ All session weights are within tolerance.")
        sys.exit(0)


if __name__ == "__main__":
    main()
