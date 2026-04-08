#!/bin/bash
#
# Distributes nox sessions across shards using greedy bin-packing (LPT) so
# that total estimated runtime per shard is balanced. Session weights come from
# a companion file (session-weights.json). Unknown sessions get a default weight.
#

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
NOXFILE=$ROOT_DIR/py/noxfile.py
WEIGHTS_FILE=$ROOT_DIR/py/scripts/session-weights.json

# Parse command line arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <shard_index> <number_of_shards> [--dry-run]"
  exit 1
fi

INDEX=$1
TOTAL=$2
DRY_RUN=false
shift 2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 <shard_index> <number_of_shards> [--dry-run]"
      exit 1
      ;;
  esac
done

if [ "$INDEX" -ge "$TOTAL" ]; then
  echo "Error: shard_index ($INDEX) must be less than number_of_shards ($TOTAL)"
  exit 1
fi

# Nox formats the sessions like:
# * test_foo
# * test_bar -> Optional description
# We need to strip the description part after " -> "
all_sessions=$(nox -l -f "$NOXFILE" | grep "^\* " | cut -c 3- | sed 's/ ->.*$//' | sort)

# Use Python for the greedy LPT assignment — it's already available.
matches=$(python3 -c "
import json, sys

sessions = '''$all_sessions'''.strip().split('\n')
total_shards = $TOTAL
my_shard = $INDEX

# Load weights; fall back to default for unknown sessions
try:
    with open('$WEIGHTS_FILE') as f:
        weights = json.load(f)
except FileNotFoundError:
    weights = {}
default_weight = weights.get('_default', 15)

# Sort sessions by weight descending (LPT)
sessions_weighted = [(s, weights.get(s, default_weight)) for s in sessions]
sessions_weighted.sort(key=lambda x: -x[1])

# Greedy assignment: always put next session into the lightest shard
shard_totals = [0] * total_shards
shard_assignments = [[] for _ in range(total_shards)]
for name, weight in sessions_weighted:
    lightest = min(range(total_shards), key=lambda i: shard_totals[i])
    shard_assignments[lightest].append(name)
    shard_totals[lightest] += weight

# Print summary to stderr
for i in range(total_shards):
    marker = ' <-- this shard' if i == my_shard else ''
    print(f'  shard {i}: {len(shard_assignments[i])} sessions, ~{shard_totals[i]}s{marker}', file=sys.stderr)

# Print this shard's sessions to stdout
for s in sorted(shard_assignments[my_shard]):
    print(s)
")

misses=$(comm -23 <(echo "$all_sessions") <(echo "$matches"))
n_matches=$(echo "$matches" | grep -c . || true)
n_all=$(echo "$all_sessions" | grep -c . || true)

printf "nox matrix idx:%d shards:%d running %d/%d sessions\n" "$INDEX" "$TOTAL" "$n_matches" "$n_all"

if [ "$DRY_RUN" = true ]; then
  echo "--------------------------------"
  echo "Would run the following sessions:"
  echo "$matches"
  echo ""
  echo "--------------------------------"
  echo "Would skip the following sessions:"
  echo "$misses"
  exit 0
fi

# Build session list and run nox once
# Quote each session name to handle parentheses in names like test_openai(latest)
session_list=$(echo "$matches" | sed 's/.*/"&"/' | tr '\n' ' ')
eval "nox -f $NOXFILE -s $session_list"
