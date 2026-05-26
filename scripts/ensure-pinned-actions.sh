#!/usr/bin/env bash
set -euo pipefail

# Verify every action reference in .github/workflows/ is pinned to a full commit SHA.
# A pinned ref looks like: uses: owner/action@<40 hex chars>
# Tags (@v4, @main) are rejected — they are mutable and can be hijacked.

unpinned=$(grep -rn --include="*.yml" --include="*.yaml" -E 'uses:\s+\S+@' .github/ \
  | grep -vE '@[a-f0-9]{40}(\s|$|#)' || true)

if [ -n "$unpinned" ]; then
  echo "ERROR: unpinned action(s) found — use a full commit SHA instead of a tag or branch:"
  echo "$unpinned"
  exit 1
fi

echo "OK: all actions are pinned to commit SHAs."
