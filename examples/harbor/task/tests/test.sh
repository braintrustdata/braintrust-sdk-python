#!/bin/sh

answer="$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null || true)"
if [ "$answer" = "102" ]; then
  reward=1
else
  reward=0
fi

printf '{"reward":%s,"answer_length":%s}\n' "$reward" "${#answer}" > /logs/verifier/reward.json
