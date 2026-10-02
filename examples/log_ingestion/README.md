# Policy-aware log ingestion demo

From `py/`, with `BRAINTRUST_API_KEY` configured through mise:

```bash
mise exec -- python ../examples/log_ingestion/demo.py --replay
```

The demo sends a root/child trace, flushes a score update, relogs in, and sends a final score and comment. `--replay` sends the same memoized feedback records again, preserving their row identities. The JSON output includes the trace link and remaining record count. Inspect the trace with:

```bash
bt view trace --object-ref project_logs:<project_id> --trace-id <trace_id> --json
```

Expect two spans, final root score `quality=1`, and one comment even after replay.

`--overflow` lowers the writer's payload limit to exercise signed uploads on servers that advertise `logs3_payload_max_bytes`. The output reports `overflow_supported`; older servers continue using ordinary ingestion. No provider SDK or model calls are needed.
