# OpenTelemetry + Braintrust

Five scripts that exercise different OTEL integration shapes:

| Script | Shape |
| --- | --- |
| `basic_otel_example.py` | bare `BraintrustSpanProcessor` with `filter_ai_spans=False` — every OTEL span flows through |
| `filtered_otel_example.py` | `BraintrustSpanProcessor` plus a `custom_filter` callback that augments the default LLM-only filtering |
| `bt-otel-context.py` | mixed-mode tracing: BT spans inside OTEL context and OTEL spans inside BT context, with `BRAINTRUST_OTEL_COMPAT=true` |
| `distributed-tracing.py` | three-service flow that uses `span.export()`, `context_from_span_export()`, and `parent_from_headers()` to maintain one trace across services |
| `otel_eval.py` | adds OTEL spans inside a Braintrust `Eval()` task using `Levenshtein` for scoring |

## Run

```bash
export BRAINTRUST_API_KEY=...
export OPENAI_API_KEY=...

uv sync
uv run python basic_otel_example.py
```
