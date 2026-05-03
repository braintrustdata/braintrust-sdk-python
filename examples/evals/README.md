# Evals

Two minimal `Eval()` examples:

- `simple_eval.py` — the smallest possible eval: hardcoded data, a regex task, a constant scorer.
- `eval_example.py` — a richer scorer that uses the `trace` argument to fetch and inspect the spans recorded for each row (configuration, span hierarchy, inputs/outputs, metadata).

## Run

```bash
export BRAINTRUST_API_KEY=...

uv sync
uv run braintrust eval simple_eval.py
uv run braintrust eval eval_example.py
```

The second example is async, so the scorer prints span info to stdout as the eval runs.
