# Evals

Two minimal `Eval()` examples:

- `simple_eval.py` — the smallest possible eval: hardcoded data, a regex task, a constant scorer.
- `eval_example.py` — a richer scorer that uses the `trace` argument to fetch and inspect the spans recorded for each row (configuration, span hierarchy, inputs/outputs, metadata).

## Run

[Install the standalone `bt` CLI](https://github.com/braintrustdata/bt#install) and run from this directory:

```bash
export BRAINTRUST_API_KEY="YOUR_API_KEY"

uv sync
bt eval simple_eval.py
bt eval eval_example.py
```

The second example is async, so the scorer prints span info to stdout as the eval runs.
