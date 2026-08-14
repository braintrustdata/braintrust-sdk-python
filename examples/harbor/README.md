# Harbor + Braintrust

Runs a small, self-contained [Harbor](https://harborframework.com/) evaluation and uses Harbor's native Braintrust job plugin to sync the result. Braintrust receives a managed dataset, an experiment row for the final trial, verifier rewards, and the Harbor lifecycle and ATIF trace.

The plugin is discovered automatically through Harbor's `braintrust` entry point. The Braintrust API key remains in the host process; it is not passed into the task container.

## Setup

Install the example's dependencies:

```bash
uv sync
```

The command below reads credentials from the repository's root `.env`. It requires:

```dotenv
BRAINTRUST_API_KEY=...
OPENAI_API_KEY=...
```

Alternatively, copy `.env.example` to `.env` in this directory and change `--env-file ../../.env` below to `--env-file .env`.

## Run

Docker must be running. From this directory, run:

```bash
uv run harbor run \
  --path task \
  --agent terminus-2 \
  --model openai/gpt-4.1-mini \
  --job-name braintrust-harbor-example \
  --jobs-dir jobs \
  --env-file ../../.env \
  --plugin braintrust \
  --yes
```

The agent solves the task in `task/`, and Harbor's verifier emits a normalized `reward` plus an `answer_length` metric. The plugin creates `jobs/braintrust-harbor-example/braintrust-sync.json` after synchronization.

By default, the plugin uses `Harbor` as the Braintrust project name. Override it with `--plugin-kwarg project_name=example-harbor` or through `.env`:

```dotenv
HARBOR_BRAINTRUST_PROJECT=example-harbor
```

## Backfill an existing job

To synchronize the persisted job again without rerunning the agent or verifier:

```bash
uv run --env-file ../../.env python backfill.py jobs/braintrust-harbor-example
```

Backfill uses the same `Harbor` project default as the initial sync. It also uses the same deterministic dataset, experiment, and span identities, so it reconciles the existing Braintrust data instead of creating duplicate rows.
