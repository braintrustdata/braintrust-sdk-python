# Harbor Framework Research: Evals, Tracing, and Configuration

Research date: 2026-07-29
Upstream repository: [`harbor-framework/harbor`](https://github.com/harbor-framework/harbor)
Inspected commit: [`e76f7e32f5644fb9f648cd23151aac5c67492ea0`](https://github.com/harbor-framework/harbor/tree/e76f7e32f5644fb9f648cd23151aac5c67492ea0)
Inspected package version: `harbor==0.20.0`

## Executive summary

Harbor is primarily a **sandboxed agent evaluation harness**, not an LLM tracing SDK. Its central abstraction is:

> A job expands tasks, agents, models, and attempts into trials; each trial runs an agent in a container, runs a verifier, emits one or more numeric rewards, and preserves the execution record on disk.

The core data flow is:

```text
dataset(s) / task(s)
        +
agent configuration(s)
        +
runtime environment configuration
        ↓
JobConfig → resolved JobPlan/JobLock
        ↓ expands attempts × tasks × agents
TrialConfig[]
        ↓ concurrent execution
sandbox setup → agent setup → agent run → artifact collection → verifier
        ↓
trajectory.json + reward.json/reward.txt + result.json + lock.json
        ↓
job-level metrics, pass@k, viewer, Hub, trace exporters, plugins
```

The most important conclusions are:

1. **Evals are task/verifier based.** A task contains an instruction, a container environment, and a test script. The verifier script decides the reward by writing `/logs/verifier/reward.json` or `reward.txt`.
2. **Trials are the unit of execution and scoring.** A job is a matrix of trials, normally `n_attempts × tasks × agents`.
3. **Configuration has three distinct layers:** task-owned TOML, run/job YAML or JSON, and CLI overrides. Resolved config and content-addressed locks are persisted for reproducibility.
4. **Tracing is file-first.** Integrated agents write an Agent Trajectory Interchange Format (ATIF) file at `agent/trajectory.json`. Harbor later visualizes or exports that file.
5. **ATIF is richer than a basic chat log.** It covers reasoning, tool calls, observations, token/cost metrics, multimodal content, continuations, copied context, and embedded or referenced subagent trajectories.
6. **Harbor has two separate trace export paths:**
   - ATIF → Hugging Face conversational datasets for SFT.
   - ATIF → OpenTelemetry/OpenInference spans through the separate `harbor-atif2otel` package.
7. **The current OTel uploader is not backend-neutral in practice.** Conversion is generic, but direct upload is hard-wired to MLflow APIs and headers.
8. **The LangSmith plugin is the best model for a Braintrust integration.** It synchronizes datasets/examples, represents trial lifecycle phases, records usage, and publishes rewards as feedback.
9. **A Braintrust integration should be a native Harbor job plugin plus ATIF ingestion**, rather than only pointing Harbor's current OTel uploader at Braintrust.

## 1. What Harbor is

Harbor describes itself as a framework for evaluating and optimizing agents and language models in sandboxed environments. It supports built-in and custom agents, local Docker and cloud sandbox providers, task/dataset publishing, RL rollout generation, SFT export, and a local/hosted results viewer.

Relevant upstream sources:

- [README](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/README.md)
- [Core concepts](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/core-concepts.mdx)
- [`JobConfig`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/models/job/config.py)
- [`TrialConfig`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/models/trial/config.py)

### Core concepts

| Concept | Meaning in Harbor |
|---|---|
| **Task** | One instruction, sandbox definition, and verifier/test implementation. |
| **Dataset** | A collection of tasks, optionally with custom aggregate metrics. |
| **Agent** | A built-in or custom implementation that operates in the sandbox. |
| **Environment** | A local or remote container runtime implementing `BaseEnvironment`. |
| **Trial** | One agent attempt on one task. Conceptually, a rollout that produces a reward. |
| **Job** | A collection of trials, often spanning datasets, agents, models, and repeated attempts. |
| **Reward** | Per-trial numeric verifier output, potentially multi-dimensional. |
| **Metric** | Job/dataset-level aggregation over trial reward dictionaries. |
| **Trajectory** | The agent interaction history, normally stored as ATIF JSON. |

Harbor is deliberately agent- and environment-agnostic. Built-in factories lazily resolve agent names and environment provider types, while custom implementations can be supplied as `module.path:ClassName` import paths.

## 2. How an evaluation works

### 2.1 Define or select tasks

A local task typically has this shape:

```text
my-task/
├── instruction.md
├── task.toml
├── environment/
│   ├── Dockerfile
│   └── ...
├── solution/                 # optional; used by the oracle agent
│   ├── solve.sh
│   └── ...
└── tests/
    ├── test.sh
    └── ...
```

The task definition owns the benchmark semantics:

- `instruction.md` is sent to the agent.
- `environment/` defines or contributes files to the sandbox.
- `task.toml` defines metadata, timeouts, resources, users, networking, MCP servers, verifier isolation, and artifacts.
- `tests/test.sh` performs grading and must write a reward file.
- `solution/solve.sh`, when present, lets the `oracle` agent sanity-check the task.

Published and Git-hosted datasets resolve to the same task model. Harbor supports:

- local task/dataset paths;
- package references such as `org/name@ref`;
- legacy registry entries;
- Git repositories, optionally pinned to a ref;
- task include/exclude glob filters and a post-filter `n_tasks` limit.

Sources:

- [Task structure](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/tasks/index.mdx)
- [Datasets](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/datasets/index.mdx)
- [`DatasetConfig`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/models/job/config.py)

### 2.2 Build the job

The normal entry point is:

```bash
harbor run -d "org/dataset@ref" -a claude-code -m anthropic/claude-sonnet-4-6
```

or:

```bash
harbor run -c job.yaml
```

A resolved job does the following before execution:

1. Resolves agent skills to local cached directories.
2. Resolves datasets into concrete task configurations.
3. Validates environment resource policies.
4. Resolves dataset and job metrics.
5. Downloads/caches package or Git tasks.
6. Builds trial configs.
7. Builds a content-addressed `JobLock`.

The trial expansion is explicit in `JobPlan.build_trial_configs()`:

```text
for each attempt
  for each task
    for each agent configuration
      create TrialConfig
```

Therefore:

```text
trial count = n_attempts × number of resolved tasks × number of agent configs
```

Multiple `--model` values become multiple agent configurations when an agent is selected on the CLI.

Source: [`JobPlan`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/job_plan.py)

### 2.3 Execute trials concurrently

`n_concurrent_trials` is the global trial limit, defaulting to 4. Each agent config can also declare `n_concurrent`, optionally sharing a named `concurrency_group` with other agent configs. A per-agent limit cannot exceed the global trial limit.

Retries use exponential backoff and can include or exclude exception types. By default, semantic failures such as agent/verifier timeouts, missing reward files, authentication failures, model-not-found failures, and API usage limits are excluded from retry.

A single-step trial follows this lifecycle:

```text
initialize config/result/lock
    ↓
start agent environment
    ↓
run environment healthcheck
    ↓
upload skills
    ↓
install/setup agent
    ↓
run agent
    ↓
download/synchronize agent logs and trajectory
    ↓
collect artifacts
    ↓
run verifier in shared or separate environment
    ↓
parse rewards
    ↓
stop environment, scrub known secrets, write result.json
```

Harbor emits hook events around this lifecycle:

- `START`
- `ENVIRONMENT_START`
- `AGENT_START`
- `AGENT_END`
- `VERIFICATION_START`
- `END`
- `CANCEL`

Those hooks power job plugins such as LangSmith, OTel export, and Harbor Hub upload.

Sources:

- [`Trial.run()` and lifecycle](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/trial/trial.py)
- [`SingleStepTrial`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/trial/single_step.py)
- [Trial hook model](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/trial/hooks.py)

### 2.4 Grade the trial

The default verifier copies the task's test directory into `/tests`, executes its OS-appropriate test script, downloads `/logs/verifier`, and parses:

1. `/logs/verifier/reward.json`, if present; otherwise
2. `/logs/verifier/reward.txt`.

`reward.txt` becomes:

```json
{"reward": 1.0}
```

`reward.json` can expose multiple numeric dimensions:

```json
{
  "correctness": 1.0,
  "quality": 0.8,
  "efficiency": 0.6,
  "reward": 0.82
}
```

The Pydantic result type restricts rewards to `dict[str, float | int]`.

A minimal verifier is:

```bash
#!/usr/bin/env bash
set -euo pipefail

if pytest -q /tests/test_solution.py; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
```

Sources:

- [Verifier implementation](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/verifier/verifier.py)
- [`VerifierResult`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/models/verifier/result.py)

### 2.5 Shared versus separate verification

By default, verification is **shared**: tests execute in the agent's container and can see its workspace and installed tools.

A task can instead define a **separate verifier environment**. This is useful for:

- hiding proprietary grading logic;
- reducing agent tampering;
- using a clean grading image;
- grading on a different OS;
- regrading a recorded trial later.

In separate mode, Harbor collects declared artifacts from the agent environment and re-materializes them at their original absolute paths in the verifier environment. The verifier image is built from `tests/` and must contain its own `/tests/test.sh` or Windows equivalent.

Example:

```toml
schema_version = "1.4"
artifacts = ["/app/output.json", "/logs/agent/trajectory.json"]

[environment]
docker_image = "python:3.12-slim"
network_mode = "no-network"

[agent]
timeout_sec = 900

[verifier]
environment_mode = "separate"
timeout_sec = 120

[verifier.environment]
docker_image = "my-org/private-grader:latest"
network_mode = "no-network"
```

Important behavior:

- Merely declaring `[verifier.environment]` implies separate mode.
- `environment_mode = "separate"` without a verifier-specific environment uses a fresh copy of the top-level task environment.
- `environment_mode = "shared"` plus `[verifier.environment]` is invalid.
- `/logs/agent` is not implicitly transferred; declare the trajectory as an artifact if the verifier grades the agent's process.
- Sidecar collection can snapshot databases or service logs before teardown.
- For tamper-sensitive sidecar evidence, single-step or final-step separate verification has the strongest isolation.

### 2.6 Regrade without rerunning the agent

`harbor job regrade` and `harbor trial regrade` fork existing execution records and run a new verifier over their recorded artifacts. The source trial is not modified, and no agent credentials or new model cost are required.

```bash
harbor job regrade jobs/old-job -p ./updated-task -e docker
```

Regrade requires:

- a completed, currently single-step source trial;
- a separate-mode new verifier;
- all newly declared artifacts to exist successfully in the old artifact manifest.

The derived config and lock record source provenance. This is a strong primitive for verifier iteration, regression analysis, and keeping agent execution fixed while testing scoring changes.

Source: [Regrade documentation](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/run-jobs/regrade.mdx)

## 3. Rewards, metrics, and eval result aggregation

### 3.1 Reward versus metric

Harbor uses these terms distinctly:

- A **reward** is emitted by one trial's verifier.
- A **metric** aggregates reward dictionaries across trials in an agent/model/dataset group.

Built-in metric types are:

- `mean`
- `sum`
- `min`
- `max`
- `uv-script`

The default is mean. Missing rewards are treated as zero by the built-in dictionary aggregation. For a single reward key, the output key is the aggregation name, such as `{"mean": 0.81}`. For multiple reward keys, each dimension is aggregated independently and missing dimensions are zero-filled.

Harbor also computes `pass@k` grouped by agent/model/dataset.

### 3.2 Custom dataset metrics

A dataset can ship a `metric.py`. Harbor invokes it through `uv` with:

```text
metric.py -i rewards.jsonl -o metric.json
```

The input has one reward dictionary or `null` per line. The script writes one JSON object containing aggregate metric names and numeric values.

This means a dataset can own benchmark-specific aggregation rather than forcing all benchmarks into mean reward.

Sources:

- [Metrics documentation](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/datasets/metrics.mdx)
- [Built-in aggregation](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/metrics/base.py)
- [`UvScript`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/metrics/uv_script.py)

### 3.3 Rewardkit

Rewardkit is a separate package in the Harbor workspace that standardizes richer verifiers. A `tests/` tree can mix:

- programmatic Python criteria;
- LLM-as-a-judge criteria;
- agent-as-a-judge criteria;
- built-in workspace, command, structured-data, image, and trajectory checks.

Each directory maps to a reward dimension. Criteria have weights, and judge rubrics can aggregate with `weighted_mean`, `all_pass`, `any_pass`, `threshold`, or `required_pass`. A root `reward.toml` can add an overall `reward` while retaining the component dimensions.

Rewardkit writes:

```text
reward.json          # numeric dimensions consumed by Harbor
reward-details.json  # criterion-level scores, reasoning, and errors
```

It can grade the process by loading an ATIF trajectory via `atif-trajectory`. This is the direct bridge between tracing and evals inside Harbor: **the trajectory itself can be an input to the verifier**.

Sources:

- [Rewardkit overview](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/rewardkit/index.mdx)
- [Judge configuration](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/rewardkit/judge-criteria.mdx)

## 4. Configuration model

Harbor configuration is easiest to understand as three layers.

### Layer A: task-owned `task.toml`

The task author controls benchmark semantics and requirements:

- package metadata and arbitrary metadata;
- agent and verifier base timeouts/users;
- agent/verifier network policies;
- environment image/build/resources/OS;
- environment variables;
- healthchecks;
- MCP servers;
- artifacts and collect hooks;
- shared/separate verifier behavior;
- multi-step task definitions and reward strategy.

This layer travels with the task and should remain stable across evaluators.

### Layer B: run-owned job or trial config

A `JobConfig` chooses how to execute tasks:

- job name/output directory;
- attempts and concurrency;
- retries;
- agent(s), model(s), kwargs, environment variables, skills, and MCP config;
- runtime environment provider and provider kwargs;
- runtime resource enforcement/overrides;
- verifier override/import path and environment variables;
- dataset/task sources and filters;
- job-level artifacts and metrics;
- timeout multipliers.

A `TrialConfig` is the corresponding single-task/single-agent execution unit.

### Layer C: CLI overrides

CLI flags mutate or replace fields loaded from `-c`. Explicit CLI values generally win. Important details from the implementation:

- `--agent` replaces the config's full `agents` list.
- Multiple `--model` values create multiple agent configs when `--agent` is supplied.
- Agent kwargs/env/skills can be merged into existing agents when no replacement agent is supplied.
- A CLI path/task/dataset source replaces configured task/dataset sources.
- `--artifact` replaces job-level artifacts rather than extending them.
- verifier/environment kwargs and env mappings update existing mappings.
- `--install-only` implies disabled verification.
- `--load-trajectory` is reserved but currently rejected as unimplemented.

Source: [`harbor run` config resolution](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/cli/jobs.py)

### 4.1 Representative current job config

```yaml
job_name: harbor-research-run
jobs_dir: jobs
n_attempts: 2
n_concurrent_trials: 8
timeout_multiplier: 1.0

retry:
  max_retries: 2
  min_wait_sec: 1
  max_wait_sec: 30

environment:
  type: docker
  force_build: false
  delete: true
  cpu_enforcement_policy: auto
  memory_enforcement_policy: auto

verifier:
  disable: false
  env:
    REWARDKIT_JUDGE: anthropic/claude-sonnet-4-6

agents:
  - name: claude-code
    model_name: anthropic/claude-sonnet-4-6
    n_concurrent: 4
    skills: []
    env:
      ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
    kwargs: {}

datasets:
  - name: terminal-bench/terminal-bench-2
    ref: latest
    task_names:
      - "*"
    n_tasks: 10

metrics:
  - type: mean

artifacts:
  - /app/output.json
```

Run it with:

```bash
harbor run -c job.yaml
```

Generate a schema-valid starting point instead of hand-authoring every field:

```bash
harbor job init --full --config-output job.yaml
harbor trial init --full --config-output trial.yaml
```

The generated config is round-trip validated through the same Pydantic model used by execution.

### 4.2 Timeouts and resources

Task timeouts are base values. Run config can apply:

- one global `timeout_multiplier`;
- phase-specific multipliers for agent execution, verification, agent setup, and environment build;
- agent/verifier override and maximum timeout values.

Task CPU and memory declarations are interpreted through runtime policies:

- `auto`
- `limit`
- `request`
- `guarantee`
- `ignore`

Runtime config can also override CPU, memory, storage, GPU, and TPU values. Provider capability validation happens before trials run when Harbor knows the provider's capabilities.

### 4.3 Network policy

Network policy has a layered model of its own:

1. Environment baseline: `public`, `no-network`, or `allowlist`.
2. Optional agent/verifier phase override.
3. Runtime host merges from CLI/run config.

A phase override that differs from the baseline requires the environment provider to support dynamic policy changes. A separate verifier environment can avoid dynamic switching by having its own baseline.

### 4.4 Environment variables and secrets

Task env values support `${VAR}` and `${VAR:-default}` templates. Harbor asks for approval before exposing host environment values unless auto-confirmed.

Sensitive agent/verifier environment values are templatized when configs are serialized. At trial finalization, Harbor also scans textual output files and replaces resolved known secret values with `[REDACTED]`. This is useful defense in depth, but it is not a complete secret scanner: binary, unreadable, unknown, or indirectly leaked values may remain.

### 4.5 Reproducibility artifacts

Each run persists both requested and resolved state:

- `config.json`: replayable configuration;
- `lock.json`: resolved, content-addressed inputs;
- `result.json`: outcomes, timings, usage, errors, and rewards.

Locks include:

- Harbor package/version/Git metadata;
- task content digest and source identity;
- agent config;
- skill names, sources, Git commits, and digests;
- environment and verifier config;
- extra instruction and compose-file digests;
- concurrency/retry behavior;
- source-trial provenance for regrades.

Task equality is content-digest based, not merely path based. This is a notably strong reproducibility design.

Source: [Job/trial lock models](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/models/job/lock.py)

## 5. Result and artifact layout

A completed single-step job looks approximately like:

```text
jobs/<job-name>/
├── config.json
├── lock.json
├── result.json
├── job.log
├── <trial-name>/
│   ├── config.json
│   ├── lock.json
│   ├── result.json
│   ├── trial.log
│   ├── agent/
│   │   ├── trajectory.json
│   │   ├── recording.cast
│   │   └── ...
│   ├── verifier/
│   │   ├── reward.json or reward.txt
│   │   ├── reward-details.json
│   │   ├── test-stdout.txt
│   │   └── test-stderr.txt
│   └── artifacts/
│       ├── manifest.json
│       └── ...
└── ...
```

Multi-step trials move agent/verifier/artifact outputs under `steps/<step-name>/`.

Artifacts are collected from:

- `/logs/artifacts` by convention, without explicit config;
- arbitrary declared paths;
- Docker Compose sidecars;
- collect-hook output generated just before teardown.

Collection is best-effort and writes status into `manifest.json`; collection failure does not itself fail a trial. Separate verification and regrade rely on the artifact record, so failed/skipped artifacts can still make those operations impossible.

Harbor's local viewer (`harbor view jobs`) can inspect job/trial results, trajectories, timing, token usage, verifier output, rewards, and artifacts. Harbor Hub provides hosted storage, comparison, sharing, and leaderboard workflows.

Sources:

- [Run evals/results](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/run-jobs/run-evals.mdx)
- [Artifact collection](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/run-jobs/results-and-artifacts.mdx)
- [Hub](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/hub/index.mdx)

## 6. Tracing and trajectories

### 6.1 Harbor tracing is post-run trajectory capture

Harbor does not primarily wrap model SDK calls at the harness level. Instead, each integrated agent is responsible for producing or converting its native logs into:

```text
<current trial>/agent/trajectory.json
```

The common format is ATIF. After the run, Harbor can:

- render the trajectory in its viewer;
- derive aggregate token/cost fields for `TrialResult`;
- grade the trajectory with Rewardkit;
- turn it into an SFT dataset;
- convert it to OTel spans.

This architecture has an important consequence: **trace completeness depends on the agent adapter**. If an agent does not emit ATIF, Harbor still runs and scores it, but standard trajectory visualization/export is unavailable.

### 6.2 Agent Trajectory Interchange Format (ATIF)

At the inspected commit, the active version is `ATIF-v1.7`.

A trajectory root contains:

- `schema_version`;
- optional run-scoped `session_id`;
- optional document-scoped `trajectory_id`;
- agent name/version/default model and optional tool definitions;
- ordered `steps`;
- optional notes and arbitrary `extra` metadata;
- optional aggregate `final_metrics`;
- optional continuation reference;
- optional embedded subagent trajectories.

A step can contain:

- sequential `step_id`;
- timestamp;
- `system`, `user`, or `agent` source;
- model name;
- message, including text/image content parts;
- reasoning content/effort;
- structured tool calls;
- observations correlated by tool-call ID;
- prompt/completion/cache tokens and cost;
- prompt/completion token IDs and log probabilities;
- an LLM call count;
- copied-context marking;
- extensible metadata.

ATIF v1.7 distinguishes:

- `session_id`: logical run identity, which may be shared;
- `trajectory_id`: unique trajectory-document identity used to resolve embedded subagent references.

It also defines:

- `llm_call_count = 0` for deterministic non-LLM dispatch;
- `llm_call_count > 1` for an aggregated multi-inference step;
- `is_copied_context = true` so SFT consumers can exclude duplicated context;
- a `context_management` convention for compaction/pruning boundaries;
- embedded or external-file subagent trajectories.

Sources:

- [ATIF documentation](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/agents/trajectory-format.mdx)
- [ATIF RFC](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/rfcs/0001-trajectory-format.md)
- [Pydantic models](https://github.com/harbor-framework/harbor/tree/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/models/trajectories)

### 6.3 ATIF validation

Harbor provides Pydantic models and a validator that checks, among other things:

- required fields and types;
- sequential step IDs starting at 1;
- ISO timestamps;
- agent-only fields appearing on valid sources;
- tool call and observation references;
- embedded subagent identity rules.

```bash
python -m harbor.utils.trajectory_validator trajectory.json
```

### 6.4 Hugging Face/SFT export

The default `harbor traces export` format converts ATIF into conversational rows in a Hugging Face `datasets.Dataset`:

```bash
harbor traces export \
  --path jobs/my-job \
  --recursive \
  --episodes last \
  --filter success \
  --sharegpt \
  --instruction-metadata \
  --verifier-metadata
```

Rows can include:

- OpenAI-style conversations;
- optional ShareGPT conversations;
- agent/model/provider;
- date, task, trial, run, and episode identity;
- result bucket;
- tool definitions;
- optional instruction and verifier output;
- subagent trace source.

Exports can be pushed to Hugging Face Hub or produced programmatically and written as Parquet. Multimodal input is rejected by the text-only path rather than silently losing images.

Source: [SFT export](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/content/docs/training-workflows/sft.mdx) and [`traces_utils.py`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/utils/traces_utils.py)

### 6.5 OpenTelemetry export

`harbor-atif2otel` is a separate package that converts ATIF into OTel protobuf `ResourceSpans` with OpenInference-style attributes.

Install and export:

```bash
pip install harbor-atif2otel

harbor traces export \
  --path jobs/my-job \
  --format otel \
  --output traces.jsonl \
  --encoding json
```

The mapping is:

| ATIF | OTel/OpenInference |
|---|---|
| trajectory | root `AGENT` span |
| multi-turn conversational turn | nested `AGENT` span |
| agent step | `LLM` span |
| tool call | `TOOL` span, sibling of the LLM span |
| context-management system step | `CHAIN` span |
| subagent | nested agent span tree |

Selected attributes include:

- `openinference.span.kind`;
- `session.id` and `trajectory.id`;
- `agent.name` and `agent.version`;
- `llm.model_name`;
- prompt/completion/cache token counts;
- total and per-step cost;
- `tool.name`;
- serialized input/output;
- reasoning content.

Conversion behavior worth noting:

- trace and span IDs are deterministic hashes;
- copied-context steps are filtered;
- `llm_call_count = 0` emits tools without an LLM span;
- large string attributes are truncated;
- image content becomes textual metadata rather than image payloads;
- missing timestamps become Unix nanoseconds `0`;
- all generated spans currently receive OTel `STATUS_CODE_OK`.

Sources:

- [`harbor-atif2otel` README](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/packages/harbor-atif2otel/README.md)
- [Converter](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/packages/harbor-atif2otel/src/harbor_atif2otel/convert.py)
- [Export orchestration](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/packages/harbor-atif2otel/src/harbor_atif2otel/export.py)

### 6.6 Streaming and batch OTel plugin

The package registers the `atif2otel` entry point under the `harbor.plugins` group. It supports:

- streaming each completed trial in `on_trial_ended`;
- batch export after the job ends;
- file output, endpoint upload, or both.

```bash
harbor run ... \
  --plugin atif2otel \
  --plugin-kwarg output_dir=./otel-traces \
  --plugin-kwarg encoding=json
```

The plugin intentionally catches export/upload failures and logs warnings so observability failure does not fail the eval job.

## 7. Job plugins and external observability

A plugin implements:

```python
class JobPlugin(Protocol):
    async def on_job_start(self, job: Job) -> None: ...
    async def on_job_end(self, job_result: JobResult) -> None: ...
```

During `on_job_start`, the plugin can subscribe to the job's trial lifecycle hooks. Plugins are discovered via Python entry points in the `harbor.plugins` group or loaded with a full `module:Class` path.

```bash
harbor plugins list
harbor run ... --plugin package.module:PluginClass
```

Current CLI behavior:

- `--plugin` is repeatable.
- `--plugin-kwarg` requires exactly one plugin because kwargs are not scoped per repeated plugin.
- Plugins in job YAML/JSON are deprecated and ignored; plugins should be supplied on the CLI.
- Plugin `on_job_end` failures are logged rather than re-raised.

Sources:

- [Plugin protocol](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/models/job/plugin.py)
- [Plugin attachment](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/cli/job_plugins.py)
- [Entry-point registry](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/src/harbor/cli/plugin_registry.py)

### LangSmith plugin behavior

The `harbor-langsmith` package is more than trace upload. It maps Harbor's eval model into LangSmith:

1. Optionally creates or finds a LangSmith dataset.
2. Upserts one example per Harbor task, including instruction and task identity.
3. Creates or reuses an experiment/session.
4. Creates one root chain run per trial.
5. Creates child phase runs for environment, agent, and verifier phases.
6. Emits a synthetic LLM child with token usage so LangSmith rolls usage up correctly.
7. Ends runs with rewards, output, cost, and errors.
8. Publishes every reward dimension as feedback.
9. Makes parent context available to in-process custom agents so their native LangSmith traces can nest beneath the Harbor trial.

This plugin is the clearest upstream precedent for a full Braintrust integration.

Source: [`LangSmithPlugin`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/packages/harbor-langsmith/src/harbor_langsmith/plugin.py)

## 8. Braintrust integration analysis

The Harbor repository has no Braintrust plugin at the inspected commit. Braintrust appears only as an OTel reference in the ATIF RFC.

### 8.1 Why the existing Harbor OTel endpoint flag is not enough

Although the CLI says it can upload to an OTLP endpoint, it constructs `MlflowProtobufUploader`. That uploader:

1. calls MLflow experiment search/create REST APIs;
2. POSTs spans to `<endpoint>/v1/traces`;
3. adds `x-mlflow-experiment-id` and `X-Mlflow-Workspace` headers.

Braintrust's OTel endpoint is instead:

```text
https://api.braintrust.dev/otel/v1/traces
```

with headers such as:

```text
Authorization: Bearer <BRAINTRUST_API_KEY>
x-bt-parent: project_name:<project>
```

Therefore, simply setting Harbor's `--endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` to Braintrust is expected to fail during the MLflow experiment lookup/create flow, even though the serialized trace payload itself is OTLP protobuf.

Relevant code:

- Harbor's [MLflow uploader](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/packages/harbor-atif2otel/src/harbor_atif2otel/uploaders/mlflow_protobuf.py)
- Braintrust's local [`OtelExporter`](../py/src/braintrust/otel/__init__.py)

### 8.2 Trace-only integration option

A narrow integration could implement `harbor_atif2otel.uploaders.base.Uploader` and POST `ExportTraceServiceRequest` bytes directly to Braintrust with the correct endpoint and headers.

Advantages:

- minimal Harbor changes;
- preserves ATIF's nested LLM/tool/subagent structure;
- uses the existing OpenInference attributes;
- can support both backfill and streaming.

Limitations in the current ATIF-to-OTel export:

- `result.json` is read only for success/failure filtering, not attached to spans;
- verifier reward dimensions are absent;
- task, dataset, job, attempt, and artifact metadata are absent;
- failures still produce `STATUS_CODE_OK` spans;
- trials with no ATIF have no exported trace;
- multi-reward filtering assumes a key named `reward` and treats its absence as zero;
- the exported root is the ATIF agent trajectory, not a complete environment/agent/verifier lifecycle trace.

A trace-only uploader is useful, but it would not provide a complete Braintrust eval experience.

### 8.3 Recommended integration: `harbor-braintrust` job plugin

The stronger design is a dedicated package registered as:

```toml
[project.entry-points."harbor.plugins"]
braintrust = "harbor_braintrust:BraintrustPlugin"
```

Conceptual CLI:

```bash
pip install harbor-braintrust
export BRAINTRUST_API_KEY=...

harbor run ... \
  --plugin braintrust \
  --plugin-kwarg project_name=harbor-evals \
  --plugin-kwarg experiment_name=my-run
```

Recommended mapping:

| Harbor | Braintrust |
|---|---|
| Job | Experiment/run grouping |
| Dataset | Braintrust dataset, optionally synchronized |
| Task | Dataset record/example |
| Trial | Root eval span |
| Instruction | Span/eval input |
| Final agent response and metadata | Span output |
| Environment, agent, verifier phases | Child spans |
| ATIF agent/LLM/tool/subagent tree | Nested child spans imported after the agent run |
| `verifier_result.rewards` | Scores, one per reward key |
| Exception | Error/status metadata |
| token/cache/cost totals | Metrics |
| config and lock | Metadata/provenance, with size/redaction controls |
| task/artifact digests | Reproducibility metadata |

The implementation can follow `LangSmithPlugin`:

1. Initialize the target project/experiment during `on_job_start`.
2. Optionally synchronize Harbor tasks into a Braintrust dataset.
3. Subscribe to trial start, environment start, agent start/end, verification start, end, and cancel hooks.
4. Create stable trial identity from Harbor's job/trial UUIDs.
5. Log phase timing and runtime configuration.
6. At trial end, log all reward dimensions as scores and attach errors and usage.
7. Parse `agent/trajectory.json` and attach its detailed child spans.
8. Flush on job end without making observability failure invalidate the Harbor run by default; offer `fail_fast` for CI.

### 8.4 Hybrid design details

A production-grade plugin should address several details not solved by the generic converter today:

- **Parenting:** allow ATIF conversion to accept an existing trace ID and parent span ID, or map ATIF directly into Braintrust child spans.
- **Lifecycle coverage:** preserve Harbor phase spans even when no ATIF is available.
- **Status:** derive root status from `exception_info`, verifier availability, and configured success semantics.
- **Scores:** attach every numeric reward key, not only a conventional `reward` key.
- **Inputs/outputs:** read task instruction and best available final agent response; do not assume ATIF always has one simple textual final answer.
- **Metadata size:** avoid blindly attaching the entire Pydantic config, raw verifier logs, or large artifacts.
- **Secrets:** rely on Harbor's serialized/redacted config and add a plugin-side metadata allowlist.
- **Idempotency:** use stable IDs so resumed jobs and plugin retries do not duplicate records.
- **Regrades:** preserve source-trial provenance and distinguish newly computed scores from original execution cost.
- **Multi-step trials:** emit one child span and score set per step, then the configured trial-level reward strategy.
- **Subagents/continuations:** support both ATIF v1.7 embedded references and external continuation files.
- **Custom agents:** accept any valid ATIF producer rather than checking only Harbor's built-in agent enum.

### 8.5 Suggested implementation sequence

1. Build a standalone ATIF → Braintrust/OTLP uploader PoC using checked-in Harbor golden trajectories.
2. Add trial result enrichment: task/job metadata, rewards, errors, and status.
3. Implement a Harbor `BraintrustPlugin` with lifecycle spans and score logging.
4. Add optional dataset synchronization.
5. Add streaming and batch/backfill modes.
6. Validate single-step, multi-step, failed, cancelled, retry, regrade, missing-trajectory, and multi-reward cases.
7. Upstream generic improvements to `harbor-atif2otel` where possible, especially pluggable endpoint upload and parent/status/resource metadata.

## 9. Important gaps and documentation/code mismatches

These findings are based on the inspected commit and may change quickly.

### 9.1 OTel conversion is generic; direct upload is MLflow-specific

The package README describes “any OTel-compatible backend,” but the bundled direct uploader and CLI endpoint path are specifically MLflow-aware. File export is backend-neutral; endpoint upload is not.

### 9.2 OTel output omits eval outcomes

The exporter reads `result.json` for filtering but does not place rewards, task identity, exceptions, or Harbor lifecycle timing onto spans. It also marks every converted span OK. An observability backend receives an agent trace, not a complete eval record.

### 9.3 Multi-reward success filtering assumes `reward`

OTel filtering looks for `verifier_result.rewards["reward"]`, defaulting to zero. A valid multi-dimensional verifier with only `correctness` and `quality` will be classified as failure for filtering purposes.

### 9.4 Custom ATIF agents may fail Hugging Face export

The SFT documentation says any ATIF-producing agent is supported, but the exporter currently coerces the agent name into the built-in `AgentName` enum and checks `AgentFactory.SUPPORTS_ATIF`. A custom import-path agent that emits valid ATIF may fail before export unless registered as a built-in name.

### 9.5 Some SFT documentation still describes legacy episode files

The SFT page describes rows as `agent/episode-*` plus `debug.json`/`response.json`, while the current utility says ATIF is preferred and discovers trials through `agent/trajectory.json`. This appears to be partially stale documentation from an older trace layout.

### 9.6 Plugin config documentation has drift

`harbor-langsmith` says plugin kwargs can come from job config, but current `JobConfig` migration explicitly removes and ignores a `plugins` key and tells users to use CLI `--plugin`. Treat CLI plugin configuration as authoritative.

### 9.7 Existing example config uses deprecated `orchestrator`

`examples/configs/features/job.yaml` still has an `orchestrator` block. The Pydantic model migrates it to top-level `n_concurrent_trials`, `quiet`, and `retry` with a deprecation warning. New configs should use top-level fields.

### 9.8 Separate verification is recommended but not yet the default

The regrade docs say separate mode is recommended and may become the default. Current resolution remains shared when neither `environment_mode` nor `[verifier.environment]` is set.

### 9.9 OTel command naming is inconsistent in one package note

One sequence document says `harbor trace export`; the actual Typer command is `harbor traces export`.

### 9.10 Timestamps are optional in ATIF but important in OTel

The converter maps missing timestamps to zero. Agents that omit timestamps can therefore generate technically serialized but poorly timed traces.

## 10. How the documentation website works

The live docs source is in the repository's `docs/` directory. It is a separate web app built with:

- Next.js 16 App Router;
- React 19;
- Fumadocs MDX/core/UI;
- Tailwind CSS 4;
- Bun;
- Vercel deployment.

The content flow is:

```text
docs/content/docs/**/*.mdx
        + meta.json navigation files
        ↓ fumadocs-mdx postinstall generation
      docs/.source/
        ↓ Fumadocs loader, base URL /docs
Next.js dynamic [[...slug]] route
        ↓
rendered docs + TOC + generated metadata/OG routes
```

Key mechanics:

- `docs/source.config.ts` defines the MDX collection and stores processed Markdown.
- `docs/src/lib/source.ts` loads content at `/docs`, applies the Lucide icon plugin, and exposes text extraction.
- `docs/src/app/docs/[[...slug]]/page.tsx` resolves a page, renders MDX, generates static params, and generates page metadata.
- `meta.json` files control ordering and nested navigation.
- `/llms-full.txt` concatenates processed Markdown for every page into an LLM-friendly endpoint.
- `next.config.mjs` contains redirects from old documentation routes and sends `/registry` to Harbor Hub.
- Vercel deploys `main`; a GitHub workflow creates docs previews from the `docs/` working directory.
- `docs-mintlify/` also exists, but the current documented and deployed app is the Next.js/Fumadocs app under `docs/`.

Relevant files:

- [`docs/package.json`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/package.json)
- [`source.config.ts`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/source.config.ts)
- [`source.ts`](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/src/lib/source.ts)
- [Dynamic docs page](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/src/app/docs/%5B%5B...slug%5D%5D/page.tsx)
- [`llms-full.txt` route](https://github.com/harbor-framework/harbor/blob/e76f7e32f5644fb9f648cd23151aac5c67492ea0/docs/src/app/llms-full.txt/route.ts)

Local docs development:

```bash
cd docs
bun install
bun dev
```

## 11. Practical command reference

### Install and inspect

```bash
uv tool install harbor
harbor --help
harbor run --help
harbor dataset list
harbor plugins list
```

### Run one local task or dataset

```bash
harbor run -p ./my-task -a claude-code -m anthropic/claude-sonnet-4-6
harbor run -p ./my-dataset -a claude-code -m anthropic/claude-sonnet-4-6 -n 8
```

### Run a published dataset

```bash
harbor run -d "org/dataset@latest" -a claude-code -m anthropic/claude-sonnet-4-6
```

### Generate configuration

```bash
harbor job init --full --config-output job.yaml
harbor trial init --full --config-output trial.yaml
harbor run -c job.yaml --print-config
```

### Inspect results

```bash
harbor view jobs
```

### Regrade

```bash
harbor job regrade jobs/<job> -p ./updated-task
harbor trial regrade jobs/<job>/<trial> -p ./updated-task
```

### Export SFT conversations

```bash
harbor traces export \
  -p jobs/<job> \
  --episodes last \
  --filter success \
  --sharegpt
```

### Export OTel files

```bash
pip install harbor-atif2otel
harbor traces export \
  -p jobs/<job> \
  --format otel \
  --encoding json \
  --output harbor-traces.jsonl
```

### Stream or batch through a plugin

```bash
harbor run ... \
  --plugin atif2otel \
  --plugin-kwarg output_dir=./otel-traces \
  --plugin-kwarg mode=batch
```

## 12. Bottom line

Harbor's strongest ideas are:

- tasks package execution and grading together;
- trials are reproducible, content-locked rollouts;
- rewards are intentionally simple numeric dictionaries;
- custom metrics and Rewardkit handle richer scoring;
- artifacts make isolated verification and regrading possible;
- ATIF gives diverse agents a common post-run trajectory format;
- plugins are the extension point for hosted eval/observability systems.

For Braintrust, the most valuable integration is not merely “OTLP export.” It is a **Harbor-aware eval plugin** that combines:

1. Harbor job/task/trial identity and reproducibility metadata;
2. complete lifecycle timing;
3. verifier rewards as Braintrust scores;
4. token/cost/error data;
5. ATIF's detailed LLM/tool/subagent trace tree;
6. dataset synchronization and regrade provenance.

That hybrid would preserve both sides of Harbor's model: **the eval record** and **the agent trajectory**.
