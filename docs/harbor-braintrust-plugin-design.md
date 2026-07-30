# Design: Braintrust Plugin for Harbor

Status: proposal
Harbor: [`harbor==0.20.0` at `e76f7e3`](https://github.com/harbor-framework/harbor/tree/e76f7e32f5644fb9f648cd23151aac5c67492ea0)
Background: [`harbor-framework-research.md`](harbor-framework-research.md)
Contract: [`braintrustdata/braintrust-spec`](https://github.com/braintrustdata/braintrust-spec/tree/main/skills/instrumentation-spec), especially its [instrumentation guide](https://github.com/braintrustdata/braintrust-spec/blob/main/skills/instrumentation-spec/references/instrumentation-guide.md) and [eval-span spec](https://github.com/braintrustdata/braintrust-spec/blob/main/skills/instrumentation-spec/references/features/eval-spans.md)

## Summary

Build a native Harbor job plugin, not an OTLP adapter. Harbor remains responsible for execution, retries, sandboxing, and verification; Braintrust provides datasets, experiments, comparisons, and trace analysis.

```text
Harbor dataset/task selection  → Braintrust Dataset
Harbor eval group              → Braintrust Experiment
Harbor final trial             → Braintrust root eval span
Harbor verifier rewards        → Braintrust scores or metrics
Harbor verifier labels         → Braintrust classifications
Harbor lifecycle + ATIF        → child spans
Harbor JobResult               → final reconciliation + job summary
```

### Key decisions

1. Construct eval traces with Braintrust SDK primitives; do not call `braintrust.Eval()`. Harbor already owns the eval loop.
2. Within each Harbor job, create one experiment per dataset and semantic system variant. Resume and backfill update those same experiments rather than creating new ones.
3. Sync the exact resolved task set to a Braintrust dataset by default.
4. Treat Harbor rewards as authoritative. Only values known to be normalized and higher-is-better become scores; other numeric values become metrics. Always retain raw rewards.
5. Make the trial the root `eval` span. Import lifecycle and conforming ATIF spans beneath its canonical `task` child.
6. Route hook events through explicit job and per-trial state machines. Reconcile from the final `JobResult`; retried failures must not become experiment rows.
7. Use deterministic IDs and names so resume and backfill converge without duplicates.
8. Keep Braintrust credentials on the host. ATIF import does not require sandbox credentials.
9. Isolate plugin failures by default; `strict=True` is opt-in.
10. Preserve user metadata under a plugin-owned namespace after normalization and redaction.

## Scope

The plugin should:

- log every final Harbor trial as one Braintrust eval row;
- support multi-dataset, multi-agent, multi-model, repeated-attempt, resume, and regrade jobs;
- preserve all verifier rewards without clamping or silent coercion;
- associate rows with stable task records and dataset versions;
- import useful lifecycle and ATIF detail when available;
- support online use and offline backfill from a Harbor job directory;
- avoid blocking Harbor's event loop or changing benchmark behavior.

Version 1 does not need to:

- replace Harbor's verifier or aggregate metric system;
- reproduce arbitrary Harbor `metric.py` or pass@k logic in Braintrust summaries;
- inject credentials or tracing SDKs into agent containers;
- upload arbitrary workspace artifacts;
- deduplicate separately instrumented native agent traces;
- depend on Harbor's MLflow-specific OTel uploader.

## Data model

| Harbor | Braintrust | Notes |
|---|---|---|
| Job | Shared metadata + optional project-log summary | A job may create several experiments. |
| Resolved dataset/task set | Dataset | Scope is the exact logical task selection. |
| Task | Dataset record | Deterministic ID and canonical task input. |
| Eval group/variant | Experiment | Partition by dataset and semantic agent config. |
| Final trial | Root `eval` span named `eval` | One experiment row. |
| Trial execution | Direct child `task` span named `task` | Canonical eval wrapper. |
| Setup, agent, verifier phases | Nested `task` spans | Use Harbor's recorded timestamps. |
| Identifiable model call | `llm` span | Only if it satisfies the instrumentation contract. |
| Tool execution | `tool` span | Correlate call arguments with its observation. |
| Ambiguous ATIF step/subagent | `task` span/tree | Never mislabel partial data as an LLM/tool call. |
| Normalized reward | Direct score child with `purpose="scorer"` | Score must be in `[0, 1]` and higher-is-better. |
| Arbitrary numeric reward | Eval-root metric | Preserve original value in metadata too. |
| Structured verifier label | Root classification + classifier child | Do not coerce labels to numbers. |
| Exception | Error on root and canonical task | Omit output on both. |
| Regrade | New experiment | Link the source experiment as base when exact. |

## Identity and partitioning

### Experiment partitions

A Harbor job may combine datasets and system variants. Mixing all rows into one experiment would blend score summaries and weaken comparisons. Partition each job by:

```text
(dataset identity, normalized agent semantic config, resolved skill digests)
```

The semantic config includes agent name/import path, model, kwargs, MCP configuration, resume behavior, non-secret environment values, sensitive-value templates, and resolved skill digests. It excludes execution controls such as concurrency, logging, output paths, and retry policy.

Prefer the resolved Harbor lock over raw config because it includes resolved task and skill content. Never include secrets in a fingerprint or name.

```text
partition_key = sha256(dataset_key + normalized_semantic_config + skill_digests)
experiment    = <job> · <agent>@<model> · <dataset> · <hash-8>
```

Include the Harbor job ID in the deterministic internal name so a reused display name cannot update an unrelated experiment. Initialize with `update=True` to support resume and backfill.

### Stable IDs

```text
dataset record ID = UUIDv5(plugin namespace, dataset scope + logical task key)
eval root ID      = Harbor TrialResult.id
child span ID     = UUIDv5(trial ID, semantic child path)
```

Examples of semantic paths:

```text
task/environment_setup
task/agent_execution/turn/0/llm/2
task/agent_execution/tool/call_123
task/verification
scorer/reward
```

IDs must depend on semantics, not import time or traversal order where a stable source identifier exists.

## Datasets

Dataset sync is enabled by default because it enables task-level comparison, provenance, and reruns.

### Modes

| Mode | Behavior |
|---|---|
| `sync` | Upsert a managed dataset and associate its version with each experiment. Default. |
| `none` | Log stable inputs without creating a dataset. |
| `existing` | Map to a user-provided dataset. Deferred until record matching is specified. |

### Dataset scope and records

Use one dataset per Harbor source and exact logical task selection:

```text
harbor · <source> · tasks-<sorted-logical-task-keys-hash-8>
```

Hash logical keys, not content. A content change then creates a new version of the same logical dataset; a different selected subset uses a different dataset and cannot prune another run's records.

Choose a logical task key in this order:

1. published package identity;
2. Git repository plus relative path;
3. Harbor task name plus source;
4. normalized local relative identity.

Do not expose absolute paths.

The record input contains task-authored semantics, not run-specific agent settings:

```json
{"task": "terminal-bench/example", "instruction": "..."}
```

For multi-step tasks:

```json
{"task": "org/task", "steps": [{"name": "build", "instruction": "..."}]}
```

Harbor has no universal expected output. Keep `expected` explicitly `null` unless a configured adapter exposes a safe expected value. Never use solution or verifier implementation files as expected output.

Record metadata should include stable source identity, task version/digest, schema version, resource requirements, and normalized user task metadata. Exclude secrets, absolute paths, full Compose files, solutions, verifiers, and arbitrary artifacts.

### Sync sequence

1. Initialize the dataset with `use_output=False`.
2. Upsert all resolved records with deterministic IDs.
3. Flush and fetch rows to resolve dataset version and record transaction metadata.
4. Initialize each experiment with its dataset.
5. Add dataset origin to each root eval span using the same shape as Braintrust's eval runner.

Hide origin plumbing behind an adapter and test it against the supported Braintrust SDK version.

## Eval trace

### Required shape

```text
eval                                      [eval]
├── task                                  [task]
│   ├── environment_setup                 [task]
│   ├── agent_setup                       [task]
│   ├── agent_execution                   [task]
│   │   └── ordered ATIF task/llm/tool tree
│   └── verification                      [task]
├── reward                                [score, purpose=scorer]
└── category                              [classifier, purpose=scorer]
```

For multi-step trials, add `step:<name>` task spans under the canonical task. Keep score/classifier spans direct children of the eval root.

Create spans with explicit parent objects, `set_current=False`, and recorded timestamps. Harbor runs trials concurrently, so do not rely on context-manager parenting.

Every span should carry Braintrust span-origin context identifying the Harbor integration and plugin version. Use the reserved `braintrust.plugin.harbor` identity only when Braintrust owns the plugin.

### Root and canonical task

The root and direct `task` child must have identical:

- canonical dataset input;
- expected value, including explicit JSON null;
- bounded final output, or the same error with output omitted.

Choose output in this order:

1. a standardized answer from agent metadata;
2. the final non-copied ATIF agent message;
3. per-step final messages;
4. a small completion/status object.

Do not put a full trajectory or artifact manifest in output. Put actual run-specific instructions on `agent_execution.input` and in metadata, not in dataset input.

If `TrialResult.exception_info` exists, log `<type>: <message>` as the root and task error. Tracebacks are opt-in metadata or attachments. A missing reward without an exception is not a zero; record it as an unevaluated warning.

### Metadata

Use a collision-safe namespace:

```json
{
  "harbor": {
    "job_id": "...",
    "trial_id": "...",
    "task_name": "...",
    "agent": "...",
    "model": "...",
    "attempt_index": 0,
    "retry_index": 0,
    "raw_rewards": {},
    "trajectory": {"present": true, "schema_version": "ATIF-v1.7"},
    "custom": {"job": {}, "task": {}, "trial": {}}
  }
}
```

Preserve explicitly supplied user metadata at the narrowest matching scope:

- job metadata → experiment and eval root;
- task metadata → dataset record and eval root;
- trial metadata → eval root;
- permitted ATIF root extras → eval-root trajectory metadata.

Normalize JSON consistently across live sync and backfill. Preserve keys, nesting, types, and explicit nulls, but apply secret-key filtering, regex redaction, path filtering, and depth/size limits. Record dropped paths as warnings. Never merge user fields into plugin-owned keys or copy arbitrary metadata to conforming LLM/tool leaves.

## Rewards and classifications

Harbor rewards are arbitrary numbers. Braintrust scores are normalized, higher-is-better values in `[0, 1]`; Braintrust metrics may be arbitrary numbers. Range alone does not establish score semantics.

### Reward classification

For each reward key, apply this precedence:

1. Exact `reward_rules` entry.
2. Configured `score_keys` or `metric_keys` glob. Reject overlaps at initialization.
3. The conventional key `reward` is a score only when it is in `[0, 1]`.
4. An invalid configured score follows `invalid_score_policy` (`metric` by default).
5. Every other numeric reward is a metric, even if its value happens to be in `[0, 1]`.
6. Preserve the complete original dictionary in `metadata.harbor.raw_rewards`.

Never clamp by default or infer direction from names. Exact rules may define explicit normalization:

```json
{
  "correctness": {"type": "score", "direction": "maximize"},
  "error_rate": {
    "type": "score",
    "direction": "minimize",
    "min": 0,
    "max": 1,
    "score_name": "error_rate_score"
  },
  "runtime_sec": {"type": "metric"}
}
```

For finite configured bounds:

```text
maximize = (value - min) / (max - min)
minimize = (max - value) / (max - min)
```

Keep a transformed source value as `harbor_reward.raw.<key>` metric as well as in raw metadata. If a Harbor reward name collides with a Braintrust standard metric but has different semantics, emit it as `harbor_reward.<key>`. Read reserved metric names from the SDK/backend source of truth rather than duplicating a snapshot here.

For every score, create a direct child span with:

```text
name = score name
type = score
purpose = scorer
scores = {<name>: <normalized value>}
```

Do not duplicate scores on the eval root. Eval metrics stay on the root. Do not invent a `success` score; users may configure one explicitly.

### Classifications

Structured categorical verifier outcomes become Braintrust classifications, never numeric surrogates. Each item has a string `id`, optional `label`, and optional JSON metadata. Preserve source order and duplicates.

Create one direct `classifier` child per source classifier with `purpose="scorer"`, then log the grouped, non-empty classification dictionary on the root. Omit the root field when no valid classifications exist.

Validate atomically per source classifier: one malformed item fails that classifier only. Log its span error and add an eval-root warning; continue syncing numeric rewards and other classifiers.

Version 1 should only read labels from known Harbor adapters or explicit `classifier_rules` pointing to documented JSON paths. Do not infer labels from arbitrary metadata or files.

### Reward details and aggregates

If `reward-details.json` exists, put a bounded summary in the scorer output and optionally attach the complete JSON. Do not create criterion-level or judge-LLM spans until Rewardkit has a stable, tested mapping.

Harbor's job metrics, custom `metric.py`, and pass@k remain authoritative aggregate results. Do not create a synthetic eval row for aggregates. Optionally write one project-log trace, `harbor.job.summary`, containing exact Harbor aggregates and links to partition experiments.

## ATIF import

Default to host-side ATIF import. It works across built-in agents, supports backfill, and keeps credentials out of sandboxes.

Modes:

| Mode | Behavior |
|---|---|
| `atif` | Import conforming ATIF detail. Default. |
| `summary` | Lifecycle and aggregate trajectory metadata only. |
| `native` | Skip ATIF because the agent is instrumented elsewhere. |

Do not attempt native/ATIF deduplication in version 1.

### Conformance gate

Imported `llm` and `tool` leaves are Braintrust instrumentation and must satisfy the linked instrumentation guide. This document does not duplicate that contract.

An ATIF step may be an `llm` span only when it represents exactly one model call and the converter can provide required identity, canonical input/output, timing, and tool configuration. Otherwise emit a `task` summary with a warning. In particular:

- `llm_call_count == 0` is deterministic work;
- unknown or multiple calls are not one LLM call unless ATIF exposes each call;
- a streaming call without measurable time-to-first-token is downgraded;
- redaction that removes required payload content also downgrades the leaf.

A tool span requires call arguments and a correlated result or error. Preserve call IDs across model output, tool spans, and tool-result messages. Preserve execution order: model → tool → model. Subagents become nested task trees.

Normalize non-Anthropic/Google providers to OpenAI Chat Completions payloads. Preserve Anthropic or Google native payloads only with the matching provider identity. Use provider operation names for spans, not model names. Tool definitions belong in LLM metadata, not the message list.

Never copy arbitrary ATIF/provider fields to leaf metadata. Required `model` and `provider`, request controls, canonical tool fields, approved prompt provenance, and allowed metrics are sufficient. Keep unsupported values in eval-root reconciliation metadata.

### Usage, cost, and timing

Leaf LLM spans are authoritative calls. Normalize only metrics allowed by both the Braintrust backend and instrumentation guide. Important mappings include:

```text
input/prompt tokens       → prompt_tokens
output/completion tokens  → completion_tokens
total tokens              → tokens
cache reads               → prompt_cached_tokens
cache writes              → prompt_cache_creation_tokens
reasoning tokens          → completion_reasoning_tokens
first-token milliseconds  → time_to_first_token (seconds)
per-call cost             → estimated_cost
```

Do not emit aliases together or add subset counts to totals. Omit unknown values rather than fabricating zero. Counts must be non-negative integers; costs and durations must be finite and non-negative.

Use explicit span timestamps rather than duplicating duration metrics. Exact aggregate usage may appear on `agent_execution`; otherwise keep it in eval-root Harbor metadata. Keep Harbor totals there for reconciliation even when leaf detail exists.

For missing ATIF timestamps, use valid values within the agent phase, clamp outliers, and interpolate missing values monotonically. Record repairs on the eval root, not on conforming leaves. Never emit epoch timestamps as a fallback.

### Content policy

| Mode | Captured |
|---|---|
| `metadata` | Structure, timing, usage, and scores; detailed leaves become task summaries. |
| `messages` | Canonical model messages and tool inputs/results. Default. |
| `full` | `messages` plus fields explicitly allowed by the instrumentation contract. |

All modes support byte limits, sensitive-key filtering, regex redaction, and reasoning exclusion. A policy must not leave an `llm` or `tool` label on a materially incomplete payload.

Convert inline media to Braintrust attachment references in place. If conversion/upload fails, preserve the original payload unless privacy policy requires removal; in that case downgrade the leaf rather than partially rewriting it.

## Hook state machines, retries, and resume

Use a small reducer-based state machine rather than independent hook callbacks mutating shared dictionaries. Maintain one job machine and one trial machine per stable logical trial identity. Use `trial_name` only if Harbor guarantees that it is unique within the job; otherwise derive the identity from the resolved trial plan.

### Job machine

```text
NEW → INITIALIZING → ACTIVE → RECONCILING → CLOSED
any nonterminal state ──unrecoverable plugin error──→ DISABLED | FAILED
```

Unrecoverable initialization or dispatcher failures transition to `DISABLED` by default and preserve diagnostics. A single malformed event or effect records a warning and leaves the machine active. In strict mode, unrecoverable failures transition to `FAILED` and raise where Harbor permits. `on_job_end` first moves the machine to `RECONCILING`, stops accepting live events, drains in-flight effects, and then closes it.

### Trial machine

`n_attempts` creates separate logical trial machines and experiment rows. Execution retries are successive attempts within one machine and produce only one final row.

```text
PENDING ──START──→ ACTIVE(phase=started, retry=n)
ACTIVE  ──phase event──→ ACTIVE(next phase)
ACTIVE  ──END, retry predicted──→ WAITING_RETRY
ACTIVE  ──END, otherwise────────→ FINAL_CANDIDATE
ACTIVE  ──CANCEL────────────────→ CANCELLED
WAITING_RETRY or FINAL_CANDIDATE ──START──→ ACTIVE(attempt=n+1)

any nonterminal state ──final JobResult contains trial──→ FINALIZING → SYNCED
any nonterminal state ──final JobResult omits trial─────→ OMITTED
```

The nested active phases are monotonic:

```text
started → environment → agent → agent_done → verification
```

A failure may skip phases, so `END` and `CANCEL` are legal from any active phase. Duplicate events are no-ops. A backward or otherwise illegal transition records a warning instead of raising unless strict mode is enabled.

Harbor 0.20 emits `END` before the queue decides whether to retry. The reducer may use public retry config to classify the next state, but `FINAL_CANDIDATE` is deliberately nonterminal: a later `START` invalidates the candidate and begins the next attempt.

Do not upload candidate eval rows to the main experiment by default. Stage their result references, then dispatch authoritative final-result events from `JobResult.trial_results` during reconciliation. This guarantees that Braintrust receives exactly Harbor's retained rows. Optional retry-attempt traces may be emitted to project logs because they cannot affect experiment counts.

Implement transitions as a pure function:

```text
(state, event) → (new_state, effects)
```

Effects include staging a result, recording retry/cancellation diagnostics, logging an operational attempt, syncing a final row, and closing an omitted trial. Serialize transitions per logical trial with an `asyncio.Lock`, then run I/O through an ordered per-trial effect queue outside that lock. Different trials may progress concurrently.

At reconciliation, sync all final results, including trials loaded by `harbor job resume`. Set root `metrics.retries` from the machine's completed execution attempts when known; intentional `n_attempts` remain separate rows. Persist terminal state, retry count, and warnings in `braintrust-sync.json` so backfill can resume safely.

Harbor should eventually add `retry_index` and `is_final_attempt` to trial events, or emit a dedicated final-attempt event. That would allow safe live row sync without changing the state-machine contract.

## Regrades

A regrade creates new experiments and preserves source output/trajectory. Link each partition to its exact source experiment as a Braintrust base using, in order:

1. explicit base experiment ID/mapping;
2. source IDs from `braintrust-sync.json`;
3. deterministic source experiment identity.

Never use fuzzy name matching. If no exact source exists, continue without a base and record a warning. Mark recorded source cost as not incurred by the regrade.

## Attachments, privacy, and failure handling

Default attachment policy:

- include bounded `reward-details.json`;
- include an artifact-manifest summary in metadata;
- exclude all other trial files.

Optional modes may include trajectory, redacted config/lock, or allowlisted artifacts. Enforce per-file and total limits. Attach values in semantically named output fields using Braintrust's attachment representation; do not invent attachment metadata fields. Read soon-to-be-deleted files into memory before constructing attachments.

Host-only credential rules:

- read Braintrust credentials from the Harbor host;
- never inject or persist them in agent/task config, locks, results, metadata, or manifests;
- redact secret-like keys and user-configured patterns;
- do not upload absolute paths, source, shell output, reasoning, or artifacts outside the selected content policy.

Default failures are isolated:

| Failure | Behavior |
|---|---|
| Initialization/auth | Disable sync with a clear error; raise only in strict mode. |
| Hook callback | Catch, record, continue. |
| Dataset sync | Fall back to unassociated experiment rows when possible. |
| Malformed ATIF | Keep the eval/lifecycle trace without detailed leaves. |
| Auxiliary attachment | Omit it and record a warning. |
| Flush | Record a local error for later backfill. |

Write `<job-dir>/braintrust-sync.json` with plugin version, job/project identity, experiment and dataset IDs, synced trial IDs, errors, and completion state. Never include credentials. This manifest supports diagnosis, exact regrade linking, and idempotent backfill.

## Plugin API and packaging

Suggested constructor:

```python
BraintrustPlugin(
    project_name=None,
    project_id=None,
    experiment_prefix=None,
    base_experiment_name=None,
    base_experiment_id=None,
    dataset_mode="sync",
    dataset_name=None,
    trajectory_mode="atif",
    content_mode="messages",
    include_custom_metadata=True,
    max_custom_metadata_bytes=100_000,
    score_keys=None,
    metric_keys=None,
    reward_rules=None,
    classifier_rules=None,
    invalid_score_policy="metric",
    include_tracebacks=False,
    attachments="verifier-details",
    artifact_include=None,
    max_attachment_bytes=5_000_000,
    max_total_attachment_bytes=20_000_000,
    max_content_bytes=20_000,
    log_job_summary=True,
    log_retry_attempts=False,
    strict=False,
)
```

Constructor options should have `HARBOR_BRAINTRUST_*` environment fallbacks, with precedence `explicit option > environment > default`. Standard Braintrust variables continue to configure API key, organization, and app URL. Complex rule values should use JSON because Harbor only supports `--plugin-kwarg` when one plugin is supplied. Validate mutually exclusive identifiers, overlapping reward rules, globs, bounds, byte limits, and mode combinations before performing network I/O.

Ship the integration in the Braintrust distribution under `py/src/braintrust/integrations/harbor/` and register `braintrust.integrations.harbor:BraintrustPlugin` in the `harbor.plugins` entry-point group. Harbor stays optional: importing Braintrust without Harbor installed must continue to work, and the integration must not raise Braintrust's minimum Python version. Test supported Harbor releases through a dedicated, explicitly pinned version matrix.

Keep lifecycle orchestration, compatibility access, state reduction, dataset/eval conversion, ATIF conversion, reward handling, attachments, and identity generation behind clear internal boundaries. The internal file layout is an implementation choice rather than part of this design.

Online sync and backfill must share the same normalization, partitioning, ID generation, reward classification, ATIF conversion, and persistence core. They should differ only in how Harbor events and final results are obtained. Keep the public surface minimal until the backfill and extension use cases establish which helpers need to be supported APIs.

### Harbor compatibility boundary

Prefer public job identity/config/directory, hook registration, event fields, and final `JobResult`. Isolate Harbor-version compatibility behind one internal boundary. Until Harbor exposes resolved plans publicly, that boundary may feature-detect read-only access to:

- `job._task_configs`;
- `job._task_download_results`;
- `job._trial_configs`.

Do not mutate private fields or depend on queue, metrics, progress, existing-trial, or lock internals. Prefer persisted `lock.json` once available and pin every tested Harbor version.

## Lifecycle

### `on_job_start`

1. Dispatch job `INITIALIZE`.
2. Validate config/auth and read resolved tasks, lock, and custom metadata.
3. Normalize metadata, build partitions, sync datasets, and initialize experiments.
4. Register all trial hooks with one thin dispatcher.
5. Dispatch job `READY`; optionally start a project-log job trace.

Keep blocking SDK and filesystem work off Harbor's event loop. Preserve per-trial effect ordering while allowing independent trials to make progress concurrently.

### Hook dispatcher

Subscribe to `START`, `ENVIRONMENT_START`, `AGENT_START`, `AGENT_END`, `VERIFICATION_START`, `END`, and `CANCEL`. Each callback only converts Harbor data to a typed state-machine event and dispatches it. It must not contain retry, synchronization, or transition logic.

The dispatcher catches callback errors because Harbor awaits hooks. State transitions remain cheap and synchronous; generated I/O effects run outside the per-trial lock.

### `on_job_end`

1. Dispatch job `RECONCILE` and drain in-flight hook effects.
2. Dispatch one authoritative `FINAL_RESULT` event for every `JobResult.trial_results` entry.
3. Mark remaining nonterminal trial machines `OMITTED` and drain final effects.
4. Verify expected root IDs and required metadata.
5. Write the optional aggregate summary, flush, and persist machine state.
6. Dispatch job `CLOSE` and report collected errors.

Harbor currently swallows finalizer errors, so fully strict end-of-job failure may require an upstream API change.

## Testing strategy

Follow [`docs/vcr-testing.md`](vcr-testing.md): real SDKs and recorded traffic are the default; mocks are a narrow exception. The primary test should exercise a real Harbor result through the real Braintrust SDK, not a hand-built `TrialResult` through a fake logger.

### Coverage layers

| Layer | Input | Braintrust side | Purpose |
|---|---|---|---|
| Pure contracts | Real Harbor/Pydantic values or JSON fixtures | No network | Reducers, IDs, reward rules, normalization, redaction. |
| Replay integration | Recorded Harbor job + hook stream | VCR-replayed Braintrust HTTP | Primary CI coverage. |
| Real Harbor smoke | Tiny real Harbor job, preferably local Docker | VCR-replayed Braintrust HTTP | Plugin discovery and actual hook ordering. |
| Live backend | Same recorded jobs | Real Braintrust project | Small opt-in round-trip suite. |

#### 1. Pure contract tests

Use table-driven tests for deterministic code: partitioning, IDs, metadata, rewards, classifications, ATIF conversion, timestamp repair, attachment policy, and state transitions. Construct typed events and run the reducer directly; do not mock Harbor callback methods or Braintrust clients.

Pure tests are appropriate here because there is no external protocol to record. Prefer loading real serialized Harbor models and ATIF documents over `MagicMock`, ad hoc objects, or invented provider responses.

#### 2. Recorded Harbor inputs + Braintrust VCR

This is the main test path. Capture small, sanitized outputs from real Harbor runs: resolved config and lock data, final results, trajectories, verifier details, and an ordered event journal ending at the final `JobResult` boundary. Version recordings when Harbor's serialized models or event behavior differ across the supported matrix.

The event journal is the equivalent of a transport cassette: it replays a non-HTTP protocol captured from the real system. Replay it through the actual hook dispatcher and state machines; do not replace it with a fake Harbor job.

Use the repository's VCR marker and shared recording infrastructure for sync and backfill tests. Let the real Braintrust SDK perform authentication/project lookup, dataset upserts/fetches, experiment creation, span/attachment uploads, and summary queries. Select HTTP recordings by the Harbor version under test.

Use deterministic job, dataset, experiment, record, and span IDs so request bodies replay reliably. Match write requests on body as well as method/path where practical; canonicalize only fields proven to be volatile. After flush, query the experiment through the Braintrust API and assert the persisted rows, origins, hierarchy, scores, metrics, classifications, and attachments. In-memory span assertions may supplement this round trip, but should not replace it.

Follow the repository record modes:

- local: `once`;
- CI: `none`;
- intentional focused re-record: `--vcr-record=all -k <test>`.

The existing VCR config removes Braintrust authentication headers. Add Harbor-specific request-body redaction and scan recorded jobs/cassettes for secrets, absolute paths, and user content before commit; header filtering alone is insufficient.

#### 3. Real Harbor smoke tests

Replay proves deterministic behavior but not that the plugin still attaches to Harbor correctly. Run at least one tiny real Harbor job using the installed Harbor package and entry point. Prefer a real built-in deterministic path such as an oracle task over a fake job object. Assert the observed hook sequence, final `JobResult`, sync manifest, and VCR-backed Braintrust result. Because live-run IDs and timestamps vary, this smoke test may use normal endpoint matching plus local payload assertions; keep strict write-body matching in the primary recorded-job tests.

Keep Docker-dependent smoke coverage in a dedicated Linux CI job if it is not portable across the normal SDK matrix. A second opt-in recording job can use a real built-in model agent to refresh the ATIF fixture; provider calls made inside Harbor's sandbox are not intercepted by host-side VCR.

#### 4. Live Braintrust tests

VCR cannot detect every backend contract change because it replays old responses. Keep a small opt-in suite that uploads one recorded successful job and verifies dataset version/origin links, experiment rows, attachments, summaries, base-experiment linkage, resume, and backfill idempotency through public Braintrust APIs. Use deterministic names in a dedicated test project and clean up when supported.

### Required scenarios

Keep recordings few but semantically dense:

1. successful single-step trial with LLM → tool → LLM ATIF, multiple rewards, metadata, and an attachment;
2. exception and verifier-disabled/missing-trajectory results;
3. retry followed by success, cancellation, duplicate hook delivery, and final reconciliation;
4. multiple dataset/agent partitions and intentional `n_attempts`;
5. multi-step and subagent trajectories;
6. regrade with exact base-experiment resolution;
7. malformed ATIF/classification data that degrades locally without losing the eval row.

Pure transition tests should enumerate skipped phases, illegal/backward events, duplicates, concurrent trials, retries, cancellation, `FINAL_RESULT`, and `OMITTED`. The recorded retry scenario remains the primary lifecycle proof.

### Mock policy

Do not use mocks/fakes as primary coverage for:

- Harbor hook ordering or retry behavior;
- `TrialResult`, `JobResult`, or ATIF response shape;
- Braintrust dataset/experiment APIs;
- emitted span shape derived from a real Harbor result;
- attachment upload or resume/backfill behavior.

Narrow monkeypatching is acceptable only for failures that cannot be recorded safely or deterministically, such as a disk write failure, task cancellation at an exact await point, or transport interruption. Keep those tests supplemental and patch the smallest effect boundary, not the Harbor or Braintrust object graph.

### Version-matrix and recording wiring

Add Harbor to the dependency version matrix, register its versioned recordings with the repository's shared cassette infrastructure, and add a dedicated parametrized nox session. Harbor requires Python 3.12+, so CI should schedule that session only on supported interpreters.

Use the nox session for playback and intentional focused re-recording so the selected Harbor version also selects compatible recordings. CI must replay with `record_mode="none"`; a missing recording is a failure, not permission to fall back to a fake.

## Delivery

1. **Eval core:** plugin entry point, config, partitioned experiments, dataset sync, final eval rows, rewards/classifications, lifecycle spans, reconciliation, manifest, and recorded/VCR coverage.
2. **ATIF:** conforming model/tool/subagent conversion, content policy, timestamp repair, and attachment handling.
3. **Advanced Harbor semantics:** multi-step detail, regrade bases, retry operational traces, exact job summaries, and selected artifacts.
4. **Upstream improvements:** public resolved plans, final-attempt event metadata, plugin-scoped YAML config, health reporting, and generic trace-parent propagation.

## Expected user experience

```bash
pip install harbor braintrust
export BRAINTRUST_API_KEY=...
export HARBOR_BRAINTRUST_PROJECT=agent-benchmarks

harbor run \
  -d terminal-bench/terminal-bench-2@latest \
  -a claude-code \
  -m anthropic/claude-sonnet-4-6 \
  -n 32 \
  --plugin braintrust
```

Braintrust should show the exact selected dataset, one experiment per system variant, one row per final trial, faithful scores/metrics/classifications, nested lifecycle and ATIF traces, useful usage/error slices, and stable links back to Harbor. Resume and backfill should update the same objects rather than duplicate them.
