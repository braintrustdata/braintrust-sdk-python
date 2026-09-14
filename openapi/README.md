# Pinned Braintrust OpenAPI specification

`spec.json` is a committed snapshot of
[`braintrustdata/braintrust-openapi`](https://github.com/braintrustdata/braintrust-openapi).
`config.json` pins the upstream commit, snapshot hash, generator versions and flags, selected endpoint
tags, and retry-policy allowlists. Builds use committed generated source and never fetch the spec or
run code generation.

## Generate and check

Run from `py/`:

```bash
make generate-api-client
make check-api-client-codegen
```

The check regenerates in a temporary directory and reports drift without changing the worktree. Generation also synchronizes the reviewed resource, method, and type inventories in the [public REST API client README](../py/src/braintrust/api/README.md).

The reviewed generated surface includes these tags:

- core resources: Projects, Experiments, Datasets, Prompts, and Functions;
- access and organization resources: Acls, Groups, ProjectGroups, Roles, Users, Organizations,
  ApiKeys, and ServiceTokens;
- configuration resources: AiSecrets, EnvVars, Environments, and McpServers;
- project resources: Agents, ProjectAutomations, OrgAutomations, ProjectScores, ProjectTags,
  SpanIframes, and Views; and
- versioned data resources: DatasetSnapshots.

Each tag produces one resource and operation registry. Models used by one resource stay in that
resource's model module; shared models live in `models/common.py`; unreachable models are omitted.

Every tag in the pinned spec must be either selected or present in `unsupported_tags` in
`config.json` with a rationale. The intentionally unsupported tags are:

- **CORS:** browser preflight `OPTIONS` operations are transport concerns rather than callable
  resource methods.
- **CrossObject:** cross-object event insertion belongs to the specialized at-least-once
  log-ingestion path.
- **Evals:** eval launch is a long-running, payload-dependent workflow that can stream and needs a
  specialized client.
- **Logs:** project-log event ingestion, fetching, and feedback remain on the specialized logging
  path.
- **Other:** the unauthenticated, text-only hello-world endpoint is a service diagnostic rather
  than a public REST resource.
- **Proxy:** provider passthrough needs proxy-target routing, streaming, and provider-specific
  response behavior. Its catch-all `proxy{path+}` operation ID is also not a valid Python
  identifier. Proxy remains on specialized SDK paths and is deliberately absent from generated
  modules and `BraintrustOpenApiClient`.

Method and inline-response names come directly from normalized OpenAPI `operationId` values. Generated
models preserve exact wire keys, including leading underscores, and methods do not add implicit request
defaults. GET and HEAD operations use the safe-read retry policy.
Logical POST reads and verified idempotent writes must be listed explicitly in `safe_reads` and
`idempotent_writes`; all other writes are non-retrying. Operations listed in `specialized_operations`
remain on their handwritten SDK paths and are excluded from the generic generated resource. Anonymous
nested objects that collide with component names receive contextual names, keeping component names stable.

## Refresh the snapshot

Fetch the configured upstream commit:

```bash
make fetch-openapi-spec
```

To fetch from a local checkout instead:

```bash
BRAINTRUST_OPENAPI_ROOT=../../braintrust-openapi make fetch-openapi-spec
```

The checkout must be at the commit pinned in `config.json`, and its spec must match the pinned hash.
To update the snapshot manually, update the commit and hash in `config.json`, fetch, regenerate, and review both the upstream spec diff and generated-source diff. Validation and generation apply only to selected tags and their transitively reachable schemas.

The scheduled and manually dispatchable [OpenAPI spec updates workflow](../.github/workflows/openapi-spec-updates.yml) checks the latest upstream commit that changed the spec. When the pin changes, it updates the snapshot, regenerates the client and public reference, runs codegen, runtime, and type tests, and opens or updates a review PR containing operation/schema summaries and a link to the upstream diff. The workflow never auto-merges its PR. If generation or validation fails, it still opens the update PR with the failure status and then fails the workflow so the new API shape can be reviewed explicitly.
