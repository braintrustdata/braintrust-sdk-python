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

The check regenerates in a temporary directory and reports drift without changing the worktree.
Currently selected tags are Projects, Experiments, and Datasets. Each tag produces one resource and
operation registry. Models used by one resource stay in that resource's model module; shared models
live in `models/common.py`; unreachable models are omitted.

Method and inline-response names come directly from normalized OpenAPI `operationId` values. Generated
models preserve exact wire keys, including leading underscores, and methods do not add implicit request
defaults. GET and HEAD operations use the safe-read retry policy.
Logical POST reads and verified idempotent writes must be listed explicitly in `safe_reads` and
`idempotent_writes`; all other writes are non-retrying.

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
To update the snapshot, update the commit and hash in `config.json`, fetch, regenerate, and review both
the upstream spec diff and generated-source diff. Validation and generation apply only to selected tags
and their transitively reachable schemas.
