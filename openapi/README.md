# Pinned Braintrust OpenAPI specification

`spec.json` is a committed snapshot of the public specification from
[`braintrustdata/braintrust-openapi`](https://github.com/braintrustdata/braintrust-openapi).
`config.json` pins the full upstream commit, snapshot SHA-256, generator tools, generator flags, and
selected endpoint tags. The generator scripts live in `py/scripts/`. Builds and package installation
use the committed generated source and never fetch or run code generation.

From `py/`, validate and regenerate the private models offline with:

```bash
make generate-api-client
make check-api-client-codegen
```

The check regenerates into a temporary directory and does not modify the worktree. Endpoint bindings
are rolled out explicitly through `endpoint_generator.generated_tags`. The current rollout supports
exactly one selected OpenAPI tag and emits its operation registry and resource class together in
`projects.py`, with reachable types in `models/projects.py`; unreachable models are omitted. Add
explicit cross-resource model partitioning before selecting a second tag. Public resource method and
inline response type names are derived mechanically from each `operationId`, and generated methods
forward request fields and parameters without implicit defaults. Writes that are safe to retry are
listed declaratively in `endpoint_generator.idempotent_writes`; reads and all other writes use
mechanical retry defaults.

To fetch the configured upstream commit explicitly:

```bash
make fetch-openapi-spec
```

For an existing local checkout, set `BRAINTRUST_OPENAPI_ROOT` to its root. The checkout must be at the
commit pinned in `config.json`, and its specification must have the pinned hash:

```bash
BRAINTRUST_OPENAPI_ROOT=../../braintrust-openapi make fetch-openapi-spec
```

To update the snapshot, first update the commit and SHA-256 in `config.json`, then fetch, regenerate,
and review both the upstream spec diff and generated model diff. Only operations selected through
`endpoint_generator.generated_tags` and their reachable schemas are validated and generated.
