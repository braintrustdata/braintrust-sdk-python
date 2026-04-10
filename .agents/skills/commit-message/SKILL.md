---
name: commit-message
description: Suggest a Braintrust SDK repo-style commit message from the current diff and conversation. Use when asked to write, suggest, or generate a commit message for the current changes.
---

# Commit Message

Generate a single commit message that matches the style used on `main` in this repo.

## Repo Style

Prefer:

```text
<type>(<scope>): <summary>
```

or, when scope does not add much:

```text
<type>: <summary>
```

Recent `main` examples:

- `feat(openai): trace images api calls`
- `fix(framework): split \`Output\` TypeVar into \`Output\` and \`Expected\``
- `ref(litellm): migrate litellm wrapper to integrations API`
- `chore: generated SDK types`
- `ci(checks): bump nox shards to 4 and introduce shard weights`
- `test(openai): add vcr regression coverage for stream helpers`
- `docs: document integrations in readme`
- `perf(json): reduce span serialization overhead`

Notes:

- This repo uses `ref`, not `refactor`.
- Scope is common and usually names the subsystem, provider, or area being changed.
- Commits on `main` often include a GitHub squash suffix like `(#245)`. Omit that for a normal local commit unless the user explicitly wants a PR title or squash-merge title.
- Most commits in this repo are title-only, but the message should still be useful. Prefer a concise subject that makes the main change obvious, and add a short body when the subject alone would feel too vague.

## Types

Use the most specific type:

- `feat` — new feature
- `fix` — bug fix
- `ref` — restructuring without behavior change
- `perf` — performance improvement
- `test` — tests only
- `docs` — documentation only
- `chore` — tooling, generated files, maintenance, config
- `ci` — GitHub Actions, nox sharding, CI wiring
- `style` — formatting only
- `revert` — reverting a prior change

## Scope Guidance

Good scopes in this repo usually look like:

- provider or integration names: `openai`, `anthropic`, `google_genai`, `claude_agent_sdk`, `langchain`
- SDK areas: `framework`, `cli`, `devserver`, `integrations`
- CI/tooling areas: `checks`, `nox`, `release`

If the change is broad or generated, omit scope instead of forcing one.

## Rules

- Keep the message useful but concise.
- Keep the subject concise and imperative.
- Prefer lowercase style and no trailing period.
- Keep the subject around 72 characters or less when practical.
- Describe the primary change, not every file touched.
- Make the subject specific enough that a reviewer can understand the change without opening the diff.
- If the diff mixes unrelated concerns, say so instead of forcing one message.
- Add a short body when non-obvious rationale, behavior changes, or breaking changes would make the message meaningfully clearer.

## Workflow

1. Inspect the current change:

```bash
git diff HEAD
git diff --cached
git status --short
```

2. Pick the main intent, choose `type` and optional `scope`, then write one useful but concise commit message.

## Output

Return exactly one commit message in a fenced code block.

If helpful, add one short sentence after the block explaining the type/scope choice.
