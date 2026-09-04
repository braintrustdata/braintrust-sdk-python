---
name: commit-message
description: Generates a meaningful conventional commit message (title + body) from the current diff and conversation context. Use when asked to write, suggest, or generate a commit message for the current changes.
---

# Commit Message Generator

Generate a conventional commit message based on the staged/unstaged diff and the surrounding conversation.

## Conventional Commits Format

```
<type>(<scope>): <short summary>

<body>

<footer>
```

**Types:**
- `feat` — new feature
- `fix` — bug fix
- `ref` — code restructuring without behavior change (prefer `ref` over `refactor`)
- `perf` — performance improvement
- `test` — adding or updating tests
- `docs` — documentation only
- `chore` — build system, tooling, dependencies, config
- `ci` — CI/CD pipeline changes
- `style` — formatting, whitespace (no logic change)
- `revert` — reverts a previous commit

**Rules:**
- Subject line ≤ 72 characters, lowercase, no trailing period
- Imperative mood: "add feature" not "added feature"
- Body wraps at 100 characters, explains *what* and *why* (not *how*)
- Footer: `BREAKING CHANGE: ...` or `Fixes #<issue>` / `Closes #<issue>` if applicable
- Scope is optional but encouraged when it adds clarity (e.g., `feat(auth):`, `fix(api):`)
- Omit body if the subject line fully communicates the intent

## Writing Style

- Lead with the point. Be direct, technical, collaborative, and lightly casual.
- Use plain words, contractions where natural, and short paragraphs.
- Prefer `we` for shared decisions and `I` for genuine opinion or uncertainty. Do not hide uncertainty behind authoritative prose.
- Ground claims in specifics: name the API, version, behavior, error, test, or file involved.
- Put code, identifiers, filenames, versions, and literal values in backticks.
- Give only the context needed to explain **context → change → reason/evidence → consequence or follow-up**.
- Surface compatibility constraints, tradeoffs, risks, and intentionally deferred work plainly when relevant.
- Use bullets for multiple distinct changes or findings; avoid unnecessary headings and polished filler.
- Do not merely restate the diff. Explain why the implementation matters or how behavior changes.

For a substantive change, actively consider including a compact code snippet, before/after example, or ASCII diagram when it explains behavior or data flow more clearly than prose. Keep it focused and omit it when it would be decorative or redundant. For example:

````text
Before: provider response -> wrapper-specific span
After:  provider response -> shared integration hook -> normalized span
````

or:

````python
# Before
wrap(client)

# After
client = wrap_client(client)
````

## Instructions

1. **Collect the diff** — run `git diff HEAD` (staged + unstaged). If empty, try `git diff --cached` (staged only). If still empty, try `git status --short` and `git log --oneline -3` to understand the trajectory.

2. **Review the conversation** — you already have the conversation history in context. Look for:
   - Explicit intent from the user ("I'm adding X", "this fixes Y")
   - Issue or ticket numbers mentioned
   - Feature names, module names, or domain language used
   - Any constraints or things the user emphasized

3. **Identify ambiguities** — before generating, check if any of these are unclear:
   - Is the primary intent a new feature, a fix, a refactor, or something else?
   - Is there a scope (module/package/component) worth calling out?
   - Are there breaking changes?
   - Is there a related issue or ticket number?
   - Does the diff span multiple unrelated concerns? (should be split into separate commits)

4. **Ask for clarifications if needed** — if the intent is genuinely ambiguous from both the diff and the conversation, ask 1–3 focused questions before generating. Do **not** ask about things already clear from the context.

5. **Generate the commit message** — produce exactly one commit message in a fenced code block:
   - Pick the most specific `type` that fits
   - Include a `scope` when it meaningfully narrows the change
   - Write a crisp subject line in imperative mood
   - Add a body if the change is non-trivial, explaining the reasoning
   - Check whether a small before/after snippet or ASCII diagram would make a substantive change easier to review, and include one when it would
   - Add footer entries for breaking changes or issue references
   - **CRITICAL when committing:** preserve **real newline characters** in the commit body
   - **Never** put literal `\n` text inside a quoted `git commit -m "..."` body and assume Git will turn it into line breaks — it will not
   - If issuing `git commit` yourself, prefer these patterns in this order:
     1. **Best for multiline bodies:** write the full message to a temp file and use `git commit -F <file>`
     2. **Good for amendments:** write the full message to a temp file and use `git commit --amend -F <file>`
     3. multiple `-m` flags, e.g. `git commit -m "subject" -m "first paragraph

second paragraph"`
     4. ANSI-C quoting, e.g. `git commit -m "subject" -m $'line 1\n\nline 2'`
   - Prefer temp-file commits by default whenever the body has multiple paragraphs, bullets, or any non-trivial formatting
   - Before finalizing, sanity-check that `git log -1 --format=medium` shows actual blank lines and wrapped paragraphs, not backslash-n sequences

## Newline safety examples

**Wrong:**

```bash
git commit -m "docs: add pi guide" -m "line 1\n\nline 2"
```

This stores the characters `\` and `n` literally in the commit message.

**Correct:**

```bash
git commit -m "docs: add pi guide" -m $'line 1\n\nline 2'
```

or

```bash
git commit -m "docs: add pi guide" -m "line 1

line 2"
```

or use a temp file/editor.

## Default execution preference

When the task is not just to suggest a commit message, but to actually run `git commit` or `git commit --amend`:

1. If the message has a body, prefer a temp file with `-F`
2. If amending a commit with a body, prefer `git commit --amend -F <file>`
3. Only use inline `-m` bodies when the formatting is trivially simple and you are certain real newlines will be preserved
4. After committing, verify with `git log -1 --format=medium`

This preference exists to avoid malformed commit bodies with literal `\n` sequences.

## Output Format

Present the final message in a fenced code block. If the message contains its own fenced snippet, use a longer outer fence so the full message remains copyable:

```
feat(auth): add OAuth2 PKCE flow for CLI login

Replace the device-code flow with PKCE so the CLI can authenticate
without opening a browser on headless machines. The previous flow
required interactive browser consent which blocked CI usage.

Closes #342
```

Then briefly (1–2 sentences) explain the key decision made (type choice, scope, whether a body was needed).

If you asked for clarifications and the user answered, incorporate those answers and output the final message immediately — no need to re-ask.

## Commands

```bash
git diff HEAD
git diff --cached
git status --short
git log --oneline -5
```
