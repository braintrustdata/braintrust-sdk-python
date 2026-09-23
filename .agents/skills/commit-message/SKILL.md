---
name: commit-message
description: Draft a conventional commit message from the current changes and conversation. Use when asked to write, suggest, or generate a commit message; do not commit unless asked.
---

# Commit Message Generator

Write a concise, accurate commit message that captures the intent of the current changes. Use the conversation for motivation and constraints; use the diff as evidence. Do not claim behavior or test results the changes do not support.

## Gather context

From the repository root, inspect `git status --short`, `git diff HEAD`, and `git diff --cached`. Check untracked paths shown by status when they are part of the requested change. If there is no relevant diff, use recent commit history for context and say when there is not enough information to draft a reliable message.

Review the conversation for the user's stated goal, issue references, and compatibility constraints. If multiple unrelated changes appear, mention that they may need separate commits instead of hiding them in one vague message.

## Format

Use Conventional Commit form:

```text
<type>(<scope>): <imperative summary>

<optional body>

<optional footer>
```

- Choose the most specific suitable type: `feat`, `fix`, `ref` (repository convention for behavior-preserving restructuring), `perf`, `test`, `docs`, `chore`, `ci`, `style`, or `revert`.
- Add a scope only when it meaningfully narrows the change.
- Keep the subject at 72 characters or fewer, lowercase, imperative, and without a trailing period.
- Add a body when the subject alone does not explain the purpose or a meaningful behavior change. Keep it focused on what changed and why; wrap prose at about 100 characters.
- Add `BREAKING CHANGE: ...` or an issue-closing footer only when the diff or conversation supports it.

Use direct, specific language. Name relevant APIs, versions, errors, or files when they clarify the change. Avoid restating the diff, unsupported claims, decorative examples, and filler.

For substantive changes, prefer a compact code snippet or ASCII diagram in the body when it explains behavior or data flow more clearly than prose. Keep it focused and omit it when it adds no clarity.

## Output and commit execution

For a message suggestion, return one copyable message in a fenced code block. Add at most one brief sentence explaining a non-obvious type or scope choice. Ask a focused question only when the intent cannot reasonably be inferred from the diff and conversation.

Do not run `git commit` or `git commit --amend` unless the user explicitly asks you to commit. When committing a multiline message, preserve real newline characters: prefer writing the message to a temporary file and using `git commit -F <file>` (or `git commit --amend -F <file>`). Multiple `-m` flags or ANSI-C quoting are acceptable for simple cases. Never assume literal `\n` inside ordinary quotes becomes a newline. After committing, verify the stored message with `git log -1 --format=medium`.
