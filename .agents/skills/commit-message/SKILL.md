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

<body>

<footer(optional)>
```

- Choose the most specific suitable type: `feat`, `fix`, `ref` (repository convention for behavior-preserving restructuring), `perf`, `test`, `docs`, `chore`, `ci`, `style`, or `revert`.
- Add a scope only when it meaningfully narrows the change.
- Keep the subject at 72 characters or fewer, lowercase, imperative, and without a trailing period.
- always add a body. The body must:
  - start with a clear AI-generated notice: `> 🤖 **This text was AI-generated.**`
  - explain the problem and the solution like you are writing docs for a user. Use simple english and link to issues/prs/docs as appropriate
  - for both the problem and solution, try to include code snippets or ASCII diagrams that illustrates the change. ASCII diagrams should be in code snippet blocks
  - mention limitations or next steps if applicable
  - do not add validation or testing sections unless there is an extremely good reason to do so
- Add `BREAKING CHANGE: ...` or an issue-closing footer only when the diff or conversation supports it. If the change is breaking, make sure to append `!` to the type, like `feat!`
- add relevant github or linear issues to the footer if applicable

Do not run `git commit` or `git commit --amend` unless the user explicitly asks you to commit. When committing a multiline message, preserve real newline characters: prefer writing the message to a temporary file and using `git commit -F <file>` (or `git commit --amend -F <file>`). Multiple `-m` flags or ANSI-C quoting are acceptable for simple cases. Never assume literal `\n` inside ordinary quotes becomes a newline. After committing, verify the stored message with `git log -1 --format=medium`.
