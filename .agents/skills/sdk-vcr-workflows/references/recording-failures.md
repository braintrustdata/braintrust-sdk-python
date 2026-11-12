# When Recording Fails

Live recording fails for reasons unrelated to the SDK change. Handle each case so a failed recording does not destroy a good cassette or leave a bad one behind.

- **Missing or invalid API key (401/403, auth errors):** stop and ask the user for a working key. Never hand-write or edit a cassette to stand in for a recording.
- **Quota or rate limits (429, 402 "no remaining credits"):** a cassette that captured an error response is worse than the old one. Delete only the cassettes that captured the error, wait or ask for another key, then refill them with the default `once` mode.
- **Retired models or upstream drift:** if the live API no longer returns what the test asserts (a retired model returns 404, a moderation block, a search result that no longer appears, a different stop reason), keep the old cassette with `git checkout` and tell the user. Do not loosen the assertion just to get a recording.
- **Hand-authored cassettes:** some cassettes are hand-edited to force a scenario such as a validation retry. Never re-record them; restore them with `git checkout`.

### Restore from git vs. re-record

- Restore from git when the cassette is hand-authored, upstream behavior drifted, or the provider is unusable locally (no key, no quota). The old cassette still matches the request the SDK sends.
- Re-record when the SDK's request shape intentionally changed, or when the new pin changes the wire format and playback fails.
- If some tests in a directory may fail to record, record without deleting the directory first, so failed tests keep their old cassettes. If a test fails partway through, `git checkout` its cassette, because a partial refresh can leave truncated files.
- After any refresh, run `git status` on the cassette directory and look for ` D` entries (cassettes deleted but never recreated).

### Traps that silently lose cassettes

- **Skipped sessions record nothing.** If a session skips on your Python version (see `_skip_if_incompatible` and the constraints in `py/noxfile.py`), deleting its cassettes first destroys them. Run nox from a compatible interpreter instead.
- **Multi-file sessions stop at the first failing file.** After a partial failure, re-run in `once` mode to fill the files that later test files never recorded.
- **Shared cassette names under `--vcr-record=all`.** When several parametrizations share a cassette name, later cases read an earlier case's response. Delete only those cassettes and re-run in `once` mode, so the first case of each name records and the rest replay.
- **Hangs.** Tests that drive long-lived live sessions can hang while recording. Exclude them with `-k "not <test>"` and keep their old cassettes.

## Debugging VCR Failures

When a cassette-backed test fails, check these in order:

1. **Wrong nox session or provider version**
   - Did you reproduce under the exact session from `py/noxfile.py`?
   - Are you accidentally testing `latest` when CI pins an older version?

2. **Cassette missing or stale**
   - Does the expected cassette file exist in the right directory?
   - Did the test move without moving the cassette?
   - Does the cassette still match the intended request shape?

3. **Behavior changed but cassette did not**
   - Did the code intentionally change the payload, headers, streaming sequence, or traced output?
   - If yes, re-record narrowly.

4. **Behavior did not change, but the test became too strict**
   - Prefer fixing brittle assertions before churning cassettes.
   - Assert the meaningful span structure, not incidental provider noise.

5. **Local-only pass, CI fail**
   - Remember CI uses `record_mode="none"`.
   - A local silent record can hide a missing cassette or wrong request shape.

6. **Optional dependency or auth confusion**
   - Do not rely on globally installed provider packages.
   - Use the active nox session.
   - Read `py/src/braintrust/conftest.py` before assuming credential behavior.
