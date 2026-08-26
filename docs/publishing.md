# Publishing the Python SDK

The Python SDK is published from GitHub Actions. Do not use a local tag-push script.

## Stable PyPI release flow

Stable releases use a version-bump PR followed by a manually dispatched, approval-gated publish.

1. Run the `Prepare Stable Python SDK Release` workflow in GitHub Actions with the stable version to release, for example `0.22.0`.
2. The workflow validates the version, updates `py/src/braintrust/version.py`, and opens a PR from `release/py-sdk-v<version>`.
3. Review and merge the PR into `main`.
4. Copy the full SHA of the version-bump merge commit on `main`.
5. Run `Publish Python SDK` with `release_type=stable`, that commit SHA, and `dry_run=false`.
6. Approve the `publish` environment.
7. The workflow builds and verifies the package, generates and attests a CycloneDX SBOM, publishes to PyPI with trusted publishing, and creates the `py-sdk-v<version>` GitHub Release.

The stable version must match `X.Y.Z`. Stable releases are published from the merge commit of the version-bump PR.

## Prereleases

Prereleases use the same manually dispatched workflow, but the prerelease version must already be committed at the release SHA.

1. Create a prerelease branch and commit `py/src/braintrust/version.py` with a version such as `0.22.0rc1`, `0.22.0a1`, or `0.22.0b1`.
2. Run `Publish Python SDK` with:
   - `release_type=prerelease`
   - `sha` set to the full commit SHA containing the version bump
   - `prev_release` set optionally to the previous tag or prerelease anchor
   - `dry_run=false`
3. Approve the `publish` environment.

Prereleases publish to the normal PyPI package, but do not create a git tag or GitHub Release. A prerelease SHA outside `main` produces a warning rather than failing validation.

If you only want to publish a prerelease build for testing, you can also use `Publish Python SDK to TestPyPI` instead. That workflow does not create a GitHub Release.

## Publish Python SDK workflow details

`Publish Python SDK` is triggered manually through `workflow_dispatch`. Its inputs are:

- `release_type`: `stable` or `prerelease`. Defaults to `stable`.
- `sha`: the full commit SHA containing the version to release. The version cannot be overridden.
- `prev_release`: an optional tag or SHA to use as the release-notes anchor.
- `dry_run`: build and validate without publishing or tagging. Defaults to `false`.

The workflow uses commit-pinned actions from [`braintrustdata/sdk-actions`](https://github.com/braintrustdata/sdk-actions) to:

1. Check out the requested SHA and read the package version from `py/src/braintrust/version.py`.
2. Determine whether the SHA is on `main`, warning rather than failing if it is not.
3. Check PyPI availability and ensure the release tag does not already exist.
4. Generate release notes and post the release approval summary.
5. Build and verify the package with `make -C py install-dev verify-build`.
6. Generate a CycloneDX SBOM and, for real publishes, create a signed SBOM attestation.
7. If `dry_run=false`, publish to PyPI through OIDC trusted publishing and create the stable GitHub Release with the SBOM attached.

The `build-and-ship` job always runs behind an environment approval gate. Real stable and prerelease publishes use the `publish` environment; dry runs use `publish-dry-run`. Configure required reviewers on both environments. The job needs `contents: write`, `id-token: write`, and `attestations: write` permissions.

## TestPyPI releases

Use the separate `Publish Python SDK to TestPyPI` workflow when you want to publish a build to TestPyPI without creating a real PyPI release, git tag, or GitHub Release.

This is useful for:

- packaging smoke tests
- validating a release candidate before the real PyPI publish
- sharing prerelease artifacts for testing without consuming the final PyPI version number

The workflow reads the version from `py/src/braintrust/version.py` and applies a workflow-controlled version override during the build so TestPyPI uploads stay unique without modifying the checked-in file. The packaged `version.py` is also templated with the exact git commit and a release channel marker.

It supports two release types:

- `prerelease`: keeps the existing TestPyPI prerelease behavior and publishes a version such as `0.8.0rc1234`
- `canary`: publishes a nightly-style development release to TestPyPI only

Run `Publish Python SDK to TestPyPI` with:

- `ref=main` or the exact branch / commit you want to test
- `release_type=prerelease` or `release_type=canary`
- `dry_run=true` if you only want to validate/build without publishing

### Canary releases

- Can be triggered manually by running `Publish Python SDK to TestPyPI` with `release_type=canary`
- Publish a PEP 440 development release in the form `<version>.dev<YYYYMMDD><run_number>`
- Only publish to TestPyPI; there is no matching canary mode in the real PyPI workflow
- Do not create a git tag or GitHub Release
- Skip publishing if the current `HEAD` commit matches the latest published TestPyPI artifact marked with release channel `canary`
- Skip publishing unless the latest completed `checks.yaml` run on the target branch succeeded

install canaries like so:

```bash
pip install -i https://test.pypi.org/simple/ braintrust==<canary-version>
```

Nightly scheduling lives in `Schedule Python SDK Canary Publish`, which only dispatches `Publish Python SDK to TestPyPI` with `release_type=canary`. The actual publish remains in `test-publish-py-sdk.yaml` so trusted publishing stays configured against a single workflow.

Install from TestPyPI with:

```bash
pip install -i https://test.pypi.org/simple/ braintrust==<version>
```

> The build will fail if you upload a package with a duplicate version number. If this happens, DO NOT update version.py. Instead, rebase your branch onto origin/main and try again. The workflow-generated prerelease or canary suffix should normally keep TestPyPI versions unique.

Just like the main PyPI workflow, the TestPyPI workflow also supports `dry_run=true`. In that mode it builds, verifies, and uploads artifacts, but it does not publish to TestPyPI.

## Dry runs

Use `dry_run=true` when you want to exercise the release workflow without publishing anything. Dry runs require approval in the `publish-dry-run` GitHub environment.

A dry run still:

- validates the selected SHA and committed version
- reports whether the release commit is on `main`
- checks the tag and PyPI version, reporting existing releases as warnings
- builds the package and runs `make -C py install-dev verify-build`
- generates a CycloneDX SBOM
- generates release notes and release summaries

A dry run does not:

- publish to PyPI
- create the `py-sdk-v<version>` tag
- create a GitHub Release

---

## Maintenance

`.github/workflows/publish-py-sdk.yaml` is generated from the `release/py/turnkey` template in [`braintrustdata/sdk-actions`](https://github.com/braintrustdata/sdk-actions). The shared actions are pinned by commit SHA. Do not hand-edit their pins to pick up upstream changes; use the workflow generator so it can preserve this repository's customizations.

### Updating sdk-actions

From an `sdk-actions` checkout with its mise tools installed:

```bash
WF=/path/to/braintrust-sdk-python/.github/workflows/publish-py-sdk.yaml
REF=$(git rev-parse origin/main)
mise exec -- bin/workflow compare --ref "$REF" "$WF"
mise exec -- bin/workflow update --ref "$REF" "$WF"
mise exec -- bin/workflow validate "$WF"
```

Pass the resolved commit SHA through `--ref`; `compare` otherwise uses the ref already recorded in the workflow header. `update` performs a three-way merge of upstream template changes, retains local edits, and updates the action pins and provenance header.

After updating:

1. Review the workflow diff and the upstream sdk-actions changes between the old and new refs. A major change to the header's `version` field indicates a breaking release-action change.
2. Run `bash scripts/ensure-pinned-actions.sh` and the workflow validator.
3. Open a PR and complete an approved `dry_run` before the next real release.

The `# sdk-actions: {...}` header at the top of the workflow records the template, pinned ref, and generation parameters. Keep it intact so `compare` and `update` can reconstruct the upstream baseline.

### Local workflow customizations

`compare` reports the intentional differences from the turnkey template. Preserve these when updating:

- the dispatch instruction reminding releasers to commit the version before publishing
- release-channel templating and wheel verification through `BRAINTRUST_RELEASE_CHANNEL` and `make install-dev verify-build`, with extended build timeouts
- the existing `py-sdk-v{version}` tag format
- the `@sdk-eng` mention in approval notifications

### Required configuration

- GitHub environments `publish` and `publish-dry-run`, with required reviewers configured. Real stable and prerelease publishes use `publish`; dry runs use `publish-dry-run`.
- A PyPI trusted publisher for `braintrust`: owner `braintrustdata`, repository `braintrust-sdk-python`, workflow `publish-py-sdk.yaml`, environment `publish`.
- Repository or organization secret `SLACK_BOT_TOKEN` and variable `SLACK_SDK_RELEASE_CHANNEL`, with the variable visible to this repository.
