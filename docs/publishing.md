# Publishing the Python SDK

The Python SDK is published from GitHub Actions. Do not use a local tag-push script.

## Stable PyPI release flow

Stable releases use a two-step GitHub Actions flow: a version-bump PR, then an approval-gated publish after the PR is merged.

1. Run the `Prepare Stable Python SDK Release` workflow in GitHub Actions with the stable version to release, for example `0.22.0`.
2. The workflow validates the version, updates `py/src/braintrust/version.py`, and opens a PR from `release/py-sdk-v<version>`.
3. Review and merge the PR into `main`.
4. Merging the release PR triggers `Publish Python SDK`.
5. The stable publish job waits for approval in the `pypi-publish` GitHub environment.
6. After approval, the workflow builds/verifies the package, publishes to PyPI, and creates the `py-sdk-v<version>` GitHub Release tag and release.

The stable version must match `X.Y.Z`. Stable releases are published from the merge commit of the release PR.

## Prereleases

Prereleases stay manual through the `Publish Python SDK` workflow.

Run `Publish Python SDK` with:

- `ref=main` or the exact commit SHA to build from
- `release_type=prerelease`
- `version` set to the prerelease version to publish, for example `0.22.0rc1`
- `dry_run=false`

Do not bump `py/src/braintrust/version.py` for prereleases. The workflow validates the requested prerelease version and passes it to the build as a version override.

Prerelease versions must match `X.Y.Zrc1`, `X.Y.Za1`, or `X.Y.Zb1`. Prereleases publish to the normal PyPI package and are marked as prereleases on the GitHub Release. They do not use the stable release PR/tag workflow, do not require a committed version bump, and do not require `pypi-publish` environment approval.

If you only want to publish a prerelease build for testing, you can also use `Publish Python SDK to TestPyPI` instead. That workflow does not create a GitHub Release.

## Publish Python SDK workflow details

`Publish Python SDK` can be triggered by:

- a merged `release/py-sdk-v*` PR
- a pushed `py-sdk-v*` tag, for rerunning or repairing a release
- manual `workflow_dispatch`

Manual inputs are:

- `ref`: the branch, tag, or commit SHA to release. Defaults to `main`.
- `release_type`: `stable`, `prerelease`, or `auto`. Defaults to `stable`.
- `version`: the version to publish for manual prerelease runs, for example `0.22.0rc1`. Stable releases read `py/src/braintrust/version.py`.
- `dry_run`: validate and build without actually publishing. Defaults to `false`.

The workflow will:

1. Check out the release PR merge commit, requested ref, or pushed tag.
2. Validate that the selected commit is on `main`.
3. Resolve the package version from the manual `version` input or from `py/src/braintrust/version.py`.
4. Enforce that:
   - `stable` uses a version like `X.Y.Z`
   - `prerelease` uses a version like `X.Y.Zrc1`, `X.Y.Za1`, or `X.Y.Zb1`
   - `auto` infers stable vs prerelease from the version
5. Verify that the version is not already published on PyPI and, for manual non-tag runs, that the release tag does not already exist.
6. Build and verify the package with `make -C py install-dev verify-build`.
7. Upload the built distribution artifacts for inspection.
8. If `dry_run=false`, publish to PyPI and create the corresponding GitHub Release.

For stable, non-dry-run publishes, the job that publishes to PyPI runs in the `pypi-publish` GitHub environment. Configure required reviewers on that environment to approve stable releases before publishing.

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

Use `dry_run=true` when you want to exercise the release workflow without publishing anything.

A dry run still:

- validates the selected ref and version
- checks that the release commit is on `main`
- checks that the tag and PyPI version do not already exist
- builds the package and runs `make -C py install-dev verify-build`
- uploads `py/dist/` as a workflow artifact
- generates release notes

A dry run does not:

- publish to PyPI
- create the `py-sdk-v<version>` tag
- create a GitHub Release
- require `pypi-publish` environment approval
