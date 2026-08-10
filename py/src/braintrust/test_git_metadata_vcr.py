import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import pytest
from braintrust import gitutil
from braintrust.conftest import get_vcr_config
from braintrust.git_fields import GitMetadataSettings
from braintrust.logger import init


PROJECT_NAME = "python-sdk-vcr-tests"
BASE_EXPERIMENT_NAME = "test-git-metadata-base-v2"
FEATURE_EXPERIMENT_NAME = "test-git-metadata-feature-v2"
BASE_SPAN_ID = "10000000-0000-4000-8000-000000000001"
FEATURE_SPAN_ID = "10000000-0000-4000-8000-000000000002"
GIT_METADATA_FIELDS = [
    "commit",
    "branch",
    "tag",
    "dirty",
    "author_name",
    "author_email",
    "commit_message",
    "commit_time",
]


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str, timestamp: str) -> str:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_DATE": timestamp,
            "GIT_COMMITTER_DATE": timestamp,
        }
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message, env=env)
    return _git(repo, "rev-parse", "HEAD")


def _initialize_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "config", "user.name", "Braintrust Git Regression")
    _git(repo, "config", "user.email", "git-regression@braintrust.dev")
    (repo / "application.py").write_text('print("seed")\n')
    _commit(repo, "Deterministic seed commit", "2024-01-31T12:00:00+00:00")
    (repo / "application.py").write_text('print("base")\n')
    base_commit = _commit(repo, "Deterministic base commit", "2024-02-01T12:00:00+00:00")
    _git(repo, "tag", "git-regression-base")

    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "--set-upstream", "origin", "main")
    return repo, base_commit


def _api_key() -> str:
    return os.environ.get("BRAINTRUST_API_KEY", "sk-dummy-for-vcr-replay")


def _git_metadata_settings() -> GitMetadataSettings:
    return GitMetadataSettings(collect="some", fields=GIT_METADATA_FIELDS)


def _normalize_vcr_request(request):
    if not request.body:
        return request

    try:
        payload = json.loads(request.body)
    except (TypeError, ValueError):
        return request

    if urlparse(request.uri).path == "/logs3":
        for row in payload.get("rows", []):
            row.pop("context", None)
            row.pop("created", None)
            row.pop("root_span_id", None)
            row.pop("span_id", None)
            metrics = row.get("metrics", {})
            metrics.pop("start", None)
            metrics.pop("end", None)

    request.body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return request


@pytest.fixture(scope="module")
def vcr_config():
    config = get_vcr_config()
    config["before_record_request"] = _normalize_vcr_request
    config["match_on"] = ["method", "scheme", "host", "port", "path", "query", "body"]
    return config


def _log_regression_span(experiment, span_id: str, case: str) -> None:
    with experiment.start_span(
        id=span_id,
        name="git-metadata-regression",
        input={"case": case},
        metadata={"git_regression_case": case},
    ) as span:
        span.log(output={"status": "recorded"})
    experiment.flush()


@pytest.fixture(autouse=True)
def clear_gitutil_caches():
    gitutil._current_repo.cache_clear()
    gitutil._get_base_branch.cache_clear()
    yield
    gitutil._current_repo.cache_clear()
    gitutil._get_base_branch.cache_clear()


@pytest.mark.vcr
def test_git_metadata_is_preserved_in_braintrust_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, base_commit = _initialize_repository(tmp_path)
    monkeypatch.chdir(repo)

    base_experiment = init(
        project=PROJECT_NAME,
        experiment=BASE_EXPERIMENT_NAME,
        api_key=_api_key(),
        git_metadata_settings=_git_metadata_settings(),
        update=True,
        set_current=False,
    )
    _log_regression_span(base_experiment, BASE_SPAN_ID, "base")

    assert base_experiment.data["repo_info"] == {
        "commit": base_commit,
        "branch": "main",
        "tag": "git-regression-base",
        "dirty": False,
        "author_name": "Braintrust Git Regression",
        "author_email": "git-regression@braintrust.dev",
        "commit_message": "Deterministic base commit",
        "commit_time": "2024-02-01T12:00:00+00:00",
    }
    assert base_experiment.data["commit"] == base_commit

    _git(repo, "checkout", "-b", "feature/git-metadata-regression")
    (repo / "application.py").write_text('print("feature")\n')
    feature_commit = _commit(repo, "Deterministic feature commit", "2024-02-02T12:00:00+00:00")
    _git(repo, "tag", "git-regression-feature")
    (repo / "application.py").write_text('print("dirty feature")\n')

    feature_experiment = init(
        project=PROJECT_NAME,
        experiment=FEATURE_EXPERIMENT_NAME,
        api_key=_api_key(),
        git_metadata_settings=_git_metadata_settings(),
        update=True,
        set_current=False,
    )
    _log_regression_span(feature_experiment, FEATURE_SPAN_ID, "feature-dirty")

    assert feature_experiment.data["repo_info"] == {
        "commit": feature_commit,
        "branch": "feature/git-metadata-regression",
        "tag": "git-regression-feature",
        "dirty": True,
        "author_name": "Braintrust Git Regression",
        "author_email": "git-regression@braintrust.dev",
        "commit_message": "Deterministic feature commit",
        "commit_time": "2024-02-02T12:00:00+00:00",
    }
    assert feature_experiment.data["commit"] == feature_commit
    assert feature_experiment.data["base_exp_id"] == base_experiment.id

    fetched_span = next(row for row in feature_experiment.fetch() if row["id"] == FEATURE_SPAN_ID)
    assert fetched_span["input"] == {"case": "feature-dirty"}
    assert fetched_span["output"] == {"status": "recorded"}
    assert fetched_span["metadata"]["git_regression_case"] == "feature-dirty"
    assert "repo_info" not in fetched_span
