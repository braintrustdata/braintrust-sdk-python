import os
import subprocess

import pytest
from braintrust import gitutil


@pytest.fixture(autouse=True)
def clear_git_executable_cache():
    gitutil._git_executable.cache_clear()
    gitutil._current_repo.cache_clear()
    yield
    gitutil._git_executable.cache_clear()
    gitutil._current_repo.cache_clear()


def test_git_executable_resolves_to_an_absolute_path():
    resolved = gitutil._git_executable()
    assert resolved is not None
    assert os.path.isabs(resolved)


def test_git_output_spawns_the_resolved_absolute_path(monkeypatch: pytest.MonkeyPatch):
    """git must be spawned by absolute path, never by bare name.

    Windows' CreateProcess searches the current directory before PATH, so spawning
    "git" from inside an untrusted repository would run a committed git.exe.
    """
    spawned = []

    def fake_run(args, **kwargs):
        spawned.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(gitutil.subprocess, "run", fake_run)

    assert gitutil._git_output(["rev-parse", "HEAD"]) == "ok"
    assert len(spawned) == 1
    assert spawned[0][0] == gitutil._git_executable()
    assert os.path.isabs(spawned[0][0])
    assert spawned[0][1:] == ["rev-parse", "HEAD"]


def test_git_executable_rejects_a_current_directory_hit(monkeypatch: pytest.MonkeyPatch):
    """shutil.which() searches the current directory on Windows; that hit must be dropped."""
    monkeypatch.setattr(gitutil.shutil, "which", lambda cmd: os.path.join(os.curdir, "git.exe"))

    assert gitutil._git_executable() is None


def test_git_output_raises_when_git_cannot_be_resolved(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gitutil.shutil, "which", lambda cmd: None)

    # FileNotFoundError is an OSError, which the callers below already treat as
    # "no git metadata available".
    with pytest.raises(FileNotFoundError):
        gitutil._git_output(["rev-parse", "HEAD"])


def test_repo_info_returns_none_when_git_cannot_be_resolved(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gitutil.shutil, "which", lambda cmd: None)

    assert gitutil._current_repo() is None
    assert gitutil.repo_info() is None
    assert list(gitutil.get_past_n_ancestors()) == []
