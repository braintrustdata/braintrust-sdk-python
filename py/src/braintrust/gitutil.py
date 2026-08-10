import logging
import os
import re
import shutil
import subprocess
import threading
from functools import lru_cache as _cache

from .git_fields import GitMetadataSettings, RepoInfo


_logger = logging.getLogger("braintrust.gitutil")
_gitlock = threading.RLock()


@_cache(1)
def _git_executable():
    """Resolve `git` from PATH, ignoring any copy in the current directory.

    Handing subprocess an absolute path avoids the current-directory-first executable
    search Windows performs at spawn time, where a `git.exe` committed to a repository
    would otherwise shadow the real one. `shutil.which()` searches the current
    directory on Windows as well, so a relative hit is rejected rather than used.
    """
    resolved = shutil.which("git")
    if resolved is not None and not os.path.isabs(resolved):
        _logger.warning("Ignoring 'git' resolved from the current directory: %s", resolved)
        return None
    return resolved


def _git_output(args, cwd=None):
    git = _git_executable()
    if git is None:
        raise FileNotFoundError("Could not find a 'git' executable on PATH")

    result = subprocess.run(
        [git, *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout
    if output.endswith(b"\n"):
        output = output[:-1]
    return output.decode("utf-8", errors="surrogateescape")


def _normalize_git_iso_datetime(value):
    return value[:-1] + "+00:00" if value.endswith("Z") else value


def _is_dirty(repo_path):
    diff_args = ["--abbrev=40", "--full-index", "--raw"]
    if _git_output(["diff", "--cached", *diff_args], cwd=repo_path):
        return True
    return bool(_git_output(["diff", *diff_args], cwd=repo_path))


@_cache(1)
def _current_repo():
    try:
        return _git_output(["rev-parse", "--show-toplevel"]).strip()
    except (OSError, subprocess.CalledProcessError):
        try:
            if _git_output(["rev-parse", "--is-bare-repository"]).strip() == "true":
                return _git_output(["rev-parse", "--absolute-git-dir"]).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        return None


@_cache(1)
def _get_base_branch(remote=None):
    repo_path = _current_repo()
    remote = remote or "origin"
    remotes = set(_git_output(["remote"], cwd=repo_path).splitlines())
    if remote not in remotes:
        raise ValueError(f"Remote named '{remote}' didn't exist")

    # NOTE: This should potentially be configuration that we derive from the project,
    # instead of spending a second or two computing it each time we run an experiment.

    # To speed this up in the short term, we pick from a list of common names
    # and only fall back to the remote origin if required.
    COMMON_BASE_BRANCHES = ["main", "master", "develop"]
    repo_branches = set(
        _git_output(["for-each-ref", "--format=%(refname:short)", "refs/heads/"], cwd=repo_path).splitlines()
    )
    if sum(b in repo_branches for b in COMMON_BASE_BRANCHES) == 1:
        for b in COMMON_BASE_BRANCHES:
            if b in repo_branches:
                return (remote, b)
        raise RuntimeError("Impossible")

    try:
        s = _git_output(["remote", "show", "origin"], cwd=repo_path)
        match = re.search(r"\s*HEAD branch:\s*(.*)$", s, re.MULTILINE)
        if match is None:
            raise RuntimeError("Could not find HEAD branch in remote " + remote)
        branch = match.group(1)
    except Exception as e:
        _logger.warning(f"Could not find base branch for remote {remote}", e)
        branch = "main"
    return (remote, branch)


def _get_base_branch_ancestor(remote=None):
    try:
        remote_name, base_branch = _get_base_branch(remote)
    except Exception as e:
        _logger.warning(
            f"Skipping git metadata. This is likely because the repository has not been published to a remote yet. {e}"
        )
        return None

    try:
        repo_path = _current_repo()
        head = "HEAD" if _is_dirty(repo_path) else "HEAD^"
        return _git_output(["merge-base", head, f"{remote_name}/{base_branch}"], cwd=repo_path).strip()
    except subprocess.CalledProcessError as e:
        # _logger.warning(f"Could not find a common ancestor with {remote_name}/{base_branch}")
        return None


def get_past_n_ancestors(n=1000, remote=None):
    with _gitlock:
        repo_path = _current_repo()
        if repo_path is None or n <= 0:
            return

        ancestor_output = _get_base_branch_ancestor()
        if ancestor_output is None:
            return
        try:
            output = _git_output(["rev-list", "--first-parent", f"--max-count={n}", ancestor_output], cwd=repo_path)
        except subprocess.CalledProcessError:
            return
        yield from output.splitlines()


def attempt(op):
    try:
        return op()
    # OSError covers FileNotFoundError, FileExistsError, etc.
    except (TypeError, ValueError, OSError, subprocess.CalledProcessError):
        return None


def truncate_to_byte_limit(input_string, byte_limit=65536):
    encoded = input_string.encode("utf-8")
    if len(encoded) <= byte_limit:
        return input_string
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def get_repo_info(settings: GitMetadataSettings | None = None):
    if settings is None:
        settings = GitMetadataSettings(collect="none")

    if settings.collect == "none":
        return None

    repo = repo_info()
    if repo is None or settings.collect == "all":
        return repo

    return RepoInfo(**{k: v if k in settings.fields else None for k, v in repo.as_dict().items()})


def repo_info():
    with _gitlock:
        repo_path = _current_repo()
        if repo_path is None:
            return None

        commit = None
        commit_message = None
        commit_time = None
        author_name = None
        author_email = None
        tag = None
        branch = None
        git_diff = None

        dirty = attempt(lambda: _is_dirty(repo_path))

        commit = attempt(lambda: _git_output(["rev-parse", "HEAD"], cwd=repo_path).strip())
        commit_message = attempt(lambda: _git_output(["log", "-1", "--format=%B", "HEAD"], cwd=repo_path).strip())
        commit_time = attempt(
            lambda: _normalize_git_iso_datetime(
                _git_output(["log", "-1", "--format=%cI", "HEAD"], cwd=repo_path).strip()
            )
        )
        author_name = attempt(lambda: _git_output(["log", "-1", "--format=%an", "HEAD"], cwd=repo_path).strip())
        author_email = attempt(lambda: _git_output(["log", "-1", "--format=%ae", "HEAD"], cwd=repo_path).strip())
        tag = attempt(lambda: _git_output(["describe", "--tags", "--exact-match", "--always"], cwd=repo_path).strip())

        branch = attempt(lambda: _git_output(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repo_path).strip())

        if dirty:
            git_diff = attempt(
                lambda: truncate_to_byte_limit(_git_output(["diff", "--no-ext-diff", "HEAD"], cwd=repo_path))
            )

        return RepoInfo(
            commit=commit,
            branch=branch,
            tag=tag,
            dirty=dirty,
            author_name=author_name,
            author_email=author_email,
            commit_message=commit_message,
            commit_time=commit_time,
            git_diff=git_diff,
        )
