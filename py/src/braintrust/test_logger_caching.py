"""Regression tests for shared login and project resolution."""

import concurrent.futures
import threading
from unittest.mock import MagicMock, patch

import pytest
from braintrust import logger
from braintrust.logger import BraintrustState, _compute_logger_metadata
from braintrust.util import GLOBAL_PROJECT


@pytest.fixture(autouse=True)
def real_metadata_resolution():
    # Some older logger helpers replace the module function with a fake.
    with patch.object(logger, "_compute_logger_metadata", _compute_logger_metadata):
        yield


@pytest.fixture
def project_state():
    state = logger.login_to_state(api_key=logger.TEST_API_KEY)
    client = MagicMock()
    client.projects.post_project.side_effect = lambda *, body: {
        "id": "project-id",
        "name": body["name"],
        "metadata": {"nested": [1]},
    }
    client.projects.get_project_id.side_effect = lambda project_id: {"id": project_id, "name": "project"}
    with patch.object(state, "api_client", return_value=client):
        yield state, client


def _observe_request_waiters(state, count):
    joined = threading.Event()
    lock = threading.Lock()
    waiting = 0
    start_request = state._start_request

    def start(future, *args, **kwargs):
        result = future.result

        def wait(*args, **kwargs):
            nonlocal waiting
            with lock:
                waiting += 1
                if waiting == count:
                    joined.set()
            return result(*args, **kwargs)

        future.result = wait
        return start_request(future, *args, **kwargs)

    return patch.object(state, "_start_request", side_effect=start), joined


def _observe_discard(state):
    discarded = threading.Event()
    start_request = state._start_request

    def start(*args, on_discard=None, **kwargs):
        def discard(value):
            if on_discard is not None:
                on_discard(value)
            discarded.set()

        return start_request(*args, on_discard=discard, **kwargs)

    return patch.object(state, "_start_request", side_effect=start), discarded


def test_project_cache_default_id_name_and_group_keys(project_state):
    state, client = project_state
    for name in (None, "", GLOBAL_PROJECT):
        _compute_logger_metadata(project_name=name, state=state)
    assert client.projects.post_project.call_count == 1
    for group in (None, "one", "two", "one"):
        _compute_logger_metadata(project_name="project", project_group_name=group, state=state)
    assert client.projects.post_project.call_count == 4
    for _ in range(2):
        _compute_logger_metadata(project_id="project", state=state)
    client.projects.get_project_id.assert_called_once_with("project")


def test_project_cache_shares_pending_lookup_and_expires_after_success(project_state, monkeypatch):
    state, client = project_state
    now = [100.0]
    monkeypatch.setattr(logger.time, "monotonic", lambda: now[0])
    entered = threading.Event()
    release = threading.Event()
    first = logger.init_logger(project="project", state=state, set_current=False)
    response = {"id": "project-id", "name": "project", "metadata": {"nested": [1]}}

    def lookup(*, body):
        entered.set()
        assert release.wait(5)
        return response

    client.projects.post_project.side_effect = lookup
    observe, joined = _observe_request_waiters(state, 2)
    with observe, concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        original = pool.submit(lambda: first.id)
        assert entered.wait(5)
        now[0] += 901
        shared = pool.submit(_compute_logger_metadata, project_name="project", state=state)
        assert joined.wait(5)
        release.set()
        assert original.result(5) == shared.result(5).project.id == "project-id"
    client.projects.post_project.assert_called_once()

    # Each logger owns a copy, including nested metadata.
    first.project.name = "changed"
    first.project.full_info["metadata"]["nested"].append(2)
    assert shared.result().project.name == "project"
    assert shared.result().project.full_info["metadata"]["nested"] == [1]
    now[0] += 899
    cached = _compute_logger_metadata(project_name="project", state=state)
    assert cached.project.name == "project"
    assert cached.project.full_info["metadata"]["nested"] == [1]
    client.projects.post_project.assert_called_once()

    # Cache hits do not slide expiry; resolved loggers keep their original ID.
    now[0] += 1
    client.projects.post_project.side_effect = None
    client.projects.post_project.return_value = {"id": "new-id", "name": "project"}
    assert _compute_logger_metadata(project_name="project", state=state).project.id == "new-id"
    assert first.id == "project-id"
    assert client.projects.post_project.call_count == 2


def test_project_cache_shares_errors_and_retries_later(project_state):
    state, client = project_state
    entered = threading.Event()
    release = threading.Event()
    error = ValueError("lookup failed")

    def lookup(*, body):
        entered.set()
        assert release.wait(5)
        raise error

    client.projects.post_project.side_effect = lookup
    observe, joined = _observe_request_waiters(state, 4)
    with observe, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = [pool.submit(_compute_logger_metadata, project_name="project", state=state) for _ in range(4)]
        assert entered.wait(5)
        assert joined.wait(5)
        release.set()
        assert all(result.exception(5) is error for result in results)
    client.projects.post_project.assert_called_once()
    client.projects.post_project.side_effect = None
    client.projects.post_project.return_value = {"id": "project-id", "name": "project"}
    for _ in range(2):
        assert _compute_logger_metadata(project_name="project", state=state).project.id == "project-id"
    assert client.projects.post_project.call_count == 2


def test_old_project_failure_does_not_remove_replacement_after_reset(project_state):
    state, client = project_state
    entered = threading.Event()
    release = threading.Event()

    def lookup(*, body):
        entered.set()
        assert release.wait(5)
        raise ValueError("old failure")

    client.projects.post_project.side_effect = lookup
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_compute_logger_metadata, project_name="project", state=state)
        assert entered.wait(5)
        state.copy_state(logger.login_to_state(api_key=logger.TEST_API_KEY))
        client.projects.post_project.side_effect = None
        client.projects.post_project.return_value = {"id": "new-id", "name": "project"}
        assert _compute_logger_metadata(project_name="project", state=state).project.id == "new-id"
        release.set()
        with pytest.raises(ValueError, match="old failure"):
            first.result(5)
    assert _compute_logger_metadata(project_name="project", state=state).project.id == "new-id"
    assert client.projects.post_project.call_count == 2


@pytest.mark.parametrize("invalidate", ["reset", "copy", "adapter"])
def test_project_cache_invalidated_by_login_context(project_state, invalidate):
    state, client = project_state
    _compute_logger_metadata(project_name="project", state=state)
    if invalidate == "reset":
        state.reset_login_info()
        state.login(api_key=logger.TEST_API_KEY)
    elif invalidate == "copy":
        state.copy_state(logger.login_to_state(api_key=logger.TEST_API_KEY))
    else:
        with patch.object(logger, "_state", state), patch.object(logger, "_http_adapter", None):
            logger.set_http_adapter(MagicMock())
    _compute_logger_metadata(project_name="project", state=state)
    assert client.projects.post_project.call_count == 2


def test_project_cache_lru_bound_and_state_isolation(project_state):
    state, client = project_state
    for index in range(1000):
        _compute_logger_metadata(project_name=str(index), state=state)
    _compute_logger_metadata(project_name="0", state=state)
    _compute_logger_metadata(project_name="1000", state=state)
    _compute_logger_metadata(project_name="0", state=state)
    assert client.projects.post_project.call_count == 1001
    _compute_logger_metadata(project_name="1", state=state)
    assert client.projects.post_project.call_count == 1002
    other = logger.login_to_state(api_key=logger.TEST_API_KEY)
    with patch.object(other, "api_client", return_value=client):
        _compute_logger_metadata(project_name="0", state=other)
    assert client.projects.post_project.call_count == 1003


def test_login_shares_cold_requests_and_preserves_forced_logins():
    state = BraintrustState()
    logged_in = logger.login_to_state(api_key=logger.TEST_API_KEY)
    entered = threading.Event()
    release = threading.Event()

    def login(**kwargs):
        entered.set()
        assert release.wait(5)
        return logged_in

    with patch.object(logger, "login_to_state", side_effect=login) as request:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = [pool.submit(state.login, api_key=logger.TEST_API_KEY) for _ in range(8)]
            assert entered.wait(5)
            release.set()
            for result in results:
                result.result(5)
        request.assert_called_once()
        state.login(api_key=logger.TEST_API_KEY)
        state.login(api_key=logger.TEST_API_KEY, force_login=True)
        state.login(api_key=logger.TEST_API_KEY, force_login=True)
        assert request.call_count == 3


def _wait_until(predicate):
    deadline = logger.time.monotonic() + 5
    while not predicate():
        assert logger.time.monotonic() < deadline
        threading.Event().wait(0.001)


def test_login_shares_failure_and_retries_later():
    state = BraintrustState()
    entered = threading.Event()
    release = threading.Event()
    error = ValueError("login failed")

    def fail(**kwargs):
        entered.set()
        assert release.wait(5)
        raise error

    with patch.object(logger, "login_to_state", side_effect=fail) as request:
        observe, joined = _observe_request_waiters(state, 4)
        with observe, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = [pool.submit(state.login, api_key=logger.TEST_API_KEY) for _ in range(4)]
            assert entered.wait(5)
            assert joined.wait(5)
            release.set()
            for result in results:
                assert result.exception(5) is error
        request.assert_called_once()
    state.login(api_key=logger.TEST_API_KEY)
    assert state.logged_in


def test_reset_cancels_pending_and_queued_logins():
    state = BraintrustState()
    old_state = logger.login_to_state(api_key=logger.TEST_API_KEY, org_name="old")
    new_state = logger.login_to_state(api_key=logger.TEST_API_KEY, org_name="new")
    entered = threading.Event()
    release = threading.Event()

    def blocked(**kwargs):
        entered.set()
        assert release.wait(5)
        return old_state

    observe, discarded = _observe_discard(state)
    with observe, patch.object(logger, "login_to_state", side_effect=blocked) as request:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(state.login, api_key=logger.TEST_API_KEY)
            assert entered.wait(5)
            queued = pool.submit(state.login, api_key=logger.TEST_API_KEY, force_login=True)
            _wait_until(lambda: len(state._login_requests) == 2)
            state.reset_login_info()
            for result in (first, queued):
                with pytest.raises(RuntimeError, match="cancelled by a reset"):
                    result.result(5)
            request.side_effect = None
            request.return_value = new_state
            state.login(api_key=logger.TEST_API_KEY)
            release.set()
            assert discarded.wait(5)
    assert state.org_name == "new"
    assert request.call_count == 2


def test_failed_login_does_not_block_queued_forced_login():
    state = BraintrustState()
    logged_in = logger.login_to_state(api_key=logger.TEST_API_KEY)
    entered = threading.Event()
    release = threading.Event()

    def fail(**kwargs):
        entered.set()
        assert release.wait(5)
        raise ValueError("login failed")

    with patch.object(logger, "login_to_state") as request:

        def dispatch(**kwargs):
            return fail(**kwargs) if request.call_count == 1 else logged_in

        request.side_effect = dispatch
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(state.login, api_key=logger.TEST_API_KEY)
            assert entered.wait(5)
            second = pool.submit(state.login, api_key=logger.TEST_API_KEY, force_login=True)
            _wait_until(lambda: len(state._login_requests) == 2)
            release.set()
            with pytest.raises(ValueError, match="login failed"):
                first.result(5)
            second.result(5)
    assert state.logged_in
    assert request.call_count == 2


def test_login_reset_during_credential_discovery():
    state = BraintrustState()

    def credentials(api_key):
        state.reset_login_info()
        return logger.TEST_API_KEY

    with patch.object(logger, "_require_api_key", side_effect=credentials):
        with pytest.raises(RuntimeError, match="cancelled by a reset"):
            state.login()
    assert not state.logged_in


def test_project_lookup_reset_before_client_capture(project_state):
    state, client = project_state
    with patch.object(state, "login", side_effect=state.reset_login_info):
        with pytest.raises(RuntimeError, match="cancelled by a login reset"):
            _compute_logger_metadata(project_name="project", state=state)
    client.projects.post_project.assert_not_called()


@pytest.mark.parametrize("kind", ["login", "project"])
def test_stalled_requests_timeout_and_allow_retry(project_state, monkeypatch, kind):
    state, client = project_state
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if kind == "login":
            return old_state
        return {"id": "stale-id", "name": "project"}

    if kind == "login":
        old_state = logger.login_to_state(api_key=logger.TEST_API_KEY, org_name="old")
        state.reset_login_info()
        monkeypatch.setattr(logger, "_LOGIN_TIMEOUT_SECONDS", 0.05)
        target = patch.object(logger, "login_to_state", side_effect=blocked)
        invoke = lambda: state.login(api_key=logger.TEST_API_KEY)
    else:
        monkeypatch.setattr(logger, "_PROJECT_LOOKUP_TIMEOUT_SECONDS", 0.05)
        target = patch.object(client.projects, "post_project", side_effect=blocked)
        invoke = lambda: _compute_logger_metadata(project_name="project", state=state)
    observe, discarded = _observe_discard(state)
    with observe:
        with target:
            with pytest.raises(TimeoutError, match="timed out"):
                invoke()
            assert entered.is_set()
        if kind == "login":
            state.login(api_key=logger.TEST_API_KEY, org_name="new")
        else:
            assert invoke().project.id == "project-id"
        release.set()
        assert discarded.wait(5)
    if kind == "login":
        assert state.org_name == "new"
    else:
        assert invoke().project.id == "project-id"
