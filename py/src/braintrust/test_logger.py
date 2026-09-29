# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
import asyncio
import builtins
import importlib
import inspect
import json
import logging
import operator
import os
import sys
import threading
import time
from collections.abc import AsyncGenerator
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import braintrust
import exceptiongroup
import pytest
import requests
from braintrust import (
    Attachment,
    BaseAttachment,
    ExternalAttachment,
    JSONAttachment,
    LazyValue,
    Prompt,
    init_logger,
    logger,
)
from braintrust.api import BraintrustTransportError
from braintrust.db_fields import AUDIT_METADATA_FIELD
from braintrust.git_fields import GitMetadataSettings, RepoInfo
from braintrust.gitutil import get_repo_info
from braintrust.id_gen import get_id_generator
from braintrust.logger import (
    BraintrustState,
    RemoteEvalParameters,
    _check_org_info,
    _extract_attachments,
    parent_context,
    render_message,
    stringify_exception,
)
from braintrust.prompt import PromptChatBlock, PromptData, PromptMessage, PromptSchema
from braintrust.prompt_cache.lru_cache import LRUCache
from braintrust.prompt_cache.parameters_cache import ParametersCache
from braintrust.prompt_cache.prompt_cache import PromptCache
from braintrust.test_helpers import (
    assert_dict_matches,
    assert_logged_out,
    init_test_exp,
    init_test_logger,
    simulate_login,  # noqa: F401 # type: ignore[reportUnusedImport]
    simulate_logout,
    with_memory_logger,  # noqa: F401 # type: ignore[reportUnusedImport]
    with_simulate_login,  # noqa: F401 # type: ignore[reportUnusedImport]
)
from braintrust.util import AugmentedHTTPError
from requests import HTTPError
from requests.exceptions import ConnectionError, SSLError


def test_login_to_state_uses_env_braintrust_api_key(tmp_path, monkeypatch):
    (tmp_path / ".env.braintrust").write_text(f"BRAINTRUST_API_KEY={logger.TEST_API_KEY}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)

    state = logger.login_to_state(org_name="test-org-name")

    assert state.login_token == logger.TEST_API_KEY
    assert state.logged_in is True


def _loader_options(api_key: str, cache_namespace: str) -> logger._LoaderLoginOptions:
    return logger._LoaderLoginOptions(
        app_url="https://app.example.com",
        api_key=api_key,
        org_name=None,
        cache_namespace=cache_namespace,
    )


def test_loader_request_state_closes_openapi_clients_on_eviction_and_reset():
    state = BraintrustState()
    state._loader_api_client_cache = LRUCache(max_size=1, on_remove=state._evict_loader_login_entry)
    first_client = MagicMock()
    second_client = MagicMock()

    with patch.object(logger, "_login_loader_client", side_effect=[first_client, second_client]):
        with state.loader_api_client(_loader_options("first-api-key", "first")) as api_client:
            assert api_client is first_client.openapi
        with state.loader_api_client(_loader_options("second-api-key", "second")) as api_client:
            assert api_client is second_client.openapi

    first_client.close.assert_called_once()
    second_client.close.assert_not_called()

    state.reset_login_info()

    second_client.close.assert_called_once()


class TestInit(TestCase):
    @staticmethod
    def _mock_api_client():
        api_client = MagicMock()
        api_client.projects.post_project.return_value = {
            "id": "test-project-id",
            "name": "test-project",
        }
        api_client.experiments.post_experiment.return_value = {
            "id": "test-exp-id",
            "name": "test-exp",
            "project_id": "test-project-id",
            "public": False,
        }
        return api_client

    def test_init_validation(self):
        with self.assertRaises(ValueError) as cm:
            braintrust.init()

        assert str(cm.exception) == "Must specify at least one of project or project_id"

        with self.assertRaises(ValueError) as cm:
            braintrust.init(project="project", open=True, update=True)

        assert str(cm.exception) == "Cannot open and update an experiment at the same time"

        with self.assertRaises(ValueError) as cm:
            braintrust.init(project="project", open=True)

        assert str(cm.exception) == "Cannot open an experiment without specifying its name"

        # duplicate tags
        tag = "Exp1"
        with self.assertRaises(ValueError) as cm:
            braintrust.init(project="project", tags=[tag, tag])

        assert str(cm.exception) == f"duplicate tag: {tag}"

        # Test 2: full Dataset object has different behavior
        # (We can't easily instantiate a Dataset here, but we can verify
        # that the isinstance check distinguishes them)

    def test_init_with_repo_info_does_not_raise(self):
        """Test that passing repo_info to init() doesn't cause an UnboundLocalError.

        Regression test for a bug where merged_git_metadata_settings was only
        defined in the else branch (when repo_info is falsy), but referenced
        unconditionally later in compute_metadata().
        ref: https://github.com/braintrustdata/braintrust-sdk-python/issues/8
        """
        api_client = self._mock_api_client()

        from braintrust.git_fields import RepoInfo

        repo_info = RepoInfo(commit="abc123", branch="main", dirty=False)

        simulate_login()
        with patch.object(logger._state, "api_client", return_value=api_client):
            exp = braintrust.init(project="test-project", repo_info=repo_info)

            # Force compute_metadata() to execute. This would raise
            # UnboundLocalError before the fix.
            metadata = exp._lazy_metadata.get()

        assert metadata.project.id == "test-project-id"
        assert metadata.experiment.name == "test-exp"

    def test_init_enable_atexit_flush(self):
        from braintrust.logger import _HTTPBackgroundLogger

        api_con_response = lambda: {
            "project": {"id": "test-project-id", "name": "test-project"},
            "experiment": {"id": "test-exp-id", "name": "test-exp"},
        }

        with patch("atexit.register") as mock_register:
            _HTTPBackgroundLogger(LazyValue(api_con_response, use_mutex=False))  # type: ignore
            mock_register.assert_called()

    def test_init_disable_atexit_flush(self):
        from braintrust.logger import _HTTPBackgroundLogger

        api_con_response = lambda: {
            "project": {"id": "test-project-id", "name": "test-project"},
            "experiment": {"id": "test-exp-id", "name": "test-exp"},
        }

        with patch.dict(os.environ, {"BRAINTRUST_DISABLE_ATEXIT_FLUSH": "True"}):
            with patch("atexit.register") as mock_register:
                _HTTPBackgroundLogger(LazyValue(api_con_response, use_mutex=False))  # type: ignore
                mock_register.assert_not_called()

        with patch.dict(os.environ, {"BRAINTRUST_DISABLE_ATEXIT_FLUSH": "1"}):
            with patch("atexit.register") as mock_register:
                _HTTPBackgroundLogger(LazyValue(api_con_response, use_mutex=False))  # type: ignore
                mock_register.assert_not_called()

        with patch.dict(os.environ, {"BRAINTRUST_DISABLE_ATEXIT_FLUSH": "yes"}):
            with patch("atexit.register") as mock_register:
                _HTTPBackgroundLogger(LazyValue(api_con_response, use_mutex=False))  # type: ignore
                mock_register.assert_not_called()

    def test_init_without_git_metadata_override_uses_org_policy(self):
        for org_settings in (
            GitMetadataSettings(collect="all"),
            GitMetadataSettings(collect="some", fields=["commit", "branch"]),
        ):
            with self.subTest(org_settings=org_settings):
                api_client = self._mock_api_client()

                simulate_login()
                logger._state.git_metadata_settings = org_settings
                with patch.object(logger._state, "api_client", return_value=api_client):
                    with patch(
                        "braintrust.logger.get_repo_info", return_value=RepoInfo(commit="abc123")
                    ) as mock_get_repo_info:
                        exp = braintrust.init(project="test-project")
                        exp._lazy_metadata.get()

                actual_settings = mock_get_repo_info.call_args.args[0]
                assert actual_settings == org_settings

    def test_init_git_metadata_override_merges_with_org_policy(self):
        api_client = self._mock_api_client()

        simulate_login()
        logger._state.git_metadata_settings = GitMetadataSettings(collect="some", fields=["commit", "branch"])
        with patch.object(logger._state, "api_client", return_value=api_client):
            with patch("braintrust.logger.get_repo_info", return_value=None) as mock_get_repo_info:
                exp = braintrust.init(
                    project="test-project",
                    git_metadata_settings=GitMetadataSettings(collect="some", fields=["commit"]),
                )
                exp._lazy_metadata.get()

        actual_settings = mock_get_repo_info.call_args.args[0]
        assert actual_settings == GitMetadataSettings(collect="some", fields=["commit"])

    def test_init_without_git_metadata_policy_collects_none(self):
        api_client = self._mock_api_client()

        simulate_login()
        assert logger._state.git_metadata_settings is None
        with patch.object(logger._state, "api_client", return_value=api_client):
            with patch("braintrust.logger.get_repo_info", return_value=None) as mock_get_repo_info:
                exp = braintrust.init(project="test-project")
                exp._lazy_metadata.get()

        actual_settings = mock_get_repo_info.call_args.args[0]
        assert actual_settings == GitMetadataSettings(collect="none")

    def test_init_with_saved_parameters_attaches_reference(self):
        api_client = self._mock_api_client()

        parameters = RemoteEvalParameters(
            id="params-123",
            project_id="project-123",
            name="Saved parameters",
            slug="saved-parameters",
            version="v1",
            schema={},
            data={},
        )

        simulate_login()
        with patch.object(logger._state, "api_client", return_value=api_client):
            exp = braintrust.init(project="test-project", parameters=parameters)
            exp._lazy_metadata.get()

        payload = api_client.experiments.post_experiment.call_args.kwargs["body"]
        assert payload["parameters_id"] == "params-123"
        assert payload["parameters_version"] == "v1"


class TestHTTPBackgroundLoggerLogs3(TestCase):
    def test_submit_logs_request_413_skips_retries(self) -> None:
        """Any 413 while publishing ``/logs3`` cannot succeed on retry with the same payload.

        ``sync_flush`` controls whether the terminal failure raises instead of printing.
        """
        from braintrust.logger import (
            LogItemWithMeta,
            Logs3OverflowInputRow,
            _HTTPBackgroundLogger,
        )

        item = LogItemWithMeta(
            str_value="{}",
            overflow_meta=Logs3OverflowInputRow(
                object_ids={},
                has_comment=False,
                is_delete=False,
                byte_size=2,
            ),
        )
        max_result = {"max_request_size": 10**9, "can_use_overflow": True}

        for response_text in ("Request Too Long", "", "Payload Too Large"):
            for sync_flush in (False, True):
                with self.subTest(response_text=response_text, sync_flush=sync_flush):
                    mock_resp = MagicMock()
                    mock_resp.ok = False
                    mock_resp.status_code = 413
                    mock_resp.text = response_text

                    mock_conn = MagicMock()
                    mock_conn.post.return_value = mock_resp

                    bg = _HTTPBackgroundLogger(LazyValue(lambda: mock_conn, use_mutex=False))
                    bg.num_tries = 5
                    bg.sync_flush = sync_flush
                    bg.failed_publish_payloads_dir = "/tmp/failed-payloads"

                    with patch.object(_HTTPBackgroundLogger, "_write_payload_to_dir") as mock_write_payload:
                        if sync_flush:
                            with self.assertRaises(Exception) as cm:
                                bg._submit_logs_request([item], max_result)
                            self.assertIn("413", str(cm.exception))
                        else:
                            bg._submit_logs_request([item], max_result)

                    self.assertEqual(mock_conn.post.call_count, 1)
                    mock_write_payload.assert_called_once()
                    self.assertEqual(
                        mock_write_payload.call_args.kwargs["payload_dir"], bg.failed_publish_payloads_dir
                    )


def test_load_prompt_async_signature_matches_load_prompt():
    assert (
        inspect.signature(braintrust.load_prompt_async).parameters
        == inspect.signature(braintrust.load_prompt).parameters
    )


def _prompt_response(slug: str):
    return {
        "objects": [
            {
                "id": f"prompt-{slug}",
                "project_id": "project-123",
                "name": "Saved prompt",
                "slug": slug,
                "_xact_id": "v1",
                "description": None,
                "tags": None,
                "prompt_data": {
                    "prompt": {
                        "type": "chat",
                        "messages": [{"role": "user", "content": "Hello {{name}}"}],
                    },
                    "options": {"model": "gpt-5-mini"},
                },
            }
        ]
    }


def _parameters_response(slug: str):
    return {
        "objects": [
            {
                "id": f"parameters-{slug}",
                "project_id": "project-123",
                "name": "Saved parameters",
                "slug": slug,
                "_xact_id": "v1",
                "function_data": {
                    "type": "parameters",
                    "data": {"prefix": slug},
                    "__schema": {"type": "object"},
                },
            }
        ]
    }


def _http_error(status_code: int) -> AugmentedHTTPError:
    error = AugmentedHTTPError(f"HTTP {status_code}")
    error.__cause__ = HTTPError(response=MagicMock(status_code=status_code))
    return error


def test_load_prompt_uses_explicit_api_key_without_changing_global_login():
    simulate_login()
    original_login_token = logger._state.login_token
    prompt_cache = PromptCache(memory_cache=LRUCache(max_size=10))
    request_client = MagicMock()
    request_client.openapi.prompts.get_prompt.return_value = _prompt_response("saved-prompt")

    with (
        patch.object(logger._state, "_prompt_cache", prompt_cache),
        patch.object(logger, "_login_loader_client", return_value=request_client) as mock_login_client,
    ):
        prompt = braintrust.load_prompt(
            project="test-project",
            slug="saved-prompt",
            api_key="prompt-api-key",
        )
        assert prompt.slug == "saved-prompt"

    request_client.openapi.prompts.get_prompt.assert_called_once_with(
        project_name="test-project",
        project_id=None,
        slug="saved-prompt",
        version=None,
        environment=None,
    )
    (called_options,) = mock_login_client.call_args.args
    assert called_options.app_url == logger._state.app_url
    assert called_options.api_key == "prompt-api-key"
    assert called_options.org_name is None
    assert logger._state.login_token == original_login_token


@pytest.mark.parametrize("configured_timeout", [0.25, 120.0])
def test_load_prompt_preserves_configured_http_timeout(monkeypatch, configured_timeout):
    monkeypatch.setenv("BRAINTRUST_HTTP_TIMEOUT", str(configured_timeout))
    simulate_login()
    response = requests.Response()
    response.status_code = 200
    response.url = "https://api.example.com/v1/prompt"
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(_prompt_response("saved-prompt")).encode()

    assert logger._state._client is not None
    with patch.object(logger._state._client.transport.session, "request", return_value=response) as request:
        prompt = braintrust.load_prompt(project="test-project", slug="saved-prompt")
        assert prompt.slug == "saved-prompt"

    assert request.call_args.kwargs["timeout"] == configured_timeout


def test_load_prompt_by_id_reports_an_empty_response_as_not_found():
    simulate_login()
    mock_api_client = MagicMock()
    mock_api_client.prompts.get_prompt_id.return_value = None

    with patch.object(logger._state, "api_client", return_value=mock_api_client):
        prompt = braintrust.load_prompt(id="missing-prompt")
        with pytest.raises(ValueError, match="Prompt with id missing-prompt not found"):
            _ = prompt.id


def test_load_parameters_uses_explicit_api_key_without_changing_global_login():
    simulate_login()
    original_login_token = logger._state.login_token
    parameters_cache = ParametersCache(memory_cache=LRUCache(max_size=10))
    request_client = MagicMock()
    request_client.openapi.functions.get_function.return_value = _parameters_response("saved-parameters")

    with (
        patch.object(logger._state, "_parameters_cache", parameters_cache),
        patch.object(logger, "_login_loader_client", return_value=request_client) as mock_login_client,
    ):
        parameters = braintrust.load_parameters(
            project="test-project",
            slug="saved-parameters",
            api_key="parameters-api-key",
        )

    assert parameters.data == {"prefix": "saved-parameters"}
    request_client.openapi.functions.get_function.assert_called_once_with(
        project_name="test-project",
        project_id=None,
        slug="saved-parameters",
        version=None,
        environment=None,
    )
    (called_options,) = mock_login_client.call_args.args
    assert called_options.app_url == logger._state.app_url
    assert called_options.api_key == "parameters-api-key"
    assert called_options.org_name is None
    assert logger._state.login_token == original_login_token


def test_load_parameters_filters_non_parameter_functions():
    simulate_login()
    mock_api_client = MagicMock()
    parameter = _parameters_response("saved-parameters")["objects"][0]
    mock_api_client.functions.get_function.return_value = {
        "objects": [
            {"id": "scorer-123", "function_data": {"type": "global"}},
            parameter,
        ]
    }

    with patch.object(logger._state, "api_client", return_value=mock_api_client):
        parameters = braintrust.load_parameters(project="test-project", slug="saved-parameters")

    assert parameters.id == "parameters-saved-parameters"
    assert parameters.data == {"prefix": "saved-parameters"}


def test_load_parameters_rejects_non_parameter_function():
    simulate_login()
    mock_api_client = MagicMock()
    mock_api_client.functions.get_function.return_value = {
        "objects": [{"id": "scorer-123", "function_data": {"type": "global"}}]
    }

    with patch.object(logger._state, "api_client", return_value=mock_api_client):
        with pytest.raises(ValueError, match="Parameters saved-parameters not found"):
            braintrust.load_parameters(project="test-project", slug="saved-parameters")


@pytest.mark.parametrize(
    "server_error",
    [
        _http_error(401),
        _http_error(501),
        json.JSONDecodeError("invalid JSON", "", 0),
        SSLError("invalid certificate"),
    ],
)
def test_load_prompt_does_not_fall_back_to_cache_for_non_transient_errors(server_error):
    simulate_login()
    prompt_cache = PromptCache(memory_cache=LRUCache(max_size=10))
    request_client = MagicMock()
    request_client.openapi.prompts.get_prompt.side_effect = [_prompt_response("saved-prompt"), server_error]

    with (
        patch.object(logger._state, "_prompt_cache", prompt_cache),
        patch.object(logger, "_login_loader_client", return_value=request_client),
    ):
        first_prompt = braintrust.load_prompt(
            project="test-project",
            slug="saved-prompt",
            api_key="prompt-api-key",
        )
        assert first_prompt.slug == "saved-prompt"

        second_prompt = braintrust.load_prompt(
            project="test-project",
            slug="saved-prompt",
            api_key="prompt-api-key",
        )
        with pytest.raises(type(server_error)):
            _ = second_prompt.slug


def _wrapped_transport_error() -> BraintrustTransportError:
    error = BraintrustTransportError(
        method="GET",
        url="https://api.example.com/v1/prompt",
        attempts=1,
        retryable=False,
    )
    error.__cause__ = ConnectionError("custom adapter exhausted its retries")
    return error


@pytest.mark.parametrize(
    "server_error",
    [
        pytest.param(_http_error(500), id="http-500"),
        pytest.param(_wrapped_transport_error(), id="wrapped-connection-error"),
    ],
)
def test_load_prompt_falls_back_to_same_api_keys_cache_for_transient_errors(server_error):
    simulate_login()
    prompt_cache = PromptCache(memory_cache=LRUCache(max_size=10))
    request_client = MagicMock()
    request_client.openapi.prompts.get_prompt.side_effect = [_prompt_response("saved-prompt"), server_error]

    with (
        patch.object(logger._state, "_prompt_cache", prompt_cache),
        patch.object(logger, "_login_loader_client", return_value=request_client),
    ):
        first_prompt = braintrust.load_prompt(
            project="test-project",
            slug="saved-prompt",
            api_key="prompt-api-key",
        )
        assert first_prompt.slug == "saved-prompt"

        cached_prompt = braintrust.load_prompt(
            project="test-project",
            slug="saved-prompt",
            api_key="prompt-api-key",
        )
        assert cached_prompt.slug == "saved-prompt"

    assert request_client.openapi.prompts.get_prompt.call_count == 2


def test_load_prompt_does_not_use_another_api_keys_transient_fallback_cache():
    simulate_login()
    prompt_cache = PromptCache(memory_cache=LRUCache(max_size=10))
    first_client = MagicMock()
    first_client.openapi.prompts.get_prompt.return_value = _prompt_response("saved-prompt")
    second_client = MagicMock()
    second_client.openapi.prompts.get_prompt.side_effect = _http_error(500)
    clients = {"first-api-key": first_client, "second-api-key": second_client}

    def login_for_api_key(options):
        return clients[options.api_key]

    with (
        patch.object(logger._state, "_prompt_cache", prompt_cache),
        patch.object(logger, "_login_loader_client", side_effect=login_for_api_key),
    ):
        first_prompt = braintrust.load_prompt(
            project="test-project",
            slug="saved-prompt",
            api_key="first-api-key",
        )
        assert first_prompt.slug == "saved-prompt"

        second_prompt = braintrust.load_prompt(
            project="test-project",
            slug="saved-prompt",
            api_key="second-api-key",
        )
        with pytest.raises(ValueError, match="not found on server or in local cache"):
            _ = second_prompt.slug


@pytest.mark.asyncio
async def test_load_prompt_async_eagerly_fetches_prompt(with_simulate_login):
    mock_api_client = MagicMock()
    mock_api_client.prompts.get_prompt.return_value = _prompt_response("saved-prompt")

    with patch.object(logger._state, "api_client", return_value=mock_api_client):
        prompt = await braintrust.load_prompt_async(
            project="test-project",
            slug="saved-prompt",
        )

        # Unlike load_prompt(), load_prompt_async() resolves the prompt metadata before returning.
        mock_api_client.prompts.get_prompt.assert_called_once_with(
            project_name="test-project",
            project_id=None,
            slug="saved-prompt",
            version=None,
            environment=None,
        )
        assert prompt.slug == "saved-prompt"
        assert prompt.build(name="Ada")["messages"][0]["content"] == "Hello Ada"


@pytest.mark.asyncio
async def test_load_prompt_async_loads_prompts_in_parallel(with_simulate_login):
    mock_api_client = MagicMock()
    barrier = threading.Barrier(2, timeout=1)

    def get_prompt(**kwargs):
        barrier.wait()
        return _prompt_response(kwargs["slug"])

    mock_api_client.prompts.get_prompt.side_effect = get_prompt

    with patch.object(logger._state, "api_client", return_value=mock_api_client):
        prompt1, prompt2 = await asyncio.gather(
            braintrust.load_prompt_async(project="test-project", slug="prompt-1"),
            braintrust.load_prompt_async(project="test-project", slug="prompt-2"),
        )

    assert [prompt1.slug, prompt2.slug] == ["prompt-1", "prompt-2"]
    assert mock_api_client.prompts.get_prompt.call_count == 2


@pytest.mark.parametrize(
    ("load", "endpoint", "response", "lookup", "expected_call"),
    [
        pytest.param(
            braintrust.load_prompt,
            "prompts.get_prompt",
            _prompt_response("saved-prompt"),
            {"project": "test-project", "slug": "saved-prompt"},
            call(project_name="test-project", project_id=None, slug="saved-prompt", version="v1", environment=None),
            id="prompt-by-slug",
        ),
        pytest.param(
            braintrust.load_prompt,
            "prompts.get_prompt_id",
            _prompt_response("saved-prompt")["objects"][0],
            {"id": "prompt-saved-prompt"},
            call("prompt-saved-prompt", version="v1", environment=None),
            id="prompt-by-id",
        ),
        pytest.param(
            braintrust.load_parameters,
            "functions.get_function",
            _parameters_response("saved-parameters"),
            {"project": "test-project", "slug": "saved-parameters"},
            call(
                project_name="test-project", project_id=None, slug="saved-parameters", version="v1", environment=None
            ),
            id="parameters-by-slug",
        ),
        pytest.param(
            braintrust.load_parameters,
            "functions.get_function_id",
            _parameters_response("saved-parameters")["objects"][0],
            {"id": "parameters-saved-parameters"},
            call("parameters-saved-parameters", version="v1", environment=None),
            id="parameters-by-id",
        ),
    ],
)
def test_load_prefers_version_over_environment(load, endpoint, response, lookup, expected_call):
    simulate_login()
    mock_api_client = MagicMock()
    endpoint_mock = operator.attrgetter(endpoint)(mock_api_client)
    endpoint_mock.return_value = response

    with patch.object(logger._state, "api_client", return_value=mock_api_client):
        loaded = load(**lookup, version="v1", environment="production")
        # Prompts load lazily; reading an attribute forces the request.
        record = response["objects"][0] if "objects" in response else response
        assert loaded.id == record["id"]

    assert endpoint_mock.call_args_list == [expected_call]


def test_load_parameters_returns_remote_object():
    simulate_login()
    mock_api_client = MagicMock()
    mock_api_client.functions.get_function.return_value = _parameters_response("saved-parameters")

    with patch.object(logger._state, "api_client", return_value=mock_api_client):
        parameters = braintrust.load_parameters(project="test-project", slug="saved-parameters")

    assert isinstance(parameters, RemoteEvalParameters)
    assert parameters.id == "parameters-saved-parameters"
    assert parameters.version == "v1"
    assert parameters.data == {"prefix": "saved-parameters"}
    cache_namespace = logger._resolve_loader_login_options(
        app_url=None,
        api_key=None,
        org_name=None,
    ).cache_namespace
    cached = logger._state._parameters_cache.get(
        slug="saved-parameters",
        version="latest",
        project_name="test-project",
        cache_namespace=cache_namespace,
    )
    assert cached.id == "parameters-saved-parameters"


def test_extract_attachments_no_op():
    attachments: list[BaseAttachment] = []

    _extract_attachments({}, attachments)
    assert len(attachments) == 0

    event = {"foo": "foo", "bar": None, "baz": [1, 2, 3]}
    baz = event["baz"]
    _extract_attachments(event, attachments)
    assert len(attachments) == 0
    assert event["baz"] is baz
    assert event == {"foo": "foo", "bar": None, "baz": [1, 2, 3]}


def test_extract_attachments_with_attachments():
    attachment1 = Attachment(
        data=b"data",
        filename="filename",
        content_type="text/plain",
    )
    attachment2 = Attachment(
        data=b"data2",
        filename="filename2",
        content_type="text/plain",
    )
    attachment3 = ExternalAttachment(
        url="s3://bucket/path/to/key.pdf",
        filename="filename3",
        content_type="application/pdf",
    )
    date = "2024-10-23T05:02:48.796Z"
    event = {
        "foo": "bar",
        "baz": [1, 2],
        "attachment1": attachment1,
        "attachment3": attachment3,
        "nested": {
            "attachment2": attachment2,
            "attachment3": attachment3,
            "info": "another string",
            "anArray": [
                attachment1,
                None,
                "string",
                attachment2,
                attachment1,
                attachment3,
                attachment3,
            ],
        },
        "null": None,
        "undefined": None,
        "date": date,
        "f": "Math.max",
        "empty": {},
    }
    saved_nested = event["nested"]

    attachments: list[BaseAttachment] = []
    _extract_attachments(event, attachments)

    expected = [
        attachment1,
        attachment3,
        attachment2,
        attachment3,
        attachment1,
        attachment2,
        attachment1,
        attachment3,
        attachment3,
    ]
    assert len(attachments) == len(expected)
    assert all(actual is want for actual, want in zip(attachments, expected))

    assert event["nested"] is saved_nested

    assert event == {
        "foo": "bar",
        "baz": [1, 2],
        "attachment1": attachment1.reference,
        "attachment3": attachment3.reference,
        "nested": {
            "attachment2": attachment2.reference,
            "attachment3": attachment3.reference,
            "info": "another string",
            "anArray": [
                attachment1.reference,
                None,
                "string",
                attachment2.reference,
                attachment1.reference,
                attachment3.reference,
                attachment3.reference,
            ],
        },
        "null": None,
        "undefined": None,
        "date": date,
        "f": "Math.max",
        "empty": {},
    }


def _test_prompt(content: str, options: dict | None = None) -> Prompt:
    """Create a lazily loaded chat prompt with a single user message."""
    prompt_schema = PromptSchema(
        id="test-id",
        project_id="test-project",
        _xact_id="test-xact",
        name="test-prompt",
        slug="test-prompt",
        description="test",
        prompt_data=PromptData(
            prompt=PromptChatBlock(messages=[PromptMessage(role="user", content=content)]),
            options=options or {"model": "gpt-4o"},
        ),
        tags=None,
    )
    return Prompt(LazyValue(lambda: prompt_schema, use_mutex=False), {}, False)


def test_prompt_build_with_structured_output_templating():
    prompt = _test_prompt(
        "Please compute {{input.expression}} and return the result in JSON.",
        options={
            "model": "gpt-4o",
            "params": {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema",
                        "schema": "{{input.schema}}",
                        "strict": True,
                    },
                },
            },
        },
    )
    schema = {
        "type": "object",
        "properties": {
            "final_answer": {"type": "string"},
        },
        "required": ["final_answer"],
        "additionalProperties": False,
    }

    result = prompt.build(input={"expression": "2 + 3", "schema": schema})

    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "schema",
            "schema": schema,
            "strict": True,
        },
    }


@pytest.mark.parametrize(
    ("template", "args", "expected"),
    [
        pytest.param(
            "Hello {{name}}, please help with {{task}}",
            {"name": "John", "task": "coding"},
            "Hello John, please help with coding",
            id="flat",
        ),
        pytest.param(
            "User {{user.name}} with email {{user.profile.email}}",
            {"user": {"name": "John", "profile": {"email": "john@example.com"}}},
            "User John with email john@example.com",
            id="nested",
        ),
        pytest.param(
            "Items: {{items.0}}, {{items.1}}",
            {"items": ["first", "second", "third"]},
            "Items: first, second",
            id="array",
        ),
    ],
)
def test_prompt_build_strict_renders_present_variables(template, args, expected):
    result = _test_prompt(template).build(strict=True, **args)
    assert result["messages"][0]["content"] == expected


@pytest.mark.parametrize(
    ("template", "args", "error"),
    [
        pytest.param(
            "Hello {{name}}, please help with {{task}}",
            {"name": "John"},
            "Template rendering failed: Could not find key 'task'",
            id="flat",
        ),
        pytest.param(
            "User {{user.name}} with email {{user.profile.email}}",
            {"user": {"name": "John"}},
            "Template rendering failed",
            id="nested",
        ),
        pytest.param(
            "Items: {{items.0}}, {{items.1}}",
            {"items": ["only_one"]},
            "Template rendering failed",
            id="array",
        ),
    ],
)
def test_prompt_build_strict_rejects_missing_variables(template, args, error):
    with pytest.raises(ValueError, match=error):
        _test_prompt(template).build(strict=True, **args)


def test_prompt_build_non_strict_renders_missing_variables_as_empty():
    result = _test_prompt("Hello {{name}}, please help with {{task}}").build(name="John")
    assert result["messages"][0]["content"] == "Hello John, please help with "


def test_render_message_with_file_content_parts():
    """Test render_message with mixed text, image, and file content parts including all file fields."""
    message = PromptMessage(
        role="user",
        content=[
            {"type": "text", "text": "Here is a {{item}}:"},
            {"type": "image_url", "image_url": {"url": "{{image_url}}"}},
            {
                "type": "file",
                "file": {
                    "file_data": "{{file_data}}",
                    "file_id": "{{file_id}}",
                    "filename": "{{filename}}",
                },
            },
        ],
    )

    rendered = render_message(
        lambda template: (
            template.replace("{{item}}", "document")
            .replace("{{image_url}}", "https://example.com/image.png")
            .replace("{{file_data}}", "base64data")
            .replace("{{file_id}}", "file-456")
            .replace("{{filename}}", "report.pdf")
        ),
        message,
    )

    assert rendered["content"] == [
        {"type": "text", "text": "Here is a document:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {
            "type": "file",
            "file": {
                "file_data": "base64data",
                "file_id": "file-456",
                "filename": "report.pdf",
            },
        },
    ]


def test_noop_permalink_issue_1837():
    # fixes issue #BRA-1837
    span = braintrust.NOOP_SPAN
    assert span.permalink() == "https://www.braintrust.dev/noop-span"

    link = braintrust.permalink(span.export())
    assert link == "https://www.braintrust.dev/noop-span"

    assert span.link() == "https://www.braintrust.dev/noop-span"


def test_span_log_accepts_pydantic_model_metadata(with_memory_logger):
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not available")

    class MetadataModel(BaseModel):
        foo: str = "bar"

    logger = init_test_logger(__name__)

    with logger.start_span(name="test_span") as span:
        span.log(input=MetadataModel(), metadata=MetadataModel())

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    assert logs[0]["input"] == {"foo": "bar"}
    assert logs[0]["metadata"] == {"foo": "bar"}


class _ModelDumpMetadata:
    def __init__(self, **values):
        self.values = values

    def model_dump(self, **kwargs):
        assert kwargs == {"exclude_none": True}
        return dict(self.values)


def _init_test_dataset():
    from braintrust.logger import Dataset, ObjectMetadata, ProjectDatasetMetadata

    project_metadata = ObjectMetadata(id="test_project", name="test_project", full_info={})
    dataset_metadata = ObjectMetadata(id="test_dataset", name="test_dataset", full_info={})
    metadata = ProjectDatasetMetadata(project=project_metadata, dataset=dataset_metadata)
    return Dataset(lazy_metadata=LazyValue(lambda: metadata, use_mutex=False))


def _span_log_metadata(metadata):
    with init_test_logger(__name__).start_span(name="test_span") as span:
        span.log(metadata=metadata)


@pytest.mark.parametrize(
    ("log_with_metadata", "metadata_field"),
    [
        pytest.param(_span_log_metadata, "metadata", id="span.log"),
        pytest.param(
            lambda m: init_test_logger(__name__).log(input="input", output="output", metadata=m),
            "metadata",
            id="logger.log",
        ),
        pytest.param(
            lambda m: init_test_exp("test-experiment", "test-project").log(
                input="input", output="output", scores={"score": 1}, metadata=m
            ),
            "metadata",
            id="experiment.log",
        ),
        pytest.param(
            lambda m: init_test_logger(__name__).log_feedback(id="event-id", scores={"score": 1}, metadata=m),
            AUDIT_METADATA_FIELD,
            id="logger.log_feedback",
        ),
        pytest.param(
            lambda m: init_test_exp("test-experiment", "test-project").log_feedback(
                id="event-id", scores={"score": 1}, metadata=m
            ),
            AUDIT_METADATA_FIELD,
            id="experiment.log_feedback",
        ),
        pytest.param(
            lambda m: _init_test_dataset().insert(input="input", expected="expected", metadata=m),
            "metadata",
            id="dataset.insert",
        ),
        pytest.param(
            lambda m: _init_test_dataset().update(id="record-id", metadata=m),
            "metadata",
            id="dataset.update",
        ),
    ],
)
def test_logging_apis_accept_model_dump_metadata(with_memory_logger, log_with_metadata, metadata_field):
    log_with_metadata(_ModelDumpMetadata(foo="bar"))

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    assert logs[0][metadata_field] == {"foo": "bar"}


def test_logger_emit_log_without_active_span(with_memory_logger):
    test_logger = init_test_logger(__name__)

    first_id = test_logger.emit_log(
        body="Payment failed",
        level="error",
        metadata={"payment_id": "pay_123"},
    )
    second_id = test_logger.emit_log(body="Retrying payment", level="info")

    logs = with_memory_logger.pop()
    assert len(logs) == 2
    first, second = logs
    assert first_id == first["id"]
    assert second_id == second["id"]
    assert first["id"] != second["id"]
    assert first["span_id"] != second["span_id"]
    assert first["root_span_id"] == second["root_span_id"]
    assert not first.get("span_parents")
    assert first["output"] == "Payment failed"
    assert "error" not in first
    assert first["metadata"] == {"payment_id": "pay_123"}
    assert "name" not in first["span_attributes"]
    assert first["span_attributes"]["type"] == "log"
    assert first["span_attributes"]["log_level"] == "error"
    assert first["metrics"]["start"] == first["metrics"]["end"]
    assert "otel" not in first.get("context", {})
    assert "error" not in second
    assert not second.get("metadata")
    assert second["span_attributes"]["log_level"] == "info"


def test_logger_emit_log_enqueues_single_row(with_memory_logger):
    test_logger = init_test_logger(__name__)

    test_logger.info("Payment completed", metadata={"payment_id": "pay_123"})

    assert len(with_memory_logger.logs) == 1
    [row] = with_memory_logger.pop()
    assert row["metrics"]["start"] == row["metrics"]["end"]
    assert row["_is_merge"] is False


def test_logger_emit_log_uses_distinct_baseline_trace_per_logger(with_memory_logger):
    first_logger = init_test_logger(f"{__name__}-first")
    second_logger = init_test_logger(f"{__name__}-second")

    first_logger.info("first")
    second_logger.info("second")

    first, second = with_memory_logger.pop()
    assert first["root_span_id"] != second["root_span_id"]


def test_logger_emit_log_uses_active_span(with_memory_logger):
    test_logger = init_test_logger(__name__)

    with test_logger.start_span(name="owner") as owner:
        log_id = test_logger.emit_log(body="Inside span", level="debug", metadata={"attempt": 1})

    rows = with_memory_logger.pop()
    log_row = next(row for row in rows if row["id"] == log_id)
    owner_row = next(row for row in rows if row["span_attributes"]["name"] == "owner")
    assert log_row["id"] != owner_row["id"]
    assert log_row["span_id"] == owner_row["span_id"]
    assert log_row["root_span_id"] == owner_row["root_span_id"]
    assert not log_row.get("span_parents")
    assert log_row["metadata"] == {"attempt": 1}
    assert log_row["span_attributes"]["log_level"] == "debug"
    assert "otel" not in log_row.get("context", {})


@pytest.mark.parametrize("level", ["trace", "debug", "info", "warn", "error", "fatal"])
def test_logger_emit_log_adds_log_level_span_attribute(with_memory_logger, level):
    test_logger = init_test_logger(__name__)

    test_logger.emit_log(body="message", level=level)

    [row] = with_memory_logger.pop()
    assert not row.get("metadata")
    assert row["span_attributes"]["log_level"] == level


@pytest.mark.parametrize("method_name", ["trace", "debug", "info", "warn", "error", "fatal"])
def test_logger_log_level_helpers(with_memory_logger, method_name):
    test_logger = init_test_logger(__name__)

    log_id = getattr(test_logger, method_name)("message", metadata={"source": method_name})

    [row] = with_memory_logger.pop()
    assert row["id"] == log_id
    assert row["output"] == "message"
    assert row["metadata"] == {"source": method_name}
    assert row["span_attributes"]["log_level"] == method_name


def test_logger_log_helpers_render_template_parameters(with_memory_logger):
    test_logger = init_test_logger(__name__)

    log_id = test_logger.info(
        "User {user_id} paid {amount:.2f} with {method}",
        metadata={"source": "checkout"},
        user_id="user-123",
        amount=12.5,
    )

    [row] = with_memory_logger.pop()
    assert row["id"] == log_id
    assert row["output"] == "User user-123 paid 12.50 with {method}"
    assert row["metadata"] == {
        "source": "checkout",
        "braintrust.template.parameter.user_id": "user-123",
        "braintrust.template.parameter.amount": 12.5,
        "braintrust.template": "User {user_id} paid {amount:.2f} with {method}",
    }
    assert row["span_attributes"]["log_level"] == "info"


@pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings require Python 3.14+")
def test_logger_log_helpers_render_t_string(with_memory_logger):
    templatelib = importlib.import_module("string.templatelib")
    template = templatelib.Template(
        "User ",
        templatelib.Interpolation("user-123", "user_id"),
        " paid ",
        templatelib.Interpolation(12.5, "amount", "r", ">8"),
        " with {card}",
    )
    test_logger = init_test_logger(__name__)

    log_id = test_logger.info(template, metadata={"source": "checkout"})

    [row] = with_memory_logger.pop()
    assert row["id"] == log_id
    assert row["output"] == "User user-123 paid     12.5 with {card}"
    assert row["metadata"] == {
        "source": "checkout",
        "braintrust.template.parameter.user_id": "user-123",
        "braintrust.template.parameter.amount": 12.5,
        "braintrust.template": "User {user_id} paid {amount!r:>8} with {{card}}",
    }
    assert row["span_attributes"]["log_level"] == "info"


@pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings require Python 3.14+")
def test_logger_t_string_retains_repeated_expression_values(with_memory_logger):
    templatelib = importlib.import_module("string.templatelib")
    template = templatelib.Template(
        templatelib.Interpolation(1, "next(it)"),
        " ",
        templatelib.Interpolation(2, "next(it)"),
    )
    test_logger = init_test_logger(__name__)

    test_logger.info(template)

    [row] = with_memory_logger.pop()
    assert row["output"] == "1 2"
    assert row["metadata"] == {
        "braintrust.template": "{next(it)} {next(it)}",
        "braintrust.template.parameter.next(it).0": 1,
        "braintrust.template.parameter.next(it).1": 2,
    }


@pytest.mark.skipif(sys.version_info < (3, 14), reason="t-strings require Python 3.14+")
def test_logger_t_string_rejects_keyword_template_parameters(with_memory_logger):
    templatelib = importlib.import_module("string.templatelib")
    template = templatelib.Template("User ", templatelib.Interpolation("user-123", "user_id"))
    test_logger = init_test_logger(__name__)

    with pytest.raises(TypeError, match="already contain their interpolation values"):
        test_logger.info(template, user_id="other-user")

    assert with_memory_logger.pop() == []


def test_logger_error_renders_template_without_error_field(with_memory_logger):
    test_logger = init_test_logger(__name__)

    test_logger.error("Payment {payment_id} failed", payment_id="pay-123")

    [row] = with_memory_logger.pop()
    assert row["output"] == "Payment pay-123 failed"
    assert "error" not in row


def test_logger_log_helpers_do_not_format_without_parameters(with_memory_logger):
    test_logger = init_test_logger(__name__)

    test_logger.info('{"key": "{value}"}')

    [row] = with_memory_logger.pop()
    assert row["output"] == '{"key": "{value}"}'
    assert not row.get("metadata")
    assert row["span_attributes"]["log_level"] == "info"


def test_logger_log_template_parameters_are_safely_serialized(with_memory_logger):
    test_logger = init_test_logger(__name__)

    test_logger.warn("Request failed: {error}", error=ValueError("bad request"))

    [row] = with_memory_logger.pop()
    assert row["output"] == "Request failed: bad request"
    assert row["metadata"]["braintrust.template.parameter.error"] == "bad request"
    assert row["span_attributes"]["log_level"] == "warn"


def test_logger_emit_log_rejects_invalid_level(with_memory_logger):
    test_logger = init_test_logger(__name__)

    with pytest.raises(ValueError, match="Invalid log level"):
        test_logger.emit_log(body="message", level="warning")

    assert with_memory_logger.pop() == []


def test_span_log_rejects_metadata_with_non_string_keys(with_memory_logger):
    logger = init_test_logger(__name__)

    with logger.start_span(name="test_span") as span:
        with pytest.raises(ValueError, match="metadata keys must be strings"):
            span.log(metadata={1: "bad"})


def test_span_log_rejects_metadata_that_serializes_to_non_dict(with_memory_logger):
    class BadMetadata:
        def model_dump(self, **kwargs):
            assert kwargs == {"exclude_none": True}
            return ["not", "metadata"]

    logger = init_test_logger(__name__)

    with logger.start_span(name="test_span") as span:
        with pytest.raises(ValueError, match="metadata must be a dictionary or serialize to a dictionary"):
            span.log(metadata=BadMetadata())


def test_span_log_replaces_circular_references(with_memory_logger):
    """Self- and nested circular references are replaced with a placeholder instead of raising."""
    logger = init_test_logger(__name__)

    self_ref = {"key": "value"}
    self_ref["self"] = self_ref
    page = {"page_number": 1, "content": "text"}
    document = {"pages": [page]}
    page["document"] = document

    with logger.start_span(name="test_span") as span:
        span.log(input=self_ref, output=document)

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    assert logs[0]["input"]["key"] == "value"
    assert "circular" in logs[0]["input"]["self"].lower()
    logged_page = logs[0]["output"]["pages"][0]
    assert logged_page["page_number"] == 1
    assert logged_page["content"] == "text"
    assert "circular" in logged_page["document"].lower()


def test_span_log_converts_non_finite_floats_to_strings(with_memory_logger):
    """NaN and +/-Infinity are logged as strings for JSON compatibility."""
    logger = init_test_logger(__name__)

    with logger.start_span(name="test_span") as span:
        span.log(output={"nan": float("nan"), "inf": float("inf"), "neg_inf": float("-inf")})

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    assert logs[0]["output"] == {"nan": "NaN", "inf": "Infinity", "neg_inf": "-Infinity"}


def test_span_log_with_extremely_deep_nesting(with_memory_logger):
    """Test that span.log() with extremely deep nesting works gracefully."""
    import sys

    logger = init_test_logger(__name__)

    with logger.start_span(name="test_span") as span:
        recursion_limit = sys.getrecursionlimit()

        # Create structure deeper than recursion limit
        deeply_nested = {"level": 0}
        current = deeply_nested
        for i in range(1, recursion_limit + 100):
            current["nested"] = {"level": i}
            current = current["nested"]

        # Should handle extremely deep nesting without RecursionError
        span.log(
            input={"test": "deep nesting"},
            output=deeply_nested,
        )

    # Verify the log was recorded (may be truncated or have placeholder for deep nesting)
    logs = with_memory_logger.pop()
    assert len(logs) == 1

    logged_output = logs[0]["output"]
    assert logged_output["level"] == 0
    # Either the structure is preserved up to a safe depth, or replaced with placeholder
    assert "nested" in logged_output


def test_span_log_handles_unstringifiable_object_gracefully(with_memory_logger):
    """Test that span.log() handles objects whose __str__ and __repr__ raise without raising itself."""
    logger = init_test_logger(__name__)

    class BadStrObject:
        def __str__(self):
            raise RuntimeError("Cannot convert to string!")

        def __repr__(self):
            raise RuntimeError("Cannot convert to repr!")

    with logger.start_span(name="test_span") as span:
        # Should NOT raise - should handle gracefully
        span.log(
            input={"test": "input"},
            output={"result": BadStrObject()},
        )

    # Verify the log was recorded with a fallback representation
    logs = with_memory_logger.pop()
    assert len(logs) == 1
    assert logs[0]["input"]["test"] == "input"
    # The bad object should have been replaced with some error placeholder
    assert "result" in logs[0]["output"]
    output_str = str(logs[0]["output"]["result"])
    # Should contain some indication of serialization failure
    assert "error" in output_str.lower() or "serializ" in output_str.lower()


def test_span_link_logged_out(with_memory_logger):
    simulate_logout()
    assert_logged_out()
    logger = init_logger(
        project="test-project",
        project_id="test-project-id",
    )
    span = logger.start_span(name="test-span")
    span.end()
    link = span.link()
    assert link == "https://www.braintrust.dev/error-generating-link?msg=login-or-provide-org-name"


def test_span_link_logged_out_org_name(with_memory_logger):
    simulate_logout()
    assert_logged_out()
    logger = init_logger(
        project_id="test-project-id",
        org_name="test-org-name",
    )
    span = logger.start_span(name="test-span")
    span.end()
    link = span.link()
    assert (
        link
        == f"https://www.braintrust.dev/app/test-org-name/object?object_type=project_logs&object_id=test-project-id&id={span._id}"
    )


def test_span_link_logged_out_org_name_env_vars(with_memory_logger):
    simulate_logout()
    assert_logged_out()
    keys = ["BRAINTRUST_APP_URL", "BRAINTRUST_ORG_NAME"]
    originals = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["BRAINTRUST_APP_URL"] = "https://my-own-thing.ca/foo/bar"
        os.environ["BRAINTRUST_ORG_NAME"] = "my-own-thing"

        logger = init_logger(project_id="test-project-id")
        span = logger.start_span(name="test-span")
        span.end()
        link = span.link()
        assert (
            link
            == f"https://my-own-thing.ca/foo/bar/app/my-own-thing/object?object_type=project_logs&object_id=test-project-id&id={span._id}"
        )
    finally:
        for k, v in originals.items():
            os.environ.pop(k, None)
            if v:
                os.environ[k] = v


def test_span_project_id_logged_in(with_memory_logger, with_simulate_login):
    logger = init_logger(
        project="test-project",
        project_id="test-project-id",
    )

    span = logger.start_span(name="test-span")
    span.end()

    link = span.link()
    assert (
        link
        == f"https://www.braintrust.dev/app/test-org-name/object?object_type=project_logs&object_id=test-project-id&id={span._id}"
    )


def test_span_export_disables_cache(with_memory_logger):
    """Test that span.export() disables the span cache."""
    logger = init_test_logger(__name__)

    with logger.start_span(name="test_span") as span:
        # Exporting should disable the span cache
        span.export()
        assert logger.state.span_cache.disabled


def test_span_project_name_logged_in(with_simulate_login, with_memory_logger):
    init_logger(project="test-project")
    span = logger.start_span(name="test-span")
    span.end()

    link = span.link()
    assert link == f"https://www.braintrust.dev/app/test-org-name/p/test-project/logs?oid={span._id}"


def test_span_link_with_resolved_experiment(with_simulate_login, with_memory_logger):
    experiment = braintrust.init(
        project="test-project",
        experiment="test-experiment",
    )

    id_lazy_value = LazyValue(lambda: "test-experiment-id", use_mutex=False)
    eid = id_lazy_value.get()
    assert eid == "test-experiment-id"

    span = experiment.start_span(name="test-span")
    span.parent_object_id = id_lazy_value
    span.end()

    link = span.link()
    assert (
        link
        == f"https://www.braintrust.dev/app/test-org-name/object?object_type=experiment&object_id=test-experiment-id&id={span._id}"
    )


def test_span_link_with_unresolved_experiment(with_simulate_login, with_memory_logger):
    experiment = braintrust.init(
        project="test-project",
        experiment="test-experiment",
    )

    span = experiment.start_span(name="test-span")
    span.end()

    link = span.link()
    assert link == "https://www.braintrust.dev/error-generating-link?msg=resolve-experiment-id"


def test_experiment_span_link_uses_env_vars_when_logged_out(with_memory_logger):
    """Verify EXPERIMENT spans use BRAINTRUST_ORG_NAME env var when not logged in."""
    simulate_logout()
    assert_logged_out()

    keys = ["BRAINTRUST_APP_URL", "BRAINTRUST_ORG_NAME"]
    originals = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["BRAINTRUST_APP_URL"] = "https://test-app.example.com"
        os.environ["BRAINTRUST_ORG_NAME"] = "env-org-name"

        experiment = braintrust.init(
            project="test-project",
            experiment="test-experiment",
        )

        # Create span with resolved experiment ID
        span = experiment.start_span(name="test-span")
        span.parent_object_id = LazyValue(lambda: "test-exp-id", use_mutex=False)
        span.end()

        link = span.link()

        # Should use env var org name and app url
        assert "env-org-name" in link
        assert "test-app.example.com" in link
        assert "test-exp-id" in link
    finally:
        for k, v in originals.items():
            os.environ.pop(k, None)
            if v:
                os.environ[k] = v


def test_permalink_with_valid_span_logged_in(with_simulate_login, with_memory_logger):
    logger = init_logger(
        project="test-project",
        project_id="test-project-id",
    )

    span = logger.start_span(name="test-span")
    span.end()

    span_export = span.export()

    link = braintrust.permalink(span_export, org_name="test-org-name", app_url="https://www.braintrust.dev")

    expected_link = f"https://www.braintrust.dev/app/test-org-name/object?object_type=project_logs&object_id=test-project-id&id={span._id}"
    assert link == expected_link


@pytest.mark.asyncio
async def test_span_link_in_async_context(with_simulate_login, with_memory_logger):
    """Test that span.link() works correctly when called from within an async function."""
    import asyncio

    logger = init_logger(
        project="test-project",
        project_id="test-project-id",
    )

    # Create a span in the main context
    span = logger.start_span(name="test-span")
    # Make it the current span so current_span() returns it
    span.set_current()

    # Define an async function that calls span.link()
    async def get_link_in_async():
        # Simulate some async work
        await asyncio.sleep(0.01)
        # This should return a valid link, not the noop link
        return braintrust.current_span().link()

    # Call the async function
    link = await get_link_in_async()

    span.end()

    # The link should NOT be the noop link
    assert link != "https://www.braintrust.dev/noop-span"
    # The link should contain the span ID
    assert span._id in link
    # The link should contain the project ID
    assert "test-project-id" in link


@pytest.mark.asyncio
async def test_current_logger_in_async_generator(with_simulate_login, with_memory_logger):
    """Test that current_logger() works within an async generator (yield)."""
    import asyncio

    logger = init_logger(project="test-project", project_id="test-project-id")

    async def logger_generator():
        for i in range(3):
            await asyncio.sleep(0.01)
            yield braintrust.current_logger()

    results = []
    async for log in logger_generator():
        results.append(log)

    assert len(results) == 3
    assert all(r is logger for r in results)


@pytest.mark.asyncio
async def test_current_logger_in_separate_task(with_simulate_login, with_memory_logger):
    """Test that current_logger() works in a separately created asyncio task."""
    import asyncio

    logger = init_logger(project="test-project", project_id="test-project-id")

    async def get_logger_in_task():
        await asyncio.sleep(0.01)
        return braintrust.current_logger()

    # Create a separate task
    task = asyncio.create_task(get_logger_in_task())
    result = await task

    assert result is logger


def test_current_logger_in_thread(with_simulate_login, with_memory_logger):
    """Test that current_logger() works correctly when called from a new thread.

    Regression test: ContextVar values don't propagate to new threads,
    so current_logger must be a plain attribute for thread access.
    """
    import threading

    logger = init_logger(project="test-project", project_id="test-project-id")
    assert braintrust.current_logger() is logger

    thread_result = {}

    def check_logger_in_thread():
        thread_result["logger"] = braintrust.current_logger()

    thread = threading.Thread(target=check_logger_in_thread)
    thread.start()
    thread.join()

    assert thread_result["logger"] is logger


@pytest.mark.asyncio
async def test_current_logger_async_context_isolation(with_simulate_login, with_memory_logger):
    """Test that different async contexts can have different loggers.

    When a child task sets its own logger, it should not affect the parent context.
    This ensures async context isolation via ContextVar.
    """
    import asyncio

    parent_logger = init_logger(project="parent-project", project_id="parent-project-id")
    assert braintrust.current_logger() is parent_logger

    child_result = {}

    async def child_task():
        # Child initially inherits parent's logger
        assert braintrust.current_logger() is parent_logger

        # Child sets its own logger
        child_logger = init_logger(project="child-project", project_id="child-project-id")
        child_result["logger"] = braintrust.current_logger()
        return child_logger

    # Run child task
    child_logger = await asyncio.create_task(child_task())

    # Child should have seen its own logger
    assert child_result["logger"] is child_logger

    # Parent should still see parent logger (not affected by child)
    assert braintrust.current_logger() is parent_logger


def test_span_set_current(with_memory_logger):
    """Test that span.set_current() makes the span accessible via current_span()."""
    init_test_logger(__name__)

    # Store initial current span
    initial_current = braintrust.current_span()

    # Start a span that can be set as current (default behavior)
    span1 = logger.start_span(name="test-span-1")

    # Initially, it should not be the current span
    assert braintrust.current_span() != span1

    # Call set_current() on the span
    span1.set_current()

    # Verify it's now the current span
    assert braintrust.current_span() == span1

    # Test that spans with set_current=False cannot be set as current
    span2 = logger.start_span(name="test-span-2", set_current=False)
    span2.set_current()  # This should not change the current span

    # Current span should still be span1
    assert braintrust.current_span() == span1

    span1.end()
    span2.end()


@pytest.mark.asyncio
async def test_traced_async_generator_with_exception(with_memory_logger):
    """Test tracing when async generator raises an exception."""
    init_test_logger(__name__)

    @logger.traced
    async def failing_async_generator() -> AsyncGenerator[int, None]:
        """An async generator that fails."""
        yield 1
        yield 2
        raise ValueError("Something went wrong")

    results = []
    start_time = time.time()
    with pytest.raises(ValueError, match="Something went wrong"):
        async for value in failing_async_generator():
            results.append(value)
    end_time = time.time()

    assert results == [1, 2]  # Should have yielded these before failing

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]

    assert_dict_matches(
        log,
        {
            "metrics": {
                "start": lambda x: start_time <= x <= end_time,
                "end": lambda x: start_time <= x <= end_time,
            },
            "error": lambda e: "ValueError" in str(e),
        },
    )


@pytest.mark.asyncio
async def test_traced_async_generator_with_subtasks(with_memory_logger):
    """
    Test async generator with current_span().log() calls - similar to user's failing case.
    Set notrace_io so we do not automatically log output and clobber the manually logged
    output "testing"
    """

    init_test_logger(__name__)

    num_loops = 3

    @logger.traced(notrace_io=True)
    async def foo(i: int) -> int:
        """Simulate some async work."""
        await asyncio.sleep(0.001)  # Small delay to simulate work
        return i * 2

    @logger.traced("main", notrace_io=True)
    async def main():
        yield 1
        logger.current_span().log(metadata={"a": "b"})
        tasks = [asyncio.create_task(foo(i)) for i in range(num_loops)]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        total = sum(task.result() for task in done)
        logger.current_span().log(metadata=dict(total=total), output="testing")
        yield total

    # consume the generator
    results: list[int] = []
    start_time = time.time()
    async for value in main():
        results.append(value)
    end_time = time.time()

    assert results == [1, 6]

    # Check logs
    logs = with_memory_logger.pop()
    assert len(logs) == num_loops + 1

    # Find the main span
    main_spans = [l for l in logs if l["span_attributes"]["name"] == "main"]
    assert len(main_spans) == 1
    main_span = main_spans[0]

    assert_dict_matches(
        main_span,
        {
            # no input because notrace_io
            "output": "testing",
            "metadata": {"a": "b", "total": 6},  # Manual metadata logging
            "metrics": {
                "start": lambda x: start_time <= x <= end_time,
                "end": lambda x: start_time <= x <= end_time,
            },
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("make_decorator", "expected_name"),
    [
        pytest.param(lambda: logger.traced, "async_multiply", id="bare"),
        pytest.param(lambda: logger.traced(), "async_multiply", id="called"),
        pytest.param(lambda: logger.traced(name="async_multiply_with_name"), "async_multiply_with_name", id="named"),
    ],
)
async def test_traced_async_function(with_memory_logger, make_decorator, expected_name):
    """Test tracing async functions with each form of the @traced decorator."""
    init_test_logger(__name__)

    @make_decorator()
    async def async_multiply(x: int, y: int) -> int:
        """An async function that multiplies two numbers."""
        await asyncio.sleep(0.001)  # Small delay to simulate async work
        logger.current_span().log(metadata={"operation": "multiply"})
        return x * y

    start_time = time.time()
    result = await async_multiply(3, 4)
    end_time = time.time()

    assert result == 12

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    assert_dict_matches(
        logs[0],
        {
            "input": {"x": 3, "y": 4},
            "output": 12,
            "metadata": {"operation": "multiply"},
            "metrics": {
                "start": lambda x: start_time <= x <= end_time,
                "end": lambda x: start_time <= x <= end_time,
            },
            "span_attributes": {
                "name": expected_name,
                "type": "function",
            },
        },
    )


def test_traced_sync_function(with_memory_logger):
    """Test tracing synchronous functions."""
    init_test_logger(__name__)

    @logger.traced
    def sync_add(a: int, b: int) -> int:
        """A sync function that adds two numbers."""
        result = a + b
        logger.current_span().log(metadata={"operation": "add"})
        return result

    start_time = time.time()
    result = sync_add(5, 7)
    end_time = time.time()

    assert result == 12

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]

    assert_dict_matches(
        log,
        {
            "input": {"a": 5, "b": 7},
            "output": 12,
            "metadata": {"operation": "add"},
            "metrics": {
                "start": lambda x: start_time <= x <= end_time,
                "end": lambda x: start_time <= x <= end_time,
            },
            "span_attributes": {
                "name": "sync_add",
                "type": "function",
            },
        },
    )


def test_traced_sync_generator(with_memory_logger):
    """Test tracing synchronous generators."""
    init_test_logger(__name__)

    @logger.traced
    def sync_number_generator(n: int):
        """A sync generator that yields numbers."""
        for i in range(n):
            yield i * 2

    results = []
    start_time = time.time()
    for value in sync_number_generator(3):
        results.append(value)
    end_time = time.time()

    assert results == [0, 2, 4]

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]

    # Should log the complete output as a list
    assert log.get("output") == [0, 2, 4]
    assert log.get("input") == {"n": 3}
    assert_dict_matches(
        log,
        {
            "metrics": {
                "start": lambda x: start_time <= x <= end_time,
                "end": lambda x: start_time <= x <= end_time,
            },
            "span_attributes": {
                "name": "sync_number_generator",
                "type": "function",
            },
        },
    )


def test_traced_sync_generator_with_exception(with_memory_logger):
    """Test sync generator that raises an exception."""
    init_test_logger(__name__)

    @logger.traced
    def failing_generator():
        yield "first"
        yield "second"
        raise RuntimeError("Generator failed")

    results = []
    start_time = time.time()
    with pytest.raises(RuntimeError, match="Generator failed"):
        for value in failing_generator():
            results.append(value)
    end_time = time.time()

    assert results == ["first", "second"]

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]

    # Should have partial output and error
    assert log.get("output") == ["first", "second"]
    assert "RuntimeError" in str(log.get("error", ""))
    assert_dict_matches(
        log,
        {
            "metrics": {
                "start": lambda x: start_time <= x <= end_time,
                "end": lambda x: start_time <= x <= end_time,
            },
        },
    )


def test_traced_sync_generator_with_subtasks(with_memory_logger):
    """
    Test sync generator with current_span().log() calls
    Set notrace_io so we do not automatically log output and clobber the manually logged
    output "testing"
    """

    init_test_logger(__name__)

    num_loops = 3

    @logger.traced(notrace_io=True)
    def foo(i: int) -> int:
        """Simulate some sync work."""
        time.sleep(0.001)
        return i * 2

    @logger.traced("main", notrace_io=True)
    def main():
        yield 1
        logger.current_span().log(metadata={"a": "b"})
        tasks = [foo(i) for i in range(num_loops)]
        total = sum(tasks)
        logger.current_span().log(metadata=dict(total=total), output="testing")
        yield total

    # consume the generator
    results: list[int] = []
    start_time = time.time()
    for value in main():
        results.append(value)
    end_time = time.time()

    assert results == [1, 6]

    # Check logs
    logs = with_memory_logger.pop()
    assert len(logs) == num_loops + 1

    # Find the main span
    main_spans = [l for l in logs if l["span_attributes"]["name"] == "main"]
    assert len(main_spans) == 1
    main_span = main_spans[0]

    assert_dict_matches(
        main_span,
        {
            # no input because notrace_io
            "output": "testing",
            "metadata": {"a": "b", "total": 6},  # Manual metadata logging
            "metrics": {
                "start": lambda x: start_time <= x <= end_time,
                "end": lambda x: start_time <= x <= end_time,
            },
        },
    )


@pytest.mark.asyncio
async def test_traced_async_generator(with_memory_logger):
    """Test async generator version of sync generator test."""
    init_test_logger(__name__)

    @logger.traced
    async def async_number_generator(n: int):
        """An async generator that yields numbers."""
        for i in range(n):
            await asyncio.sleep(0.001)
            yield i * 2

    results = []
    start_time = time.time()
    async for value in async_number_generator(3):
        results.append(value)
    end_time = time.time()

    assert results == [0, 2, 4]

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]

    # Should log the complete output as a list
    assert log.get("output") == [0, 2, 4]
    assert log.get("input") == {"n": 3}
    assert_dict_matches(
        log,
        {
            "metrics": {
                "start": lambda x: start_time <= x <= end_time,
                "end": lambda x: start_time <= x <= end_time,
            },
            "span_attributes": {
                "name": "async_number_generator",
                "type": "function",
            },
        },
    )


def _consume_traced_generator(kind: str, n: int) -> list[int]:
    """Consume a @traced sync or async generator that yields ``range(n)``."""
    if kind == "sync":

        @logger.traced
        def sync_generator():
            yield from range(n)

        return list(sync_generator())

    @logger.traced
    async def async_generator():
        for i in range(n):
            yield i

    async def collect():
        return [v async for v in async_generator()]

    return asyncio.run(collect())


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize(
    ("max_items", "logged_output", "warns"),
    [
        pytest.param("3", None, True, id="truncated"),
        pytest.param("0", None, False, id="zero-drops-output"),
        pytest.param("-1", list(range(10)), False, id="unlimited"),
    ],
)
def test_traced_generator_max_items(with_memory_logger, monkeypatch, caplog, kind, max_items, logged_output, warns):
    """BRAINTRUST_MAX_GENERATOR_ITEMS limits logged output but never what the generator yields."""
    init_test_logger(__name__)
    monkeypatch.setenv("BRAINTRUST_MAX_GENERATOR_ITEMS", max_items)

    with caplog.at_level(logging.WARNING):
        assert _consume_traced_generator(kind, 10) == list(range(10))

    [log] = with_memory_logger.pop()
    assert log.get("output") == logged_output
    assert log.get("input") == {}
    warnings = [r.message for r in caplog.records if "Generator output exceeded limit" in r.message]
    if warns:
        assert "exceeded limit of 3 items" in warnings[0]
    else:
        assert not warnings


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("value", ["", "  ", "not-a-number"])
def test_traced_generators_ignore_invalid_max_items_env(with_memory_logger, monkeypatch, kind, value):
    """An empty or non-numeric BRAINTRUST_MAX_GENERATOR_ITEMS falls back to the default instead of raising."""
    init_test_logger(__name__)
    monkeypatch.setenv("BRAINTRUST_MAX_GENERATOR_ITEMS", value)

    assert _consume_traced_generator(kind, 3) == [0, 1, 2]

    [log] = with_memory_logger.pop()
    assert log.get("output") == [0, 1, 2]


@pytest.mark.parametrize(
    "name",
    [
        "BRAINTRUST_PROMPT_CACHE_MEMORY_MAX_SIZE",
        "BRAINTRUST_PROMPT_CACHE_DISK_MAX_SIZE",
        "BRAINTRUST_PARAMETERS_CACHE_MEMORY_MAX_SIZE",
        "BRAINTRUST_PARAMETERS_CACHE_DISK_MAX_SIZE",
    ],
)
@pytest.mark.parametrize("value", ["", "not-a-number"])
def test_state_ignores_invalid_cache_size_env(monkeypatch, name, value):
    """An empty or non-numeric cache size variable keeps the default instead of failing state creation."""
    monkeypatch.setenv(name, value)
    logger.BraintrustState()


def _redact_secrets(data):
    """Test masking function: redacts "secret" substrings and every "api_key" value."""
    if isinstance(data, str):
        return data.replace("secret", "REDACTED")
    if isinstance(data, dict):
        return {k: "REDACTED" if k == "api_key" else _redact_secrets(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_redact_secrets(item) for item in data]
    return data


_SENSITIVE_INPUT = {"api_key": "sk-12345", "query": "a secret query"}
_SENSITIVE_OUTPUT = {"response": "secret data", "api_key": "sk-67890", "items": ["secret", 42]}


def _log_sensitive_data_to_child_span():
    with init_test_logger("test_project").start_span(name="parent_span") as parent:
        parent.log(input=_SENSITIVE_INPUT)
        with parent.start_span(name="child_span") as child:
            child.log(output=_SENSITIVE_OUTPUT)


@pytest.mark.parametrize(
    ("log_sensitive_data", "output_field", "unmasked"),
    [
        pytest.param(
            lambda: init_test_logger("test_project").log(
                input=_SENSITIVE_INPUT, output=_SENSITIVE_OUTPUT, metadata={"user": "secret_user", "safe": "normal"}
            ),
            "output",
            {"metadata": {"user": "REDACTED_user", "safe": "normal"}},
            id="logger",
        ),
        pytest.param(
            lambda: init_test_exp("test_experiment", "test_project").log(
                input=_SENSITIVE_INPUT, output=_SENSITIVE_OUTPUT, scores={"accuracy": 0.95}
            ),
            "output",
            {"scores": {"accuracy": 0.95}},
            id="experiment",
        ),
        pytest.param(_log_sensitive_data_to_child_span, "output", {}, id="child-span"),
        pytest.param(
            lambda: _init_test_dataset().insert(
                input=_SENSITIVE_INPUT, expected=_SENSITIVE_OUTPUT, metadata={"admin": "secret"}
            ),
            "expected",
            {"metadata": {"admin": "REDACTED"}},
            id="dataset",
        ),
    ],
)
def test_masking_function_applies_to_logged_data(
    with_memory_logger, with_simulate_login, log_sensitive_data, output_field, unmasked
):
    """The global masking function is applied to every logged field, including in child spans."""
    braintrust.set_masking_function(_redact_secrets)

    log_sensitive_data()

    rows = with_memory_logger.pop()
    input_row = next(row for row in rows if row.get("input"))
    assert input_row["input"] == {"api_key": "REDACTED", "query": "a REDACTED query"}
    assert next(row[output_field] for row in rows if row.get(output_field)) == {
        "response": "REDACTED data",
        "api_key": "REDACTED",
        "items": ["REDACTED", 42],
    }
    for field, expected in unmasked.items():
        assert input_row[field] == expected
    serialized = json.dumps(rows)
    assert "secret" not in serialized
    assert "sk-" not in serialized


def test_masking_function_with_error(with_memory_logger, with_simulate_login):
    def broken_masking_function(data):
        if data == "safe output":
            return data
        raise TypeError("private exception detail")

    braintrust.set_masking_function(broken_masking_function)
    test_logger = init_test_logger("test_masking_errors_logger")
    test_logger.log(
        input={"password": "private input"},
        output="safe output",
        expected="private expected",
        metadata={"token": "private metadata"},
        scores={"accuracy": 0.85},
        metrics={"accuracy": 0.95},
        error="existing application error",
        tags=["untouched"],
    )

    [record] = with_memory_logger.pop()
    assert isinstance(record["input"], str)
    assert isinstance(record["expected"], str)
    assert isinstance(record["metadata"]["error"], str)
    assert record["output"] == "safe output"
    assert record["tags"] == ["untouched"]
    assert "scores" not in record
    assert "metrics" not in record
    assert "existing application error" in record["error"]
    assert "scores" in record["error"]
    assert "metrics" in record["error"]
    assert "private" not in json.dumps(record)


def test_attachment_unreadable_path_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="braintrust"):
        Attachment(
            data="unreadable.txt",
            filename="unreadable.txt",
            content_type="text/plain",
        )

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Failed to read file" in caplog.records[0].message


def test_attachment_readable_path_returns_data(tmp_path):
    file_path = tmp_path / "attachments" / "hello.txt"
    file_path.parent.mkdir(parents=True)
    file_path.write_bytes(b"hello world")

    a = Attachment(data=str(file_path), filename="hello.txt", content_type="text/plain")
    assert a.data == b"hello world"


def test_parent_precedence_with_parent_context_and_traced(with_memory_logger, with_simulate_login):
    """Test that with parent_context + traced, child spans attach to current span (not directly to parent context)."""
    init_test_logger(__name__)

    # Create exported parent context
    with logger.start_span(name="outer") as outer:
        outer_export = outer.export()

    @logger.traced("inner", notrace_io=True)
    def inner():
        s = logger.start_span(name="child")
        s.end()

    with parent_context(outer_export):
        inner()

    logs = with_memory_logger.pop()
    outer_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "outer")
    inner_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "inner")
    child_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "child")

    # child should have inner as a parent
    assert inner_log["span_id"] in (child_log.get("span_parents") or [])
    # child and outer should share the same root
    assert child_log["root_span_id"] == outer_log["root_span_id"]


def test_parent_precedence_traced_baseline(with_memory_logger, with_simulate_login):
    """Test that traced baseline nests child under current span."""
    init_test_logger(__name__)

    @logger.traced("top", notrace_io=True)
    def top():
        s = logger.start_span(name="child")
        s.end()

    top()
    logs = with_memory_logger.pop()
    top_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "top")
    child_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "child")

    assert top_log["span_id"] in (child_log.get("span_parents") or [])


def test_parent_precedence_explicit_parent_overrides(with_memory_logger, with_simulate_login):
    """Test that explicit parent overrides current span."""
    init_test_logger(__name__)

    with logger.start_span(name="outer") as outer:
        outer_export = outer.export()

    @logger.traced("inner", notrace_io=True)
    def inner():
        s = braintrust.start_span(name="forced", parent=outer_export)
        s.end()

    inner()
    logs = with_memory_logger.pop()
    outer_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "outer")
    inner_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "inner")
    forced_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "forced")

    parents = forced_log.get("span_parents") or []
    assert outer_log["span_id"] in parents
    assert inner_log["span_id"] not in parents


@pytest.fixture
def reset_id_generator_state(monkeypatch):
    """Clear ID-format env vars and reset the cached ID generator and context manager."""
    monkeypatch.delenv("BRAINTRUST_OTEL_COMPAT", raising=False)
    monkeypatch.delenv("BRAINTRUST_LEGACY_IDS", raising=False)
    logger._state._reset_id_generator()
    logger._state._reset_context_manager()
    yield
    logger._state._reset_id_generator()
    logger._state._reset_context_manager()


def _is_hex(s: str) -> bool:
    return all(c in "0123456789abcdef" for c in s)


def test_span_with_otel_ids_export_import(reset_id_generator_state, monkeypatch):
    """Test that actual Span objects with OTEL IDs can export and be used as parent context."""
    init_test_logger(__name__)
    monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "true")

    assert get_id_generator().share_root_span_id() is False

    with logger.start_span(name="test") as span:
        # OTEL spans do not share span_id and root_span_id
        assert span.span_id != span.root_span_id
        assert len(span.span_id) == 16  # 8-byte hex
        assert len(span.root_span_id) == 32  # 16-byte hex
        assert _is_hex(span.span_id)
        assert _is_hex(span.root_span_id)

        from braintrust.span_identifier_v4 import SpanComponentsV4

        imported = SpanComponentsV4.from_str(span.export())
        assert imported.span_id == span.span_id
        assert imported.root_span_id == span.root_span_id


def test_span_with_uuid_ids_share_root_span_id(reset_id_generator_state, monkeypatch):
    """Test that legacy UUID generators share span_id as root_span_id for backwards compatibility."""
    monkeypatch.setenv("BRAINTRUST_LEGACY_IDS", "true")
    init_test_logger(__name__)

    assert get_id_generator().share_root_span_id() is True

    with logger.start_span(name="test") as span:
        assert span.span_id == span.root_span_id


def test_parent_context_with_otel_ids(with_memory_logger, reset_id_generator_state, monkeypatch):
    """Test that parent_context works correctly with OTEL-compatible IDs."""
    monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "true")
    init_test_logger(__name__)

    # Create a span and export it
    with logger.start_span(name="parent") as parent_span:
        parent_export = parent_span.export()
        original_span_id = parent_span.span_id
        original_root_span_id = parent_span.root_span_id

    assert _is_hex(original_span_id)
    assert _is_hex(original_root_span_id)

    # Use the exported span as parent context
    with parent_context(parent_export):
        with logger.start_span(name="child") as child_span:
            # Child should inherit the root_span_id from parent
            assert child_span.root_span_id == original_root_span_id
            assert original_span_id in child_span.span_parents

    # Verify logs were created correctly
    logs = with_memory_logger.pop()
    parent_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "parent")
    child_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "child")

    assert parent_log["span_id"] == original_span_id
    assert parent_log["root_span_id"] == original_root_span_id
    assert child_log["root_span_id"] == original_root_span_id
    assert parent_log["span_id"] in child_log.get("span_parents", [])


def test_nested_spans_with_export(with_memory_logger):
    """Test nested spans with login triggered during span execution.

    This reproduces a bug where calling state.login() during an active span
    calls copy_state(), which would overwrite _context_manager with None,
    causing a ContextVar token mismatch error when the span exits.
    """
    from braintrust import logger
    from braintrust.test_helpers import init_test_exp

    experiment = init_test_exp("test-experiment", "test-project")

    # Start a span, then trigger login which calls copy_state()
    with experiment.start_span(name="s1") as span1:
        span1.log(input="one")
        # Trigger login with TEST_API_KEY and force_login=True
        # This calls copy_state() which should NOT overwrite _context_manager
        experiment.state.login(api_key=logger.TEST_API_KEY, force_login=True)
        # Continue with nested spans to ensure context manager still works
        with experiment.start_span(name="s2") as span2:
            span2.log(input="two")


def test_span_start_span_with_explicit_parent(with_memory_logger):
    """Test that span.start_span() with explicit parent doesn't inherit from context.

    This verifies the fix where span.start_span(parent=exported) should use the
    exported parent, not the current span from the context manager.
    """
    from braintrust.test_helpers import init_test_exp

    experiment = init_test_exp("test-experiment", "test-project")

    # Create a root span, log to it (creates row_id), and export it
    with experiment.start_span(name="root") as root_span:
        root_span.log(input="root input")
        root_export = root_span.export()
        root_span_id = root_span.span_id
        root_root_span_id = root_span.root_span_id

    # Create another span
    with experiment.start_span(name="span2") as span2:
        span2_span_id = span2.span_id

        # Within span2's context, create span3 with explicit parent=root_export
        # span3 should NOT inherit from span2 (the active context)
        # span3 should inherit from root (because root_export has row_id after logging)
        with span2.start_span(parent=root_export, name="span3") as span3:
            span3.log(input="test")

    logs = with_memory_logger.pop()
    span3_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "span3")

    # span3 should NOT have span2 as parent (would happen if it inherited from context)
    assert span2_span_id not in span3_log.get("span_parents", []), (
        "span3 should not inherit from span2 context when explicit parent is provided"
    )

    # span3 should inherit from root (the explicit parent)
    assert root_span_id in span3_log.get("span_parents", []), (
        "span3 should have root_span_id in span_parents from explicit parent"
    )
    assert span3_log["root_span_id"] == root_root_span_id, "span3 should have root's root_span_id"


def test_span_start_span_inherits_from_self(with_memory_logger):
    """Test that span.start_span() without explicit parent inherits from self.

    When no explicit parent is provided, the child should inherit from the current span.
    """
    from braintrust.test_helpers import init_test_exp

    experiment = init_test_exp("test-experiment", "test-project")

    # Create a parent span
    with experiment.start_span(name="parent") as parent_span:
        parent_span_id = parent_span.span_id
        parent_root_span_id = parent_span.root_span_id

        # Create a child span without explicit parent - should inherit from parent_span
        with parent_span.start_span(name="child") as child_span:
            child_span.log(input="test")

    logs = with_memory_logger.pop()
    child_log = next(l for l in logs if l.get("span_attributes", {}).get("name") == "child")

    # Child should inherit parent's root_span_id and have parent_span_id in span_parents
    assert child_log["root_span_id"] == parent_root_span_id
    assert parent_span_id in child_log.get("span_parents", []), (
        "child should have parent_span_id in span_parents when no explicit parent is provided"
    )


def test_update_span_includes_span_id_and_root_span_id_from_export(with_memory_logger):
    experiment = init_test_exp("test-experiment", "test-project")

    with experiment.start_span(name="span") as span:
        span.log(input="input")
        exported = span.export()
        span_id = span.span_id
        root_span_id = span.root_span_id

    with_memory_logger.pop()

    braintrust.update_span(exported=exported, output="updated output", metadata=_ModelDumpMetadata(foo="bar"))

    logs = with_memory_logger.pop()
    updated_log = next(log for log in logs if log.get("output") == "updated output")
    assert updated_log["span_id"] == span_id
    assert updated_log["root_span_id"] == root_span_id
    assert updated_log["metadata"] == {"foo": "bar"}


@pytest.mark.parametrize(
    ("env", "expected_version"),
    [
        pytest.param({}, 4, id="default"),
        pytest.param({"BRAINTRUST_LEGACY_IDS": "true"}, 3, id="legacy-ids"),
        pytest.param({"BRAINTRUST_OTEL_COMPAT": "true"}, 4, id="otel-compat"),
        pytest.param({"BRAINTRUST_OTEL_COMPAT": "true", "BRAINTRUST_LEGACY_IDS": "true"}, 4, id="otel-compat-wins"),
    ],
)
def test_export_format_follows_id_mode(monkeypatch, env, expected_version):
    """_get_exporter(), Experiment.export(), and Logger.export() use V3 only in legacy UUID mode."""
    from braintrust.logger import _get_exporter
    from braintrust.span_identifier_v3 import SpanComponentsV3
    from braintrust.span_identifier_v4 import SpanComponentsV4

    monkeypatch.delenv("BRAINTRUST_OTEL_COMPAT", raising=False)
    monkeypatch.delenv("BRAINTRUST_LEGACY_IDS", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert _get_exporter() is (SpanComponentsV3 if expected_version == 3 else SpanComponentsV4)
    assert SpanComponentsV4.get_version(init_test_exp("test-exp").export()) == expected_version
    assert SpanComponentsV4.get_version(init_test_logger(__name__).export()) == expected_version


def test_register_otel_flush_callback():
    """flush_otel() is a no-op until a callback is registered, then invokes it."""
    from braintrust import register_otel_flush
    from braintrust.logger import _internal_get_global_state

    init_test_logger(__name__)
    state = _internal_get_global_state()
    asyncio.run(state.flush_otel())

    callback_invoked = False

    async def mock_flush():
        nonlocal callback_invoked
        callback_invoked = True

    register_otel_flush(mock_flush)
    asyncio.run(state.flush_otel())

    assert callback_invoked is True


def test_register_otel_flush_permanently_disables_cache():
    """Test that register_otel_flush permanently disables the cache."""
    from braintrust import register_otel_flush
    from braintrust.logger import _internal_get_global_state
    from braintrust.test_helpers import init_test_logger

    init_test_logger(__name__)
    state = _internal_get_global_state()

    # Enable the cache
    state.span_cache.start()
    assert state.span_cache.disabled is False

    async def mock_flush():
        pass

    # Register OTEL flush
    register_otel_flush(mock_flush)
    assert state.span_cache.disabled is True

    # Try to start again - should still be disabled because of explicit disable
    state.span_cache.start()
    assert state.span_cache.disabled is True


class TestJSONAttachment(TestCase):
    def test_create_attachment_from_json_data(self):
        """Test creating an attachment from JSON data."""
        test_data = {
            "foo": "bar",
            "nested": {
                "array": [1, 2, 3],
                "bool": True,
            },
        }

        attachment = JSONAttachment(test_data)

        self.assertEqual(attachment.reference["type"], "braintrust_attachment")
        self.assertEqual(attachment.reference["filename"], "data.json")
        self.assertEqual(attachment.reference["content_type"], "application/json")
        self.assertIn("key", attachment.reference)

        data = attachment.data
        parsed = json.loads(data.decode("utf-8"))
        self.assertEqual(parsed, test_data)

    def test_custom_filename(self):
        """Test that custom filename is respected."""
        attachment = JSONAttachment({"test": "data"}, filename="custom.json")

        self.assertEqual(attachment.reference["filename"], "custom.json")

    def test_pretty_print(self):
        """Test pretty printing JSON data."""
        test_data = {"a": 1, "b": 2}
        attachment = JSONAttachment(test_data, pretty=True)

        data = attachment.data
        text = data.decode("utf-8")
        self.assertEqual(text, '{\n  "a": 1,\n  "b": 2\n}')

    def test_arrays_and_primitives(self):
        """Test handling arrays and primitive values."""
        array_data = [1, 2, 3, 4, 5]
        attachment = JSONAttachment(array_data)

        data = attachment.data
        parsed = json.loads(data.decode("utf-8"))
        self.assertEqual(parsed, array_data)

    def test_extract_attachments_with_json_attachment(self):
        """Test that JSONAttachment works with _extract_attachments."""
        json_attachment = JSONAttachment({"foo": "bar"}, filename="test.json")
        event = {
            "input": {
                "data": json_attachment,
            },
        }

        attachments: list[BaseAttachment] = []
        _extract_attachments(event, attachments)

        self.assertEqual(len(attachments), 1)
        self.assertIs(attachments[0], json_attachment)
        self.assertEqual(event["input"]["data"], json_attachment.reference)


class TestExperimentGeneratedAPI(TestCase):
    def test_init_uses_generated_registration_and_resolves_named_base(self):
        mock_state = MagicMock(spec=BraintrustState)
        mock_state.org_id = "test-org-id"
        mock_state.org_name = "test-org"
        api_client = mock_state.api_client.return_value
        api_client.projects.post_project.return_value = {
            "id": "test-project-id",
            "name": "test-project",
        }
        api_client.experiments.get_experiment.return_value = {
            "objects": [
                {
                    "id": "base-experiment-id",
                    "name": "base-experiment",
                    "project_id": "test-project-id",
                    "public": False,
                }
            ]
        }
        api_client.experiments.post_experiment.return_value = {
            "id": "test-experiment-id",
            "name": "test-experiment",
            "project_id": "test-project-id",
            "public": False,
        }

        experiment = braintrust.init(
            project="test-project",
            experiment="test-experiment",
            description="test description",
            base_experiment="base-experiment",
            metadata={"source": "unit-test"},
            tags=["generated"],
            update=False,
            set_current=False,
            repo_info=RepoInfo(commit="abc123"),
            state=mock_state,
        )

        assert experiment.id == "test-experiment-id"
        assert experiment.project.id == "test-project-id"
        api_client.projects.post_project.assert_called_once_with(body={"name": "test-project", "org_name": "test-org"})
        api_client.experiments.get_experiment.assert_called_once_with(
            experiment_name="base-experiment",
            project_id="test-project-id",
            org_name="test-org",
        )
        body = api_client.experiments.post_experiment.call_args.kwargs["body"]
        assert body == {
            "project_id": "test-project-id",
            "name": "test-experiment",
            "description": "test description",
            "repo_info": RepoInfo(commit="abc123").as_dict(),
            "base_exp_id": "base-experiment-id",
            "public": False,
            "metadata": {"source": "unit-test"},
            "tags": ["generated"],
            "ensure_new": True,
        }
        mock_state.app_conn.assert_not_called()

    def test_open_uses_generated_experiment_lookup(self):
        mock_state = MagicMock(spec=BraintrustState)
        mock_state.org_name = "test-org"
        api_client = mock_state.api_client.return_value
        api_client.experiments.get_experiment.return_value = {
            "objects": [
                {
                    "id": "test-experiment-id",
                    "name": "test-experiment",
                    "project_id": "test-project-id",
                    "public": False,
                }
            ]
        }

        experiment = braintrust.init(
            project="test-project",
            experiment="test-experiment",
            open=True,
            set_current=False,
            state=mock_state,
        )

        assert experiment.id == "test-experiment-id"
        api_client.experiments.get_experiment.assert_called_once_with(
            experiment_name="test-experiment",
            project_name="test-project",
            org_name="test-org",
        )
        mock_state.app_conn.assert_not_called()

    def test_experiment_fetch_uses_generated_paginated_fetch(self):
        from braintrust.logger import Experiment, ObjectMetadata, ProjectExperimentMetadata

        mock_state = MagicMock(spec=BraintrustState)
        api_client = mock_state.api_client.return_value
        first_event = {
            "id": "first-event",
            "_xact_id": "1",
            "project_id": "test-project-id",
            "experiment_id": "test-experiment-id",
            "created": "2026-01-01T00:00:00Z",
            "span_id": "first-event",
            "root_span_id": "first-event",
        }
        second_event = {**first_event, "id": "second-event", "_xact_id": "2"}
        api_client.experiments.post_experiment_id_fetch.side_effect = [
            {"events": [first_event], "cursor": "next-page"},
            {"events": [second_event]},
        ]
        metadata = ProjectExperimentMetadata(
            project=ObjectMetadata(id="test-project-id", name="test-project", full_info={}),
            experiment=ObjectMetadata(id="test-experiment-id", name="test-experiment", full_info={}),
        )
        experiment = Experiment(LazyValue(lambda: metadata, use_mutex=False), state=mock_state)

        events = list(experiment.fetch(batch_size=1))

        assert [event["id"] for event in events] == ["first-event", "second-event"]
        assert api_client.experiments.post_experiment_id_fetch.call_args_list == [
            call("test-experiment-id", body={"limit": 1}),
            call("test-experiment-id", body={"limit": 1, "cursor": "next-page"}),
        ]
        mock_state.api_conn.assert_not_called()


class TestProjectGeneratedAPI(TestCase):
    def test_lazy_project_uses_generated_registration_and_lookup(self):
        mock_state = MagicMock()
        mock_state.org_name = "test-org"
        api_client = mock_state.api_client.return_value
        api_client.projects.post_project.return_value = {
            "id": "created-project-id",
            "name": "created-project",
        }
        api_client.projects.get_project_id.return_value = {
            "id": "looked-up-project-id",
            "name": "looked-up-project",
        }

        with patch.object(logger, "_state", mock_state):
            created = logger.Project(name="created-project")
            looked_up = logger.Project(id="looked-up-project-id")
            self.assertEqual(created.id, "created-project-id")
            self.assertEqual(looked_up.name, "looked-up-project")

        api_client.projects.post_project.assert_called_once_with(
            body={"name": "created-project", "org_name": "test-org"}
        )
        api_client.projects.get_project_id.assert_called_once_with("looked-up-project-id")
        mock_state.app_conn.assert_not_called()


class TestDatasetGeneratedAPI(TestCase):
    def test_init_dataset_uses_generated_project_and_dataset_resources(self):
        mock_state = MagicMock()
        mock_state.org_name = "test-org"
        api_client = mock_state.api_client.return_value
        api_client.projects.post_project.return_value = {
            "id": "test-project-id",
            "name": "test-project",
        }
        api_client.datasets.post_dataset.return_value = {
            "id": "test-dataset-id",
            "project_id": "test-project-id",
            "name": "test-dataset",
        }

        dataset = braintrust.init_dataset(
            project="test-project",
            name="test-dataset",
            description="description",
            metadata={"purpose": "test"},
            use_output=False,
            state=mock_state,
        )

        self.assertEqual(dataset.id, "test-dataset-id")
        api_client.projects.post_project.assert_called_once_with(body={"name": "test-project", "org_name": "test-org"})
        api_client.datasets.post_dataset.assert_called_once_with(
            body={
                "project_id": "test-project-id",
                "name": "test-dataset",
                "description": "description",
                "metadata": {"purpose": "test"},
            }
        )
        mock_state.app_conn.assert_not_called()

    def test_init_dataset_without_name_uses_logs_name_with_generated_resources(self):
        mock_state = MagicMock()
        mock_state.org_name = "test-org"
        api_client = mock_state.api_client.return_value
        api_client.projects.post_project.return_value = {
            "id": "test-project-id",
            "name": "test-project",
        }
        api_client.datasets.post_dataset.return_value = {
            "id": "test-dataset-id",
            "project_id": "test-project-id",
            "name": "logs",
        }

        dataset = braintrust.init_dataset(
            project="test-project",
            description="description",
            use_output=False,
            state=mock_state,
        )

        self.assertEqual(dataset.name, "logs")
        api_client.projects.post_project.assert_called_once_with(body={"name": "test-project", "org_name": "test-org"})
        api_client.datasets.post_dataset.assert_called_once_with(
            body={"project_id": "test-project-id", "name": "logs", "description": "description"}
        )
        mock_state.app_conn.assert_not_called()

    def test_init_dataset_looks_up_explicit_dataset_id(self):
        mock_state = MagicMock()
        api_client = mock_state.api_client.return_value
        api_client.datasets.get_dataset_id.return_value = {
            "id": "test-dataset-id",
            "project_id": "test-project-id",
            "name": "test-dataset",
        }
        api_client.projects.get_project_id.return_value = {
            "id": "test-project-id",
            "name": "test-project",
        }

        dataset = braintrust.init_dataset(
            dataset_id="test-dataset-id",
            use_output=False,
            state=mock_state,
        )

        self.assertEqual(dataset.id, "test-dataset-id")
        self.assertEqual(dataset.name, "test-dataset")
        self.assertEqual(dataset.project.name, "test-project")
        api_client.datasets.get_dataset_id.assert_called_once_with("test-dataset-id")
        api_client.projects.get_project_id.assert_called_once_with("test-project-id")
        api_client.datasets.post_dataset.assert_not_called()
        api_client.projects.post_project.assert_not_called()

    def test_init_dataset_looks_up_explicit_project_id(self):
        mock_state = MagicMock()
        api_client = mock_state.api_client.return_value
        api_client.projects.get_project_id.return_value = {
            "id": "test-project-id",
            "name": "test-project",
        }
        api_client.datasets.post_dataset.return_value = {
            "id": "test-dataset-id",
            "project_id": "test-project-id",
            "name": "test-dataset",
        }

        dataset = braintrust.init_dataset(
            project="ignored-project-name",
            project_id="test-project-id",
            name="test-dataset",
            use_output=False,
            state=mock_state,
        )

        self.assertEqual(dataset.project.name, "test-project")
        api_client.projects.get_project_id.assert_called_once_with("test-project-id")
        api_client.projects.post_project.assert_not_called()

    def test_dataset_fetch_uses_generated_resource(self):
        mock_state = MagicMock()
        api_client = mock_state.api_client.return_value
        api_client.datasets.post_dataset_id_fetch.side_effect = [
            {"events": [{"id": "row-1", "expected": "first"}], "cursor": "next"},
            {"events": [{"id": "row-2", "expected": "second"}]},
        ]
        metadata = logger.ProjectDatasetMetadata(
            project=logger.ObjectMetadata(id="test-project-id", name="test-project", full_info={}),
            dataset=logger.ObjectMetadata(id="test-dataset-id", name="test-dataset", full_info={}),
        )
        dataset = logger.Dataset(
            lazy_metadata=LazyValue(lambda: metadata, use_mutex=False),
            version=123,
            legacy=False,
            state=mock_state,
        )

        self.assertEqual([row["id"] for row in dataset.fetch(batch_size=2)], ["row-1", "row-2"])
        self.assertEqual(
            api_client.datasets.post_dataset_id_fetch.call_args_list,
            [
                call("test-dataset-id", body={"limit": 2, "version": "123"}),
                call("test-dataset-id", body={"limit": 2, "cursor": "next", "version": "123"}),
            ],
        )
        mock_state.api_conn.assert_not_called()

    def test_dataset_summary_uses_generated_resource_and_configured_public_url(self):
        mock_state = MagicMock()
        mock_state.app_public_url = "https://public.example.com"
        mock_state.org_name = "test org"
        mock_state.api_client.return_value.datasets.get_dataset_id_summarize.return_value = {
            "project_name": "backend-project",
            "dataset_name": "backend-dataset",
            "project_url": "https://backend.example.com/project",
            "dataset_url": "https://backend.example.com/dataset",
            "data_summary": {"total_records": 3},
        }
        metadata = logger.ProjectDatasetMetadata(
            project=logger.ObjectMetadata(id="test-project-id", name="test project", full_info={}),
            dataset=logger.ObjectMetadata(id="test-dataset-id", name="test dataset", full_info={}),
        )
        dataset = logger.Dataset(
            lazy_metadata=LazyValue(lambda: metadata, use_mutex=False),
            legacy=False,
            state=mock_state,
        )
        dataset.new_records = 1

        summary = dataset.summarize()

        self.assertEqual(summary.project_name, "test project")
        self.assertEqual(summary.dataset_name, "test dataset")
        self.assertEqual(summary.project_url, "https://public.example.com/app/test%20org/p/test%20project")
        self.assertEqual(
            summary.dataset_url,
            "https://public.example.com/app/test%20org/p/test%20project/datasets/test%20dataset",
        )
        self.assertEqual(summary.data_summary, logger.DataSummary(new_records=1, total_records=3))
        mock_state.api_client.return_value.datasets.get_dataset_id_summarize.assert_called_once_with(
            "test-dataset-id", summarize_data=True
        )
        mock_state.api_conn.assert_not_called()


class TestDatasetInternalBtql(TestCase):
    """Test that _internal_btql parameters (especially limit) are properly passed through to BTQL queries."""

    def test_init_dataset_applies_bt_eval_internal_btql_runtime_value(self):
        """Test that bt eval runtime BTQL is injected into dataset BTQL."""
        from braintrust.logger import init_dataset

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(builtins, "__bt_eval_internal_btql", {"sample": 5}, raising=False)
        try:
            dataset = init_dataset(project="test-project", name="test-dataset", use_output=False, state=MagicMock())

            self.assertEqual(dataset._internal_btql, {"sample": 5})
        finally:
            monkeypatch.undo()

    def test_init_dataset_merges_bt_eval_internal_btql_with_internal_btql(self):
        """Test that bt eval runtime BTQL is added to existing BTQL filters."""
        from braintrust.logger import init_dataset

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(builtins, "__bt_eval_internal_btql", {"sample": 5, "limit": 10}, raising=False)
        try:
            internal_btql = {"where": {"op": "eq", "left": "metadata.kind", "right": "synthetic"}}
            dataset = init_dataset(
                project="test-project",
                name="test-dataset",
                use_output=False,
                _internal_btql=internal_btql,
                state=MagicMock(),
            )

            self.assertEqual(
                dataset._internal_btql,
                {
                    "where": {"op": "eq", "left": "metadata.kind", "right": "synthetic"},
                    "sample": 5,
                    "limit": 10,
                },
            )
            self.assertEqual(internal_btql, {"where": {"op": "eq", "left": "metadata.kind", "right": "synthetic"}})
        finally:
            monkeypatch.undo()

    def test_init_dataset_merges_bt_eval_internal_btql_without_overriding_explicit_keys(self):
        """Test that explicit BTQL keys override bt eval runtime BTQL."""
        from braintrust.logger import init_dataset

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(builtins, "__bt_eval_internal_btql", {"sample": 5, "limit": 10}, raising=False)
        try:
            dataset = init_dataset(
                project="test-project",
                name="test-dataset",
                use_output=False,
                _internal_btql={"filter": "metadata.kind = 'synthetic'", "sample": 2},
                state=MagicMock(),
            )

            self.assertEqual(
                dataset._internal_btql,
                {"filter": "metadata.kind = 'synthetic'", "sample": 2, "limit": 10},
            )
        finally:
            monkeypatch.undo()

    def test_init_dataset_keeps_btql_unchanged_without_eval_internal_btql_runtime_value(self):
        """Test that ordinary init_dataset calls are unchanged without runtime BTQL."""
        from braintrust.logger import init_dataset

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.delattr(builtins, "__bt_eval_internal_btql", raising=False)
        try:
            dataset = init_dataset(project="test-project", name="test-dataset", use_output=False, state=MagicMock())

            self.assertIsNone(dataset._internal_btql)
        finally:
            monkeypatch.undo()

    def test_init_dataset_forwards_bt_eval_internal_btql_runtime_value_to_fetch(self):
        """Test that bt eval runtime BTQL is included in fetched dataset BTQL."""
        from braintrust.logger import init_dataset

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(builtins, "__bt_eval_internal_btql", {"sample": 5}, raising=False)
        try:
            mock_state = MagicMock()
            mock_state.org_id = "test-org"

            mock_app_conn = MagicMock()
            mock_app_conn.post_json.return_value = {
                "project": {"id": "test-project-id", "name": "test-project"},
                "dataset": {"id": "test-dataset-id", "name": "test-dataset"},
            }
            mock_state.app_conn.return_value = mock_app_conn

            mock_api_conn = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"data": [], "cursor": None}
            mock_api_conn.post.return_value = mock_response
            mock_state.api_conn.return_value = mock_api_conn

            dataset = init_dataset(project="test-project", name="test-dataset", use_output=False, state=mock_state)
            list(dataset.fetch())

            query_json = mock_api_conn.post.call_args[1]["json"]["query"]
            self.assertEqual(query_json["sample"], 5)
        finally:
            monkeypatch.undo()

    @patch("braintrust.logger.BraintrustState")
    def test_dataset_internal_btql_limit_not_overwritten(self, mock_state_class):
        """Test that custom limit in _internal_btql is not overwritten by DEFAULT_FETCH_BATCH_SIZE."""
        # Set up mock state
        mock_state = MagicMock()
        mock_state_class.return_value = mock_state

        # Mock the API connection and response
        mock_api_conn = MagicMock()
        mock_state.api_conn.return_value = mock_api_conn

        # Mock response object
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "1", "input": "test1", "expected": "output1"},
                {"id": "2", "input": "test2", "expected": "output2"},
            ],
            "cursor": None,
        }
        mock_api_conn.post.return_value = mock_response

        # Create dataset with custom limit in _internal_btql
        from braintrust.logger import Dataset, LazyValue, ObjectMetadata, ProjectDatasetMetadata

        project_metadata = ObjectMetadata(id="test-project", name="test-project", full_info={})
        dataset_metadata = ObjectMetadata(id="test-dataset", name="test-dataset", full_info={})
        lazy_metadata = LazyValue(
            lambda: ProjectDatasetMetadata(project=project_metadata, dataset=dataset_metadata),
            use_mutex=False,
        )

        custom_limit = 50
        dataset = Dataset(
            lazy_metadata=lazy_metadata,
            _internal_btql={"limit": custom_limit, "where": {"op": "eq", "left": "foo", "right": "bar"}},
            state=mock_state,
        )

        # Trigger a fetch which will make the BTQL query
        list(dataset.fetch())

        # Verify the API was called
        mock_api_conn.post.assert_called_once()

        # Get the actual call arguments
        call_args = mock_api_conn.post.call_args
        query_json = call_args[1]["json"]["query"]

        # Verify that the custom limit is present (not overwritten by DEFAULT_FETCH_BATCH_SIZE)
        self.assertEqual(query_json["limit"], custom_limit)

        # Verify that other _internal_btql fields are also present
        self.assertEqual(query_json["where"], {"op": "eq", "left": "foo", "right": "bar"})

    def test_dataset_internal_btql_zero_limit_skips_fetch(self):
        from braintrust.logger import Dataset, LazyValue, ObjectMetadata, ProjectDatasetMetadata

        project_metadata = ObjectMetadata(id="test-project", name="test-project", full_info={})
        dataset_metadata = ObjectMetadata(id="test-dataset", name="test-dataset", full_info={})
        compute_metadata = MagicMock(
            return_value=ProjectDatasetMetadata(project=project_metadata, dataset=dataset_metadata)
        )
        mock_state = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [], "cursor": "unexpected-cursor"}
        mock_state.api_conn.return_value.post.return_value = mock_response
        dataset = Dataset(
            lazy_metadata=LazyValue(compute_metadata, use_mutex=False),
            _internal_btql={"limit": 0},
            state=mock_state,
        )

        self.assertEqual(list(dataset), [])
        compute_metadata.assert_not_called()
        mock_state.api_conn.assert_not_called()

    def test_dataset_default_limit_when_not_specified(self):
        """Default dataset fetches use the generated API's standard batch size."""
        from braintrust.logger import (
            DEFAULT_FETCH_BATCH_SIZE,
            Dataset,
            LazyValue,
            ObjectMetadata,
            ProjectDatasetMetadata,
        )

        mock_state = MagicMock()
        mock_state.api_client.return_value.datasets.post_dataset_id_fetch.return_value = {"events": []}

        # Create dataset without custom limit
        project_metadata = ObjectMetadata(id="test-project", name="test-project", full_info={})
        dataset_metadata = ObjectMetadata(id="test-dataset", name="test-dataset", full_info={})
        lazy_metadata = LazyValue(
            lambda: ProjectDatasetMetadata(project=project_metadata, dataset=dataset_metadata),
            use_mutex=False,
        )

        dataset = Dataset(
            lazy_metadata=lazy_metadata,
            _internal_btql=None,
            state=mock_state,
        )

        list(dataset.fetch())

        mock_state.api_client.return_value.datasets.post_dataset_id_fetch.assert_called_once_with(
            "test-dataset", body={"limit": DEFAULT_FETCH_BATCH_SIZE}
        )


@pytest.mark.vcr
def test_dataset_internal_btql_limit_caps_total_results():
    dataset = braintrust.init_dataset(
        project="python-sdk-vcr-tests",
        name="test-dataset-internal-btql-total-limit",
        api_key=os.environ.get("BRAINTRUST_API_KEY", "sk-dummy-for-vcr-replay"),
        use_output=False,
        _internal_btql={"limit": 1},
    )
    dataset.insert(id="internal-btql-limit-record-1", input="first", expected="first")
    dataset.insert(id="internal-btql-limit-record-2", input="second", expected="second")
    dataset.flush()

    assert len(list(dataset)) == 1


def test_attachment_upload_tracked_on_flush(with_memory_logger, with_simulate_login):
    """Test that attachment upload is tracked when attachments are logged and flushed."""
    attachment = Attachment(data=b"test data", filename="test.txt", content_type="text/plain")

    logger = init_test_logger(__name__)
    span = logger.start_span(name="test_span")
    span.log(input={"file": attachment})
    span.end()

    # No upload attempts yet
    assert len(with_memory_logger.upload_attempts) == 0

    # Flush should track upload attempt
    logger.flush()

    # Now upload should be tracked
    assert len(with_memory_logger.upload_attempts) == 1
    assert with_memory_logger.upload_attempts[0] is attachment


def test_same_attachment_logged_twice_tracked_twice(with_memory_logger, with_simulate_login):
    """Test that same attachment logged twice appears twice in upload attempts."""
    attachment = Attachment(data=b"data", filename="file.txt", content_type="text/plain")

    logger = init_test_logger(__name__)
    span = logger.start_span(name="test_span")
    span.log(input={"file": attachment})
    span.log(metadata={"same_file": attachment})
    span.end()
    logger.flush()

    # Same attachment should be tracked twice (once for each log call)
    assert len(with_memory_logger.upload_attempts) == 2
    assert with_memory_logger.upload_attempts[0] is attachment
    assert with_memory_logger.upload_attempts[1] is attachment


def test_multiple_attachment_types_tracked(with_memory_logger, with_simulate_login):
    """Test that different attachment types are all tracked."""
    attachment = Attachment(data=b"data", filename="file.txt", content_type="text/plain")
    json_attachment = JSONAttachment({"key": "value"}, filename="data.json")
    ext_attachment = ExternalAttachment(url="s3://bucket/key", filename="file.pdf", content_type="application/pdf")

    logger = init_test_logger(__name__)
    span = logger.start_span(name="test_span")
    span.log(input=attachment, output=json_attachment, metadata={"file": ext_attachment})
    span.end()
    logger.flush()

    # All three types should be tracked
    assert len(with_memory_logger.upload_attempts) == 3
    assert attachment in with_memory_logger.upload_attempts
    assert json_attachment in with_memory_logger.upload_attempts
    assert ext_attachment in with_memory_logger.upload_attempts


# --- Tests for Span.name property ---


def test_span_name_returns_explicit_name(with_memory_logger):
    """Test that span.name returns the name passed to start_span()."""
    test_logger = init_test_logger(__name__)

    with test_logger.start_span(name="my-span") as span:
        assert span.name == "my-span"


def test_span_name_returns_inferred_root_name(with_memory_logger):
    """Test that a root span with no explicit name gets the default 'root' name."""
    test_logger = init_test_logger(__name__)

    span = test_logger.start_span()
    assert span.name == "root"
    span.end()


def test_span_name_returns_inferred_subspan_name(with_memory_logger):
    """Test that a child span with no explicit name gets a caller-location-based name."""
    test_logger = init_test_logger(__name__)

    with test_logger.start_span(name="parent") as parent:
        child = parent.start_span()
        # The inferred name is based on caller location: "funcname:filename:lineno"
        assert child.name is not None
        assert len(child.name) > 0
        child.end()


def test_span_name_updated_by_set_attributes(with_memory_logger):
    """Test that span.name reflects changes made via set_attributes()."""
    test_logger = init_test_logger(__name__)

    with test_logger.start_span(name="original") as span:
        assert span.name == "original"
        span.set_attributes(name="renamed")
        assert span.name == "renamed"


def test_span_name_consistent_with_logged_data(with_memory_logger):
    """Test that span.name matches the name in the logged span_attributes."""
    test_logger = init_test_logger(__name__)

    with test_logger.start_span(name="logged-name") as span:
        assert span.name == "logged-name"

    logs = with_memory_logger.pop()
    logged_name = logs[0].get("span_attributes", {}).get("name")
    assert logged_name == "logged-name"


def test_noop_span_name_returns_none():
    """Test that the noop span's name property returns None."""
    span = braintrust.NOOP_SPAN
    assert span.name == ""


def test_current_span_name_accessible(with_memory_logger):
    """Test that current_span().name works inside a traced context."""
    test_logger = init_test_logger(__name__)

    captured_name = None
    with test_logger.start_span(name="active-span") as span:
        span.set_current()
        captured_name = braintrust.current_span().name

    assert captured_name == "active-span"


def test_traced_decorator_span_name(with_memory_logger):
    """Test that @traced sets span name to the function name by default."""
    test_logger = init_test_logger(__name__)

    captured_name = None

    @logger.traced
    def my_traced_function():
        nonlocal captured_name
        captured_name = braintrust.current_span().name
        return "done"

    my_traced_function()

    assert captured_name == "my_traced_function"


def _raise_test_exception_group():
    """Raise and return a standard ExceptionGroup with a traceback for testing."""
    try:
        raise exceptiongroup.ExceptionGroup(
            "Multiple failures",
            [
                ConnectionRefusedError("[Errno 61] Connection refused"),
                ValueError("Invalid configuration"),
            ],
        )
    except exceptiongroup.ExceptionGroup as eg:
        return eg


def _assert_test_exception_group_contents(error_str):
    """Assert that error_str contains the expected sub-exception details."""
    assert "ExceptionGroup: Multiple failures" in error_str
    assert "ConnectionRefusedError" in error_str
    assert "[Errno 61] Connection refused" in error_str
    assert "ValueError" in error_str
    assert "Invalid configuration" in error_str


def test_stringify_exception_with_exception_group():
    eg = _raise_test_exception_group()
    result = stringify_exception(type(eg), eg, eg.__traceback__)
    _assert_test_exception_group_contents(result)
    assert "(2 sub-exceptions)" in result


def test_stringify_exception_with_nested_exception_group():
    result = ""
    try:
        inner = exceptiongroup.ExceptionGroup("inner", [TypeError("bad type")])
        raise exceptiongroup.ExceptionGroup(
            "outer",
            [inner, RuntimeError("top-level error")],
        )
    except exceptiongroup.ExceptionGroup as eg:
        result = stringify_exception(type(eg), eg, eg.__traceback__)

    assert result, "ExceptionGroup was not raised"
    assert "outer" in result
    assert "inner" in result
    assert "TypeError" in result
    assert "bad type" in result
    assert "RuntimeError" in result
    assert "top-level error" in result


def test_span_exit_logs_exception_group_sub_exceptions(with_memory_logger):
    """Verify sub-exceptions are captured when an ExceptionGroup propagates through span.__exit__."""
    init_test_logger(__name__)

    with pytest.raises(exceptiongroup.ExceptionGroup):
        with braintrust.current_logger().start_span(name="eg-span"):
            raise _raise_test_exception_group()

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    _assert_test_exception_group_contents(logs[0].get("error", ""))


@pytest.mark.parametrize(
    "exception_type",
    [GeneratorExit, asyncio.CancelledError, KeyboardInterrupt, SystemExit],
)
def test_span_exit_logs_base_exceptions(with_memory_logger, exception_type):
    init_test_logger(__name__)

    with pytest.raises(exception_type):
        with braintrust.current_logger().start_span(name="base-exception-span"):
            raise exception_type

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    assert exception_type.__name__ in logs[0]["error"]


def test_traced_logs_exception_group_sub_exceptions(with_memory_logger):
    """Verify sub-exceptions are captured when an ExceptionGroup propagates through @traced."""
    init_test_logger(__name__)

    @logger.traced
    def failing_function():
        raise _raise_test_exception_group()

    with pytest.raises(exceptiongroup.ExceptionGroup):
        failing_function()

    logs = with_memory_logger.pop()
    assert len(logs) == 1
    _assert_test_exception_group_contents(logs[0].get("error", ""))


def test_check_org_info_no_git_metadata_leaves_settings_none():
    """When org has no git_metadata, state.git_metadata_settings should be None (no org restriction)."""
    state = BraintrustState()
    org_info = [
        {
            "id": "org-1",
            "name": "org1",
            "api_url": "https://api.example.com",
            "proxy_url": "https://proxy.example.com",
        }
    ]
    _check_org_info(state, org_info, None)
    assert state.git_metadata_settings is None


def test_check_org_info_with_git_metadata_uses_server_settings():
    """When org provides git_metadata, it is used as-is."""
    state = BraintrustState()
    org_info = [
        {
            "id": "org-1",
            "name": "org1",
            "api_url": "https://api.example.com",
            "proxy_url": "https://proxy.example.com",
            "git_metadata": {"collect": "some", "fields": ["commit", "branch"]},
        }
    ]
    _check_org_info(state, org_info, None)
    assert state.git_metadata_settings.collect == "some"
    assert set(state.git_metadata_settings.fields) == {"commit", "branch"}


def test_proxy_conn_strips_v1_proxy_suffix():
    """EU/self-hosted proxy_url ends in /v1/proxy; proxy_conn must target the API host root."""
    state = BraintrustState()
    state.proxy_url = "https://api-eu.braintrust.dev/v1/proxy"
    assert state.proxy_conn().base_url == "https://api-eu.braintrust.dev"


def test_proxy_conn_leaves_bare_host_unchanged():
    """A bare proxy host (US default) is used as-is."""
    state = BraintrustState()
    state.proxy_url = "https://api.braintrust.dev"
    assert state.proxy_conn().base_url == "https://api.braintrust.dev"


def test_get_repo_info_without_settings_returns_none():
    """Direct call to get_repo_info with settings=None should return None."""
    assert get_repo_info(None) is None
