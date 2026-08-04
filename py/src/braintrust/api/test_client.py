import contextlib
import http.server
import json
import socketserver
import threading

import pytest
import requests
from braintrust import logger
from braintrust.api import BraintrustClient, BraintrustHTTPError, EndpointRouter, RequestTarget
from braintrust.api._transport import RetryRequestExceptionsAdapter
from braintrust.logger import BraintrustState, login_to_state
from requests.adapters import HTTPAdapter


@contextlib.contextmanager
def login_server(orgs, *, status=200, response_headers=None):
    body = json.dumps({"org_info": orgs}).encode()

    class LoginHandler(http.server.BaseHTTPRequestHandler):
        request_count = 0
        authorization = None

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            type(self).request_count += 1
            type(self).authorization = self.headers.get("Authorization")
            self.send_response(status)
            for name, value in (response_headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), LoginHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", LoginHandler
    finally:
        server.shutdown()
        server.server_close()


def test_endpoint_router_preserves_origins_and_proxy_fallback():
    router = EndpointRouter(
        app_url="https://app.example.com/",
        api_url="https://api.example.com/",
    )

    assert router.resolve(RequestTarget.APP, "/api/apikey/login") == "https://app.example.com/api/apikey/login"
    assert router.resolve(RequestTarget.API, "ping") == "https://api.example.com/ping"
    assert router.resolve(RequestTarget.PROXY, "function/invoke") == "https://api.example.com/function/invoke"

    router.proxy_url = "https://universal.example.com/v1/proxy"
    assert router.resolve(RequestTarget.PROXY, "function/invoke") == "https://universal.example.com/function/invoke"


def test_client_bootstraps_selected_org_on_one_session(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_URL", raising=False)
    monkeypatch.delenv("BRAINTRUST_PROXY_URL", raising=False)
    orgs = [
        {
            "id": "org-1",
            "name": "first",
            "api_url": "https://api-1.example.com",
            "proxy_url": None,
        },
        {
            "id": "org-2",
            "name": "selected",
            "api_url": "https://api-2.example.com",
            "proxy_url": "https://api-2.example.com/v1/proxy",
            "is_universal_api": True,
            "new_server_field": {"preserved": True},
        },
    ]
    session = requests.Session()
    session.cookies.set("existing", "yes")
    cookie_policy = session.cookies.get_policy()
    with login_server(orgs, response_headers={"Set-Cookie": "accepted=yes"}) as (app_url, handler):
        client = BraintrustClient(api_key="secret\n", org_name="selected", app_url=app_url, session=session)

    assert handler.request_count == 1
    assert handler.authorization == "Bearer secret"
    assert client.org_id == "org-2"
    assert client.org_name == "selected"
    assert client.router.api_url == "https://api-2.example.com"
    assert client.router.is_universal_api is True
    assert client.router.resolve(RequestTarget.PROXY, "ping") == "https://api-2.example.com/ping"
    assert client.login_result.organization.raw["new_server_field"] == {"preserved": True}
    assert not hasattr(client, "auth")
    assert "Authorization" not in session.headers
    assert session.cookies.get("existing") == "yes"
    assert session.cookies.get_policy() is cookie_policy
    assert all(
        service._transport is client.transport
        for service in (
            client.projects,
            client.experiments,
            client.datasets,
            client.prompts,
            client.functions,
            client.queries,
            client.attachments,
        )
    )


def test_sdk_owned_session_rejects_response_cookies():
    orgs = [{"id": "org-1", "name": "org", "api_url": "https://api.example.com"}]
    with login_server(orgs, response_headers={"Set-Cookie": "ignored=yes"}) as (app_url, _):
        client = BraintrustClient(api_key="secret", app_url=app_url)

    assert not client.transport.session.cookies


def test_client_url_override_precedence(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_URL", "https://api-env.example.com")
    monkeypatch.setenv("BRAINTRUST_PROXY_URL", "https://proxy-env.example.com")
    orgs = [
        {
            "id": "org-1",
            "name": "org",
            "api_url": "https://api-discovered.example.com",
            "proxy_url": "https://proxy-discovered.example.com",
        }
    ]
    with login_server(orgs) as (app_url, _):
        env_client = BraintrustClient(api_key="secret", app_url=app_url)
        explicit_client = BraintrustClient(
            api_key="secret",
            app_url=app_url,
            api_url="https://api-explicit.example.com",
            proxy_url="https://proxy-explicit.example.com",
        )

    assert env_client.router.api_url == "https://api-env.example.com"
    assert env_client.router.proxy_url == "https://proxy-env.example.com"
    assert explicit_client.router.api_url == "https://api-explicit.example.com"
    assert explicit_client.router.proxy_url == "https://proxy-explicit.example.com"


def test_custom_adapter_disables_bootstrap_retries():
    orgs = [{"id": "org-1", "name": "org", "api_url": "https://api.example.com"}]
    with login_server(orgs, status=503) as (app_url, handler):
        with pytest.raises(BraintrustHTTPError):
            BraintrustClient(api_key="secret", app_url=app_url, adapter=HTTPAdapter())

    assert handler.request_count == 1


def test_login_to_state_hydrates_isolated_legacy_connections(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_URL", raising=False)
    monkeypatch.delenv("BRAINTRUST_PROXY_URL", raising=False)
    with login_server([]) as (app_url, _):
        orgs = [
            {
                "id": "org-1",
                "name": "org",
                "api_url": app_url,
                "proxy_url": app_url,
                "git_metadata": {},
            }
        ]
        with login_server(orgs) as (login_url, _):
            state = login_to_state(api_key="secret", app_url=login_url, org_name="org")

    assert state.api_client().org_id == "org-1"
    assert state.git_metadata_settings is None
    assert state._client.transport.session is not state.api_conn().session
    assert state._client.transport.session is not state.app_conn().session
    assert state.api_conn().session.headers["Authorization"] == "Bearer secret"
    assert state.app_conn().session.headers["Authorization"] == "Bearer secret"
    assert state.proxy_conn().session.headers["Authorization"] == "Bearer secret"
    assert isinstance(state.api_conn().adapter, RetryRequestExceptionsAdapter)
    assert isinstance(state.app_conn().adapter, RetryRequestExceptionsAdapter)
    assert isinstance(state.proxy_conn().adapter, RetryRequestExceptionsAdapter)


def test_legacy_adapter_mutation_keeps_characterized_target_scope(monkeypatch):
    state = BraintrustState()
    state.app_url = "https://app.example.com"
    state.api_url = "https://api.example.com"
    state.proxy_url = "https://proxy.example.com"
    monkeypatch.setattr(logger, "_state", state)
    monkeypatch.setattr(logger, "_http_adapter", None)

    app_connection = state.app_conn()
    api_connection = state.api_conn()
    proxy_connection = state.proxy_conn()
    adapter = HTTPAdapter()
    logger.set_http_adapter(adapter)

    assert app_connection.adapter is adapter
    assert api_connection.adapter is adapter
    assert proxy_connection.adapter is None


def test_state_concurrent_lazy_access_bootstraps_once(monkeypatch):
    monkeypatch.delenv("BRAINTRUST_API_URL", raising=False)
    monkeypatch.delenv("BRAINTRUST_PROXY_URL", raising=False)
    state = BraintrustState()
    with login_server([]) as (api_url, _):
        orgs = [{"id": "org-1", "name": "org", "api_url": api_url, "proxy_url": None}]
        with login_server(orgs) as (app_url, handler):
            monkeypatch.setenv("BRAINTRUST_API_KEY", "secret")
            monkeypatch.setenv("BRAINTRUST_APP_URL", app_url)
            clients = []
            threads = [threading.Thread(target=lambda: clients.append(state.api_client())) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    assert handler.request_count == 1
    assert len(clients) == 8
    assert all(client is clients[0] for client in clients)
