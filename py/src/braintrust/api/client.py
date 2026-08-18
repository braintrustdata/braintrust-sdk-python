"""Synchronous Braintrust API clients."""

from typing import Any

import requests
from requests.adapters import HTTPAdapter

from ..env import BraintrustEnv, resolve_app_url
from ._routing import EndpointRouter
from ._transport import HTTPConnection, Transport
from .auth import AuthAPI


def _resolve_api_key(api_key: str | None) -> str:
    resolved_api_key = api_key or BraintrustEnv.API_KEY.get(None, use_dotenv=True)
    if not resolved_api_key:
        raise ValueError(
            "Could not initialize the Braintrust API client. Set BRAINTRUST_API_KEY in your environment "
            "or nearest .env.braintrust file, or pass api_key explicitly."
        )
    return HTTPConnection.sanitize_token(resolved_api_key)


def _create_transport(
    *,
    session: requests.Session | None,
    adapter: HTTPAdapter | None,
    transport: Transport | None,
    enable_sdk_retries: bool | None,
) -> tuple[Transport, bool]:
    if transport is not None and (session is not None or adapter is not None or enable_sdk_retries is not None):
        raise ValueError("transport cannot be combined with session, adapter, or enable_sdk_retries")
    if transport is not None:
        return transport, False
    return (
        Transport(
            session=session,
            adapter=adapter,
            enable_sdk_retries=enable_sdk_retries,
            persist_cookies=False,
        ),
        True,
    )


class BraintrustClient:
    """Client for generated and handwritten Braintrust API services.

    Construction performs no network requests. Call ``client.auth.login()``
    to discover organization routing when ``api_url`` is not configured.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_url: str | None = None,
        api_url: str | None = None,
        proxy_url: str | None = None,
        session: requests.Session | None = None,
        adapter: HTTPAdapter | None = None,
        transport: Transport | None = None,
        enable_sdk_retries: bool | None = None,
    ):
        self.api_key = _resolve_api_key(api_key)
        self.transport, self._owns_transport = _create_transport(
            session=session,
            adapter=adapter,
            transport=transport,
            enable_sdk_retries=enable_sdk_retries,
        )
        self.router = EndpointRouter(
            app_url=resolve_app_url(app_url),
            api_url=api_url or BraintrustEnv.API_URL.get(None),
            proxy_url=proxy_url or BraintrustEnv.PROXY_URL.get(None),
        )
        self.auth = AuthAPI(
            self.transport,
            self.router,
            self.api_key,
            api_url=api_url,
            proxy_url=proxy_url,
        )
        self.openapi = BraintrustOpenApiClient.from_transport(
            transport=self.transport,
            router=self.router,
            api_key=self.api_key,
        )

    def close(self) -> None:
        """Close the transport when it was created by this client."""

        if self._owns_transport:
            self.transport.close()

    def __enter__(self) -> "BraintrustClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class BraintrustOpenApiClient:
    """Synchronous resource-oriented client for the Braintrust REST API.

    Construction performs no network requests. Use :class:`BraintrustClient`
    when authentication and generated resources should share one transport.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        proxy_url: str | None = None,
        session: requests.Session | None = None,
        adapter: HTTPAdapter | None = None,
        transport: Transport | None = None,
        enable_sdk_retries: bool | None = None,
    ):
        resolved_api_key = _resolve_api_key(api_key)
        resolved_api_url = api_url or BraintrustEnv.API_URL.get(None)
        if not resolved_api_url:
            raise ValueError("api_url is required when constructing BraintrustOpenApiClient")
        resolved_proxy_url = proxy_url or BraintrustEnv.PROXY_URL.get(None)
        self.transport, self._owns_transport = _create_transport(
            session=session,
            adapter=adapter,
            transport=transport,
            enable_sdk_retries=enable_sdk_retries,
        )
        self.router = EndpointRouter(
            app_url=resolve_app_url(None),
            api_url=resolved_api_url,
            proxy_url=resolved_proxy_url,
        )
        self._initialize_services(resolved_api_key)

    @classmethod
    def from_transport(
        cls,
        *,
        transport: Transport,
        router: EndpointRouter,
        api_key: str,
    ) -> "BraintrustOpenApiClient":
        """Build a client around an already-configured transport and router."""

        client = cls.__new__(cls)
        client._owns_transport = False
        client.transport = transport
        client.router = router
        client._initialize_services(HTTPConnection.sanitize_token(api_key))
        return client

    def _initialize_services(self, api_key: str) -> None:
        from ._generated.datasets import DatasetsAPI
        from ._generated.experiments import ExperimentsAPI
        from ._generated.projects import ProjectsAPI

        self.api_key = api_key
        self.datasets = DatasetsAPI(self.transport, self.router, api_key)
        self.experiments = ExperimentsAPI(self.transport, self.router, api_key)
        self.projects = ProjectsAPI(self.transport, self.router, api_key)

    def close(self) -> None:
        """Close the transport when it was created by this client."""

        if self._owns_transport:
            self.transport.close()

    def __enter__(self) -> "BraintrustOpenApiClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
