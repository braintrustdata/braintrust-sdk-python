"""Synchronous Braintrust API client facade."""

from typing import Any

import requests
from requests.adapters import HTTPAdapter

from ..env import BraintrustEnv, resolve_app_url, resolve_org_name
from ._routing import EndpointRouter
from ._service import ClientContext
from ._transport import HTTPConnection, Transport
from .attachments import AttachmentsAPI
from .auth import AuthAPI, LoginResult
from .datasets import DatasetsAPI
from .experiments import ExperimentsAPI
from .functions import FunctionsAPI
from .projects import ProjectsAPI
from .prompts import PromptsAPI
from .queries import QueriesAPI


class BraintrustClient:
    """Synchronous resource-oriented client for the Braintrust API.

    The convenience constructor authenticates through the app origin, selects an
    organization, and configures API and proxy routing on one shared transport.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        org_name: str | None = None,
        app_url: str | None = None,
        api_url: str | None = None,
        proxy_url: str | None = None,
        session: requests.Session | None = None,
        adapter: HTTPAdapter | None = None,
        transport: Transport | None = None,
        enable_sdk_retries: bool | None = None,
    ):
        if transport is not None and (session is not None or adapter is not None or enable_sdk_retries is not None):
            raise ValueError("transport cannot be combined with session, adapter, or enable_sdk_retries")

        resolved_api_key = api_key or BraintrustEnv.API_KEY.get(None, use_dotenv=True)
        if not resolved_api_key:
            raise ValueError(
                "Could not login to Braintrust. You may need to set BRAINTRUST_API_KEY in your environment "
                "or nearest .env.braintrust file."
            )
        resolved_api_key = HTTPConnection.sanitize_token(resolved_api_key)
        resolved_org_name = resolve_org_name(org_name)
        resolved_app_url = resolve_app_url(app_url)

        self._owns_transport = transport is None
        self.transport = transport or Transport(
            session=session,
            adapter=adapter,
            enable_sdk_retries=enable_sdk_retries,
            persist_cookies=False,
        )
        self.router = EndpointRouter(app_url=resolved_app_url)
        auth = AuthAPI(self.transport, self.router)
        try:
            result = auth.login(
                resolved_api_key,
                org_name=resolved_org_name,
                api_url=api_url,
                proxy_url=proxy_url,
            )
        except Exception:
            if self._owns_transport:
                self.transport.close()
            raise

        self._initialize_services(result, resolved_api_key)

    @classmethod
    def from_transport(
        cls,
        *,
        transport: Transport,
        router: EndpointRouter,
        api_key: str,
        org_id: str,
        org_name: str,
        login_result: LoginResult | None = None,
    ) -> "BraintrustClient":
        """Build a client around an already-authenticated transport and router."""

        client = cls.__new__(cls)
        client._owns_transport = False
        client.transport = transport
        client.router = router
        client._initialize_services_from_context(
            ClientContext(org_id=org_id, org_name=org_name),
            HTTPConnection.sanitize_token(api_key),
            login_result,
        )
        return client

    def _initialize_services(self, result: LoginResult, api_key: str) -> None:
        organization = result.organization
        self._initialize_services_from_context(
            ClientContext(org_id=organization.id, org_name=organization.name),
            api_key,
            result,
        )

    def _initialize_services_from_context(
        self,
        context: ClientContext,
        api_key: str,
        login_result: LoginResult | None,
    ) -> None:
        self.context = context
        self.api_key = api_key
        self._login_result = login_result
        service_args: tuple[Any, ...] = (self.transport, self.router, context, api_key)
        self.projects = ProjectsAPI(*service_args)
        self.experiments = ExperimentsAPI(*service_args)
        self.datasets = DatasetsAPI(*service_args)
        self.prompts = PromptsAPI(*service_args)
        self.functions = FunctionsAPI(*service_args)
        self.queries = QueriesAPI(*service_args)
        self.attachments = AttachmentsAPI(*service_args)

    @property
    def login_result(self) -> LoginResult:
        """Return organization discovery details for a bootstrapped client."""

        if self._login_result is None:
            raise RuntimeError("Login details are unavailable for a pre-authenticated client")
        return self._login_result

    @property
    def org_id(self) -> str:
        return self.context.org_id

    @property
    def org_name(self) -> str:
        return self.context.org_name

    def close(self) -> None:
        """Close the transport when it was created by this client."""

        if self._owns_transport:
            self.transport.close()

    def __enter__(self) -> "BraintrustClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
