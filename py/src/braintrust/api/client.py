"""Synchronous Braintrust API clients."""

from threading import Lock
from typing import TYPE_CHECKING, Any, TypeVar, cast

import requests
from requests.adapters import HTTPAdapter

from ..env import BraintrustEnv, resolve_app_url
from ._routing import EndpointRouter
from ._service import ResourceAPI
from ._transport import HTTPConnection, Transport
from .auth import AuthAPI


if TYPE_CHECKING:
    from ._generated.acls import AclsAPI
    from ._generated.agents import AgentsAPI
    from ._generated.ai_secrets import AiSecretsAPI
    from ._generated.api_keys import ApiKeysAPI
    from ._generated.dataset_snapshots import DatasetSnapshotsAPI
    from ._generated.datasets import DatasetsAPI
    from ._generated.env_vars import EnvVarsAPI
    from ._generated.environments import EnvironmentsAPI
    from ._generated.experiments import ExperimentsAPI
    from ._generated.functions import FunctionsAPI
    from ._generated.groups import GroupsAPI
    from ._generated.mcp_servers import McpServersAPI
    from ._generated.org_automations import OrgAutomationsAPI
    from ._generated.organizations import OrganizationsAPI
    from ._generated.project_automations import ProjectAutomationsAPI
    from ._generated.project_groups import ProjectGroupsAPI
    from ._generated.project_scores import ProjectScoresAPI
    from ._generated.project_tags import ProjectTagsAPI
    from ._generated.projects import ProjectsAPI
    from ._generated.prompts import PromptsAPI
    from ._generated.roles import RolesAPI
    from ._generated.service_tokens import ServiceTokensAPI
    from ._generated.span_iframes import SpanIframesAPI
    from ._generated.users import UsersAPI
    from ._generated.views import ViewsAPI


_ServiceT = TypeVar("_ServiceT", bound=ResourceAPI)


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
            request_timeout=BraintrustEnv.HTTP_TIMEOUT.get(None),
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

    Construction performs no network requests. Generated resources are imported and cached on
    first access. Use :class:`BraintrustClient` when authentication and generated resources should
    share one transport.
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
        self._initialize_service_cache(resolved_api_key)

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
        client._initialize_service_cache(HTTPConnection.sanitize_token(api_key))
        return client

    def _initialize_service_cache(self, api_key: str) -> None:
        self.api_key = api_key
        self._services: dict[str, ResourceAPI] = {}
        self._services_lock = Lock()

    def _service(self, name: str, service_type: type[_ServiceT]) -> _ServiceT:
        with self._services_lock:
            service = self._services.get(name)
            if service is None:
                service = service_type(self.transport, self.router, self.api_key)
                self._services[name] = service
        return cast(_ServiceT, service)

    @property
    def acls(self) -> "AclsAPI":
        from ._generated.acls import AclsAPI

        return self._service("acls", AclsAPI)

    @property
    def agents(self) -> "AgentsAPI":
        from ._generated.agents import AgentsAPI

        return self._service("agents", AgentsAPI)

    @property
    def ai_secrets(self) -> "AiSecretsAPI":
        from ._generated.ai_secrets import AiSecretsAPI

        return self._service("ai_secrets", AiSecretsAPI)

    @property
    def api_keys(self) -> "ApiKeysAPI":
        from ._generated.api_keys import ApiKeysAPI

        return self._service("api_keys", ApiKeysAPI)

    @property
    def dataset_snapshots(self) -> "DatasetSnapshotsAPI":
        from ._generated.dataset_snapshots import DatasetSnapshotsAPI

        return self._service("dataset_snapshots", DatasetSnapshotsAPI)

    @property
    def datasets(self) -> "DatasetsAPI":
        from ._generated.datasets import DatasetsAPI

        return self._service("datasets", DatasetsAPI)

    @property
    def env_vars(self) -> "EnvVarsAPI":
        from ._generated.env_vars import EnvVarsAPI

        return self._service("env_vars", EnvVarsAPI)

    @property
    def environments(self) -> "EnvironmentsAPI":
        from ._generated.environments import EnvironmentsAPI

        return self._service("environments", EnvironmentsAPI)

    @property
    def experiments(self) -> "ExperimentsAPI":
        from ._generated.experiments import ExperimentsAPI

        return self._service("experiments", ExperimentsAPI)

    @property
    def functions(self) -> "FunctionsAPI":
        from ._generated.functions import FunctionsAPI

        return self._service("functions", FunctionsAPI)

    @property
    def groups(self) -> "GroupsAPI":
        from ._generated.groups import GroupsAPI

        return self._service("groups", GroupsAPI)

    @property
    def mcp_servers(self) -> "McpServersAPI":
        from ._generated.mcp_servers import McpServersAPI

        return self._service("mcp_servers", McpServersAPI)

    @property
    def org_automations(self) -> "OrgAutomationsAPI":
        from ._generated.org_automations import OrgAutomationsAPI

        return self._service("org_automations", OrgAutomationsAPI)

    @property
    def organizations(self) -> "OrganizationsAPI":
        from ._generated.organizations import OrganizationsAPI

        return self._service("organizations", OrganizationsAPI)

    @property
    def project_automations(self) -> "ProjectAutomationsAPI":
        from ._generated.project_automations import ProjectAutomationsAPI

        return self._service("project_automations", ProjectAutomationsAPI)

    @property
    def project_groups(self) -> "ProjectGroupsAPI":
        from ._generated.project_groups import ProjectGroupsAPI

        return self._service("project_groups", ProjectGroupsAPI)

    @property
    def project_scores(self) -> "ProjectScoresAPI":
        from ._generated.project_scores import ProjectScoresAPI

        return self._service("project_scores", ProjectScoresAPI)

    @property
    def project_tags(self) -> "ProjectTagsAPI":
        from ._generated.project_tags import ProjectTagsAPI

        return self._service("project_tags", ProjectTagsAPI)

    @property
    def projects(self) -> "ProjectsAPI":
        from ._generated.projects import ProjectsAPI

        return self._service("projects", ProjectsAPI)

    @property
    def prompts(self) -> "PromptsAPI":
        from ._generated.prompts import PromptsAPI

        return self._service("prompts", PromptsAPI)

    @property
    def roles(self) -> "RolesAPI":
        from ._generated.roles import RolesAPI

        return self._service("roles", RolesAPI)

    @property
    def service_tokens(self) -> "ServiceTokensAPI":
        from ._generated.service_tokens import ServiceTokensAPI

        return self._service("service_tokens", ServiceTokensAPI)

    @property
    def span_iframes(self) -> "SpanIframesAPI":
        from ._generated.span_iframes import SpanIframesAPI

        return self._service("span_iframes", SpanIframesAPI)

    @property
    def users(self) -> "UsersAPI":
        from ._generated.users import UsersAPI

        return self._service("users", UsersAPI)

    @property
    def views(self) -> "ViewsAPI":
        from ._generated.views import ViewsAPI

        return self._service("views", ViewsAPI)

    def close(self) -> None:
        """Close the transport when it was created by this client."""

        if self._owns_transport:
            self.transport.close()

    def __enter__(self) -> "BraintrustOpenApiClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
