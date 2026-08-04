"""Authentication and organization discovery service."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..env import BraintrustEnv
from ._routing import EndpointRouter, RequestTarget
from ._transport import HTTPConnection, Transport
from .policies import RetryMode


@dataclass(frozen=True)
class OrganizationInfo:
    """Organization routing information returned by API-key login."""

    id: str
    name: str
    api_url: str | None
    proxy_url: str | None
    realtime_url: str | None
    is_universal_api: bool
    git_metadata: Mapping[str, Any] | None
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OrganizationInfo":
        """Parse an additive login response while retaining unknown fields."""

        org_id = value.get("id")
        name = value.get("name")
        if not isinstance(org_id, str) or not isinstance(name, str):
            raise ValueError("Organization login data must include string id and name fields")

        def optional_string(field: str) -> str | None:
            result = value.get(field)
            return result if isinstance(result, str) and result else None

        git_metadata = value.get("git_metadata")
        if not isinstance(git_metadata, Mapping):
            git_metadata = None

        return cls(
            id=org_id,
            name=name,
            api_url=optional_string("api_url"),
            proxy_url=optional_string("proxy_url"),
            realtime_url=optional_string("realtime_url"),
            is_universal_api=bool(value.get("is_universal_api", False)),
            git_metadata=MappingProxyType(dict(git_metadata)) if git_metadata is not None else None,
            raw=MappingProxyType(dict(value)),
        )


@dataclass(frozen=True)
class LoginResult:
    """Selected organization and the complete login response."""

    organization: OrganizationInfo
    response: Mapping[str, Any]


class AuthAPI:
    """Authenticate an API key and configure an endpoint router."""

    def __init__(self, transport: Transport, router: EndpointRouter):
        self._transport = transport
        self._router = router

    def login(
        self,
        api_key: str,
        *,
        org_name: str | None = None,
        api_url: str | None = None,
        proxy_url: str | None = None,
    ) -> LoginResult:
        """Log in, select an organization, and apply routing override precedence."""

        api_key = HTTPConnection.sanitize_token(api_key)
        response = self._transport.request_json(
            "POST",
            self._router.resolve(RequestTarget.APP, "/api/apikey/login"),
            headers={"Authorization": f"Bearer {api_key}"},
            retry_mode=RetryMode.SAFE_READ,
        )
        if not isinstance(response, Mapping):
            raise ValueError("API-key login returned a non-object response")
        raw_orgs = response.get("org_info")
        if not isinstance(raw_orgs, Sequence) or isinstance(raw_orgs, (str, bytes)):
            raise ValueError("API-key login response did not include an organization list")

        organizations = [OrganizationInfo.from_dict(org) for org in raw_orgs if isinstance(org, Mapping)]
        organization = self._select_organization(organizations, org_name)

        resolved_api_url = api_url or BraintrustEnv.API_URL.get(organization.api_url)
        resolved_proxy_url = proxy_url or BraintrustEnv.PROXY_URL.get(organization.proxy_url)
        if not resolved_api_url:
            if org_name:
                raise ValueError(
                    f"Unable to log into organization '{org_name}'."
                    " Are you sure this credential is scoped to the organization?"
                )
            raise ValueError("Unable to log into any organization with the provided credential.")

        self._router.configure(
            api_url=resolved_api_url,
            proxy_url=resolved_proxy_url,
            is_universal_api=organization.is_universal_api,
        )
        return LoginResult(organization=organization, response=MappingProxyType(dict(response)))

    @staticmethod
    def _select_organization(organizations: Sequence[OrganizationInfo], org_name: str | None) -> OrganizationInfo:
        if not organizations:
            raise ValueError("This user is not part of any organizations.")
        for organization in organizations:
            if org_name is None or organization.name == org_name:
                return organization
        choices = ", ".join(organization.name for organization in organizations)
        raise ValueError(f"Organization {org_name} not found. Must be one of {choices}")
