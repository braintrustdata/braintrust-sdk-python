"""Endpoint routing for Braintrust API requests."""

import enum
from dataclasses import dataclass

from ..util import _urljoin


_V1_PROXY_SUFFIX = "/v1/proxy"


class RequestTarget(enum.Enum):
    """A logical Braintrust request destination."""

    APP = "app"
    API = "api"
    PROXY = "proxy"


def normalize_proxy_url(proxy_url: str) -> str:
    """Normalize a Universal Proxy URL to the API host used by SDK routes."""

    if proxy_url.endswith(_V1_PROXY_SUFFIX):
        return proxy_url[: -len(_V1_PROXY_SUFFIX)]
    return proxy_url


@dataclass
class EndpointRouter:
    """Resolve logical Braintrust targets without changing their configured origins."""

    app_url: str
    api_url: str | None = None
    proxy_url: str | None = None
    is_universal_api: bool = False

    def configure(
        self,
        *,
        api_url: str | None,
        proxy_url: str | None,
        is_universal_api: bool = False,
    ) -> None:
        """Apply URLs discovered during authentication."""

        self.api_url = api_url
        self.proxy_url = proxy_url
        self.is_universal_api = is_universal_api

    def base_url(self, target: RequestTarget) -> str:
        """Return the configured origin for ``target``."""

        if target is RequestTarget.APP:
            return self.app_url
        if target is RequestTarget.API:
            if not self.api_url:
                raise RuntimeError("API URL is unavailable before organization discovery")
            return self.api_url
        if target is RequestTarget.PROXY:
            base_url = self.proxy_url or self.api_url
            if not base_url:
                raise RuntimeError("Proxy URL is unavailable before organization discovery")
            return normalize_proxy_url(base_url)
        raise ValueError(f"Unknown request target: {target!r}")

    def resolve(self, target: RequestTarget, path: str) -> str:
        """Resolve ``path`` against the origin for ``target``."""

        return _urljoin(self.base_url(target), path)
