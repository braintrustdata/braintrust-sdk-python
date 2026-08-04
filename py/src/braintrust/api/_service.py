"""Shared resource service primitives."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._routing import EndpointRouter, RequestTarget
from ._transport import Transport


@dataclass(frozen=True)
class ClientContext:
    """Organization and credential context shared by resource services."""

    org_id: str
    org_name: str


class ResourceAPI:
    """Base class for synchronous resource services."""

    def __init__(
        self,
        transport: Transport,
        router: EndpointRouter,
        context: ClientContext,
        api_key: str,
    ):
        self._transport = transport
        self._router = router
        self._context = context
        self._api_key = api_key

    def _request_json(
        self,
        target: RequestTarget,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        request_headers = {"Authorization": f"Bearer {self._api_key}"}
        if headers:
            request_headers.update(headers)
        return self._transport.request_json(
            method,
            self._router.resolve(target, path),
            headers=request_headers,
            **kwargs,
        )
