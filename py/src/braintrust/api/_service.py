"""Shared resource service primitives."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from ._routing import EndpointRouter, RequestTarget
from ._transport import Transport
from .errors import BraintrustHTTPError
from .policies import RetryMode


@dataclass(frozen=True)
class Parameter:
    """Serialization metadata for one generated operation parameter."""

    argument_name: str
    name: str
    location: str
    required: bool


@dataclass(frozen=True)
class Operation:
    """Resolved wire and runtime-policy metadata for a generated operation."""

    operation_id: str
    method: str
    path: str
    parameters: tuple[Parameter, ...]
    has_request_body: bool
    success_statuses: tuple[int, ...]
    json_success_statuses: tuple[int, ...]
    retry_mode: RetryMode


class ResourceAPI:
    """Base class for synchronous resource services and generated operations."""

    def __init__(
        self,
        transport: Transport,
        router: EndpointRouter,
        api_key: str,
    ):
        self._transport = transport
        self._router = router
        self._api_key = api_key

    def _request(
        self,
        target: RequestTarget,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        request_headers = {"Authorization": f"Bearer {self._api_key}"}
        if headers:
            request_headers.update(headers)
        return self._transport.request(
            method,
            self._router.resolve(target, path),
            headers=request_headers,
            **kwargs,
        )

    def _request_json(
        self,
        target: RequestTarget,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        response = self._request(target, method, path, headers=headers, **kwargs)
        return self._transport.decode_json_response(response, method=method, url=response.url)

    def execute(
        self,
        operation: Operation,
        *,
        path_parameters: Mapping[str, Any] | None = None,
        query_parameters: Mapping[str, Any] | None = None,
        body: Any = None,
    ) -> Any:
        """Execute a generated operation through this resource's transport."""

        path_values = path_parameters or {}
        query_values = query_parameters or {}
        path = operation.path
        query_parts: list[str] = []

        for parameter in operation.parameters:
            values = {"path": path_values, "query": query_values}.get(parameter.location)
            if values is None:
                raise ValueError(f"Unsupported generated parameter location: {parameter.location!r}")
            value = values.get(parameter.argument_name)
            if value is None:
                if parameter.required:
                    raise TypeError(f"Missing required parameter: {parameter.argument_name}")
                continue
            if parameter.location == "path":
                encoded = _encode_path_parameter(value)
                path = path.replace("{" + parameter.name + "}", encoded)
            else:
                query_parts.extend(_encode_query_parameter(value, parameter))

        if query_parts:
            path += ("&" if "?" in path else "?") + "&".join(query_parts)

        request_kwargs: dict[str, Any] = {"retry_mode": operation.retry_mode}
        if operation.has_request_body and body is not None:
            request_kwargs["json"] = body

        response = self._request(RequestTarget.API, operation.method, path, **request_kwargs)
        if response.status_code not in operation.success_statuses:
            raise BraintrustHTTPError(
                method=operation.method,
                url=response.url,
                status_code=response.status_code,
                response_body=response.text,
                response_headers=response.headers,
                attempts=getattr(response, "_braintrust_attempts", 1),
                retryable=False,
            )
        if response.status_code in operation.json_success_statuses:
            return self._transport.decode_json_response(response, method=operation.method, url=response.url)
        return None


def _encode_path_parameter(value: Any) -> str:
    if _is_array(value):
        raise TypeError("Path parameters must be scalar")
    return quote(_scalar_string(value), safe="")


def _encode_query_parameter(value: Any, parameter: Parameter) -> list[str]:
    encoded_name = quote(parameter.name, safe="")
    if _is_array(value):
        encoded_values = [quote(_scalar_string(item), safe="") for item in value]
        return [f"{encoded_name}={item}" for item in encoded_values]
    encoded_value = quote(_scalar_string(value), safe="")
    return [f"{encoded_name}={encoded_value}"]


def _is_array(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _scalar_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
