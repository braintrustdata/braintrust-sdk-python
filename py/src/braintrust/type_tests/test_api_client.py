"""Static and runtime checks for the public API client facade."""

from typing import TYPE_CHECKING

from braintrust.api import BraintrustClient, EndpointRouter, RequestTarget


if TYPE_CHECKING:
    client = BraintrustClient(api_key="key", org_name="org")
    client_with_overrides = BraintrustClient(
        api_key="key",
        app_url="https://app.example.com",
        api_url="https://api.example.com",
        proxy_url="https://proxy.example.com",
    )
    org_id: str = client.org_id
    org_name: str = client.org_name
    api_url: str | None = client_with_overrides.router.api_url


def test_api_client_public_types() -> None:
    router = EndpointRouter(app_url="https://app.example.com", api_url="https://api.example.com")

    assert router.resolve(RequestTarget.API, "ping") == "https://api.example.com/ping"
    assert BraintrustClient.__name__ == "BraintrustClient"
