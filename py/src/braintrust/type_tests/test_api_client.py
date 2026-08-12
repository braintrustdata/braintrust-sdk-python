"""Static and runtime checks for the public API client facade."""

from typing import TYPE_CHECKING

from braintrust.api import (
    BaseExperiment,
    BraintrustClient,
    EndpointRouter,
    ExperimentComparison,
    ExperimentRecord,
    RequestTarget,
)


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
    experiment: ExperimentRecord = client.experiments.get("experiment-id")
    base_experiment: BaseExperiment | None = client.experiments.get_base("experiment-id")
    comparison: ExperimentComparison = client.experiments.compare(
        "experiment-id", base_experiment_id="base-experiment-id"
    )


def test_api_client_public_types() -> None:
    router = EndpointRouter(app_url="https://app.example.com", api_url="https://api.example.com")

    assert router.resolve(RequestTarget.API, "ping") == "https://api.example.com/ping"
    assert BraintrustClient.__name__ == "BraintrustClient"
    assert ExperimentRecord.__name__ == "ExperimentRecord"
    assert BaseExperiment.__name__ == "BaseExperiment"
    assert ExperimentComparison.__name__ == "ExperimentComparison"
