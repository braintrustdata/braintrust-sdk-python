"""Static and runtime checks for the public API client facade."""

from typing import TYPE_CHECKING

from braintrust.api import BraintrustClient, BraintrustOpenApiClient, EndpointRouter, RequestTarget
from braintrust.api.types import (
    CreateDataset,
    CreateExperiment,
    CreateProject,
    Dataset,
    Experiment,
    FetchDatasetEventsResponse,
    FetchEventsRequest,
    FetchExperimentEventsResponse,
    GetDatasetResponse,
    GetExperimentResponse,
    GetProjectResponse,
    InsertDatasetEventRequest,
    PatchDataset,
    PatchExperiment,
    PatchProject,
    Project,
    SummarizeDatasetResponse,
    SummarizeExperimentResponse,
)


if TYPE_CHECKING:
    client = BraintrustClient(api_key="key", app_url="https://app.example.com")
    discovery = client.auth.login(org_name="org")
    openapi_client: BraintrustOpenApiClient = client.openapi
    org_id: str = discovery.organization.id
    org_name: str = discovery.organization.name
    api_url: str | None = client.router.api_url
    create_project: CreateProject = {"name": "typed-project"}
    patch_project: PatchProject = {"description": "updated"}
    project: Project = openapi_client.projects.post_project(body=create_project)
    projects: GetProjectResponse = openapi_client.projects.get_project(
        ids=[project["id"]], project_name=project["name"]
    )
    fetched_project: Project = openapi_client.projects.get_project_id(project["id"])
    updated_project: Project = openapi_client.projects.patch_project_id(project["id"], body=patch_project)
    deleted_project: Project = openapi_client.projects.delete_project_id(project["id"])

    create_dataset: CreateDataset = {"project_id": project["id"], "name": "typed-dataset"}
    dataset: Dataset = openapi_client.datasets.post_dataset(body=create_dataset)
    datasets: GetDatasetResponse = openapi_client.datasets.get_dataset(ids=[dataset["id"]], project_id=project["id"])
    fetched_dataset: Dataset = openapi_client.datasets.get_dataset_id(dataset["id"])
    patch_dataset: PatchDataset = {"description": "updated"}
    updated_dataset: Dataset = openapi_client.datasets.patch_dataset_id(dataset["id"], body=patch_dataset)
    insert_dataset_events: InsertDatasetEventRequest = {
        "events": [
            {
                "id": "row-id",
                "_is_merge": True,
                "_merge_paths": [["input"]],
                "_array_delete": [{"path": ["tags"], "delete": ["old"]}],
                "_object_delete": True,
                "_parent_id": "parent-id",
            }
        ]
    }
    openapi_client.datasets.post_dataset_id_insert(dataset["id"], body=insert_dataset_events)
    fetched_dataset_events: FetchDatasetEventsResponse = openapi_client.datasets.post_dataset_id_fetch(
        dataset["id"], body={"limit": 10}
    )
    fetched_dataset_xact_id: str | None = fetched_dataset_events["events"][0].get("_xact_id")
    dataset_summary: SummarizeDatasetResponse = openapi_client.datasets.get_dataset_id_summarize(dataset["id"])
    deleted_dataset: Dataset = openapi_client.datasets.delete_dataset_id(dataset["id"])

    create_experiment: CreateExperiment = {"project_id": project["id"], "name": "typed-experiment"}
    experiment: Experiment = openapi_client.experiments.post_experiment(body=create_experiment)
    experiments: GetExperimentResponse = openapi_client.experiments.get_experiment(
        ids=[experiment["id"]], project_id=project["id"]
    )
    fetched_experiment: Experiment = openapi_client.experiments.get_experiment_id(experiment["id"])
    patch_experiment: PatchExperiment = {"description": "updated"}
    updated_experiment: Experiment = openapi_client.experiments.patch_experiment_id(
        experiment["id"], body=patch_experiment
    )
    fetch_request: FetchEventsRequest = {"limit": 10}
    fetched_events: FetchExperimentEventsResponse = openapi_client.experiments.post_experiment_id_fetch(
        experiment["id"], body=fetch_request
    )
    summary: SummarizeExperimentResponse = openapi_client.experiments.get_experiment_id_summarize(
        experiment["id"], summarize_scores=True
    )
    deleted_experiment: Experiment = openapi_client.experiments.delete_experiment_id(experiment["id"])


def test_api_client_router() -> None:
    router = EndpointRouter(app_url="https://app.example.com", api_url="https://api.example.com")

    assert router.resolve(RequestTarget.API, "ping") == "https://api.example.com/ping"
