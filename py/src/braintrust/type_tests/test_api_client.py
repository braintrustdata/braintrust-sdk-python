"""Static and runtime checks for the public API client facade."""

from typing import TYPE_CHECKING

from braintrust.api import BraintrustClient, BraintrustOpenApiClient, EndpointRouter, RequestTarget
from braintrust.api.types import CreateProject, GetProjectResponse, PatchProject, Project


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


def test_api_client_router() -> None:
    router = EndpointRouter(app_url="https://app.example.com", api_url="https://api.example.com")

    assert router.resolve(RequestTarget.API, "ping") == "https://api.example.com/ping"
