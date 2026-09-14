"""Static and runtime checks for the public API client facade."""

from typing import TYPE_CHECKING

from braintrust.api import BraintrustClient, BraintrustOpenApiClient, EndpointRouter, RequestTarget
from braintrust.api.types import (
    CreateDataset,
    CreateExperiment,
    CreateFunction,
    CreateProject,
    CreatePrompt,
    Dataset,
    Experiment,
    FetchDatasetEventsResponse,
    FetchEventsRequest,
    FetchExperimentEventsResponse,
    Function,
    GetDatasetResponse,
    GetExperimentResponse,
    GetFunctionResponse,
    GetProjectResponse,
    GetPromptResponse,
    InsertDatasetEventRequest,
    PatchDataset,
    PatchExperiment,
    PatchFunction,
    PatchProject,
    PatchPrompt,
    Project,
    Prompt,
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

    openapi_client.acls.get_acl_id("acl-id")
    openapi_client.agents.get_agent()
    openapi_client.ai_secrets.get_ai_secret()
    openapi_client.api_keys.get_api_key()
    openapi_client.dataset_snapshots.get_dataset_snapshot()
    openapi_client.env_vars.get_env_var()
    openapi_client.environments.list_environments()
    openapi_client.groups.get_group()
    openapi_client.mcp_servers.get_mcp_server()
    openapi_client.org_automations.get_org_automation()
    openapi_client.organizations.get_organization()
    openapi_client.project_automations.get_project_automation()
    openapi_client.project_groups.get_project_group()
    openapi_client.project_scores.get_project_score(score_type=["slider", "categorical"])
    openapi_client.project_tags.get_project_tag()
    openapi_client.roles.get_role()
    openapi_client.service_tokens.get_service_token()
    openapi_client.span_iframes.get_span_iframe()
    openapi_client.users.get_user()
    openapi_client.views.get_view_id("view-id", "project", "project-id")

    create_project: CreateProject = {"name": "typed-project"}
    patch_project: PatchProject = {"description": "updated"}
    project: Project = openapi_client.projects.post_project(body=create_project)
    projects: GetProjectResponse = openapi_client.projects.get_project(
        ids=[project["id"]], project_name=project["name"]
    )
    fetched_project: Project = openapi_client.projects.get_project_id(project["id"])
    updated_project: Project = openapi_client.projects.patch_project_id(project["id"], body=patch_project)
    deleted_project: Project = openapi_client.projects.delete_project_id(project["id"])

    create_prompt: CreatePrompt = {
        "project_id": project["id"],
        "name": "Typed prompt",
        "slug": "typed-prompt",
        "prompt_data": {
            "prompt": {
                "type": "chat",
                "messages": [{"role": "user", "content": "Hello {{name}}"}],
            },
            "options": {"model": "gpt-5-mini"},
        },
    }
    prompt: Prompt = openapi_client.prompts.post_prompt(body=create_prompt)
    replaced_prompt: Prompt = openapi_client.prompts.put_prompt(body=create_prompt)
    prompts: GetPromptResponse = openapi_client.prompts.get_prompt(
        project_id=project["id"], slug=prompt["slug"], limit=1
    )
    fetched_prompt: Prompt = openapi_client.prompts.get_prompt_id(prompt["id"], version=prompt["_xact_id"])
    patch_prompt: PatchPrompt = {"description": "updated"}
    updated_prompt: Prompt = openapi_client.prompts.patch_prompt_id(prompt["id"], body=patch_prompt)
    deleted_prompt: Prompt = openapi_client.prompts.delete_prompt_id(prompt["id"])

    create_function: CreateFunction = {
        "project_id": project["id"],
        "name": "Typed parameters",
        "slug": "typed-parameters",
        "function_type": "parameters",
        "function_data": {
            "type": "parameters",
            "data": {"prefix": "hello"},
            "__schema": {"type": "object", "properties": {}},
        },
    }
    function: Function = openapi_client.functions.post_function(body=create_function)
    replaced_function: Function = openapi_client.functions.put_function(body=create_function)
    functions: GetFunctionResponse = openapi_client.functions.get_function(
        project_id=project["id"], slug=function["slug"], limit=1
    )
    fetched_function: Function = openapi_client.functions.get_function_id(function["id"], version=function["_xact_id"])
    patch_function: PatchFunction = {"description": "updated"}
    updated_function: Function = openapi_client.functions.patch_function_id(function["id"], body=patch_function)
    deleted_function: Function = openapi_client.functions.delete_function_id(function["id"])

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
