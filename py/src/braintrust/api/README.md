# Braintrust synchronous REST API client

`braintrust.api` provides a synchronous, typed, resource-oriented client for the reviewed Braintrust REST API. It uses the SDK's existing `requests` transport, authentication, organization routing, retry policies, and typed errors. Client construction does not make a network request.

> [!NOTE]
> Most applications should continue to use the higher-level APIs exported from `braintrust`. Use this REST client when you need direct access to a reviewed REST resource.

## Create a client

### Share authentication and routing

Use `BraintrustClient` when organization discovery and REST resources should share one transport. `auth.login()` configures the API route selected for the organization.

```python
from braintrust.api import BraintrustClient
from braintrust.api.types import CreateProject, Project

with BraintrustClient(api_key="...") as client:
    client.auth.login(org_name="my-organization")

    request: CreateProject = {"name": "my-project"}
    project: Project = client.openapi.projects.post_project(body=request)
    print(project["id"])
```

The API key can be omitted when `BRAINTRUST_API_KEY` is set. `BraintrustClient` also accepts `app_url`, `api_url`, and `proxy_url` for custom deployments.

### Connect directly to an API URL

Use `BraintrustOpenApiClient` when organization discovery is unnecessary and the API URL is already known.

```python
from braintrust.api import BraintrustOpenApiClient

with BraintrustOpenApiClient(
    api_key="...",
    api_url="https://api.example.com",
) as client:
    response = client.projects.get_project(project_name="my-project")
```

A direct client requires `api_url` or `BRAINTRUST_API_URL`. Both clients accept an existing `requests.Session`, an `HTTPAdapter`, or a Braintrust `Transport`; a supplied transport is not owned or closed by the client.

## Calling resources

Resources are loaded and cached on first access. Generated method names are the snake-case form of the OpenAPI `operationId`. Arguments mirror path and query parameters. JSON request bodies are passed through `body=` and results are mappings matching the exported `TypedDict` return type.

The client does not inject organization names or other request defaults. Pass `org_name`, project identifiers, pagination parameters, and other filters explicitly when the operation exposes them. List responses are returned as one response page; the client does not create an implicit iterator.

Responses may contain additive server fields that are not yet in the pinned types. Those keys are preserved at runtime.

## Errors and retries

REST failures use the error classes exported from `braintrust.api`, including `BraintrustHTTPError`, `BraintrustJSONDecodeError`, `BraintrustTransportError`, and the retry-exhaustion variants. HTTP errors preserve response status, opaque response bodies, and Braintrust request IDs when available.

GET and HEAD operations use the safe-read retry policy. A small reviewed allowlist of logical POST reads and verified idempotent writes is also retried. Other writes are not retried automatically. Set `enable_sdk_retries=False` on either client to disable SDK retries.

## Stability

The handwritten clients, routing types, retry types, and error classes listed in the public exports section are supported public API.

The generated resource methods and `braintrust.api.types` are a **preview API**. Every published tag and operation is reviewed before it is added, and pinned-spec updates are reviewed rather than auto-merged. While the API is in preview, method signatures and generated type shapes may change between SDK minor releases as the REST contract evolves. Presence in the upstream OpenAPI document alone does not make an operation public.

The `braintrust.api._generated` package is private implementation detail. Do not import resource classes, operation metadata, or models from it directly.

## REST types versus SDK payload types

Import request and response types for this client from `braintrust.api.types`:

```python
from braintrust.api.types import CreateExperiment, Experiment
```

These are dependency-free `TypedDict`s and type aliases generated from the pinned REST specification. They describe direct REST wire payloads.

`braintrust.generated_types` is a separate public surface for high-level SDK logging and evaluation payloads. Some names intentionally overlap, but the shapes and compatibility contracts can differ. Do not substitute a type from one module for the same-named type in the other module.

## Resource reference

The table lists every reviewed generated method. Consult the method's Python signature in your editor for its path parameters, keyword-only query parameters, `body` type, and return type.

<!-- BEGIN GENERATED RESOURCE REFERENCE -->
| Client property | Methods |
| --- | --- |
| `client.projects` | `post_project` — `POST /v1/project`<br>`get_project` — `GET /v1/project`<br>`get_project_id` — `GET /v1/project/{project_id}`<br>`patch_project_id` — `PATCH /v1/project/{project_id}`<br>`delete_project_id` — `DELETE /v1/project/{project_id}` |
| `client.experiments` | `post_experiment` — `POST /v1/experiment`<br>`get_experiment` — `GET /v1/experiment`<br>`get_experiment_id` — `GET /v1/experiment/{experiment_id}`<br>`patch_experiment_id` — `PATCH /v1/experiment/{experiment_id}`<br>`delete_experiment_id` — `DELETE /v1/experiment/{experiment_id}`<br>`post_experiment_id_insert` — `POST /v1/experiment/{experiment_id}/insert`<br>`post_experiment_id_fetch` — `POST /v1/experiment/{experiment_id}/fetch`<br>`get_experiment_id_fetch` — `GET /v1/experiment/{experiment_id}/fetch`<br>`post_experiment_id_feedback` — `POST /v1/experiment/{experiment_id}/feedback`<br>`get_experiment_id_summarize` — `GET /v1/experiment/{experiment_id}/summarize` |
| `client.datasets` | `post_dataset` — `POST /v1/dataset`<br>`get_dataset` — `GET /v1/dataset`<br>`get_dataset_id` — `GET /v1/dataset/{dataset_id}`<br>`patch_dataset_id` — `PATCH /v1/dataset/{dataset_id}`<br>`delete_dataset_id` — `DELETE /v1/dataset/{dataset_id}`<br>`post_dataset_id_insert` — `POST /v1/dataset/{dataset_id}/insert`<br>`post_dataset_id_fetch` — `POST /v1/dataset/{dataset_id}/fetch`<br>`get_dataset_id_fetch` — `GET /v1/dataset/{dataset_id}/fetch`<br>`post_dataset_id_feedback` — `POST /v1/dataset/{dataset_id}/feedback`<br>`get_dataset_id_summarize` — `GET /v1/dataset/{dataset_id}/summarize` |
| `client.prompts` | `post_prompt` — `POST /v1/prompt`<br>`put_prompt` — `PUT /v1/prompt`<br>`get_prompt` — `GET /v1/prompt`<br>`get_prompt_id` — `GET /v1/prompt/{prompt_id}`<br>`patch_prompt_id` — `PATCH /v1/prompt/{prompt_id}`<br>`delete_prompt_id` — `DELETE /v1/prompt/{prompt_id}` |
| `client.functions` | `post_function` — `POST /v1/function`<br>`put_function` — `PUT /v1/function`<br>`get_function` — `GET /v1/function`<br>`get_function_id` — `GET /v1/function/{function_id}`<br>`patch_function_id` — `PATCH /v1/function/{function_id}`<br>`delete_function_id` — `DELETE /v1/function/{function_id}` |
| `client.acls` | `post_acl` — `POST /v1/acl`<br>`delete_acl` — `DELETE /v1/acl`<br>`get_acl` — `GET /v1/acl`<br>`get_acl_id` — `GET /v1/acl/{acl_id}`<br>`delete_acl_id` — `DELETE /v1/acl/{acl_id}`<br>`acl_batch_update` — `POST /v1/acl/batch_update`<br>`acl_list_org` — `GET /v1/acl/list_org` |
| `client.agents` | `post_agent` — `POST /v1/agent`<br>`put_agent` — `PUT /v1/agent`<br>`get_agent` — `GET /v1/agent`<br>`get_agent_id` — `GET /v1/agent/{agent_id}`<br>`patch_agent_id` — `PATCH /v1/agent/{agent_id}`<br>`delete_agent_id` — `DELETE /v1/agent/{agent_id}` |
| `client.ai_secrets` | `post_ai_secret` — `POST /v1/ai_secret`<br>`put_ai_secret` — `PUT /v1/ai_secret`<br>`delete_ai_secret` — `DELETE /v1/ai_secret`<br>`get_ai_secret` — `GET /v1/ai_secret`<br>`get_ai_secret_id` — `GET /v1/ai_secret/{ai_secret_id}`<br>`patch_ai_secret_id` — `PATCH /v1/ai_secret/{ai_secret_id}`<br>`delete_ai_secret_id` — `DELETE /v1/ai_secret/{ai_secret_id}` |
| `client.api_keys` | `get_api_key` — `GET /v1/api_key`<br>`get_api_key_id` — `GET /v1/api_key/{api_key_id}`<br>`delete_api_key_id` — `DELETE /v1/api_key/{api_key_id}` |
| `client.dataset_snapshots` | `post_dataset_snapshot` — `POST /v1/dataset_snapshot`<br>`put_dataset_snapshot` — `PUT /v1/dataset_snapshot`<br>`get_dataset_snapshot` — `GET /v1/dataset_snapshot`<br>`get_dataset_snapshot_id` — `GET /v1/dataset_snapshot/{dataset_snapshot_id}`<br>`patch_dataset_snapshot_id` — `PATCH /v1/dataset_snapshot/{dataset_snapshot_id}`<br>`delete_dataset_snapshot_id` — `DELETE /v1/dataset_snapshot/{dataset_snapshot_id}` |
| `client.env_vars` | `post_env_var` — `POST /v1/env_var`<br>`put_env_var` — `PUT /v1/env_var`<br>`get_env_var` — `GET /v1/env_var`<br>`get_env_var_id` — `GET /v1/env_var/{env_var_id}`<br>`patch_env_var_id` — `PATCH /v1/env_var/{env_var_id}`<br>`delete_env_var_id` — `DELETE /v1/env_var/{env_var_id}` |
| `client.environments` | `list_environments` — `GET /environment`<br>`create_environment` — `POST /environment`<br>`get_environment` — `GET /environment/{environment_id}`<br>`update_environment` — `PATCH /environment/{environment_id}`<br>`delete_environment` — `DELETE /environment/{environment_id}` |
| `client.groups` | `post_group` — `POST /v1/group`<br>`put_group` — `PUT /v1/group`<br>`get_group` — `GET /v1/group`<br>`get_group_id` — `GET /v1/group/{group_id}`<br>`patch_group_id` — `PATCH /v1/group/{group_id}`<br>`delete_group_id` — `DELETE /v1/group/{group_id}` |
| `client.mcp_servers` | `post_mcp_server` — `POST /v1/mcp_server`<br>`put_mcp_server` — `PUT /v1/mcp_server`<br>`get_mcp_server` — `GET /v1/mcp_server`<br>`get_mcp_server_id` — `GET /v1/mcp_server/{mcp_server_id}`<br>`patch_mcp_server_id` — `PATCH /v1/mcp_server/{mcp_server_id}`<br>`delete_mcp_server_id` — `DELETE /v1/mcp_server/{mcp_server_id}` |
| `client.org_automations` | `post_org_automation` — `POST /v1/org_automation`<br>`put_org_automation` — `PUT /v1/org_automation`<br>`get_org_automation` — `GET /v1/org_automation`<br>`get_org_automation_id` — `GET /v1/org_automation/{org_automation_id}`<br>`patch_org_automation_id` — `PATCH /v1/org_automation/{org_automation_id}`<br>`delete_org_automation_id` — `DELETE /v1/org_automation/{org_automation_id}` |
| `client.organizations` | `get_organization` — `GET /v1/organization`<br>`get_organization_id` — `GET /v1/organization/{organization_id}`<br>`patch_organization_id` — `PATCH /v1/organization/{organization_id}`<br>`patch_organization_members` — `PATCH /v1/organization/members` |
| `client.project_automations` | `post_project_automation` — `POST /v1/project_automation`<br>`put_project_automation` — `PUT /v1/project_automation`<br>`get_project_automation` — `GET /v1/project_automation`<br>`get_project_automation_id` — `GET /v1/project_automation/{project_automation_id}`<br>`patch_project_automation_id` — `PATCH /v1/project_automation/{project_automation_id}`<br>`delete_project_automation_id` — `DELETE /v1/project_automation/{project_automation_id}` |
| `client.project_groups` | `post_project_group` — `POST /v1/project_group`<br>`put_project_group` — `PUT /v1/project_group`<br>`get_project_group` — `GET /v1/project_group`<br>`get_project_group_id` — `GET /v1/project_group/{project_group_id}`<br>`patch_project_group_id` — `PATCH /v1/project_group/{project_group_id}`<br>`delete_project_group_id` — `DELETE /v1/project_group/{project_group_id}` |
| `client.project_scores` | `post_project_score` — `POST /v1/project_score`<br>`put_project_score` — `PUT /v1/project_score`<br>`get_project_score` — `GET /v1/project_score`<br>`get_project_score_id` — `GET /v1/project_score/{project_score_id}`<br>`patch_project_score_id` — `PATCH /v1/project_score/{project_score_id}`<br>`delete_project_score_id` — `DELETE /v1/project_score/{project_score_id}` |
| `client.project_tags` | `post_project_tag` — `POST /v1/project_tag`<br>`put_project_tag` — `PUT /v1/project_tag`<br>`get_project_tag` — `GET /v1/project_tag`<br>`get_project_tag_id` — `GET /v1/project_tag/{project_tag_id}`<br>`patch_project_tag_id` — `PATCH /v1/project_tag/{project_tag_id}`<br>`delete_project_tag_id` — `DELETE /v1/project_tag/{project_tag_id}` |
| `client.roles` | `post_role` — `POST /v1/role`<br>`put_role` — `PUT /v1/role`<br>`get_role` — `GET /v1/role`<br>`get_role_id` — `GET /v1/role/{role_id}`<br>`patch_role_id` — `PATCH /v1/role/{role_id}`<br>`delete_role_id` — `DELETE /v1/role/{role_id}` |
| `client.service_tokens` | `post_service_token` — `POST /v1/service_token`<br>`put_service_token` — `PUT /v1/service_token`<br>`delete_service_token` — `DELETE /v1/service_token`<br>`get_service_token` — `GET /v1/service_token`<br>`get_service_token_id` — `GET /v1/service_token/{service_token_id}`<br>`delete_service_token_id` — `DELETE /v1/service_token/{service_token_id}` |
| `client.span_iframes` | `post_span_iframe` — `POST /v1/span_iframe`<br>`put_span_iframe` — `PUT /v1/span_iframe`<br>`get_span_iframe` — `GET /v1/span_iframe`<br>`get_span_iframe_id` — `GET /v1/span_iframe/{span_iframe_id}`<br>`patch_span_iframe_id` — `PATCH /v1/span_iframe/{span_iframe_id}`<br>`delete_span_iframe_id` — `DELETE /v1/span_iframe/{span_iframe_id}` |
| `client.users` | `get_user` — `GET /v1/user`<br>`get_user_id` — `GET /v1/user/{user_id}` |
| `client.views` | `post_view` — `POST /v1/view`<br>`put_view` — `PUT /v1/view`<br>`get_view` — `GET /v1/view`<br>`get_view_id` — `GET /v1/view/{view_id}`<br>`patch_view_id` — `PATCH /v1/view/{view_id}`<br>`delete_view_id` — `DELETE /v1/view/{view_id}` |
<!-- END GENERATED RESOURCE REFERENCE -->

Streaming, log ingestion, eval launch, cross-object insertion, browser CORS, diagnostics, provider proxying, and function invocation remain on specialized SDK paths and are intentionally absent from this client.

## Public `braintrust.api` exports

<!-- BEGIN GENERATED API EXPORTS -->
| Name | Name | Name |
| --- | --- | --- |
| `BraintrustAPIError` | `BraintrustClient` | `BraintrustOpenApiClient` |
| `BraintrustHTTPError` | `BraintrustJSONDecodeError` | `BraintrustRetryExhaustedError` |
| `BraintrustTransportError` | `BraintrustTransportRetryExhaustedError` | `EndpointRouter` |
| `LoginResult` | `OrganizationInfo` | `RequestTarget` |
| `RetryMode` | `RetryPolicy` |  |
<!-- END GENERATED API EXPORTS -->

## Public `braintrust.api.types` exports

Only request and response models reachable directly from reviewed resource method signatures are exported.

<!-- BEGIN GENERATED REST TYPES -->
| Name | Name | Name |
| --- | --- | --- |
| `AISecret` | `Acl` | `AclBatchUpdateRequest` |
| `AclBatchUpdateResponse` | `AclItem` | `AclListOrgResponse` |
| `Agent` | `ApiKey` | `CreateAISecret` |
| `CreateAgent` | `CreateDataset` | `CreateDatasetSnapshot` |
| `CreateEnvironment` | `CreateExperiment` | `CreateFunction` |
| `CreateGroup` | `CreateMCPServer` | `CreateOrgAutomation` |
| `CreateProject` | `CreateProjectAutomation` | `CreateProjectGroup` |
| `CreateProjectScore` | `CreateProjectTag` | `CreatePrompt` |
| `CreateRole` | `CreateServiceTokenOutput` | `CreateSpanIFrame` |
| `CreateView` | `Dataset` | `DatasetSnapshot` |
| `DeleteAISecret` | `DeleteServiceToken` | `DeleteView` |
| `EnvVar` | `Environment` | `Experiment` |
| `FeedbackDatasetEventRequest` | `FeedbackExperimentEventRequest` | `FeedbackResponseSchema` |
| `FetchDatasetEventsResponse` | `FetchEventsRequest` | `FetchExperimentEventsResponse` |
| `Function` | `GetAclResponse` | `GetAgentResponse` |
| `GetAiSecretResponse` | `GetApiKeyResponse` | `GetDatasetResponse` |
| `GetDatasetSnapshotResponse` | `GetEnvVarResponse` | `GetExperimentResponse` |
| `GetFunctionResponse` | `GetGroupResponse` | `GetMcpServerResponse` |
| `GetOrgAutomationResponse` | `GetOrganizationResponse` | `GetProjectAutomationResponse` |
| `GetProjectGroupResponse` | `GetProjectResponse` | `GetProjectScoreResponse` |
| `GetProjectTagResponse` | `GetPromptResponse` | `GetRoleResponse` |
| `GetServiceTokenResponse` | `GetSpanIframeResponse` | `GetUserResponse` |
| `GetViewResponse` | `Group` | `InsertDatasetEventRequest` |
| `InsertEventsResponse` | `InsertExperimentEventRequest` | `ListEnvironmentsResponse` |
| `MCPServer` | `OrgAutomation` | `Organization` |
| `PatchAISecret` | `PatchAgent` | `PatchDataset` |
| `PatchDatasetSnapshot` | `PatchEnvironment` | `PatchExperiment` |
| `PatchFunction` | `PatchGroup` | `PatchMCPServer` |
| `PatchOrgAutomation` | `PatchOrganization` | `PatchOrganizationMembers` |
| `PatchOrganizationMembersOutput` | `PatchProject` | `PatchProjectAutomation` |
| `PatchProjectGroup` | `PatchProjectScore` | `PatchProjectTag` |
| `PatchPrompt` | `PatchRole` | `PatchSpanIFrame` |
| `PatchView` | `Project` | `ProjectAutomation` |
| `ProjectGroup` | `ProjectScore` | `ProjectTag` |
| `Prompt` | `Role` | `ServiceToken` |
| `SpanIFrame` | `SummarizeDatasetResponse` | `SummarizeExperimentResponse` |
| `User` | `View` |  |
<!-- END GENERATED REST TYPES -->

The reference sections above are synchronized with the Python surface by `make generate-api-client`; do not edit those sections manually.
