import contextlib

import braintrust
import pytest
from braintrust.api import BraintrustClient, BraintrustOpenApiClient
from braintrust.api._generated.prompts import OPERATIONS
from braintrust.api._test_server import scripted_server
from braintrust.api.policies import RetryMode


def _create_prompt_body(project_id: str, description: str) -> dict:
    return {
        "project_id": project_id,
        "name": "Generated prompts API",
        "slug": "generated-prompts-api",
        "description": description,
        "prompt_data": {
            "prompt": {
                "type": "chat",
                "messages": [{"role": "user", "content": "Hello {{name}}"}],
            },
            "options": {"model": "gpt-5-mini"},
        },
        "tags": ["python-sdk-vcr"],
    }


def test_all_prompt_operations_have_complete_retry_classification():
    assert {name: operation.retry_mode for name, operation in OPERATIONS.items()} == {
        "postPrompt": RetryMode.NONE,
        "putPrompt": RetryMode.NONE,
        "getPrompt": RetryMode.SAFE_READ,
        "getPromptId": RetryMode.SAFE_READ,
        "patchPromptId": RetryMode.NONE,
        "deletePromptId": RetryMode.NONE,
    }


def test_get_prompt_preserves_exact_wire_query_and_additive_response_fields():
    response = b'{"objects":[{"id":"prompt-id","future_field":{"preserved":true}}]}'
    with scripted_server([(200, {"Content-Type": "application/json"}, response)]) as (api_url, handler):
        with BraintrustOpenApiClient(api_key="test-key", api_url=api_url) as client:
            prompts = client.prompts.get_prompt(
                limit=1,
                project_name="test project",
                slug="prompt/slug",
                version="123",
            )

    assert handler.requests[0][:2] == (
        "GET",
        "/v1/prompt?limit=1&project_name=test%20project&slug=prompt%2Fslug&version=123",
    )
    assert prompts["objects"][0]["future_field"] == {"preserved": True}


@pytest.mark.vcr
def test_prompts_end_to_end_with_real_backend(api_key):
    project_name = "python-sdk-generated-prompts-vcr"
    cleanup_project_id = None
    cleanup_prompt_id = None

    with BraintrustClient(api_key=api_key) as client:
        try:
            discovery = client.auth.login()
            project = client.openapi.projects.post_project(
                body={"name": project_name, "org_name": discovery.organization.name}
            )
            cleanup_project_id = project["id"]
            created = client.openapi.prompts.post_prompt(
                body=_create_prompt_body(
                    project["id"],
                    description="created by the Python SDK VCR test",
                )
            )
            cleanup_prompt_id = created["id"]
            listed = client.openapi.prompts.get_prompt(
                project_id=project["id"],
                slug=created["slug"],
                limit=1,
            )
            fetched = client.openapi.prompts.get_prompt_id(created["id"], version=created["_xact_id"])
            updated = client.openapi.prompts.patch_prompt_id(
                created["id"], body={"description": "updated by the Python SDK VCR test"}
            )
            replaced = client.openapi.prompts.put_prompt(
                body=_create_prompt_body(
                    project["id"],
                    description="replaced by the Python SDK VCR test",
                )
            )
            deleted = client.openapi.prompts.delete_prompt_id(created["id"])
            cleanup_prompt_id = None
            client.openapi.projects.delete_project_id(project["id"])
            cleanup_project_id = None
        finally:
            if cleanup_prompt_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.prompts.delete_prompt_id(cleanup_prompt_id)
            if cleanup_project_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.projects.delete_project_id(cleanup_project_id)

    assert created["slug"] == "generated-prompts-api"
    assert [prompt["id"] for prompt in listed["objects"]] == [created["id"]]
    assert fetched["id"] == created["id"]
    assert updated["description"] == "updated by the Python SDK VCR test"
    assert replaced["id"] == created["id"]
    assert replaced["description"] == "replaced by the Python SDK VCR test"
    assert deleted["id"] == created["id"]


@pytest.mark.vcr
def test_high_level_load_prompt_uses_generated_resources(api_key):
    project_name = "python-sdk-high-level-prompts-vcr"
    cleanup_project_id = None
    cleanup_prompt_id = None

    with BraintrustClient(api_key=api_key) as client:
        try:
            discovery = client.auth.login()
            project = client.openapi.projects.post_project(
                body={"name": project_name, "org_name": discovery.organization.name}
            )
            cleanup_project_id = project["id"]
            created = client.openapi.prompts.post_prompt(
                body=_create_prompt_body(project["id"], description="loaded through braintrust.load_prompt")
            )
            cleanup_prompt_id = created["id"]

            by_slug = braintrust.load_prompt(
                project_id=project["id"],
                slug=created["slug"],
                version=created["_xact_id"],
                api_key=api_key,
                org_name=discovery.organization.name,
            )
            by_id = braintrust.load_prompt(
                id=created["id"],
                api_key=api_key,
                org_name=discovery.organization.name,
            )

            assert by_slug.build(name="Ada")["messages"][0]["content"] == "Hello Ada"
            assert by_id.id == created["id"]
        finally:
            if cleanup_prompt_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.prompts.delete_prompt_id(cleanup_prompt_id)
            if cleanup_project_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.projects.delete_project_id(cleanup_project_id)
