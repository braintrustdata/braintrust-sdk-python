import contextlib

import braintrust
import pytest
from braintrust.api import BraintrustClient, BraintrustOpenApiClient
from braintrust.api._generated.functions import OPERATIONS, FunctionsAPI
from braintrust.api._test_server import scripted_server
from braintrust.api.policies import RetryMode


def _create_function_body(project_id: str, prefix: str) -> dict:
    return {
        "project_id": project_id,
        "name": "Generated parameters API",
        "slug": "generated-parameters-api",
        "function_type": "parameters",
        "function_data": {
            "type": "parameters",
            "data": {"prefix": prefix},
            "__schema": {
                "type": "object",
                "properties": {"prefix": {"type": "string"}},
            },
        },
        "tags": ["python-sdk-vcr"],
    }


def test_all_function_metadata_operations_have_complete_retry_classification():
    assert {name: operation.retry_mode for name, operation in OPERATIONS.items()} == {
        "postFunction": RetryMode.NONE,
        "putFunction": RetryMode.NONE,
        "getFunction": RetryMode.SAFE_READ,
        "getFunctionId": RetryMode.SAFE_READ,
        "patchFunctionId": RetryMode.NONE,
        "deleteFunctionId": RetryMode.NONE,
    }
    assert "postFunctionIdInvoke" not in OPERATIONS
    assert not hasattr(FunctionsAPI, "post_function_id_invoke")


def test_get_function_preserves_exact_wire_query_and_additive_response_fields():
    response = b'{"objects":[{"id":"function-id","future_field":{"preserved":true}}]}'
    with scripted_server([(200, {"Content-Type": "application/json"}, response)]) as (api_url, handler):
        with BraintrustOpenApiClient(api_key="test-key", api_url=api_url) as client:
            functions = client.functions.get_function(
                limit=1,
                project_name="test project",
                slug="function/slug",
                version="123",
            )

    assert handler.requests[0][:2] == (
        "GET",
        "/v1/function?limit=1&project_name=test%20project&slug=function%2Fslug&version=123",
    )
    assert functions["objects"][0]["future_field"] == {"preserved": True}


@pytest.mark.vcr
def test_functions_end_to_end_with_real_backend(api_key):
    project_name = "python-sdk-generated-functions-vcr"
    cleanup_project_id = None
    cleanup_function_id = None

    with BraintrustClient(api_key=api_key) as client:
        try:
            discovery = client.auth.login()
            project = client.openapi.projects.post_project(
                body={"name": project_name, "org_name": discovery.organization.name}
            )
            cleanup_project_id = project["id"]
            created = client.openapi.functions.post_function(
                body=_create_function_body(project["id"], prefix="created")
            )
            cleanup_function_id = created["id"]
            listed = client.openapi.functions.get_function(
                project_id=project["id"],
                slug=created["slug"],
                limit=1,
            )
            fetched = client.openapi.functions.get_function_id(created["id"], version=created["_xact_id"])
            updated = client.openapi.functions.patch_function_id(
                created["id"], body={"description": "updated by the Python SDK VCR test"}
            )
            replaced = client.openapi.functions.put_function(
                body=_create_function_body(project["id"], prefix="replaced")
            )
            deleted = client.openapi.functions.delete_function_id(created["id"])
            cleanup_function_id = None
            client.openapi.projects.delete_project_id(project["id"])
            cleanup_project_id = None
        finally:
            if cleanup_function_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.functions.delete_function_id(cleanup_function_id)
            if cleanup_project_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.projects.delete_project_id(cleanup_project_id)

    assert created["function_data"]["data"] == {"prefix": "created"}
    assert [function["id"] for function in listed["objects"]] == [created["id"]]
    assert fetched["id"] == created["id"]
    assert updated["description"] == "updated by the Python SDK VCR test"
    assert replaced["id"] == created["id"]
    assert replaced["function_data"]["data"] == {"prefix": "replaced"}
    assert deleted["id"] == created["id"]


@pytest.mark.vcr
def test_high_level_load_parameters_uses_generated_resources(api_key):
    project_name = "python-sdk-high-level-parameters-vcr"
    cleanup_project_id = None
    cleanup_function_id = None

    with BraintrustClient(api_key=api_key) as client:
        try:
            discovery = client.auth.login()
            project = client.openapi.projects.post_project(
                body={"name": project_name, "org_name": discovery.organization.name}
            )
            cleanup_project_id = project["id"]
            created = client.openapi.functions.post_function(
                body=_create_function_body(project["id"], prefix="loaded")
            )
            cleanup_function_id = created["id"]

            by_slug = braintrust.load_parameters(
                project_id=project["id"],
                slug=created["slug"],
                version=created["_xact_id"],
                api_key=api_key,
                org_name=discovery.organization.name,
            )
            by_id = braintrust.load_parameters(
                id=created["id"],
                api_key=api_key,
                org_name=discovery.organization.name,
            )

            assert by_slug.data == {"prefix": "loaded"}
            assert by_id.id == created["id"]
        finally:
            if cleanup_function_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.functions.delete_function_id(cleanup_function_id)
            if cleanup_project_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.projects.delete_project_id(cleanup_project_id)
