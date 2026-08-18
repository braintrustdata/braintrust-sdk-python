import contextlib
import json
import os
from urllib.parse import urlsplit

import pytest
from braintrust.api import BraintrustClient, BraintrustHTTPError, BraintrustOpenApiClient
from braintrust.api._test_server import scripted_server
from braintrust.api._transport import Transport


@contextlib.contextmanager
def project_server():
    project = {
        "id": "project-id",
        "org_id": "test-org-id",
        "name": "project/name",
        "new_backend_field": {"preserved": True},
    }

    def respond(method, request_path, _body, _headers):
        path = urlsplit(request_path).path
        response = (
            {"objects": [project], "new_list_field": ["preserved"]}
            if path == "/v1/project" and method == "GET"
            else project
        )
        return 200, {"Content-Type": "application/json"}, json.dumps(response).encode()

    with scripted_server(respond) as server:
        yield server


def _client_for_server(url, transport):
    return BraintrustOpenApiClient(api_key="test-key", api_url=url, transport=transport)


def test_generated_operation_rejects_undeclared_success_statuses_as_http_errors():
    response_body = b'{"queued": true}'
    headers = {"Content-Type": "application/json", "x-request-id": "unexpected-status-id"}
    with scripted_server([(202, headers, response_body)]) as (url, _):
        projects = _client_for_server(url, Transport()).projects
        with pytest.raises(BraintrustHTTPError) as exc_info:
            projects.get_project()

    assert exc_info.value.status_code == 202
    assert exc_info.value.response_body == response_body.decode()
    assert exc_info.value.request_id == "unexpected-status-id"
    assert exc_info.value.request_id_header == "x-request-id"


def test_projects_use_generated_bindings_with_exact_wire_shape_and_additive_responses():
    with project_server() as (url, handler):
        projects = _client_for_server(url, Transport()).projects
        create_project = {"name": "project/name", "description": "created"}
        created = projects.post_project(body=create_project)
        listed = projects.get_project(
            limit=2,
            ids=["project-id", "other/id"],
            project_name="project/name",
        )
        projects.get_project_id("project/id")
        projects.patch_project_id("project/id", body={"description": "updated"})
        projects.delete_project_id("project/id")

    assert create_project == {"name": "project/name", "description": "created"}
    assert created["new_backend_field"] == {"preserved": True}
    assert listed["new_list_field"] == ["preserved"]

    assert handler.requests == [
        (
            "POST",
            "/v1/project",
            b'{"name": "project/name", "description": "created"}',
            "Bearer test-key",
        ),
        (
            "GET",
            "/v1/project?limit=2&ids=project-id&ids=other%2Fid&project_name=project%2Fname",
            b"",
            "Bearer test-key",
        ),
        ("GET", "/v1/project/project%2Fid", b"", "Bearer test-key"),
        ("PATCH", "/v1/project/project%2Fid", b'{"description": "updated"}', "Bearer test-key"),
        ("DELETE", "/v1/project/project%2Fid", b"", "Bearer test-key"),
    ]


def test_projects_allow_explicit_organization_overrides():
    with project_server() as (url, handler):
        projects = _client_for_server(url, Transport()).projects
        projects.post_project(body={"name": "project/name", "org_name": "other org"})
        projects.get_project(org_name="other org")

    assert handler.requests == [
        (
            "POST",
            "/v1/project",
            b'{"name": "project/name", "org_name": "other org"}',
            "Bearer test-key",
        ),
        ("GET", "/v1/project?org_name=other%20org", b"", "Bearer test-key"),
    ]


def test_projects_percent_encode_query_values():
    with project_server() as (url, handler):
        projects = _client_for_server(url, Transport()).projects
        projects.get_project(project_name="R&D#research/v1+beta?x=y", org_name="org&name#one + two")

    assert handler.requests == [
        (
            "GET",
            "/v1/project?project_name=R%26D%23research%2Fv1%2Bbeta%3Fx%3Dy&org_name=org%26name%23one%20%2B%20two",
            b"",
            "Bearer test-key",
        )
    ]


def _api_key():
    return os.environ.get("BRAINTRUST_API_KEY", "sk-dummy-for-vcr-replay")


@pytest.mark.vcr
def test_projects_end_to_end_with_real_backend():
    project_name = "python-sdk-generated-projects-vcr"
    with BraintrustClient(api_key=_api_key()) as client:
        discovery = client.auth.login()
        created = client.openapi.projects.post_project(
            body={
                "name": project_name,
                "description": "created by the Python SDK VCR test",
                "org_name": discovery.organization.name,
            }
        )
        listed = client.openapi.projects.get_project(
            project_name=project_name,
            org_name=discovery.organization.name,
        )
        fetched = client.openapi.projects.get_project_id(created["id"])
        updated = client.openapi.projects.patch_project_id(
            created["id"], body={"description": "updated by the Python SDK VCR test"}
        )
        deleted = client.openapi.projects.delete_project_id(created["id"])

    assert created["name"] == project_name
    assert [project["id"] for project in listed["objects"]] == [created["id"]]
    assert fetched["id"] == created["id"]
    assert updated["description"] == "updated by the Python SDK VCR test"
    assert deleted["id"] == created["id"]
