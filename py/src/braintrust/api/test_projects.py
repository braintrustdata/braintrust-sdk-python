import pytest
from braintrust import init_logger
from braintrust.api import BraintrustClient
from braintrust.logger import BraintrustState


@pytest.mark.vcr
def test_projects_end_to_end_with_real_backend(api_key):
    project_name = "python-sdk-generated-projects-vcr"
    create_project = {
        "name": project_name,
        "description": "created by the Python SDK VCR test",
    }

    with BraintrustClient(api_key=api_key) as client:
        discovery = client.auth.login()
        create_project["org_name"] = discovery.organization.name
        created = client.openapi.projects.post_project(body=create_project)
        listed = client.openapi.projects.get_project(
            project_name=project_name,
            org_name=discovery.organization.name,
        )
        fetched = client.openapi.projects.get_project_id(created["id"])
        updated = client.openapi.projects.patch_project_id(
            created["id"], body={"description": "updated by the Python SDK VCR test"}
        )

        state = BraintrustState()
        state.login(api_key=api_key, org_name=discovery.organization.name)
        registered_logger = init_logger(project=project_name, state=state, set_current=False)
        looked_up_logger = init_logger(project_id=created["id"], state=state, set_current=False)
        assert registered_logger.id == created["id"]
        assert looked_up_logger.project.name == project_name

        deleted = client.openapi.projects.delete_project_id(created["id"])

    assert create_project == {
        "name": project_name,
        "description": "created by the Python SDK VCR test",
        "org_name": discovery.organization.name,
    }
    assert created["name"] == project_name
    assert [project["id"] for project in listed["objects"]] == [created["id"]]
    assert fetched["id"] == created["id"]
    assert updated["description"] == "updated by the Python SDK VCR test"
    assert deleted["id"] == created["id"]
