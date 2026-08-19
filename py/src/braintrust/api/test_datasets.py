import contextlib
import json

import braintrust
import pytest
from braintrust.api import BraintrustClient, BraintrustOpenApiClient
from braintrust.api._generated.datasets import OPERATIONS
from braintrust.api._test_server import scripted_server
from braintrust.api.policies import RetryMode
from braintrust.api.types import InsertDatasetEventRequest


def test_all_dataset_operations_have_complete_retry_classification():
    assert {name: operation.retry_mode for name, operation in OPERATIONS.items()} == {
        "postDataset": RetryMode.IDEMPOTENT_WRITE,
        "getDataset": RetryMode.SAFE_READ,
        "getDatasetId": RetryMode.SAFE_READ,
        "patchDatasetId": RetryMode.NONE,
        "deleteDatasetId": RetryMode.NONE,
        "postDatasetIdInsert": RetryMode.NONE,
        "postDatasetIdFetch": RetryMode.SAFE_READ,
        "getDatasetIdFetch": RetryMode.SAFE_READ,
        "postDatasetIdFeedback": RetryMode.NONE,
        "getDatasetIdSummarize": RetryMode.SAFE_READ,
    }


def test_dataset_insert_preserves_underscore_prefixed_wire_keys():
    body: InsertDatasetEventRequest = {
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
    with scripted_server([(200, {"Content-Type": "application/json"}, b'{"row_ids":["row-id"]}')]) as (
        api_url,
        handler,
    ):
        with BraintrustOpenApiClient(api_key="test-key", api_url=api_url) as client:
            client.datasets.post_dataset_id_insert("dataset-id", body=body)

    assert handler.requests[0][:2] == ("POST", "/v1/dataset/dataset-id/insert")
    assert json.loads(handler.requests[0][2]) == body


@pytest.mark.vcr
def test_datasets_end_to_end_with_real_backend(api_key):
    project_name = "python-sdk-generated-datasets-vcr"
    dataset_name = "generated-datasets-api"
    event_id = "generated-datasets-row"

    with BraintrustClient(api_key=api_key) as client:
        cleanup_project_id = None
        cleanup_dataset_id = None
        try:
            discovery = client.auth.login()
            project = client.openapi.projects.post_project(
                body={"name": project_name, "org_name": discovery.organization.name}
            )
            cleanup_project_id = project["id"]
            created = client.openapi.datasets.post_dataset(
                body={
                    "project_id": project["id"],
                    "name": dataset_name,
                    "description": "created by the Python SDK VCR test",
                }
            )
            cleanup_dataset_id = created["id"]
            listed = client.openapi.datasets.get_dataset(
                dataset_name=dataset_name,
                project_id=project["id"],
            )
            fetched = client.openapi.datasets.get_dataset_id(created["id"])
            updated = client.openapi.datasets.patch_dataset_id(
                created["id"], body={"description": "updated by the Python SDK VCR test"}
            )
            inserted = client.openapi.datasets.post_dataset_id_insert(
                created["id"],
                body={
                    "events": [
                        {
                            "id": event_id,
                            "input": {
                                "question": "What is the answer?",
                                "context": {"preserved": True},
                                "replace_me": {"old": True},
                            },
                            "expected": "42",
                            "tags": ["keep", "remove"],
                        }
                    ]
                },
            )
            merged = client.openapi.datasets.post_dataset_id_insert(
                created["id"],
                body={
                    "events": [
                        {
                            "id": event_id,
                            "_is_merge": True,
                            "_merge_paths": [["input", "replace_me"]],
                            "_array_delete": [{"path": ["tags"], "delete": ["remove"]}],
                            "input": {
                                "question": "What is the updated answer?",
                                "replace_me": {"new": True},
                            },
                        }
                    ]
                },
            )
            fetched_post = client.openapi.datasets.post_dataset_id_fetch(created["id"], body={"limit": 10})
            fetched_get = client.openapi.datasets.get_dataset_id_fetch(created["id"], limit=10)
            feedback = client.openapi.datasets.post_dataset_id_feedback(
                created["id"], body={"feedback": [{"id": event_id, "comment": "useful example"}]}
            )
            summary = client.openapi.datasets.get_dataset_id_summarize(created["id"], summarize_data=True)
            removed = client.openapi.datasets.post_dataset_id_insert(
                created["id"], body={"events": [{"id": event_id, "_object_delete": True}]}
            )
            fetched_after_delete = client.openapi.datasets.post_dataset_id_fetch(created["id"], body={"limit": 10})
            deleted = client.openapi.datasets.delete_dataset_id(created["id"])
            cleanup_dataset_id = None
            client.openapi.projects.delete_project_id(project["id"])
            cleanup_project_id = None
        finally:
            if cleanup_dataset_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.datasets.delete_dataset_id(cleanup_dataset_id)
            if cleanup_project_id is not None:
                with contextlib.suppress(Exception):
                    client.openapi.projects.delete_project_id(cleanup_project_id)

    assert created["name"] == dataset_name
    assert [dataset["id"] for dataset in listed["objects"]] == [created["id"]]
    assert fetched["id"] == created["id"]
    assert updated["description"] == "updated by the Python SDK VCR test"
    assert inserted["row_ids"] == [event_id]
    assert merged["row_ids"] == [event_id]
    assert [event["id"] for event in fetched_post["events"]] == [event_id]
    assert fetched_post["events"][0]["input"] == {
        "question": "What is the updated answer?",
        "context": {"preserved": True},
        "replace_me": {"new": True},
    }
    assert fetched_post["events"][0]["tags"] == ["keep"]
    assert [event["id"] for event in fetched_get["events"]] == [event_id]
    assert feedback == {"status": "success"}
    assert summary["project_name"] == project_name
    assert summary["dataset_name"] == dataset_name
    assert removed["row_ids"] == [event_id]
    assert fetched_after_delete["events"] == []
    assert deleted["id"] == created["id"]


@pytest.mark.vcr
def test_high_level_dataset_uses_generated_resources(api_key):
    project_name = "python-sdk-high-level-datasets-vcr"
    event_id = "high-level-generated-row"
    cleanup_project_id = None
    cleanup_dataset_id = None

    dataset = braintrust.init_dataset(
        project=project_name,
        description="created through braintrust.init_dataset",
        api_key=api_key,
        use_output=False,
    )
    try:
        cleanup_dataset_id = dataset.id
        cleanup_project_id = dataset.project.id
        api_client = dataset.state.api_client()
        api_client.datasets.post_dataset_id_insert(
            dataset.id,
            body={
                "events": [
                    {
                        "id": event_id,
                        "input": {"question": "What is the answer?"},
                        "expected": "42",
                    }
                ]
            },
        )

        events = list(dataset.fetch(batch_size=10))
        summary = dataset.summarize()

        assert [event["id"] for event in events] == [event_id]
        assert summary.project_name == project_name
        assert dataset.name == "logs"
        assert summary.dataset_name == "logs"
        assert summary.data_summary is not None
        assert summary.data_summary.total_records == 1
    finally:
        if cleanup_dataset_id is not None:
            with contextlib.suppress(Exception):
                dataset.state.api_client().datasets.delete_dataset_id(cleanup_dataset_id)
        if cleanup_project_id is not None:
            with contextlib.suppress(Exception):
                dataset.state.api_client().projects.delete_project_id(cleanup_project_id)
