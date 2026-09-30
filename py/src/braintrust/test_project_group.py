"""Tests for the `project_group_name` option on the entrypoints that can create a project.

These run against a real local HTTP server (`scripted_server`) rather than a mocked client, so the
assertions are made on the bytes the SDK actually puts on the wire.
"""

import json
from contextlib import contextmanager
from unittest.mock import patch

import braintrust
import pytest
from braintrust import logger
from braintrust.api._test_server import scripted_server
from braintrust.logger import BraintrustState, span_components_to_object_id
from braintrust.span_identifier_v4 import SpanComponentsV4


PROJECT_ID = "00000000-0000-0000-0000-000000000001"

# `test_helpers.init_test_logger` replaces this module-level function with a fake and never puts it
# back, so capture the real one at import time (before any test body has run) and restore it for
# each test here. These tests are about the requests the real implementation makes.
_REAL_COMPUTE_LOGGER_METADATA = logger._compute_logger_metadata


@pytest.fixture(autouse=True)
def use_real_compute_logger_metadata():
    with patch.object(logger, "_compute_logger_metadata", _REAL_COMPUTE_LOGGER_METADATA):
        yield


@contextmanager
def braintrust_server():
    """Serve the handful of endpoints these entrypoints touch, and yield a logged-in state."""

    base_url = []

    def handle(command, path, body, headers):
        route = path.split("?")[0]
        if route == "/api/apikey/login":
            response = {
                "org_info": [{"id": "org-id", "name": "org-name", "api_url": base_url[0], "proxy_url": None}]
            }
        elif route == "/v1/project":
            response = {"id": PROJECT_ID, "name": "project"}
        elif route == f"/v1/project/{PROJECT_ID}":
            response = {"id": PROJECT_ID, "name": "project"}
        elif route == "/v1/experiment":
            response = {"id": "experiment-id", "name": "experiment", "project_id": PROJECT_ID}
        elif route == "/v1/dataset":
            response = {"id": "dataset-id", "name": "dataset", "project_id": PROJECT_ID}
        else:
            response = {}
        return (200, {"Content-Type": "application/json"}, json.dumps(response).encode())

    with scripted_server(handle) as (url, handler):
        base_url.append(url)
        state = BraintrustState()
        state.login(api_key="test-key", app_url=url)
        try:
            yield state, handler
        finally:
            state.flush()


def bodies_for(handler, path):
    return [json.loads(body) for method, request_path, body, _ in handler.requests if request_path.split("?")[0] == path]


def test_init_logger_forwards_project_group_name():
    with braintrust_server() as (state, handler):
        log = braintrust.init_logger(
            project="project", project_group_name="my-group", set_current=False, state=state
        )
        assert log.id == PROJECT_ID

    assert bodies_for(handler, "/v1/project") == [
        {"name": "project", "org_name": "org-name", "project_group_name": "my-group"}
    ]


def test_init_logger_omits_project_group_name_when_unspecified():
    with braintrust_server() as (state, handler):
        braintrust.init_logger(project="project", set_current=False, state=state).id

    assert bodies_for(handler, "/v1/project") == [{"name": "project", "org_name": "org-name"}]


def test_init_creates_the_project_in_the_group_then_registers_by_id():
    with braintrust_server() as (state, handler):
        experiment = braintrust.init(
            project="project",
            experiment="experiment",
            project_group_name="my-group",
            set_current=False,
            state=state,
        )
        assert experiment.id == "experiment-id"

    assert bodies_for(handler, "/v1/project") == [
        {"name": "project", "org_name": "org-name", "project_group_name": "my-group"}
    ]
    (experiment_body,) = bodies_for(handler, "/v1/experiment")
    assert experiment_body["project_id"] == PROJECT_ID
    assert "project_name" not in experiment_body


def test_init_omits_project_group_name_when_unspecified():
    with braintrust_server() as (state, handler):
        braintrust.init(project="project", experiment="experiment", set_current=False, state=state).id

    assert bodies_for(handler, "/v1/project") == [{"name": "project", "org_name": "org-name"}]


def test_init_dataset_creates_the_project_in_the_group_then_registers_by_id():
    with braintrust_server() as (state, handler):
        dataset = braintrust.init_dataset(
            project="project", name="dataset", project_group_name="my-group", state=state
        )
        assert dataset.id == "dataset-id"

    assert bodies_for(handler, "/v1/project") == [
        {"name": "project", "org_name": "org-name", "project_group_name": "my-group"}
    ]
    (dataset_body,) = bodies_for(handler, "/v1/dataset")
    assert dataset_body["project_id"] == PROJECT_ID
    assert "project_name" not in dataset_body


def test_project_group_name_is_ignored_when_project_id_is_specified():
    with braintrust_server() as (state, handler):
        braintrust.init_dataset(
            project_id=PROJECT_ID, name="dataset", project_group_name="my-group", state=state
        ).id
        braintrust.init_logger(
            project_id=PROJECT_ID, project_group_name="my-group", set_current=False, state=state
        ).id

    assert bodies_for(handler, "/v1/project") == []
    assert bodies_for(handler, "/v1/dataset")[0]["project_id"] == PROJECT_ID


def test_eval_creates_the_project_in_the_group_before_the_experiment():
    with braintrust_server() as (state, handler):
        result = braintrust.Eval(
            "project",
            data=[{"input": 1, "expected": 2}],
            task=lambda input: input * 2,
            scores=[],
            project_group_name="my-group",
            state=state,
        )
        assert result.summary.experiment_id == "experiment-id"

    assert bodies_for(handler, "/v1/project") == [
        {"name": "project", "org_name": "org-name", "project_group_name": "my-group"}
    ]
    assert bodies_for(handler, "/v1/experiment")[0]["project_id"] == PROJECT_ID


def test_exported_span_components_carry_the_project_group_for_lazy_resolution():
    with braintrust_server() as (state, handler):
        log = braintrust.init_logger(
            project="project", project_group_name="my-group", set_current=False, state=state
        )

        # The logger has not resolved its id yet, so `export()` defers project resolution (and
        # creation) to whoever consumes the exported components.
        components = SpanComponentsV4.from_str(log.export())
        assert components.compute_object_metadata_args == {
            "project_name": "project",
            "project_id": None,
            "project_group_name": "my-group",
        }

        with patch.object(logger, "_state", state):
            assert span_components_to_object_id(components) == PROJECT_ID

    assert bodies_for(handler, "/v1/project") == [
        {"name": "project", "org_name": "org-name", "project_group_name": "my-group"}
    ]


def test_exported_span_components_omit_the_project_group_when_unspecified():
    with braintrust_server() as (state, _handler):
        log = braintrust.init_logger(project="project", set_current=False, state=state)
        components = SpanComponentsV4.from_str(log.export())
        assert components.compute_object_metadata_args == {"project_name": "project", "project_id": None}
