"""Tests for sandbox registration."""

import os

import pytest

from .logger import BraintrustState
from .sandbox import RegisterSandboxResult, SandboxConfig, register_sandbox


SNAPSHOT_REF = "im-icRxmsk1Sz9XPP2f8OblVU"
PROJECT = "My Project"
ENTRYPOINTS = ["./local/js/optimization/evals/btql-generation/btql-queries.eval.ts"]

# Paths that register_sandbox actually calls — used to filter out
# unrelated background logger traffic that VCR would otherwise capture.
_SANDBOX_PATHS = {"/api/project/register", "/function/sandbox-list", "/insert-functions"}


@pytest.fixture(scope="module")
def vcr_config():
    from .conftest import get_vcr_config

    config = get_vcr_config()
    config["before_record_request"] = lambda req: req if any(p in req.path for p in _SANDBOX_PATHS) else None
    return config


@pytest.fixture
def logged_in_state():
    """Create a pre-configured BraintrustState that skips the login HTTP call."""
    state = BraintrustState()
    state.app_url = "https://www.braintrust.dev"
    state.api_url = "https://api.braintrust.dev"
    state.proxy_url = "https://api.braintrust.dev"
    state.org_name = "nate test"
    state.org_id = "bf61839e-94bb-4a81-85ed-26b533a24b61"
    state.logged_in = True

    token = os.environ.get("BRAINTRUST_API_KEY", "sk-dummy-for-vcr-replay")
    state.app_conn().set_token(token)
    state.api_conn().set_token(token)
    state.proxy_conn().set_token(token)
    return state


@pytest.mark.vcr
def test_register_sandbox_single_evaluator(logged_in_state):
    result = register_sandbox(
        name="test-sandbox",
        project=PROJECT,
        sandbox=SandboxConfig(provider="modal", snapshot_ref=SNAPSHOT_REF),
        entrypoints=ENTRYPOINTS,
        state=logged_in_state,
    )

    assert isinstance(result, RegisterSandboxResult)
    assert result.project_id
    assert len(result.functions) >= 1
    assert result.functions[0].eval_name
    assert result.functions[0].id
    assert result.functions[0].slug


@pytest.mark.vcr
def test_register_sandbox_with_metadata_and_description(logged_in_state):
    result = register_sandbox(
        name="test-sandbox",
        project=PROJECT,
        sandbox=SandboxConfig(provider="modal", snapshot_ref=SNAPSHOT_REF),
        entrypoints=ENTRYPOINTS,
        metadata={"version": "1.0"},
        description="A test sandbox",
        state=logged_in_state,
    )

    assert isinstance(result, RegisterSandboxResult)
    assert result.project_id
    assert len(result.functions) >= 1


@pytest.mark.vcr
def test_register_sandbox_with_entrypoints(logged_in_state):
    result = register_sandbox(
        name="test-sandbox",
        project=PROJECT,
        sandbox=SandboxConfig(provider="modal", snapshot_ref=SNAPSHOT_REF),
        entrypoints=ENTRYPOINTS,
        state=logged_in_state,
    )

    assert isinstance(result, RegisterSandboxResult)
    assert result.project_id
    assert len(result.functions) >= 1
