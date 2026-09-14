import importlib.resources
import json
import subprocess
import sys
import threading
from typing import get_type_hints, is_typeddict


GENERATED_RESOURCE_NAMES = (
    "acls",
    "agents",
    "ai_secrets",
    "api_keys",
    "dataset_snapshots",
    "datasets",
    "env_vars",
    "environments",
    "experiments",
    "functions",
    "groups",
    "mcp_servers",
    "org_automations",
    "organizations",
    "project_automations",
    "project_groups",
    "project_scores",
    "project_tags",
    "projects",
    "prompts",
    "roles",
    "service_tokens",
    "span_iframes",
    "users",
    "views",
)


def test_import_braintrust_is_lazy_about_generated_api_modules():
    script = """
import json
import sys
import braintrust
print(json.dumps(sorted(name for name in sys.modules if name.startswith('braintrust.api._generated'))))
"""

    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    assert json.loads(result.stdout) == []


def test_generated_models_import_on_supported_python():
    from braintrust.api._generated import datasets as dataset_bindings
    from braintrust.api._generated import experiments as experiment_bindings
    from braintrust.api._generated import functions as function_bindings
    from braintrust.api._generated import models
    from braintrust.api._generated import projects as project_bindings
    from braintrust.api._generated import prompts as prompt_bindings

    assert is_typeddict(models.Dataset)
    assert is_typeddict(models.Experiment)
    assert is_typeddict(models.Function)
    assert is_typeddict(models.Project)
    assert is_typeddict(models.Prompt)
    assert models.DatasetIdParam is str
    assert models.ExperimentIdParam is str
    assert models.FunctionIdParam is str
    assert models.ProjectIdParam is str
    assert models.PromptIdParam is str
    assert get_type_hints(dataset_bindings.DatasetsAPI.get_dataset)["return"] is models.GetDatasetResponse
    assert get_type_hints(experiment_bindings.ExperimentsAPI.get_experiment)["return"] is models.GetExperimentResponse
    assert get_type_hints(function_bindings.FunctionsAPI.get_function)["return"] is models.GetFunctionResponse
    assert get_type_hints(project_bindings.ProjectsAPI.get_project)["return"] is models.GetProjectResponse
    assert get_type_hints(prompt_bindings.PromptsAPI.get_prompt)["return"] is models.GetPromptResponse


def test_openapi_client_lazily_loads_and_caches_generated_resources():
    script = """
import json
import sys
from braintrust.api import BraintrustOpenApiClient

client = BraintrustOpenApiClient(api_key="test-key", api_url="https://api.example.com")
before = sorted(name for name in sys.modules if name.startswith("braintrust.api._generated"))
first = client.projects
second = client.projects
after = sorted(name for name in sys.modules if name.startswith("braintrust.api._generated"))
print(json.dumps({"before": before, "cached": first is second, "after": after}))
client.close()
"""

    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    loaded = json.loads(result.stdout)

    assert loaded["before"] == []
    assert loaded["cached"] is True
    assert "braintrust.api._generated.projects" in loaded["after"]
    assert "braintrust.api._generated.models.projects" in loaded["after"]
    assert "braintrust.api._generated.datasets" not in loaded["after"]
    assert "braintrust.api._generated.models.datasets" not in loaded["after"]
    assert "braintrust.api._generated.models.project_automations" not in loaded["after"]


def test_openapi_client_caches_one_resource_across_threads():
    from braintrust.api import BraintrustOpenApiClient

    client = BraintrustOpenApiClient(api_key="test-key", api_url="https://api.example.com")
    barrier = threading.Barrier(8)
    resources = []

    def load_projects():
        barrier.wait()
        resources.append(client.projects)

    threads = [threading.Thread(target=load_projects) for _ in range(barrier.parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    client.close()

    assert len(resources) == barrier.parties
    assert all(resource is resources[0] for resource in resources)


def test_openapi_client_exposes_only_reviewed_generated_resources():
    from braintrust.api import BraintrustOpenApiClient

    with BraintrustOpenApiClient(api_key="test-key", api_url="https://api.example.com") as client:
        assert all(hasattr(client, resource_name) for resource_name in GENERATED_RESOURCE_NAMES)
        assert not any(
            hasattr(client, resource_name)
            for resource_name in ("cors", "cross_object", "evals", "logs", "other", "proxy")
        )


def test_generated_package_content_is_installed():
    generated = importlib.resources.files("braintrust.api._generated")

    assert generated.joinpath("__init__.py").is_file()
    assert generated.joinpath("models", "__init__.py").is_file()
    assert generated.joinpath("models", "common.py").is_file()
    for resource_name in GENERATED_RESOURCE_NAMES:
        assert generated.joinpath("models", f"{resource_name}.py").is_file()
        assert generated.joinpath(f"{resource_name}.py").is_file()


def test_rest_and_logging_type_surfaces_have_reviewed_overlap():
    from braintrust import generated_types
    from braintrust.api import types

    overlap = set(generated_types.__all__) & set(types.__all__)

    assert overlap == {
        "AISecret",
        "Acl",
        "Agent",
        "ApiKey",
        "Dataset",
        "DatasetSnapshot",
        "EnvVar",
        "Experiment",
        "Function",
        "Group",
        "MCPServer",
        "OrgAutomation",
        "Organization",
        "Project",
        "ProjectAutomation",
        "ProjectGroup",
        "ProjectScore",
        "ProjectTag",
        "Prompt",
        "Role",
        "ServiceToken",
        "SpanIFrame",
        "User",
        "View",
    }
