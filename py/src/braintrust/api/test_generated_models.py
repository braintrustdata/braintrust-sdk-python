import importlib.resources
import json
import subprocess
import sys
from typing import get_type_hints, is_typeddict


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
    from braintrust.api._generated import models
    from braintrust.api._generated import projects as project_bindings

    assert is_typeddict(models.Dataset)
    assert is_typeddict(models.Experiment)
    assert is_typeddict(models.Project)
    assert models.DatasetIdParam is str
    assert models.ExperimentIdParam is str
    assert models.ProjectIdParam is str
    assert get_type_hints(dataset_bindings.DatasetsAPI.get_dataset)["return"] is models.GetDatasetResponse
    assert get_type_hints(experiment_bindings.ExperimentsAPI.get_experiment)["return"] is models.GetExperimentResponse
    assert get_type_hints(project_bindings.ProjectsAPI.get_project)["return"] is models.GetProjectResponse


def test_generated_package_content_is_installed():
    generated = importlib.resources.files("braintrust.api._generated")

    assert generated.joinpath("__init__.py").is_file()
    assert generated.joinpath("models", "__init__.py").is_file()
    assert generated.joinpath("models", "common.py").is_file()
    assert generated.joinpath("models", "datasets.py").is_file()
    assert generated.joinpath("models", "experiments.py").is_file()
    assert generated.joinpath("models", "projects.py").is_file()
    assert generated.joinpath("datasets.py").is_file()
    assert generated.joinpath("experiments.py").is_file()
    assert generated.joinpath("projects.py").is_file()


def test_rest_and_logging_type_surfaces_have_reviewed_overlap():
    from braintrust import generated_types
    from braintrust.api import types

    overlap = set(generated_types.__all__) & set(types.__all__)

    assert overlap == {"Dataset", "Experiment", "Project"}
