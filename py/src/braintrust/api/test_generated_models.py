import importlib.resources
import json
import subprocess
import sys
from typing import is_typeddict


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
    from braintrust.api._generated import models

    assert is_typeddict(models.Project)
    assert models.ProjectIdParam is str


def test_generated_package_content_is_installed():
    generated = importlib.resources.files("braintrust.api._generated")

    assert generated.joinpath("__init__.py").is_file()
    assert generated.joinpath("models.py").is_file()


def test_rest_and_logging_type_surfaces_have_reviewed_overlap():
    from braintrust import generated_types
    from braintrust.api import types

    overlap = set(generated_types.__all__) & set(types.__all__)

    assert overlap == set()
