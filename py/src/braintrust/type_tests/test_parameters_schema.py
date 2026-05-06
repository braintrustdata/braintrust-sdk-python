"""Type-check tests for saved parameter schema definitions."""

from braintrust import projects
from braintrust.parameters import EvalParameters


def test_parameters_create_accepts_discriminated_schema_entries() -> None:
    project = projects.create("test-project")

    schema: EvalParameters = {
        "model": {
            "type": "model",
            "default": "gpt-5-mini",
        },
        "main": {
            "type": "prompt",
            "default": {
                "prompt": {
                    "type": "chat",
                    "messages": [{"role": "user", "content": "{{input}}"}],
                },
                "options": {
                    "model": "gpt-5-mini",
                },
            },
        },
    }

    created = project.parameters.create(name="test-parameters", schema=schema)

    assert created is schema
