import importlib.util

import pytest
from braintrust.parameters import (
    RemoteEvalParameters,
    parameters_to_json_schema,
    validate_parameters,
)


HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_validate_local_parameters_with_prompt_and_model_defaults():
    from pydantic import BaseModel

    class PrefixParam(BaseModel):
        value: str = "hello"

    result = validate_parameters(
        {},
        {
            "prefix": PrefixParam,
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
        },
    )

    assert result["prefix"] == "hello"
    assert result["model"] == "gpt-5-mini"
    assert hasattr(result["main"], "build")


def test_validate_remote_parameters_merges_saved_data_and_runtime_overrides():
    parameters = RemoteEvalParameters(
        id="params-123",
        project_id="project-123",
        name="Saved parameters",
        slug="saved-parameters",
        version="v1",
        schema={
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "default": "saved-prefix",
                },
                "model": {
                    "type": "string",
                    "x-bt-type": "model",
                    "default": "gpt-5-mini",
                },
            },
            "additionalProperties": True,
        },
        data={"prefix": "saved-prefix"},
    )

    result = validate_parameters({"model": "gpt-5-nano"}, parameters)

    assert result == {
        "prefix": "saved-prefix",
        "model": "gpt-5-nano",
    }


def test_validate_remote_parameters_hydrates_prompt_values():
    parameters = RemoteEvalParameters(
        id="params-123",
        project_id="project-123",
        name="Saved parameters",
        slug="saved-parameters",
        version="v1",
        schema={
            "type": "object",
            "properties": {
                "main": {
                    "type": "object",
                    "x-bt-type": "prompt",
                },
            },
            "additionalProperties": True,
        },
        data={
            "main": {
                "prompt": {
                    "type": "chat",
                    "messages": [{"role": "user", "content": "{{input}}"}],
                },
                "options": {
                    "model": "gpt-5-mini",
                },
            },
        },
    )

    result = validate_parameters({}, parameters)

    assert hasattr(result["main"], "build")


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_parameters_to_json_schema_uses_scalar_schema_for_single_value_models():
    from pydantic import BaseModel

    class PrefixParam(BaseModel):
        value: str = "hello"

    schema = parameters_to_json_schema({"prefix": PrefixParam})

    assert schema["properties"]["prefix"] == {
        "type": "string",
        "default": "hello",
        "title": "Value",
    }
