import importlib.util

import pytest
from braintrust.parameters import (
    RemoteEvalParameters,
    parameters_to_json_schema,
    serialize_eval_parameters,
    validate_parameters,
)


HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None


def _contains_json_schema_ref(node):
    if isinstance(node, dict):
        if "$ref" in node or "$defs" in node or "definitions" in node:
            return True
        return any(_contains_json_schema_ref(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_json_schema_ref(value) for value in node)
    return False


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


def test_validate_remote_parameters_keeps_prompt_values_as_dicts():
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

    assert isinstance(result["main"], dict)


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


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_parameters_to_json_schema_keeps_complex_single_value_models_self_contained():
    from pydantic import BaseModel, Field

    class ComplexValue(BaseModel):
        a: int = Field(default=1)
        b: list[int] = Field(default=[2, 3])

    class ComplexParameter(BaseModel):
        value: ComplexValue = Field(
            default=ComplexValue(),
            description="Complex example parameter",
        )

    schema = parameters_to_json_schema({"complex": ComplexParameter})

    assert schema["properties"]["complex"] == {
        "type": "object",
        "properties": {
            "a": {
                "default": 1,
                "title": "A",
                "type": "integer",
            },
            "b": {
                "default": [2, 3],
                "items": {
                    "type": "integer",
                },
                "title": "B",
                "type": "array",
            },
        },
        "default": {
            "a": 1,
            "b": [2, 3],
        },
        "description": "Complex example parameter",
        "title": "ComplexValue",
    }


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_serialize_eval_parameters_does_not_emit_dangling_ref_for_complex_single_value_models():
    from pydantic import BaseModel, Field

    class ComplexValue(BaseModel):
        a: int = Field(default=1)
        b: list[int] = Field(default=[2, 3])

    class ComplexParameter(BaseModel):
        value: ComplexValue = Field(
            default=ComplexValue(),
            description="Complex example parameter",
        )

    serialized = serialize_eval_parameters({"complex": ComplexParameter})

    assert serialized["complex"] == {
        "type": "data",
        "schema": {
            "type": "object",
            "properties": {
                "a": {
                    "default": 1,
                    "title": "A",
                    "type": "integer",
                },
                "b": {
                    "default": [2, 3],
                    "items": {
                        "type": "integer",
                    },
                    "title": "B",
                    "type": "array",
                },
            },
            "default": {
                "a": 1,
                "b": [2, 3],
            },
            "description": "Complex example parameter",
            "title": "ComplexValue",
        },
        "default": {
            "a": 1,
            "b": [2, 3],
        },
        "description": "Complex example parameter",
    }


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_parameters_to_json_schema_keeps_array_of_objects_single_value_models_self_contained():
    from pydantic import BaseModel, Field

    class ComplexItem(BaseModel):
        a: int = Field(default=1)
        b: str = Field(default="x")

    class ComplexParameter(BaseModel):
        value: list[ComplexItem] = Field(
            default=[ComplexItem()],
            description="Array example parameter",
        )

    schema = parameters_to_json_schema({"complex_array": ComplexParameter})

    assert schema["properties"]["complex_array"] == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "a": {
                    "default": 1,
                    "title": "A",
                    "type": "integer",
                },
                "b": {
                    "default": "x",
                    "title": "B",
                    "type": "string",
                },
            },
            "title": "ComplexItem",
        },
        "default": [
            {
                "a": 1,
                "b": "x",
            }
        ],
        "description": "Array example parameter",
        "title": "Value",
    }
    assert not _contains_json_schema_ref(schema["properties"]["complex_array"])


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_parameters_to_json_schema_inlines_refs_for_multi_field_models():
    from pydantic import BaseModel, Field

    class Address(BaseModel):
        street: str = Field(default="Main")
        zip_code: int = Field(default=12345)

    class ComplexParameter(BaseModel):
        address: Address = Field(default=Address())
        enabled: bool = Field(default=True)

    schema = parameters_to_json_schema({"complex": ComplexParameter})

    assert not _contains_json_schema_ref(schema["properties"]["complex"])
    assert schema["properties"]["complex"]["properties"]["address"] == {
        "type": "object",
        "properties": {
            "street": {
                "default": "Main",
                "title": "Street",
                "type": "string",
            },
            "zip_code": {
                "default": 12345,
                "title": "Zip Code",
                "type": "integer",
            },
        },
        "default": {
            "street": "Main",
            "zip_code": 12345,
        },
        "title": "Address",
    }


def test_parameters_to_json_schema_inlines_legacy_definitions_refs():
    class _FakeField:
        required = False

    class LegacyParameter:
        __fields__ = {"value": _FakeField()}

        @classmethod
        def parse_obj(cls, value):
            return value

        @classmethod
        def schema(cls):
            return {
                "type": "object",
                "properties": {
                    "value": {
                        "$ref": "#/definitions/ComplexValue",
                        "description": "Legacy example parameter",
                        "default": {"a": 1},
                    },
                },
                "definitions": {
                    "ComplexValue": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "integer",
                                "title": "A",
                                "default": 1,
                            },
                        },
                        "title": "ComplexValue",
                    },
                },
            }

    schema = parameters_to_json_schema({"legacy": LegacyParameter})

    assert schema["properties"]["legacy"] == {
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "title": "A",
                "default": 1,
            },
        },
        "title": "ComplexValue",
        "description": "Legacy example parameter",
        "default": {"a": 1},
    }
    assert not _contains_json_schema_ref(schema["properties"]["legacy"])


def test_parameters_to_json_schema_raises_for_cyclic_local_refs():
    class _FakeField:
        required = False

    class CyclicParameter:
        __fields__ = {"value": _FakeField()}

        @classmethod
        def parse_obj(cls, value):
            return value

        @classmethod
        def schema(cls):
            return {
                "type": "object",
                "properties": {
                    "value": {
                        "$ref": "#/definitions/Node",
                    },
                },
                "definitions": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "child": {
                                "$ref": "#/definitions/Node",
                            },
                        },
                    },
                },
            }

    with pytest.raises(ValueError, match="Cyclic JSON schema ref"):
        parameters_to_json_schema({"cyclic": CyclicParameter})


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_parameters_to_json_schema_marks_prompt_and_model_without_defaults_required():
    schema = parameters_to_json_schema(
        {
            "prompt_required": {"type": "prompt"},
            "prompt_optional": {
                "type": "prompt",
                "default": {
                    "prompt": {
                        "type": "chat",
                        "messages": [{"role": "user", "content": "{{input}}"}],
                    },
                    "options": {"model": "gpt-5-mini"},
                },
            },
            "model_required": {"type": "model"},
            "model_optional": {"type": "model", "default": "gpt-5-mini"},
        }
    )

    assert schema["required"] == ["prompt_required", "model_required"]


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_parameters_to_json_schema_marks_single_value_models_required_from_value_field():
    from pydantic import BaseModel

    class RequiredScalarParam(BaseModel):
        value: int

    class OptionalScalarParam(BaseModel):
        value: int = 3

    schema = parameters_to_json_schema(
        {
            "required_scalar": RequiredScalarParam,
            "optional_scalar": OptionalScalarParam,
        }
    )

    assert schema["required"] == ["required_scalar"]


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_parameters_to_json_schema_does_not_mark_multi_field_models_required():
    from pydantic import BaseModel

    class RequiredObjectParam(BaseModel):
        x: int
        y: int = 2

    class OptionalObjectParam(BaseModel):
        x: int = 1
        y: int = 2

    schema = parameters_to_json_schema(
        {
            "required_object": RequiredObjectParam,
            "optional_object": OptionalObjectParam,
        }
    )

    assert "required" not in schema


@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
def test_parameters_to_json_schema_does_not_mark_passthrough_values_required():
    schema = parameters_to_json_schema(
        {
            "passthrough": None,
        }
    )

    assert "required" not in schema
