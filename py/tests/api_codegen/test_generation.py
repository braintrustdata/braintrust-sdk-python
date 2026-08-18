import copy
import re

import pytest
from openapi_codegen import CodegenError, atomic_replace_tree, compare_generated, generate_tree


def _generate(tmp_path, name, config, spec):
    output = tmp_path / name / "_generated"
    generate_tree(output, config, spec)
    return output


def _models_text(generated):
    return "\n".join(path.read_text() for path in sorted((generated / "models").glob("*.py")))


def test_generation_is_byte_for_byte_deterministic(tmp_path, codegen_config, minimal_spec):
    first = _generate(tmp_path, "first", codegen_config, minimal_spec)
    second = _generate(tmp_path, "second", codegen_config, minimal_spec)

    assert compare_generated(first, second) == []


def test_generation_selects_generated_tag_regardless_of_tag_order(tmp_path, codegen_config, minimal_spec):
    minimal_spec["paths"]["/widgets/{widget_id}"]["get"]["tags"] = ["Internal", "Widgets"]

    generated = _generate(tmp_path, "secondary-generated-tag", codegen_config, minimal_spec)

    assert "def get_widget(" in (generated / "widgets.py").read_text()


def test_declarative_post_reads_use_safe_read_retry_mode(tmp_path, codegen_config, minimal_spec):
    minimal_spec["paths"]["/widgets"] = {
        "post": {
            "operationId": "postWidgetFetch",
            "tags": ["Widgets"],
            "responses": {
                "200": {
                    "description": "OK",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Widget"}}},
                }
            },
        }
    }
    codegen_config["endpoint_generator"]["safe_reads"] = ["postWidgetFetch"]

    generated = _generate(tmp_path, "safe-read", codegen_config, minimal_spec)

    assert "retry_mode=RetryMode.SAFE_READ" in (generated / "widgets.py").read_text()


def test_idempotent_writes_use_idempotent_write_retry_mode(tmp_path, codegen_config, minimal_spec):
    minimal_spec["paths"]["/widgets"] = {
        "post": {
            "operationId": "postWidget",
            "tags": ["Widgets"],
            "responses": {
                "200": {
                    "description": "OK",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Widget"}}},
                }
            },
        }
    }
    codegen_config["endpoint_generator"]["idempotent_writes"] = ["postWidget"]

    generated = _generate(tmp_path, "idempotent-write", codegen_config, minimal_spec)
    bindings = (generated / "widgets.py").read_text()

    assert "retry_mode=RetryMode.IDEMPOTENT_WRITE" in bindings


def test_multiple_generated_resources_partition_shared_models_deterministically(
    tmp_path, codegen_config, minimal_spec
):
    spec = copy.deepcopy(minimal_spec)
    spec["components"]["schemas"]["Widget"]["properties"]["details"] = {"$ref": "#/components/schemas/WidgetDetails"}
    spec["components"]["schemas"]["WidgetDetails"] = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    spec["components"]["schemas"]["Gadget"] = {
        "type": "object",
        "properties": {
            "widget": {"$ref": "#/components/schemas/Widget"},
            "serial": {"type": "string"},
        },
        "required": ["widget", "serial"],
    }
    spec["paths"]["/gadgets"] = {
        "get": {
            "operationId": "getGadget",
            "tags": ["Gadgets"],
            "responses": {
                "200": {
                    "description": "OK",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Gadget"}}},
                }
            },
        }
    }
    codegen_config["endpoint_generator"]["generated_tags"] = ["Widgets", "Gadgets"]

    generated = _generate(tmp_path, "multiple-model-resources", codegen_config, spec)

    common_models = (generated / "models" / "common.py").read_text()
    assert "class Widget(TypedDict):" in common_models
    assert "class WidgetDetails(TypedDict):" in common_models
    assert "from .common import" not in common_models
    assert "class Gadget(TypedDict):" in (generated / "models" / "gadgets.py").read_text()
    assert "from .common import Widget" in (generated / "models" / "gadgets.py").read_text()
    assert "from .models.common import Widget" in (generated / "widgets.py").read_text()
    assert "from .models.gadgets import Gadget" in (generated / "gadgets.py").read_text()
    model_exports = (generated / "models" / "__init__.py").read_text()
    assert "from .common import Widget, WidgetDetails" in model_exports
    assert "from .gadgets import Gadget" in model_exports


def test_unreachable_models_are_omitted_but_transitive_references_are_kept(tmp_path, codegen_config, minimal_spec):
    minimal_spec["components"]["schemas"]["Widget"]["properties"]["details"] = {
        "$ref": "#/components/schemas/WidgetDetails"
    }
    minimal_spec["components"]["schemas"]["WidgetDetails"] = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    minimal_spec["components"]["schemas"]["Unused"] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }
    generated = _generate(tmp_path, "reachable-models", codegen_config, minimal_spec)
    models = _models_text(generated)

    assert "class Widget(TypedDict):" in models
    assert "class WidgetDetails(TypedDict):" in models
    assert "class Unused(TypedDict):" not in models


def test_inline_response_models_use_the_model_generator(tmp_path, codegen_config, minimal_spec):
    spec = copy.deepcopy(minimal_spec)
    response = spec["paths"]["/widgets/{widget_id}"]["get"]["responses"]["200"]
    response["content"]["application/json"]["schema"] = {
        "type": "object",
        "properties": {
            "objects": {"type": "array", "items": {"$ref": "#/components/schemas/Widget"}},
        },
        "required": ["objects"],
    }

    generated = _generate(tmp_path, "inline-response", codegen_config, spec)
    models = _models_text(generated)
    bindings = (generated / "widgets.py").read_text()

    assert "class GetWidgetResponse(TypedDict):" in models
    assert "objects: Sequence[Widget]" in models
    assert "def get_widget(" in bindings
    assert ') -> "GetWidgetResponse":' in bindings


def test_inline_response_generated_name_collisions_fail(tmp_path, codegen_config, minimal_spec):
    spec = copy.deepcopy(minimal_spec)
    spec["components"]["schemas"]["get_widget_response"] = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }
    response = spec["paths"]["/widgets/{widget_id}"]["get"]["responses"]["200"]
    response["content"]["application/json"]["schema"] = {
        "type": "object",
        "properties": {
            "nested": {"$ref": "#/components/schemas/get_widget_response"},
        },
    }

    with pytest.raises(
        CodegenError,
        match="Inline response model 'GetWidgetResponse'.*component schema 'get_widget_response'",
    ):
        _generate(tmp_path, "inline-response-name-collision", codegen_config, spec)


def test_mixed_json_and_empty_success_responses_track_statuses(tmp_path, codegen_config, minimal_spec):
    responses = minimal_spec["paths"]["/widgets/{widget_id}"]["get"]["responses"]
    responses["200"]["content"]["application/json"]["schema"] = {
        "type": "object",
        "properties": {"widget": {"$ref": "#/components/schemas/Widget"}},
    }
    responses["204"] = {"description": "No content"}

    generated = _generate(tmp_path, "mixed-success-responses", codegen_config, minimal_spec)
    models = _models_text(generated)
    bindings = (generated / "widgets.py").read_text()

    assert "class GetWidgetResponse(TypedDict):" in models
    assert "success_statuses=(200, 204)" in bindings
    assert "json_success_statuses=(200,)" in bindings
    assert ') -> "GetWidgetResponse | None":' in bindings


def test_differing_inline_success_response_schemas_fail(tmp_path, codegen_config, minimal_spec):
    spec = copy.deepcopy(minimal_spec)
    responses = spec["paths"]["/widgets/{widget_id}"]["get"]["responses"]
    responses["200"]["content"]["application/json"]["schema"] = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    responses["201"] = {
        "description": "Created",
        "content": {"application/json": {"schema": {"type": "object", "properties": {"id": {"type": "string"}}}}},
    }
    with pytest.raises(CodegenError, match="conflicting inline success response schemas"):
        _generate(tmp_path, "conflicting-inline-responses", codegen_config, spec)


def test_stale_artifacts_are_reported_and_removed(tmp_path, codegen_config, minimal_spec):
    """Files the generator no longer emits count as drift, whatever their extension."""
    regenerated = _generate(tmp_path, "regenerated", codegen_config, minimal_spec)
    committed = _generate(tmp_path, "committed", codegen_config, minimal_spec)
    (committed / "endpoints.pyi").write_text("stale\n", encoding="utf-8")
    stale_package = committed / "services"
    stale_package.mkdir()
    (stale_package / "__init__.py").write_text("stale\n", encoding="utf-8")
    (stale_package / "__pycache__").mkdir()

    differences = compare_generated(regenerated, committed)
    assert any("endpoints.pyi" in line for line in differences)
    assert any("services/__init__.py" in line for line in differences)

    atomic_replace_tree(regenerated, committed)

    assert compare_generated(regenerated, committed) == []
    assert not (committed / "endpoints.pyi").exists()
    assert not stale_package.exists()


def test_nullable_and_missing_fields_remain_distinct(tmp_path, codegen_config, minimal_spec):
    spec = copy.deepcopy(minimal_spec)
    widget = spec["components"]["schemas"]["Widget"]
    widget["properties"] = {
        "required_nullable": {"type": "string", "nullable": True},
        "optional_nullable": {"type": "string", "nullable": True},
        "optional_non_null": {"type": "string"},
    }
    widget["required"] = ["required_nullable"]

    generated = _generate(tmp_path, "nullable", codegen_config, spec)
    models = _models_text(generated)

    assert re.search(r"required_nullable: str \| None", models)
    assert re.search(r"optional_nullable: NotRequired\[str \| None\]", models)
    assert re.search(r"optional_non_null: NotRequired\[str\]", models)


def test_composition_types_and_json_scalars_generate(tmp_path, codegen_config, minimal_spec):
    spec = copy.deepcopy(minimal_spec)
    spec["components"]["schemas"].update(
        {
            "Cat": {
                "type": "object",
                "properties": {"lives": {"type": "integer"}},
                "required": ["lives"],
            },
            "Dog": {
                "type": "object",
                "properties": {"good": {"type": "boolean"}},
                "required": ["good"],
            },
            "Pet": {"oneOf": [{"$ref": "#/components/schemas/Cat"}, {"$ref": "#/components/schemas/Dog"}]},
            "PetEnvelope": {
                "allOf": [
                    {"type": "object", "properties": {"pet": {"$ref": "#/components/schemas/Pet"}}},
                    {
                        "type": "object",
                        "properties": {
                            "ratio": {"type": "number"},
                            "labels": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                ]
            },
        }
    )
    spec["components"]["schemas"]["Widget"]["properties"]["pet_envelope"] = {
        "$ref": "#/components/schemas/PetEnvelope"
    }

    generated = _generate(tmp_path, "composition", codegen_config, spec)
    models = _models_text(generated)

    assert re.search(r"Pet: TypeAlias = Cat \| Dog", models)
    assert "class PetEnvelope" in models
    assert "ratio: NotRequired[float]" in models
    assert "labels: NotRequired[Sequence[str]]" in models
