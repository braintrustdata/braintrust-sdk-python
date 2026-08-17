import copy
import re

from openapi_codegen import atomic_replace_tree, compare_generated, generate_tree


def _generate(tmp_path, name, config, spec):
    output = tmp_path / name / "_generated"
    generate_tree(output, config, spec)
    return output


def test_generation_is_byte_for_byte_deterministic(tmp_path, codegen_config, minimal_spec):
    first = _generate(tmp_path, "first", codegen_config, minimal_spec)
    second = _generate(tmp_path, "second", codegen_config, minimal_spec)

    assert compare_generated(first, second) == []


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
    models = (generated / "models.py").read_text()

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

    generated = _generate(tmp_path, "composition", codegen_config, spec)
    models = (generated / "models.py").read_text()

    assert re.search(r"Pet: TypeAlias = Cat \| Dog", models)
    assert "class PetEnvelope" in models
    assert "ratio: NotRequired[float]" in models
    assert "labels: NotRequired[Sequence[str]]" in models
