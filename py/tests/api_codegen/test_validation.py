import copy
import hashlib
import json

import pytest
from openapi_codegen import (
    CodegenError,
    normalize_spec,
    read_and_verify_spec,
    validate_config,
    validate_spec,
)


def test_hash_and_full_commit_pin_validation(tmp_path, codegen_config, minimal_spec):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(minimal_spec, sort_keys=True), encoding="utf-8")
    codegen_config["spec"]["sha256"] = hashlib.sha256(spec_path.read_bytes()).hexdigest()

    assert read_and_verify_spec(codegen_config, spec_path) == minimal_spec

    codegen_config["spec"]["sha256"] = "0" * 64
    with pytest.raises(CodegenError, match="spec hash mismatch"):
        read_and_verify_spec(codegen_config, spec_path)

    codegen_config["spec"]["commit"] = "main"
    with pytest.raises(CodegenError, match="full lowercase 40-character commit SHA"):
        validate_config(codegen_config, check_installed_tools=False)


def test_normalization_removes_only_options_and_exact_skips(minimal_spec, codegen_config):
    spec = copy.deepcopy(minimal_spec)
    spec["paths"]["/widgets/{widget_id}"]["options"] = {
        "operationId": "optionsWidget",
        "tags": ["CORS"],
        "responses": {"200": {"description": "OK", "content": {"text/plain": {"schema": {"type": "string"}}}}},
    }
    spec["paths"]["/proxy"] = {
        "post": {
            "operationId": "proxyRequest",
            "tags": ["Proxy"],
            "responses": {
                "200": {
                    "description": "OK",
                    "content": {"application/json": {"schema": {"type": "object"}}},
                }
            },
        }
    }
    codegen_config["endpoint_generator"]["skip_tags"] = {
        "Proxy": {"reason": "Specialized streaming transport", "operation_ids": ["proxyRequest"]}
    }

    normalized = normalize_spec(spec, validate_spec(spec, codegen_config).skip_ids)

    assert set(normalized["paths"]) == {"/widgets/{widget_id}"}
    assert set(normalized["paths"]["/widgets/{widget_id}"]) == {"get"}

    non_cors_options = copy.deepcopy(spec)
    non_cors_options["paths"]["/widgets/{widget_id}"]["options"]["tags"] = ["Other"]
    with pytest.raises(CodegenError, match="is not tagged only as CORS"):
        validate_spec(non_cors_options, codegen_config)

    codegen_config["endpoint_generator"]["skip_tags"]["Proxy"]["operation_ids"] = []
    with pytest.raises(CodegenError, match="does not match the spec exactly"):
        validate_spec(spec, codegen_config)


def test_invalid_and_duplicate_operation_ids_fail(minimal_spec, codegen_config):
    spec = copy.deepcopy(minimal_spec)
    spec["paths"]["/widgets/{widget_id}"]["get"]["operationId"] = "get-widget"
    with pytest.raises(CodegenError, match="invalid operationId"):
        validate_spec(spec, codegen_config)

    spec = copy.deepcopy(minimal_spec)
    spec["paths"]["/other"] = copy.deepcopy(spec["paths"]["/widgets/{widget_id}"])
    spec["paths"]["/other"]["get"]["parameters"] = []
    with pytest.raises(CodegenError, match="Duplicate operationId"):
        validate_spec(spec, codegen_config)


def test_duplicate_operation_ids_fail_even_when_one_copy_is_skipped(minimal_spec, codegen_config):
    """A skipped duplicate must not silently take its supported twin out of the generated client."""
    spec = copy.deepcopy(minimal_spec)
    duplicate = copy.deepcopy(spec["paths"]["/widgets/{widget_id}"]["get"])
    duplicate["parameters"] = []
    duplicate["tags"] = ["Proxy"]
    spec["paths"]["/proxy"] = {"get": duplicate}
    codegen_config["endpoint_generator"]["skip_tags"] = {
        "Proxy": {"reason": "Specialized streaming transport", "operation_ids": ["getWidget"]}
    }

    with pytest.raises(CodegenError, match="Duplicate operationId"):
        validate_spec(spec, codegen_config)


def test_inline_operation_name_collisions_fail(minimal_spec, codegen_config):
    spec = copy.deepcopy(minimal_spec)
    second = copy.deepcopy(spec["paths"]["/widgets/{widget_id}"]["get"])
    second["operationId"] = "GetWidget"
    second["parameters"] = []
    spec["paths"]["/other"] = {"get": second}

    with pytest.raises(CodegenError, match="Inline operation name collision"):
        validate_spec(spec, codegen_config)


def test_schema_name_collisions_fail(minimal_spec, codegen_config):
    spec = copy.deepcopy(minimal_spec)
    spec["components"]["schemas"]["foo-bar"] = {"type": "string"}
    spec["components"]["schemas"]["foo_bar"] = {"type": "string"}

    with pytest.raises(CodegenError, match="Schema name collision"):
        validate_spec(spec, codegen_config)


def test_media_types_and_success_statuses_are_validated(minimal_spec, codegen_config):
    spec = copy.deepcopy(minimal_spec)
    operation = spec["paths"]["/widgets/{widget_id}"]["get"]
    operation["requestBody"] = {
        "content": {"application/xml": {"schema": {"type": "string"}}},
        "required": True,
    }
    with pytest.raises(CodegenError, match="unsupported request media type"):
        validate_spec(spec, codegen_config)

    spec = copy.deepcopy(minimal_spec)
    response = spec["paths"]["/widgets/{widget_id}"]["get"]["responses"].pop("200")
    spec["paths"]["/widgets/{widget_id}"]["get"]["responses"]["206"] = response
    with pytest.raises(CodegenError, match="unsupported success status"):
        validate_spec(spec, codegen_config)

    spec = copy.deepcopy(minimal_spec)
    content = spec["paths"]["/widgets/{widget_id}"]["get"]["responses"]["200"]["content"]
    content["application/octet-stream"] = content.pop("application/json")
    with pytest.raises(CodegenError, match="unsupported success response media type"):
        validate_spec(spec, codegen_config)


def test_referenced_parameters_resolve_and_match_path(minimal_spec, codegen_config):
    validate_spec(minimal_spec, codegen_config)

    spec = copy.deepcopy(minimal_spec)
    spec["components"]["parameters"]["WidgetId"]["schema"] = {"type": "object"}
    with pytest.raises(CodegenError, match="must be scalar"):
        validate_spec(spec, codegen_config)

    spec = copy.deepcopy(minimal_spec)
    spec["paths"]["/widgets/{widget_id}"]["get"]["parameters"][0]["$ref"] = "#/components/parameters/Missing"
    with pytest.raises(CodegenError, match="Unresolved OpenAPI reference"):
        validate_spec(spec, codegen_config)


def test_only_json_compatible_schema_types_and_values_are_supported(minimal_spec, codegen_config):
    spec = copy.deepcopy(minimal_spec)
    spec["components"]["schemas"]["Widget"]["properties"]["created"] = {"type": "date"}
    with pytest.raises(CodegenError, match="Unsupported non-JSON schema type"):
        validate_spec(spec, codegen_config)

    spec = copy.deepcopy(minimal_spec)
    spec["components"]["schemas"]["Widget"]["properties"]["count"] = {
        "type": "number",
        "default": float("nan"),
    }
    with pytest.raises(CodegenError, match="not JSON-compatible"):
        validate_spec(spec, codegen_config)


@pytest.mark.parametrize(
    "keyword,subschema",
    [
        ("patternProperties", {"^x-": {"type": "date"}}),
        ("dependentSchemas", {"name": {"type": "date"}}),
        ("propertyNames", {"type": "date"}),
        ("prefixItems", [{"type": "date"}]),
        ("if", {"type": "date"}),
        ("contains", {"type": "date"}),
    ],
)
def test_unsupported_types_are_caught_under_every_schema_keyword(minimal_spec, codegen_config, keyword, subschema):
    spec = copy.deepcopy(minimal_spec)
    spec["components"]["schemas"]["Widget"][keyword] = subschema

    with pytest.raises(CodegenError, match="Unsupported non-JSON schema type"):
        validate_spec(spec, codegen_config)


def test_malformed_specs_and_configs_raise_actionable_errors(minimal_spec, codegen_config):
    """Shape problems have to surface as CodegenError; a bare KeyError escapes the scripts' handler."""
    # A spec without any components is empty, not malformed -- it must not blow up on a missing key.
    assert validate_spec({"openapi": "3.0.3", "paths": {}}, codegen_config).schema_count == 0
    with pytest.raises(CodegenError, match="components.schemas must be an object"):
        validate_spec({"openapi": "3.0.3", "paths": {}, "components": {"schemas": []}}, codegen_config)

    spec = copy.deepcopy(minimal_spec)
    spec["paths"]["/widgets/{widget_id}"]["get"]["responses"]["200"]["content"]["application/json"] = None
    with pytest.raises(CodegenError, match="must be an object"):
        validate_spec(spec, codegen_config)

    for key in ("supported_request_media_types", "supported_response_media_types", "supported_success_statuses"):
        broken = copy.deepcopy(codegen_config)
        del broken["endpoint_generator"][key]
        with pytest.raises(CodegenError, match=f"endpoint_generator.{key} must be a non-empty list"):
            validate_spec(minimal_spec, broken)
        with pytest.raises(CodegenError, match=f"endpoint_generator.{key} must be a non-empty list"):
            validate_config(broken, check_installed_tools=False)
