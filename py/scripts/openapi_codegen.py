"""Shared validation and generation helpers for the pinned Braintrust OpenAPI spec."""

import copy
import difflib
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, NamedTuple, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "openapi" / "config.json"
SPEC_PATH = REPO_ROOT / "openapi" / "spec.json"
GENERATED_ROOT = REPO_ROOT / "py" / "src" / "braintrust" / "api" / "_generated"
HTTP_METHODS = {"delete", "get", "head", "options", "patch", "post", "put", "trace"}
OPERATION_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
JSON_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
# Keywords whose values hold nested schemas. Everything reachable through them has to be walked,
# otherwise an unsupported type hidden under, say, `patternProperties` sails past validation.
_SCHEMA_MAP_KEYWORDS = ("properties", "patternProperties", "dependentSchemas", "definitions", "$defs")
_SCHEMA_CHILD_KEYWORDS = (
    "items",
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
)
_SCHEMA_LIST_KEYWORDS = ("allOf", "anyOf", "oneOf", "prefixItems", "items")


class CodegenError(RuntimeError):
    """An actionable OpenAPI validation or generation failure."""


class ValidationReport(NamedTuple):
    operation_count: int
    options_operation_count: int
    schema_count: int
    skip_ids: FrozenSet[str]

    def __str__(self) -> str:
        return (
            f"{self.operation_count} supported operations, "
            f"{self.options_operation_count} CORS OPTIONS operations removed, "
            f"{len(self.skip_ids)} explicitly skipped operations, "
            f"{self.schema_count} schemas"
        )


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CodegenError(f"Unable to read OpenAPI config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise CodegenError(f"OpenAPI config {path} must contain a JSON object")
    return config


def verify_spec_hash(spec_bytes: bytes, config: Mapping[str, Any], source: str) -> None:
    expected_hash = _required_string(config, "spec", "sha256")
    if not SHA256_RE.fullmatch(expected_hash):
        raise CodegenError("openapi/config.json spec.sha256 must be a lowercase 64-character SHA-256")
    actual_hash = hashlib.sha256(spec_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise CodegenError(
            f"OpenAPI spec hash mismatch for {source}: expected {expected_hash}, got {actual_hash}. "
            "Update the pinned hash intentionally or run make fetch-openapi-spec."
        )


def read_and_verify_spec(config: Mapping[str, Any], path: Path = SPEC_PATH) -> Dict[str, Any]:
    try:
        spec_bytes = path.read_bytes()
    except OSError as exc:
        raise CodegenError(f"Unable to read pinned OpenAPI spec {path}: {exc}") from exc
    verify_spec_hash(spec_bytes, config, str(path))
    try:
        spec = json.loads(spec_bytes)
    except json.JSONDecodeError as exc:
        raise CodegenError(f"Pinned OpenAPI spec {path} is not valid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise CodegenError(f"Pinned OpenAPI spec {path} must contain a JSON object")
    return spec


def validate_config(config: Mapping[str, Any], check_installed_tools: bool = True) -> None:
    if config.get("schema_version") != 1:
        raise CodegenError("Unsupported openapi/config.json schema_version; expected 1")
    commit = _required_string(config, "spec", "commit")
    if not SHA_RE.fullmatch(commit):
        raise CodegenError("openapi/config.json spec.commit must be a full lowercase 40-character commit SHA")
    _required_string(config, "spec", "repository")
    _required_string(config, "spec", "path")
    flags = _model_flags(config)
    if not flags or not all(isinstance(flag, str) and flag.startswith("--") for flag in flags):
        raise CodegenError("model_generator.flags must be a non-empty list of command-line flags")
    _endpoint_config(config)

    if check_installed_tools:
        for distribution, config_key in (("datamodel-code-generator", "datamodel-code-generator"), ("ruff", "ruff")):
            expected = _required_string(config, "tools", config_key)
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise CodegenError(
                    f"Required generator tool {distribution}=={expected} is not installed; "
                    "run the make target through the api-codegen dependency group"
                ) from exc
            if actual != expected:
                raise CodegenError(f"Generator tool pin mismatch: expected {distribution}=={expected}, got {actual}")


def validate_spec(spec: Mapping[str, Any], config: Mapping[str, Any]) -> ValidationReport:
    if not isinstance(spec.get("paths"), dict):
        raise CodegenError("OpenAPI spec must define a paths object")
    components = spec.get("components", {})
    if not isinstance(components, dict) or not isinstance(components.get("schemas", {}), dict):
        raise CodegenError("OpenAPI spec components.schemas must be an object")
    schemas = components.get("schemas", {})

    _validate_refs(spec)
    operations = list(_iter_operations(spec))
    # Uniqueness is checked before the skip set is resolved: skipped and supported operations share
    # one operationId namespace, so a collision between them would otherwise drop the supported
    # operation from the normalized spec without any error.
    _validate_unique_operation_ids(operations)
    endpoint = _endpoint_config(config)
    skip_ids = _validate_skip_set(operations, endpoint)

    generated_names: Dict[str, str] = {}
    supported_count = 0
    options_count = 0
    for method, path, operation_id, operation, path_item in operations:
        if method == "options":
            if operation.get("tags") != ["CORS"]:
                raise CodegenError(
                    f"OPTIONS {path} is not tagged only as CORS and cannot be removed during normalization"
                )
            options_count += 1
            continue
        if operation_id in skip_ids:
            continue
        if not operation_id or not OPERATION_ID_RE.fullmatch(operation_id):
            raise CodegenError(f"Operation {method.upper()} {path} has an invalid operationId: {operation_id!r}")
        tags = operation.get("tags")
        if not isinstance(tags, list) or len(tags) != 1 or not isinstance(tags[0], str) or not tags[0].strip():
            raise CodegenError(f"Operation {operation_id!r} must have exactly one usable tag")
        generated_name = _python_type_name(operation_id)
        previous = generated_names.setdefault(generated_name, operation_id)
        if previous != operation_id:
            raise CodegenError(
                f"Inline operation name collision: {previous!r} and {operation_id!r} both generate {generated_name!r}"
            )
        _validate_operation_media(operation_id, operation, endpoint, spec)
        _validate_path_parameters(operation_id, path, path_item, operation, spec)
        supported_count += 1

    _validate_component_names(schemas)
    _validate_json_values_and_types(spec)
    return ValidationReport(supported_count, options_count, len(schemas), frozenset(skip_ids))


def normalize_spec(spec: Mapping[str, Any], skip_ids: FrozenSet[str]) -> Dict[str, Any]:
    """Remove only CORS OPTIONS and the exact configured skip set."""
    normalized = copy.deepcopy(spec)
    for path, path_item in list(normalized["paths"].items()):
        for method, operation in list(path_item.items()):
            lower_method = method.lower()
            if lower_method in HTTP_METHODS and (
                lower_method == "options" or operation.get("operationId") in skip_ids
            ):
                del path_item[method]
        if not any(key.lower() in HTTP_METHODS for key in path_item):
            del normalized["paths"][path]
    return normalized


def generate_tree(output_root: Path, config: Mapping[str, Any], spec: Mapping[str, Any]) -> ValidationReport:
    validate_config(config)
    report = validate_spec(spec, config)
    normalized = normalize_spec(spec, report.skip_ids)
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_path = output_root.parent / "normalized-spec.json"
    normalized_path.write_text(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    try:
        _generate_models(normalized_path, output_root / "models.py", config)
    finally:
        normalized_path.unlink(missing_ok=True)
    _write_generated_file(output_root / "__init__.py", _GENERATED_INIT_BODY, config)
    return report


def compare_generated(expected_root: Path, actual_root: Path) -> List[str]:
    expected_files = _generated_source_files(expected_root)
    actual_files = _generated_source_files(actual_root)
    differences: List[str] = []
    for relative in sorted(expected_files | actual_files):
        expected = expected_root / relative
        actual = actual_root / relative
        if not expected.exists():
            differences.append(f"Unexpected generated file: {relative}")
            continue
        if not actual.exists():
            differences.append(f"Missing generated file: {relative}")
            continue
        if expected.read_bytes() == actual.read_bytes():
            continue
        differences.extend(
            difflib.unified_diff(
                actual.read_text(encoding="utf-8").splitlines(keepends=True),
                expected.read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile=f"committed/{relative}",
                tofile=f"regenerated/{relative}",
            )
        )
    return differences


def _generated_source_files(root: Path) -> Set[Path]:
    """Every committed file in a generated tree, not just ``*.py``.

    Restricting this to ``*.py`` would let a file the generator has stopped emitting survive both
    the drift check and the tree replacement, so a stale artifact could sit in the committed tree
    while ``check-api-client-codegen`` reported it as current.
    """
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith(".")
    }


def atomic_replace_tree(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    source_files = _generated_source_files(source_root)
    destination_files = _generated_source_files(destination_root)
    for relative in destination_files - source_files:
        (destination_root / relative).unlink()
    for relative in source_files:
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination)
    _prune_empty_directories(destination_root)


def _prune_empty_directories(root: Path) -> None:
    """Drop directories the generator no longer populates, deepest first.

    A directory left behind after its last module is removed still imports as a namespace package,
    so emptied packages have to go with their files.
    """
    directories = sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda p: len(p.parts), reverse=True)
    for directory in directories:
        if directory.name == "__pycache__":
            continue
        if any(child.name != "__pycache__" for child in directory.iterdir()):
            continue
        shutil.rmtree(directory)


def _generate_models(spec_path: Path, output_path: Path, config: Mapping[str, Any]) -> None:
    placeholder = "CONTENT_HASH_PLACEHOLDER"
    header = _generated_header(config, placeholder).rstrip()
    command = [
        sys.executable,
        "-m",
        "datamodel_code_generator",
        "--input",
        str(spec_path),
        "--output",
        str(output_path),
        *_model_flags(config),
        "--custom-file-header",
        header,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise CodegenError(f"datamodel-code-generator failed: {detail}") from exc
    # datamodel-code-generator's own `--formatters=ruff-format` pass is not a fixed point; running
    # the pinned ruff again is what makes the committed output stable.
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(output_path)], check=True, capture_output=True, text=True
    )
    generated = output_path.read_text(encoding="utf-8")
    marker = f"# Content SHA-256: {placeholder}"
    if marker not in generated:
        raise CodegenError("Generated models did not contain the expected content hash marker")
    body = generated.split(marker, 1)[1].lstrip("\n")
    content_hash = hashlib.sha256(body.encode()).hexdigest()
    # Substituting the hash only rewrites characters inside a comment, so the file stays formatted.
    _write_checked(output_path, generated.replace(placeholder, content_hash, 1))


def _write_generated_file(path: Path, body: str, config: Mapping[str, Any]) -> None:
    content_hash = hashlib.sha256(body.encode()).hexdigest()
    _write_checked(path, _generated_header(config, content_hash) + "\n\n" + body)


def _write_checked(path: Path, text: str) -> None:
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def _generated_header(config: Mapping[str, Any], content_hash: str) -> str:
    return "\n".join(
        [
            "# Generated by scripts/generate-api-client.py. DO NOT EDIT.",
            f"# OpenAPI commit: {_required_string(config, 'spec', 'commit')}",
            f"# OpenAPI spec SHA-256: {_required_string(config, 'spec', 'sha256')}",
            f"# datamodel-code-generator: {_required_string(config, 'tools', 'datamodel-code-generator')}",
            f"# ruff: {_required_string(config, 'tools', 'ruff')}",
            f"# Generator Python: {_required_string(config, 'tools', 'python')}",
            f"# Content SHA-256: {content_hash}",
        ]
    )


_GENERATED_INIT_BODY = '''"""Private generated REST API implementation.

Import submodules explicitly (``from braintrust.api._generated import models``); importing this
package pulls in no models.
"""
'''


def _iter_operations(
    spec: Mapping[str, Any],
) -> Iterator[Tuple[str, str, Any, Mapping[str, Any], Mapping[str, Any]]]:
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            raise CodegenError(f"Path item {path!r} must be an object")
        for method, operation in path_item.items():
            lower_method = method.lower()
            if lower_method not in HTTP_METHODS:
                continue
            if not isinstance(operation, dict):
                raise CodegenError(f"Operation {method.upper()} {path} must be an object")
            yield lower_method, path, operation.get("operationId"), operation, path_item


def _endpoint_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    endpoint = config.get("endpoint_generator")
    if not isinstance(endpoint, dict) or endpoint.get("schema_version") != 1:
        raise CodegenError("Unsupported endpoint_generator schema_version; expected 1")
    if not isinstance(endpoint.get("skip_tags"), dict):
        raise CodegenError("endpoint_generator.skip_tags must be an object")
    for key in ("supported_request_media_types", "supported_response_media_types", "supported_success_statuses"):
        values = endpoint.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
            raise CodegenError(f"Config key endpoint_generator.{key} must be a non-empty list of strings")
    return endpoint


def _validate_unique_operation_ids(
    operations: Sequence[Tuple[str, str, Any, Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    seen: Dict[str, str] = {}
    for method, path, operation_id, _, _ in operations:
        if method == "options" or operation_id is None:
            continue
        location = f"{method.upper()} {path}"
        previous = seen.setdefault(operation_id, location)
        if previous != location:
            raise CodegenError(f"Duplicate operationId {operation_id!r} on {previous} and {location}")


def _validate_skip_set(
    operations: Sequence[Tuple[str, str, Any, Mapping[str, Any], Mapping[str, Any]]], endpoint: Mapping[str, Any]
) -> Set[str]:
    skip_tags = endpoint["skip_tags"]
    configured_ids: Set[str] = set()
    operation_tags: Dict[str, Set[str]] = {}
    for method, path, operation_id, operation, _ in operations:
        if method == "options":
            continue
        if operation_id is not None:
            operation_tags[operation_id] = set(operation.get("tags", []))
    for tag, skip_config in skip_tags.items():
        if not isinstance(tag, str) or not isinstance(skip_config, dict):
            raise CodegenError("Each endpoint_generator.skip_tags entry must be an object keyed by a tag")
        reason = skip_config.get("reason")
        ids = skip_config.get("operation_ids")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(ids, list)
            or not all(isinstance(value, str) for value in ids)
        ):
            raise CodegenError(f"Skip tag {tag!r} must have a reason and an operation_ids list")
        if len(ids) != len(set(ids)):
            raise CodegenError(f"Skip tag {tag!r} contains duplicate operation IDs")
        actual_ids = {operation_id for operation_id, tags in operation_tags.items() if tag in tags}
        expected_ids = set(ids)
        if actual_ids != expected_ids:
            missing = sorted(actual_ids - expected_ids)
            stale = sorted(expected_ids - actual_ids)
            raise CodegenError(
                f"Skip tag {tag!r} does not match the spec exactly; unlisted={missing}, stale={stale}. "
                "Update the explicit skip set and review each operation."
            )
        overlap = configured_ids & expected_ids
        if overlap:
            raise CodegenError(f"Operations occur in more than one skip tag: {', '.join(sorted(overlap))}")
        configured_ids.update(expected_ids)
    return configured_ids


def _validate_refs(spec: Mapping[str, Any]) -> None:
    for value in _walk_values(spec):
        if not isinstance(value, dict) or "$ref" not in value:
            continue
        reference = value["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise CodegenError(f"Only local OpenAPI references are supported, got {reference!r}")
        _resolve_ref(reference, spec)


def _resolve_ref(reference: str, spec: Mapping[str, Any]) -> Any:
    current: Any = spec
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise CodegenError(f"Unresolved OpenAPI reference {reference!r}")
        current = current[part]
    return current


def _resolve_object(value: Mapping[str, Any], spec: Mapping[str, Any]) -> Mapping[str, Any]:
    seen: Set[str] = set()
    while "$ref" in value:
        reference = value["$ref"]
        if reference in seen:
            raise CodegenError(f"Cyclic direct OpenAPI reference {reference!r}")
        seen.add(reference)
        resolved = _resolve_ref(reference, spec)
        if not isinstance(resolved, dict):
            raise CodegenError(f"OpenAPI reference {reference!r} must resolve to an object")
        value = resolved
    return value


def _validate_operation_media(
    operation_id: str, operation: Mapping[str, Any], endpoint: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    request_body = operation.get("requestBody")
    if request_body is not None:
        if not isinstance(request_body, dict):
            raise CodegenError(f"Operation {operation_id!r} has an invalid request body")
        request_body = _resolve_object(request_body, spec)
        content = request_body.get("content", {})
        if not isinstance(content, dict) or not content:
            raise CodegenError(f"Operation {operation_id!r} request body must define content")
        unsupported = set(content) - set(endpoint["supported_request_media_types"])
        if unsupported:
            raise CodegenError(
                f"Operation {operation_id!r} has unsupported request media type(s): {sorted(unsupported)}"
            )

    responses = operation.get("responses")
    if not isinstance(responses, dict):
        raise CodegenError(f"Operation {operation_id!r} must define responses")
    success_count = 0
    for status, response in responses.items():
        status_string = str(status)
        if not status_string.startswith("2"):
            continue
        success_count += 1
        if status_string not in endpoint["supported_success_statuses"]:
            raise CodegenError(f"Operation {operation_id!r} has unsupported success status {status_string}")
        if not isinstance(response, dict):
            raise CodegenError(f"Operation {operation_id!r} response {status_string} must be an object")
        response = _resolve_object(response, spec)
        content = response.get("content", {})
        if status_string == "204" and not content:
            continue
        if not isinstance(content, dict) or not content:
            raise CodegenError(f"Operation {operation_id!r} response {status_string} must define content")
        unsupported = set(content) - set(endpoint["supported_response_media_types"])
        if unsupported:
            raise CodegenError(
                f"Operation {operation_id!r} has unsupported success response media type(s): {sorted(unsupported)}"
            )
        for media_type, media in content.items():
            if not isinstance(media, dict):
                raise CodegenError(
                    f"Operation {operation_id!r} response {status_string} media type {media_type!r} must be an object"
                )
            if media_type == "text/plain":
                schema = _resolve_object(media.get("schema", {}), spec)
                if schema.get("type") != "string":
                    raise CodegenError(f"Operation {operation_id!r} text/plain success response must be a string")
    if not success_count:
        raise CodegenError(f"Operation {operation_id!r} has no supported success response")


def _validate_path_parameters(
    operation_id: str,
    path: str,
    path_item: Mapping[str, Any],
    operation: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    template_names = set(re.findall(r"\{([^{}]+)\}", path))
    path_parameters: Dict[str, Mapping[str, Any]] = {}
    for parameter in [*path_item.get("parameters", []), *operation.get("parameters", [])]:
        if not isinstance(parameter, dict):
            raise CodegenError(f"Operation {operation_id!r} has an invalid parameter")
        parameter = _resolve_object(parameter, spec)
        if parameter.get("in") == "path":
            name = parameter.get("name")
            if not isinstance(name, str):
                raise CodegenError(f"Operation {operation_id!r} has a path parameter without a name")
            path_parameters[name] = parameter
    if template_names != set(path_parameters):
        raise CodegenError(
            f"Operation {operation_id!r} path template/parameter mismatch: "
            f"template={sorted(template_names)}, declared={sorted(path_parameters)}"
        )
    for name, parameter in path_parameters.items():
        if parameter.get("required") is not True:
            raise CodegenError(f"Operation {operation_id!r} path parameter {name!r} must be required")
        schema = parameter.get("schema")
        if not isinstance(schema, dict):
            raise CodegenError(f"Operation {operation_id!r} path parameter {name!r} must define a schema")
        schema = _resolve_object(schema, spec)
        if schema.get("type") not in {"boolean", "integer", "number", "string"}:
            raise CodegenError(f"Operation {operation_id!r} path parameter {name!r} must be scalar")


def _validate_component_names(schemas: Mapping[str, Any]) -> None:
    names: Dict[str, str] = {}
    for schema_name in schemas:
        generated_name = _python_type_name(schema_name)
        previous = names.setdefault(generated_name, schema_name)
        if previous != schema_name:
            raise CodegenError(
                f"Schema name collision: {previous!r} and {schema_name!r} both generate {generated_name!r}"
            )


def _validate_json_values_and_types(spec: Mapping[str, Any]) -> None:
    """Validate every schema in the spec: the named components plus every inline ``schema`` value."""
    schema_roots: List[Mapping[str, Any]] = list(spec.get("components", {}).get("schemas", {}).values())
    schema_roots.extend(
        value["schema"]
        for value in _walk_values(spec)
        if isinstance(value, dict) and isinstance(value.get("schema"), dict)
    )
    for schema_root in schema_roots:
        _validate_schema_values(schema_root)


def _validate_schema_values(schema: Mapping[str, Any]) -> None:
    schema_type = schema.get("type")
    if schema_type is not None and schema_type not in JSON_SCHEMA_TYPES:
        raise CodegenError(f"Unsupported non-JSON schema type {schema_type!r}")
    for key in ("default", "enum", "example"):
        if key in schema:
            try:
                json.dumps(schema[key], allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise CodegenError(f"Schema {key} value is not JSON-compatible: {schema[key]!r}") from exc

    children: List[Any] = []
    for key in _SCHEMA_MAP_KEYWORDS:
        mapping = schema.get(key)
        if isinstance(mapping, dict):
            children.extend(mapping.values())
    for key in _SCHEMA_CHILD_KEYWORDS:
        child = schema.get(key)
        if isinstance(child, dict):
            children.append(child)
    # `items` is a single schema in 3.0 and may be a tuple of schemas in 3.1, so it appears in both
    # the single-child and the list keyword sets.
    for key in _SCHEMA_LIST_KEYWORDS:
        child_list = schema.get(key)
        if isinstance(child_list, list):
            children.extend(child_list)
    for child in children:
        if isinstance(child, dict):
            _validate_schema_values(child)


def _walk_values(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)


def _python_type_name(value: str) -> str:
    words = [word for word in re.split(r"[^A-Za-z0-9]+", value) if word]
    if not words:
        return "Model"
    return "".join(word[:1].upper() + word[1:] for word in words)


def _model_flags(config: Mapping[str, Any]) -> Sequence[str]:
    flags = config.get("model_generator", {}).get("flags")
    if not isinstance(flags, list):
        raise CodegenError("Missing required config key model_generator.flags")
    return flags


def _required_string(config: Mapping[str, Any], *path: str) -> str:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise CodegenError(f"Missing required config key {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, str) or not value:
        raise CodegenError(f"Config key {'.'.join(path)} must be a non-empty string")
    return value
