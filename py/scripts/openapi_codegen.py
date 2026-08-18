"""Shared validation and generation helpers for the pinned Braintrust OpenAPI spec."""

import ast
import copy
import difflib
import hashlib
import importlib.metadata
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Sequence, Set, Tuple


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
    schema_count: int

    def __str__(self) -> str:
        return f"{self.operation_count} selected operations, {self.schema_count} reachable schemas"


class GeneratedParameter(NamedTuple):
    argument_name: str
    name: str
    location: str
    type_name: str
    required: bool


class GeneratedOperation(NamedTuple):
    operation_id: str
    constant_name: str
    method: str
    path: str
    tag: str
    parameters: Tuple[GeneratedParameter, ...]
    request_body_type: str | None
    request_body_required: bool
    response_type: str | None
    success_statuses: Tuple[int, ...]
    json_success_statuses: Tuple[int, ...]
    retry_mode: str


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

    endpoint = _endpoint_config(config)
    all_operations = list(_iter_operations(spec))
    _validate_unique_operation_ids(all_operations)
    operations = _selected_operations(all_operations, endpoint["generated_tags"])
    reference_roots = []
    for _, _, _, operation, path_item in operations:
        reference_roots.append(operation)
        reference_roots.extend(path_item.get("parameters", []))
    _validate_refs(reference_roots, spec)

    generated_names: Dict[str, str] = {}
    method_identifiers: Dict[str, str] = {}
    constant_identifiers: Dict[str, str] = {}
    for method, path, operation_id, operation, path_item in operations:
        if not operation_id or not OPERATION_ID_RE.fullmatch(operation_id):
            raise CodegenError(f"Operation {method.upper()} {path} has an invalid operationId: {operation_id!r}")
        tags = operation.get("tags")
        if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag.strip() for tag in tags):
            raise CodegenError(f"Operation {operation_id!r} must have usable tags")
        if len(_generated_operation_tags(operation, endpoint["generated_tags"])) != 1:
            raise CodegenError(f"Operation {operation_id!r} must have exactly one generated OpenAPI tag")
        generated_name = _python_type_name(operation_id)
        previous = generated_names.setdefault(generated_name, operation_id)
        if previous != operation_id:
            raise CodegenError(
                f"Inline operation name collision: {previous!r} and {operation_id!r} both generate {generated_name!r}"
            )
        for namespace, identifier, seen in (
            ("method", _snake_case(operation_id), method_identifiers),
            ("constant", _snake_case(operation_id).upper(), constant_identifiers),
        ):
            previous = seen.setdefault(identifier, operation_id)
            if previous != operation_id:
                raise CodegenError(
                    f"Generated operation identifier collision: {previous!r} and {operation_id!r} "
                    f"both emit {namespace} {identifier!r}"
                )
        _validate_operation_media(operation_id, operation, endpoint, spec)
        _validate_parameters(operation_id, path, path_item, operation, spec)

    _validate_selected_operations(operations, endpoint)
    operation_ids = {operation_id for _, _, operation_id, _, _ in operations}
    selected_spec = _slice_model_spec(spec, operation_ids)
    schemas = selected_spec.get("components", {}).get("schemas", {})
    _validate_component_names(schemas)
    _validate_json_values_and_types(selected_spec)
    return ValidationReport(len(operations), len(schemas))


def _generated_operation_tags(operation: Mapping[str, Any], generated_tags: Sequence[str]) -> List[str]:
    tags = operation.get("tags")
    if not isinstance(tags, list):
        return []
    selected_tags = set(generated_tags)
    return [tag for tag in tags if isinstance(tag, str) and tag in selected_tags]


def _selected_operations(
    operations: Sequence[Tuple[str, str, Any, Mapping[str, Any], Mapping[str, Any]]],
    generated_tags: Sequence[str],
) -> List[Tuple[str, str, Any, Mapping[str, Any], Mapping[str, Any]]]:
    return [
        operation_entry
        for operation_entry in operations
        if operation_entry[0] != "options" and _generated_operation_tags(operation_entry[3], generated_tags)
    ]


def _slice_model_spec(spec: Mapping[str, Any], operation_ids: Set[str]) -> Dict[str, Any]:
    """Keep selected operations and the transitive component closure they reference."""
    selected_paths: Dict[str, Any] = {}
    for path, path_item in spec.get("paths", {}).items():
        selected_item = {
            key: copy.deepcopy(value)
            for key, value in path_item.items()
            if key.lower() not in HTTP_METHODS or value.get("operationId") in operation_ids
        }
        if any(key.lower() in HTTP_METHODS for key in selected_item):
            selected_paths[path] = selected_item

    selected_components: Dict[str, Dict[str, Any]] = {}
    seen_components: Set[Tuple[str, str]] = set()

    def collect_components(value: Any) -> None:
        if isinstance(value, dict):
            reference = value.get("$ref")
            component_key = _component_key_from_ref(reference) if isinstance(reference, str) else None
            if component_key is not None and component_key not in seen_components:
                seen_components.add(component_key)
                component_type, component_name = component_key
                components = spec.get("components", {})
                component_group = components.get(component_type, {})
                if not isinstance(component_group, dict) or component_name not in component_group:
                    raise CodegenError(f"Unresolved OpenAPI reference {reference!r}")
                component = component_group[component_name]
                selected_components.setdefault(component_type, {})[component_name] = copy.deepcopy(component)
                collect_components(component)
            for key, child in value.items():
                if key != "$ref":
                    collect_components(child)
        elif isinstance(value, list):
            for child in value:
                collect_components(child)

    collect_components(selected_paths)
    model_spec = {key: copy.deepcopy(spec[key]) for key in ("openapi", "info", "jsonSchemaDialect") if key in spec}
    model_spec["paths"] = selected_paths
    model_spec["components"] = selected_components
    return model_spec


def _component_key_from_ref(reference: str) -> Tuple[str, str] | None:
    prefix = "#/components/"
    if not reference.startswith(prefix):
        return None
    parts = reference[len(prefix) :].split("/", 2)
    if len(parts) < 2:
        return None
    return tuple(part.replace("~1", "/").replace("~0", "~") for part in parts[:2])


def _with_inline_models(
    spec: Mapping[str, Any], inline_models: Sequence[Tuple[str, Mapping[str, Any]]]
) -> Dict[str, Any]:
    """Expose named inline response schemas to the existing model generator."""
    model_spec = copy.deepcopy(spec)
    schemas = model_spec.setdefault("components", {}).setdefault("schemas", {})
    generated_names = {_python_type_name(name): name for name in schemas}
    for name, schema in inline_models:
        generated_name = _python_type_name(name)
        existing_name = generated_names.get(generated_name)
        if existing_name is not None:
            raise CodegenError(
                f"Inline response model {name!r} collides with component schema {existing_name!r}; "
                f"both generate Python type {generated_name!r}"
            )
        schemas[name] = copy.deepcopy(schema)
        generated_names[generated_name] = name
    return model_spec


_NON_MODEL_ANNOTATION_NAMES = {"Any", "Literal", "Mapping", "None", "Sequence"}


def _operation_annotation_names(operation: GeneratedOperation) -> Set[str]:
    annotation_names: Set[str] = set()
    for type_name in [
        operation.request_body_type,
        operation.response_type,
        *(parameter.type_name for parameter in operation.parameters),
    ]:
        if type_name:
            annotation_names.update(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", type_name))
    return annotation_names


def _operation_model_roots(operations: Sequence[GeneratedOperation]) -> Dict[str, Set[str]]:
    roots: Dict[str, Set[str]] = {}
    for operation in operations:
        roots.setdefault(operation.tag, set()).update(
            _operation_annotation_names(operation) - _NON_MODEL_ANNOTATION_NAMES
        )
    return roots


def _partition_model_source(
    source: str, operations: Sequence[GeneratedOperation]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Partition one deterministic model-generator output by resource dependency closure.

    Definitions reached by more than one generated tag live in ``common.py``. Resource-specific
    modules import those shared definitions explicitly, avoiding duplicate runtime type identities.
    """
    tree = ast.parse(source)
    imports: List[ast.stmt] = []
    definitions: List[Tuple[str, List[ast.stmt]]] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
            continue
        if isinstance(node, ast.ClassDef):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if definitions:
                definitions[-1][1].append(node)
            continue
        else:
            raise CodegenError(f"Unsupported generated model statement: {type(node).__name__}")
        if len(names) != 1:
            raise CodegenError("Generated model definitions must bind exactly one public name")
        definitions.append((names[0], [node]))

    definition_names = {name for name, _ in definitions}
    dependencies: Dict[str, Set[str]] = {}
    for name, nodes in definitions:
        dependencies[name] = {
            child.id
            for node in nodes
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in definition_names and child.id != name
        }

    owners: Dict[str, Set[str]] = {name: set() for name in definition_names}
    for tag, roots in _operation_model_roots(operations).items():
        pending = list(roots)
        seen: Set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            if name not in definition_names:
                raise CodegenError(f"Generated resource {tag!r} references unknown model {name!r}")
            seen.add(name)
            owners[name].add(tag)
            pending.extend(dependencies[name])

    unreachable = sorted(name for name, tags in owners.items() if not tags)
    if unreachable:
        raise CodegenError(f"Generated models are unreachable from resource methods: {unreachable}")

    common_names = {name for name, tags in owners.items() if len(tags) > 1}
    module_for_name = {
        name: "common" if name in common_names else _snake_case(next(iter(tags))) for name, tags in owners.items()
    }

    def source_for(node: ast.stmt) -> str:
        segment = ast.get_source_segment(source, node)
        if segment is None:
            raise CodegenError(f"Could not recover generated model source for {type(node).__name__}")
        return segment

    import_source = "\n".join(source_for(node) for node in imports)
    bodies: Dict[str, List[str]] = {}
    for name, nodes in definitions:
        bodies.setdefault(module_for_name[name], []).append("\n".join(source_for(node) for node in nodes))

    modules: Dict[str, str] = {}
    for module, blocks in sorted(bodies.items()):
        common_imports = sorted(
            dependency
            for name, _ in definitions
            if module_for_name[name] == module
            for dependency in dependencies[name]
            if dependency in common_names
        )
        sections = [import_source]
        if common_imports and module != "common":
            sections.append(f"from .common import {', '.join(dict.fromkeys(common_imports))}")
        sections.append("\n\n".join(blocks))
        modules[module] = "\n\n".join(section for section in sections if section) + "\n"
    return modules, module_for_name


def generate_tree(output_root: Path, config: Mapping[str, Any], spec: Mapping[str, Any]) -> ValidationReport:
    validate_config(config)
    report = validate_spec(spec, config)
    operations, inline_models = _collect_generated_operations(spec, config)
    selected_spec = _slice_model_spec(spec, {operation.operation_id for operation in operations})
    model_spec = _with_inline_models(selected_spec, inline_models)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_spec_path = output_root.parent / "selected-spec.json"
    monolithic_models_path = output_root.parent / "models.py"
    selected_spec_path.write_text(
        json.dumps(model_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    try:
        _generate_models(selected_spec_path, monolithic_models_path, config)
        model_sources, model_modules = _partition_model_source(monolithic_models_path.read_text(), operations)
    finally:
        selected_spec_path.unlink(missing_ok=True)
        monolithic_models_path.unlink(missing_ok=True)
    model_paths = []
    for module, body in model_sources.items():
        model_path = output_root / "models" / f"{module}.py"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _write_generated_file(model_path, body, config)
        model_paths.append(model_path)
    _write_generated_file(output_root / "__init__.py", _GENERATED_INIT_BODY, config)
    model_init_path = output_root / "models" / "__init__.py"
    _write_generated_file(model_init_path, _model_package_source(model_modules), config)
    resource_files = _generate_resources(output_root, operations, model_modules, config)
    _format_generated_files([*model_paths, model_init_path, *resource_files])
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = _generated_header(config, "CONTENT_HASH_PLACEHOLDER").rstrip()
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
    if not output_path.is_file():
        raise CodegenError("datamodel-code-generator did not emit the model module")
    model_paths = [output_path]
    # datamodel-code-generator's own formatter pass is not a fixed point; one pinned Ruff pass over
    # the complete module tree makes the committed output stable and finalizes each content hash.
    _format_generated_files(model_paths)


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


def _model_package_source(model_modules: Mapping[str, str]) -> str:
    by_module: Dict[str, List[str]] = {}
    for name, module in model_modules.items():
        by_module.setdefault(module, []).append(name)

    lines = ['"""Generated private model types with stable package-level imports."""', ""]
    for module, names in sorted(by_module.items()):
        lines.append(f"from .{module} import {', '.join(sorted(names))}")
    lines.extend(["", "", "__all__ = ["])
    lines.extend(f"    {name!r}," for name in sorted(model_modules))
    lines.extend(["]", ""])
    return "\n".join(lines)


def _validate_selected_operations(
    operations: Sequence[Tuple[str, str, Any, Mapping[str, Any], Mapping[str, Any]]],
    endpoint: Mapping[str, Any],
) -> None:
    supported = {operation_id: operation for _, _, operation_id, operation, _ in operations}
    configured_tags = set(endpoint["generated_tags"])
    actual_tags = {
        tag
        for operation in supported.values()
        for tag in _generated_operation_tags(operation, endpoint["generated_tags"])
    }
    missing_tags = configured_tags - actual_tags
    if missing_tags:
        raise CodegenError(f"endpoint_generator.generated_tags contains unknown tags: {sorted(missing_tags)}")

    safe_reads = set(endpoint["safe_reads"])
    stale_safe_reads = safe_reads - set(supported)
    if stale_safe_reads:
        operation_id = sorted(stale_safe_reads)[0]
        raise CodegenError(f"endpoint_generator.safe_reads references non-generated operation {operation_id!r}")
    non_post_safe_reads = sorted(
        operation_id for method, _, operation_id, _, _ in operations if operation_id in safe_reads and method != "post"
    )
    if non_post_safe_reads:
        raise CodegenError(
            f"endpoint_generator.safe_reads must reference POST operations; got {non_post_safe_reads[0]!r}"
        )

    idempotent_writes = set(endpoint["idempotent_writes"])
    stale_idempotent_writes = idempotent_writes - set(supported)
    if stale_idempotent_writes:
        operation_id = sorted(stale_idempotent_writes)[0]
        raise CodegenError(f"endpoint_generator.idempotent_writes references non-generated operation {operation_id!r}")
    overlapping_retry_modes = safe_reads & idempotent_writes
    if overlapping_retry_modes:
        operation_id = sorted(overlapping_retry_modes)[0]
        raise CodegenError(
            f"Operation {operation_id!r} cannot appear in both endpoint_generator.safe_reads "
            "and endpoint_generator.idempotent_writes"
        )
    non_writes = sorted(
        operation_id
        for method, _, operation_id, _, _ in operations
        if operation_id in idempotent_writes and method in {"get", "head"}
    )
    if non_writes:
        raise CodegenError(f"endpoint_generator.idempotent_writes references read operation {non_writes[0]!r}")


def _operation_retry_mode(method: str, operation_id: str, safe_reads: Set[str], idempotent_writes: Set[str]) -> str:
    if method in {"get", "head"} or operation_id in safe_reads:
        return "SAFE_READ"
    if operation_id in idempotent_writes:
        return "IDEMPOTENT_WRITE"
    return "NONE"


def _collect_generated_operations(
    spec: Mapping[str, Any], config: Mapping[str, Any]
) -> Tuple[List[GeneratedOperation], List[Tuple[str, Mapping[str, Any]]]]:
    endpoint = _endpoint_config(config)
    safe_reads = set(endpoint["safe_reads"])
    idempotent_writes = set(endpoint["idempotent_writes"])
    operations: List[GeneratedOperation] = []
    inline_models: Dict[str, Mapping[str, Any]] = {}
    for method, path, operation_id, operation, path_item in _iter_operations(spec):
        operation_generated_tags = _generated_operation_tags(operation, endpoint["generated_tags"])
        if method == "options" or not operation_generated_tags:
            continue
        parameters = _operation_parameters(path_item, operation, spec)
        request_body_type, request_body_required = _operation_request_body(operation, spec)
        response_type, statuses, json_statuses, inline_schema = _operation_response(operation_id, operation, spec)
        if inline_schema is not None:
            inline_model_name = _python_type_name(operation_id + "Response")
            previous_schema = inline_models.setdefault(inline_model_name, inline_schema)
            if previous_schema != inline_schema:
                raise CodegenError(f"Inline response model {inline_model_name!r} has conflicting schemas")
        operations.append(
            GeneratedOperation(
                operation_id=operation_id,
                constant_name=_snake_case(operation_id).upper(),
                method=method.upper(),
                path=path,
                tag=operation_generated_tags[0],
                parameters=tuple(parameters),
                request_body_type=request_body_type,
                request_body_required=request_body_required,
                response_type=response_type,
                success_statuses=statuses,
                json_success_statuses=json_statuses,
                retry_mode=_operation_retry_mode(method, operation_id, safe_reads, idempotent_writes),
            )
        )
    return operations, list(inline_models.items())


def _operation_parameters(
    path_item: Mapping[str, Any], operation: Mapping[str, Any], spec: Mapping[str, Any]
) -> List[GeneratedParameter]:
    by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for raw_parameter in [*path_item.get("parameters", []), *operation.get("parameters", [])]:
        parameter = _resolve_object(raw_parameter, spec)
        location = parameter.get("in")
        name = parameter.get("name")
        if not isinstance(location, str) or not isinstance(name, str):
            raise CodegenError("Generated operation parameter must have string in and name fields")
        by_key[(location, name)] = parameter
    generated = []
    for (location, name), parameter in by_key.items():
        generated.append(
            GeneratedParameter(
                argument_name=_python_argument_name(name),
                name=name,
                location=location,
                type_name=_schema_annotation(parameter.get("schema", {}), spec),
                required=bool(parameter.get("required", False)),
            )
        )
    return generated


def _operation_request_body(operation: Mapping[str, Any], spec: Mapping[str, Any]) -> Tuple[str | None, bool]:
    request_body = operation.get("requestBody")
    if request_body is None:
        return None, False
    request_body = _resolve_object(request_body, spec)
    media = request_body["content"]["application/json"]
    return _schema_annotation(media.get("schema", {}), spec), bool(request_body.get("required", False))


def _operation_response(
    operation_id: str,
    operation: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> Tuple[str | None, Tuple[int, ...], Tuple[int, ...], Mapping[str, Any] | None]:
    successes = [
        (int(status), response) for status, response in operation["responses"].items() if str(status).startswith("2")
    ]
    successes.sort(key=lambda item: item[0])
    response_types: List[str | None] = []
    json_statuses: List[int] = []
    inline_schema: Mapping[str, Any] | None = None
    for status, raw_response in successes:
        response = _resolve_object(raw_response, spec)
        content = response.get("content", {})
        if not content:
            response_types.append(None)
            continue
        media_type = "application/json" if "application/json" in content else sorted(content)[0]
        json_statuses.append(status)
        schema = content[media_type].get("schema", {})
        if "$ref" in schema:
            response_type = _schema_annotation(schema, spec)
        elif schema:
            response_type = _python_type_name(operation_id + "Response")
            if inline_schema is not None and inline_schema != schema:
                raise CodegenError(f"Operation {operation_id!r} has conflicting inline success response schemas")
            inline_schema = schema
        else:
            response_type = "Any"
        response_types.append(response_type)
    unique_types = list(dict.fromkeys(response_types))
    if len(unique_types) == 1:
        response_type = unique_types[0]
    elif len(unique_types) == 2 and None in unique_types:
        response_type = f"{next(type_name for type_name in unique_types if type_name is not None)} | None"
    else:
        response_type = "Any"
    return (
        response_type,
        tuple(status for status, _ in successes),
        tuple(json_statuses),
        inline_schema if response_type != "Any" else None,
    )


def _schema_annotation(schema: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    if "$ref" in schema:
        return _python_type_name(str(schema["$ref"]).rsplit("/", 1)[-1])
    resolved = _resolve_object(schema, spec)
    if "oneOf" in resolved or "anyOf" in resolved:
        choices = resolved.get("oneOf", resolved.get("anyOf", []))
        annotation = " | ".join(_schema_annotation(choice, spec) for choice in choices) or "Any"
    elif "allOf" in resolved:
        choices = resolved["allOf"]
        annotation = _schema_annotation(choices[0], spec) if len(choices) == 1 else "Any"
    else:
        schema_type = resolved.get("type")
        if schema_type == "array":
            annotation = f"Sequence[{_schema_annotation(resolved.get('items', {}), spec)}]"
        elif schema_type == "object":
            annotation = "Mapping[str, Any]"
        else:
            annotation = {
                "boolean": "bool",
                "integer": "int",
                "number": "float",
                "string": "str",
            }.get(schema_type, "Any")
    if resolved.get("nullable") is True and "None" not in annotation:
        annotation += " | None"
    return annotation


def _generate_resources(
    root: Path,
    operations: Sequence[GeneratedOperation],
    model_modules: Mapping[str, str],
    config: Mapping[str, Any],
) -> List[Path]:
    by_tag: Dict[str, List[GeneratedOperation]] = {}
    for operation in operations:
        by_tag.setdefault(operation.tag, []).append(operation)

    generated_paths = []
    for tag, tag_operations in sorted(by_tag.items()):
        resource_path = root / f"{_snake_case(tag)}.py"
        _write_generated_file(resource_path, _resource_module_source(tag, tag_operations, model_modules), config)
        generated_paths.append(resource_path)
    return generated_paths


def _resource_module_source(
    tag: str, operations: Sequence[GeneratedOperation], model_modules: Mapping[str, str]
) -> str:
    annotation_names = set().union(*(_operation_annotation_names(operation) for operation in operations))
    collections_imports = sorted(annotation_names & {"Mapping", "Sequence"})
    typing_imports = sorted(annotation_names & {"Any", "Literal"})
    model_type_names = annotation_names - _NON_MODEL_ANNOTATION_NAMES
    model_imports: Dict[str, Set[str]] = {}
    for type_name in model_type_names:
        module = model_modules.get(type_name)
        if module is None:
            raise CodegenError(f"Generated resource {tag!r} references unknown model {type_name!r}")
        model_imports.setdefault(module, set()).add(type_name)

    lines = [f'"""Generated {tag} REST operations and resource."""', ""]
    if collections_imports:
        lines.extend([f"from collections.abc import {', '.join(collections_imports)}", ""])
    lines.append(f"from typing import {', '.join([*typing_imports, 'cast'])}")
    lines.extend(
        [
            "",
            "from .._service import Operation, Parameter, ResourceAPI",
            "from ..policies import RetryMode",
        ]
    )
    for module, names in sorted(model_imports.items()):
        lines.append(f"from .models.{module} import {', '.join(sorted(names))}")
    for operation in operations:
        lines.extend(["", "", *_operation_definition_source(operation)])
    lines.extend(["", "", "OPERATIONS = {"])
    lines.extend(f"    {operation.operation_id!r}: {operation.constant_name}," for operation in operations)
    lines.append("}")
    lines.extend(["", "", f"class {_python_type_name(tag)}API(ResourceAPI):", f'    """Generated {tag} REST API."""'])
    for operation in operations:
        lines.extend(["", *_resource_method_source(operation)])
    return "\n".join(lines) + "\n"


def _operation_definition_source(operation: GeneratedOperation) -> List[str]:
    lines = [
        f"{operation.constant_name} = Operation(",
        f"    operation_id={operation.operation_id!r},",
        f"    method={operation.method!r},",
        f"    path={operation.path!r},",
        "    parameters=(",
    ]
    for parameter in operation.parameters:
        lines.extend(
            [
                "        Parameter(",
                f"            argument_name={parameter.argument_name!r},",
                f"            name={parameter.name!r},",
                f"            location={parameter.location!r},",
                f"            required={parameter.required!r},",
                "        ),",
            ]
        )
    lines.extend(
        [
            "    ),",
            f"    has_request_body={operation.request_body_type is not None!r},",
            f"    success_statuses={operation.success_statuses!r},",
            f"    json_success_statuses={operation.json_success_statuses!r},",
            f"    retry_mode=RetryMode.{operation.retry_mode},",
            ")",
        ]
    )
    return lines


def _resource_method_source(operation: GeneratedOperation) -> List[str]:
    required_parameters = [parameter for parameter in operation.parameters if parameter.required]
    optional_parameters = [parameter for parameter in operation.parameters if not parameter.required]
    arguments = ["self"]
    arguments.extend(f'{parameter.argument_name}: "{parameter.type_name}"' for parameter in required_parameters)
    keyword_arguments = [
        f'{parameter.argument_name}: "{parameter.type_name} | None" = None' for parameter in optional_parameters
    ]
    if operation.request_body_type is not None:
        default = "" if operation.request_body_required else " = None"
        body_type = (
            operation.request_body_type if operation.request_body_required else f"{operation.request_body_type} | None"
        )
        keyword_arguments.append(f'body: "{body_type}"{default}')
    if keyword_arguments:
        arguments.append("*")
        arguments.extend(keyword_arguments)
    return_type = operation.response_type or "None"
    lines = [
        f"    def {_snake_case(operation.operation_id)}(",
        *(f"        {argument}," for argument in arguments),
        f'    ) -> "{return_type}":',
    ]
    call_arguments = [operation.constant_name]
    for location, keyword_name in (("path", "path_parameters"), ("query", "query_parameters")):
        parameters = [parameter for parameter in operation.parameters if parameter.location == location]
        if parameters:
            values = ", ".join(f"{parameter.argument_name!r}: {parameter.argument_name}" for parameter in parameters)
            call_arguments.append(f"{keyword_name}={{{values}}}")
    if operation.request_body_type is not None:
        call_arguments.append("body=body")
    lines.append(f'        return cast("{return_type}", self.execute(')
    lines.extend(f"            {argument}," for argument in call_arguments)
    lines.append("        ))")
    return lines


def _format_generated_files(paths: Sequence[Path]) -> None:
    if not paths:
        return
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", *(str(path) for path in paths)],
        check=True,
        capture_output=True,
        text=True,
    )
    for path in paths:
        generated = path.read_text(encoding="utf-8")
        marker_match = re.search(r"^# Content SHA-256: ([^\n]+)$", generated, flags=re.MULTILINE)
        if marker_match is None:
            raise CodegenError(f"Formatted generated file {path} lost its content hash")
        body_after_marker = generated[marker_match.end() :].lstrip("\n")
        content_hash = hashlib.sha256(body_after_marker.encode()).hexdigest()
        _write_checked(path, generated[: marker_match.start(1)] + content_hash + generated[marker_match.end(1) :])


def _python_argument_name(value: str) -> str:
    result = re.sub(r"\W", "_", value)
    if not result or result[0].isdigit():
        result = "_" + result
    if keyword.iskeyword(result):
        result += "_"
    return result


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


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
    generated_tags = endpoint.get("generated_tags")
    if (
        not isinstance(generated_tags, list)
        or not all(isinstance(value, str) and value for value in generated_tags)
        or len(generated_tags) != len(set(generated_tags))
    ):
        raise CodegenError("endpoint_generator.generated_tags must be a unique list of non-empty strings")
    for key in ("safe_reads", "idempotent_writes"):
        values = endpoint.get(key)
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise CodegenError(f"endpoint_generator.{key} must be a unique list of non-empty strings")
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


def _validate_refs(value: Any, spec: Mapping[str, Any], seen_refs: Set[str] | None = None) -> None:
    seen_refs = seen_refs if seen_refs is not None else set()
    for child in _walk_values(value):
        if not isinstance(child, dict) or "$ref" not in child:
            continue
        reference = child["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise CodegenError(f"Only local OpenAPI references are supported, got {reference!r}")
        resolved = _resolve_ref(reference, spec)
        if reference not in seen_refs:
            seen_refs.add(reference)
            _validate_refs(resolved, spec, seen_refs)


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


def _validate_parameters(
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
        location = parameter.get("in")
        if location not in {"path", "query"}:
            raise CodegenError(f"Operation {operation_id!r} has unsupported parameter location {location!r}")
        if location == "path":
            name = parameter.get("name")
            if not isinstance(name, str):
                raise CodegenError(f"Operation {operation_id!r} has a path parameter without a name")
            path_parameters[name] = parameter
            continue

        style = parameter.get("style", "form")
        if style != "form":
            raise CodegenError(f"Operation {operation_id!r} has unsupported query parameter style {style!r}")
        schema = parameter.get("schema")
        if not isinstance(schema, dict):
            raise CodegenError(f"Operation {operation_id!r} query parameter must define a schema")
        schema_kinds = _parameter_schema_kinds(schema, spec)
        if "array" in schema_kinds and parameter.get("explode", True) is not True:
            raise CodegenError(f"Operation {operation_id!r} query array parameters must be exploded")
    if template_names != set(path_parameters):
        raise CodegenError(
            f"Operation {operation_id!r} path template/parameter mismatch: "
            f"template={sorted(template_names)}, declared={sorted(path_parameters)}"
        )
    for name, parameter in path_parameters.items():
        style = parameter.get("style", "simple")
        if style != "simple":
            raise CodegenError(f"Operation {operation_id!r} has unsupported path parameter style {style!r}")
        if parameter.get("explode", False) is not False:
            raise CodegenError(f"Operation {operation_id!r} path parameters cannot be exploded")
        if parameter.get("required") is not True:
            raise CodegenError(f"Operation {operation_id!r} path parameter {name!r} must be required")
        schema = parameter.get("schema")
        if not isinstance(schema, dict):
            raise CodegenError(f"Operation {operation_id!r} path parameter {name!r} must define a schema")
        schema = _resolve_object(schema, spec)
        if schema.get("type") not in {"boolean", "integer", "number", "string"}:
            raise CodegenError(f"Operation {operation_id!r} path parameter {name!r} must be scalar")


def _parameter_schema_kinds(schema: Mapping[str, Any], spec: Mapping[str, Any]) -> Set[str]:
    schema = _resolve_object(schema, spec)
    choices = schema.get("oneOf", schema.get("anyOf"))
    if isinstance(choices, list):
        kinds: Set[str] = set()
        for choice in choices:
            if not isinstance(choice, dict):
                raise CodegenError("Query parameter alternatives must be schemas")
            kinds.update(_parameter_schema_kinds(choice, spec))
        return kinds

    schema_type = schema.get("type")
    if schema_type in {"boolean", "integer", "number", "string"}:
        return {"scalar"}
    if schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, dict) or _parameter_schema_kinds(items, spec) != {"scalar"}:
            raise CodegenError("Query parameter arrays must contain scalar values")
        return {"array"}
    raise CodegenError(f"Unsupported query parameter schema type {schema_type!r}")


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
