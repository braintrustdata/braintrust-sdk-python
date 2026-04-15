"""Evaluation parameters support for Python SDK."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from typing_extensions import NotRequired

from .prompt import PromptData
from .serializable_data_class import SerializableDataClass


if TYPE_CHECKING:
    from .logger import Prompt


class PromptParameter(TypedDict):
    """A prompt parameter specification."""

    type: Literal["prompt"]
    name: NotRequired[str | None]
    default: NotRequired[PromptData | None]
    description: NotRequired[str | None]


class ModelParameter(TypedDict):
    """A model parameter specification."""

    type: Literal["model"]
    default: NotRequired[str | None]
    description: NotRequired[str | None]


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
ValidatedParameters = dict[str, Any]
ParameterSchema = PromptParameter | ModelParameter | type[object] | None
EvalParameters = Mapping[str, ParameterSchema]
ParametersSchema = Mapping[str, Any]


@dataclass
class RemoteEvalParameters(SerializableDataClass):
    id: str | None
    project_id: str | None
    name: str
    slug: str
    version: str | None
    schema: ParametersSchema
    data: dict[str, Any]

    @classmethod
    def from_function_row(cls, row: dict[str, Any]) -> "RemoteEvalParameters":
        function_data = row.get("function_data") or {}
        return cls(
            id=row.get("id"),
            project_id=row.get("project_id"),
            name=row["name"],
            slug=row["slug"],
            version=row.get("_xact_id"),
            schema=function_data.get("__schema") or {},
            data=function_data.get("data") or {},
        )

    def validate(self, data: Any) -> bool:
        try:
            validate_json_schema(data, self.schema)
            return True
        except ValueError:
            return False


def _pydantic_to_json_schema(model: Any) -> dict[str, Any]:
    """Convert a pydantic model to JSON schema."""
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    if hasattr(model, "schema"):
        return model.schema()
    raise ValueError(f"Cannot convert {model} to JSON schema - not a pydantic model")


def _is_prompt_parameter(schema: Any) -> bool:
    return isinstance(schema, dict) and schema.get("type") == "prompt"


def _is_model_parameter(schema: Any) -> bool:
    return isinstance(schema, dict) and schema.get("type") == "model"


def _is_pydantic_model(schema: Any) -> bool:
    return hasattr(schema, "parse_obj") or hasattr(schema, "model_validate")


def _get_pydantic_fields(schema: Any) -> dict[str, Any]:
    model_fields = getattr(schema, "model_fields", None)
    if model_fields is not None:
        return model_fields
    return getattr(schema, "__fields__", {})


def _resolve_json_pointer(document: dict[str, JSONValue], pointer: str) -> JSONValue:
    if pointer == "#":
        return document
    if not pointer.startswith("#/"):
        raise ValueError(f"Unsupported JSON schema ref '{pointer}'")

    current: JSONValue = document
    for raw_part in pointer[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"JSON schema ref '{pointer}' could not be resolved")
        current = current[part]
    return current


def _resolve_local_json_schema_refs(
    node: JSONValue,
    root: dict[str, JSONValue],
    resolving: tuple[str, ...] = (),
) -> JSONValue:
    if isinstance(node, list):
        return [_resolve_local_json_schema_refs(item, root, resolving) for item in node]

    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in resolving:
            raise ValueError(f"Cyclic JSON schema ref '{ref}'")

        resolved = deepcopy(_resolve_json_pointer(root, ref))
        resolved = _resolve_local_json_schema_refs(resolved, root, resolving + (ref,))

        siblings = {
            key: _resolve_local_json_schema_refs(value, root, resolving)
            for key, value in node.items()
            if key != "$ref"
        }
        if siblings:
            if not isinstance(resolved, dict):
                raise ValueError(f"Cannot merge sibling keys into non-object JSON schema ref '{ref}'")
            merged = dict(resolved)
            merged.update(siblings)
            return merged
        return resolved

    return {key: _resolve_local_json_schema_refs(value, root, resolving) for key, value in node.items()}


def _serialize_pydantic_parameter_schema(schema: Any) -> dict[str, Any]:
    schema_json = _pydantic_to_json_schema(schema)
    schema_json = _resolve_local_json_schema_refs(schema_json, schema_json)
    schema_json.pop("$defs", None)
    schema_json.pop("definitions", None)
    fields = _get_pydantic_fields(schema)
    if len(fields) == 1 and "value" in fields:
        properties = schema_json.get("properties")
        if isinstance(properties, dict) and isinstance(properties.get("value"), dict):
            return dict(properties["value"])
    return schema_json


def _pydantic_field_required(field: Any) -> bool:
    is_required = getattr(field, "is_required", None)
    if callable(is_required):
        return bool(is_required())
    return bool(getattr(field, "required", False))


def is_eval_parameter_schema(schema: Any) -> bool:
    if isinstance(schema, RemoteEvalParameters):
        return True
    if not isinstance(schema, Mapping):
        return False
    if len(schema) == 0:
        return True

    for value in schema.values():
        if _is_prompt_parameter(value) or _is_model_parameter(value) or value is None or _is_pydantic_model(value):
            continue
        return False
    return True


def _prompt_data_to_dict(
    prompt_data: PromptData | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if prompt_data is None:
        return None
    if isinstance(prompt_data, PromptData):
        return prompt_data.as_dict()
    return prompt_data


def _create_prompt(name: str, prompt_data: dict[str, Any]) -> "Prompt":
    from .logger import Prompt

    return Prompt.from_prompt_data(name, PromptData.from_dict_deep(prompt_data))


def _apply_defaults_to_json_schema_instance(instance: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(instance, dict):
        return instance
    if schema.get("type") != "object":
        return instance

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return instance

    for name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue

        if name not in instance and "default" in property_schema:
            instance[name] = property_schema["default"]

        if name in instance:
            value = instance[name]
            if isinstance(value, dict):
                _apply_defaults_to_json_schema_instance(value, property_schema)
            elif isinstance(value, list):
                items_schema = property_schema.get("items")
                if isinstance(items_schema, dict):
                    for item in value:
                        _apply_defaults_to_json_schema_instance(item, items_schema)

    return instance


def validate_json_schema(parameters: dict[str, Any], schema: ParametersSchema) -> dict[str, Any]:
    candidate = dict(parameters)
    _apply_defaults_to_json_schema_instance(candidate, schema)

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(candidate), key=lambda error: list(error.path))
    if errors:
        messages = []
        for error in errors:
            path = ".".join(str(part) for part in error.path)
            messages.append(f"{path or 'root'}: {error.message}")
        raise ValueError(f"Invalid parameters: {', '.join(messages)}")

    return candidate


def _rehydrate_remote_parameters(
    parameters: dict[str, Any],
    schema: ParametersSchema,
) -> ValidatedParameters:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return parameters

    result: ValidatedParameters = dict(parameters)
    for name, property_schema in properties.items():
        if not isinstance(property_schema, dict) or name not in result:
            continue

        if property_schema.get("x-bt-type") == "prompt":
            prompt_data = result[name]
            if not isinstance(prompt_data, dict):
                raise ValueError(f"Invalid parameter '{name}': prompt value must be an object")
            result[name] = _create_prompt(name, prompt_data)

    return result


def _validate_local_parameters(
    parameters: dict[str, Any],
    parameter_schema: EvalParameters,
) -> ValidatedParameters:
    result: ValidatedParameters = {}

    for name, schema in parameter_schema.items():
        value = parameters.get(name)

        try:
            if _is_prompt_parameter(schema):
                prompt_data = None
                if value is not None:
                    prompt_data = value
                elif schema.get("default") is not None:
                    prompt_data = _prompt_data_to_dict(schema["default"])
                else:
                    raise ValueError(f"Parameter '{name}' is required")

                if prompt_data is None:
                    raise ValueError(f"Parameter '{name}' is required")

                result[name] = _create_prompt(schema.get("name") or name, prompt_data)
            elif _is_model_parameter(schema):
                model = value if value is not None else schema.get("default")
                if model is None:
                    raise ValueError(f"Parameter '{name}' is required")
                if not isinstance(model, str):
                    raise ValueError(f"Parameter '{name}' must be a string model identifier")
                result[name] = model
            elif schema is None:
                result[name] = value
            elif _is_pydantic_model(schema):
                fields = _get_pydantic_fields(schema)
                if len(fields) == 1 and "value" in fields:
                    if value is None:
                        try:
                            default_instance = schema()
                            result[name] = default_instance.value
                        except Exception as exc:
                            raise ValueError(f"Parameter '{name}' is required") from exc
                    elif hasattr(schema, "parse_obj"):
                        result[name] = schema.parse_obj({"value": value}).value
                    else:
                        result[name] = schema.model_validate({"value": value}).value
                else:
                    if value is None:
                        try:
                            result[name] = schema()
                        except Exception as exc:
                            raise ValueError(f"Parameter '{name}' is required") from exc
                    elif hasattr(schema, "parse_obj"):
                        result[name] = schema.parse_obj(value)
                    else:
                        result[name] = schema.model_validate(value)
            else:
                result[name] = value
        except JSONSchemaValidationError as exc:
            raise ValueError(f"Invalid parameter '{name}': {exc.message}") from exc
        except Exception as exc:
            raise ValueError(f"Invalid parameter '{name}': {str(exc)}") from exc

    return result


def validate_parameters(
    parameters: dict[str, Any],
    parameter_schema: EvalParameters | RemoteEvalParameters | None,
) -> ValidatedParameters:
    """
    Validate parameters against the schema.

    Args:
        parameters: The parameters to validate.
        parameter_schema: The schema to validate against.

    Returns:
        Validated parameters.

    Raises:
        ValueError: If validation fails.
    """
    if parameter_schema is None:
        return dict(parameters)

    if isinstance(parameter_schema, RemoteEvalParameters):
        merged = dict(parameter_schema.data)
        merged.update(parameters)
        validated = validate_json_schema(merged, parameter_schema.schema)
        return _rehydrate_remote_parameters(validated, parameter_schema.schema)

    return _validate_local_parameters(parameters, parameter_schema)


def serialize_eval_parameters(parameters: EvalParameters) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for name, schema in parameters.items():
        if _is_prompt_parameter(schema):
            parameter_data = {
                "type": "prompt",
                "description": schema.get("description"),
            }
            default = schema.get("default")
            if default is not None:
                parameter_data["default"] = _prompt_data_to_dict(default)
            result[name] = parameter_data
        elif _is_model_parameter(schema):
            parameter_data = {
                "type": "model",
                "description": schema.get("description"),
            }
            default = schema.get("default")
            if default is not None:
                parameter_data["default"] = default
            result[name] = parameter_data
        elif schema is None:
            result[name] = {
                "type": "data",
                "schema": {},
            }
        else:
            schema_json = _serialize_pydantic_parameter_schema(schema)
            parameter_data = {
                "type": "data",
                "schema": schema_json,
                "description": schema_json.get("description"),
            }
            if "default" in schema_json:
                parameter_data["default"] = schema_json["default"]
            result[name] = parameter_data

    return result


def _parameter_required(schema: ParameterSchema) -> bool:
    if _is_prompt_parameter(schema) or _is_model_parameter(schema):
        return schema.get("default") is None

    if schema is None:
        return False

    if _is_pydantic_model(schema):
        fields = _get_pydantic_fields(schema)
        if len(fields) == 1 and "value" in fields:
            return _pydantic_field_required(fields["value"])
        return False

    return False


def parameters_to_json_schema(parameters: EvalParameters) -> ParametersSchema:
    """
    Convert EvalParameters to JSON Schema for saved parameter validation.
    """
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for name, schema in parameters.items():
        if _is_prompt_parameter(schema):
            property_schema: dict[str, Any] = {
                "type": "object",
                "x-bt-type": "prompt",
            }
            default = _prompt_data_to_dict(schema.get("default"))
            if default is not None:
                property_schema["default"] = default
            if schema.get("description") is not None:
                property_schema["description"] = schema["description"]
            properties[name] = property_schema
        elif _is_model_parameter(schema):
            property_schema = {
                "type": "string",
                "x-bt-type": "model",
            }
            if "default" in schema:
                property_schema["default"] = schema.get("default")
            if schema.get("description") is not None:
                property_schema["description"] = schema["description"]
            properties[name] = property_schema
        elif schema is None:
            properties[name] = {}
        else:
            properties[name] = _serialize_pydantic_parameter_schema(schema)

        if _parameter_required(schema):
            required.append(name)

    result: ParametersSchema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }
    if required:
        result["required"] = required
    return result


def get_default_data_from_parameters_schema(schema: ParametersSchema) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}

    defaults: dict[str, Any] = {}
    for name, value in properties.items():
        if isinstance(value, dict) and "default" in value:
            defaults[name] = value["default"]
    return defaults


def serialize_remote_eval_parameters_container(
    parameters: EvalParameters | RemoteEvalParameters,
) -> dict[str, Any]:
    if isinstance(parameters, RemoteEvalParameters):
        return {
            "type": "braintrust.parameters",
            "schema": parameters.schema,
            "source": {
                "parametersId": parameters.id,
                "slug": parameters.slug,
                "name": parameters.name,
                "projectId": parameters.project_id,
                "version": parameters.version,
            },
        }

    return {
        "type": "braintrust.staticParameters",
        "schema": serialize_eval_parameters(parameters),
        "source": None,
    }
