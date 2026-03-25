import dataclasses
import json
import math
import warnings
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple, cast, overload


# Try to import orjson for better performance
# If not available, we'll use standard json
try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False


_BT_SAFE_SPECIAL_TYPES: tuple[type[Any], type[Any], type[Any], type[Any], type[Any], type[Any]] | None = None


def _get_bt_safe_special_types() -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any], type[Any]]:
    global _BT_SAFE_SPECIAL_TYPES
    if _BT_SAFE_SPECIAL_TYPES is None:
        # avoid circular imports
        from braintrust.logger import BaseAttachment, Dataset, Experiment, Logger, ReadonlyAttachment, Span

        _BT_SAFE_SPECIAL_TYPES = (Span, Experiment, Dataset, Logger, BaseAttachment, ReadonlyAttachment)

    return _BT_SAFE_SPECIAL_TYPES


def _to_bt_safe(v: Any) -> Any:
    """
    Converts the object to a Braintrust-safe representation (i.e. Attachment objects are safe (specially handled by background logger)).
    """
    v_type = type(v)
    if v_type is str or v_type is int or v_type is bool or v is None:
        # Skip all richer object checks for primitive scalar values.
        return v

    if v_type is float:
        # Handle NaN and Infinity for JSON compatibility
        if math.isfinite(v):
            return v

        if math.isnan(v):
            return "NaN"

        return "Infinity" if v > 0 else "-Infinity"

    if isinstance(v, str):
        return v

    dataclass_fields = getattr(v_type, "__dataclass_fields__", None)
    if dataclass_fields is not None:
        # Use manual field iteration instead of dataclasses.asdict() because
        # asdict() deep-copies values, which breaks objects like Attachment
        # that contain non-copyable items (thread locks, file handles, etc.)
        instance_dict = getattr(v, "__dict__", None)
        if instance_dict is not None and len(instance_dict) == len(dataclass_fields):
            return bt_safe_deep_copy(instance_dict)
        return {f.name: _to_bt_safe(getattr(v, f.name)) for f in dataclass_fields.values()}

    # Pydantic model classes (not instances) with model_json_schema
    if isinstance(v, type):
        model_json_schema = getattr(v, "model_json_schema", None)
        if callable(model_json_schema):
            try:
                return model_json_schema()
            except Exception:
                pass

    # Attempt to dump a Pydantic v2 `BaseModel`.
    # Suppress Pydantic serializer warnings that arise from generic/discriminated-union
    # models (e.g. OpenAI's ParsedResponse[T]).  See
    # https://github.com/braintrustdata/braintrust-sdk-python/issues/60
    model_dump = getattr(v_type, "model_dump", None)
    if callable(model_dump):
        try:
            if hasattr(v_type, "__pydantic_serializer__"):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
                    return model_dump(v, exclude_none=True)
            return model_dump(v, exclude_none=True)
        except TypeError:
            pass

    # Attempt to dump a Pydantic v1 `BaseModel`.
    dict_method = getattr(v_type, "dict", None)
    if callable(dict_method):
        try:
            return dict_method(v, exclude_none=True)
        except TypeError:
            pass

    if isinstance(v, (str, bool, int)):
        return v

    if isinstance(v, float):
        if math.isfinite(v):
            return v

        if math.isnan(v):
            return "NaN"

        return "Infinity" if v > 0 else "-Infinity"

    Span, Experiment, Dataset, Logger, BaseAttachment, ReadonlyAttachment = _get_bt_safe_special_types()

    if isinstance(v, Span):
        return "<span>"

    if isinstance(v, Experiment):
        return "<experiment>"

    if isinstance(v, Dataset):
        return "<dataset>"

    if isinstance(v, Logger):
        return "<logger>"

    if isinstance(v, BaseAttachment):
        return v

    if isinstance(v, ReadonlyAttachment):
        return v.reference

    # Note: we avoid using copy.deepcopy, because it's difficult to
    # guarantee the independence of such copied types from their origin.
    # E.g. the original type could have a `__del__` method that alters
    # some shared internal state, and we need this deep copy to be
    # fully-independent from the original.

    # We pass `encoder=_str_encoder` since we've already tried converting rich objects to json safe objects.
    return bt_loads(bt_dumps(v, encoder=_str_encoder))


@overload
def bt_safe_deep_copy(
    obj: Mapping[str, Any],
    max_depth: int = ...,
) -> dict[str, Any]: ...


@overload
def bt_safe_deep_copy(
    obj: list[Any],
    max_depth: int = ...,
) -> list[Any]: ...


@overload
def bt_safe_deep_copy(
    obj: Any,
    max_depth: int = ...,
) -> Any: ...
def bt_safe_deep_copy(obj: Any, max_depth: int = 200):
    """
    Creates a deep copy of the given object and converts rich objects to Braintrust-safe representations. See `_to_bt_safe` for more details.

    Args:
        obj: Object to deep copy and sanitize.
        to_json_safe: Function to ensure the object is json safe.
        max_depth: Maximum depth to copy.

    Returns:
        Deep copy of the object.
    """
    # Track visited objects to detect circular references
    visited: set[int] = set()

    def _deep_copy_object(v: Any, depth: int = 0) -> Any:
        # Check depth limit - use >= to stop before exceeding
        if depth >= max_depth:
            return "<max depth exceeded>"

        v_type = type(v)

        # Check for circular references in mutable containers.
        # Fast-path the built-in container types we expect most often.
        if v_type is dict:
            # Prevent dict keys from holding references to user data. Note that
            # `bt_json` already coerces keys to string, a behavior that comes from
            # `json.dumps`. However, that runs at log upload time, while we want to
            # cut out all the references to user objects synchronously in this
            # function.
            result = {}
            next_depth = depth + 1
            if next_depth >= max_depth:
                for k in v:
                    if type(k) is str:
                        key_str = k
                    else:
                        try:
                            key_str = str(k)
                        except Exception:
                            # If str() fails on the key, use a fallback representation
                            key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                    result[key_str] = "<max depth exceeded>"
                return result

            items = iter(v.items())
            for k, value in items:
                if type(k) is not str:
                    try:
                        key_str = str(k)
                    except Exception:
                        # If str() fails on the key, use a fallback representation
                        key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                    obj_id = id(v)
                    if obj_id in visited:
                        return "<circular reference>"
                    visited.add(obj_id)
                    try:
                        result[key_str] = _deep_copy_object(value, next_depth)
                        for k, value in items:
                            if type(k) is str:
                                key_str = k
                            else:
                                try:
                                    key_str = str(k)
                                except Exception:
                                    # If str() fails on the key, use a fallback representation
                                    key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                            result[key_str] = _deep_copy_object(value, next_depth)
                        return result
                    finally:
                        visited.remove(obj_id)

                value_type = type(value)
                if value_type is str or value_type is int or value_type is bool or value is None:
                    result[k] = value
                    continue

                if value_type is float:
                    if math.isfinite(value):
                        result[k] = value
                    elif math.isnan(value):
                        result[k] = "NaN"
                    else:
                        result[k] = "Infinity" if value > 0 else "-Infinity"
                    continue

                obj_id = id(v)
                if obj_id in visited:
                    return "<circular reference>"
                visited.add(obj_id)
                try:
                    result[k] = _deep_copy_object(value, next_depth)
                    for k, value in items:
                        if type(k) is str:
                            key_str = k
                        else:
                            try:
                                key_str = str(k)
                            except Exception:
                                # If str() fails on the key, use a fallback representation
                                key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                        result[key_str] = _deep_copy_object(value, next_depth)
                    return result
                finally:
                    visited.remove(obj_id)

            return result

        if v_type is list:
            obj_id = id(v)
            added_to_visited = False
            try:
                next_depth = depth + 1
                result = []
                for value in v:
                    value_type = type(value)
                    if value_type is dict:
                        if not added_to_visited:
                            if obj_id in visited:
                                return "<circular reference>"
                            visited.add(obj_id)
                            added_to_visited = True
                        nested_result = {}
                        if next_depth >= max_depth:
                            for k in value:
                                if type(k) is str:
                                    key_str = k
                                else:
                                    try:
                                        key_str = str(k)
                                    except Exception:
                                        key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                                nested_result[key_str] = "<max depth exceeded>"
                            result.append(nested_result)
                            continue

                        items = iter(value.items())
                        for k, nested_value in items:
                            if type(k) is not str:
                                try:
                                    key_str = str(k)
                                except Exception:
                                    key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                                value_id = id(value)
                                if value_id in visited:
                                    result.append("<circular reference>")
                                    break
                                visited.add(value_id)
                                try:
                                    nested_result[key_str] = _deep_copy_object(nested_value, next_depth)
                                    for k, nested_value in items:
                                        if type(k) is str:
                                            key_str = k
                                        else:
                                            try:
                                                key_str = str(k)
                                            except Exception:
                                                key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                                        nested_result[key_str] = _deep_copy_object(nested_value, next_depth)
                                    result.append(nested_result)
                                finally:
                                    visited.remove(value_id)
                                break

                            nested_value_type = type(nested_value)
                            if nested_value_type is str or nested_value_type is int or nested_value_type is bool or nested_value is None:
                                nested_result[k] = nested_value
                                continue

                            if nested_value_type is float:
                                if math.isfinite(nested_value):
                                    nested_result[k] = nested_value
                                elif math.isnan(nested_value):
                                    nested_result[k] = "NaN"
                                else:
                                    nested_result[k] = "Infinity" if nested_value > 0 else "-Infinity"
                                continue

                            value_id = id(value)
                            if value_id in visited:
                                result.append("<circular reference>")
                                break
                            visited.add(value_id)
                            try:
                                nested_result[k] = _deep_copy_object(nested_value, next_depth)
                                for k, nested_value in items:
                                    if type(k) is str:
                                        key_str = k
                                    else:
                                        try:
                                            key_str = str(k)
                                        except Exception:
                                            key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                                    nested_result[key_str] = _deep_copy_object(nested_value, next_depth)
                                result.append(nested_result)
                            finally:
                                visited.remove(value_id)
                            break
                        else:
                            result.append(nested_result)
                        continue

                    if value_type is str or value_type is int or value_type is bool or value is None:
                        result.append(value)
                        continue

                    if value_type is float:
                        if math.isfinite(value):
                            result.append(value)
                        elif math.isnan(value):
                            result.append("NaN")
                        else:
                            result.append("Infinity" if value > 0 else "-Infinity")
                        continue

                    if not added_to_visited:
                        if obj_id in visited:
                            return "<circular reference>"
                        visited.add(obj_id)
                        added_to_visited = True
                    result.append(_deep_copy_object(value, next_depth))
                return result
            finally:
                if added_to_visited:
                    visited.remove(obj_id)

        if v_type is tuple or v_type is set:
            obj_id = id(v)
            if obj_id in visited:
                return "<circular reference>"
            visited.add(obj_id)
            try:
                next_depth = depth + 1
                return [_deep_copy_object(x, next_depth) for x in v]
            finally:
                visited.remove(obj_id)

        if v_type is str or v_type is int or v_type is bool or v is None:
            return v

        if v_type is float:
            if math.isfinite(v):
                return v
            if math.isnan(v):
                return "NaN"
            return "Infinity" if v > 0 else "-Infinity"

        if isinstance(v, (str, bool, int)):
            return v

        if isinstance(v, float):
            if math.isfinite(v):
                return v
            if math.isnan(v):
                return "NaN"
            return "Infinity" if v > 0 else "-Infinity"

        if isinstance(v, dict):
            obj_id = id(v)
            if obj_id in visited:
                return "<circular reference>"
            visited.add(obj_id)
            try:
                result = {}
                next_depth = depth + 1
                for k, value in v.items():
                    if type(k) is str:
                        key_str = k
                    else:
                        try:
                            key_str = str(k)
                        except Exception:
                            key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                    result[key_str] = _deep_copy_object(value, next_depth)
                return result
            finally:
                visited.remove(obj_id)

        if isinstance(v, list):
            obj_id = id(v)
            if obj_id in visited:
                return "<circular reference>"
            visited.add(obj_id)
            try:
                next_depth = depth + 1
                return [_deep_copy_object(x, next_depth) for x in v]
            finally:
                visited.remove(obj_id)

        if isinstance(v, (tuple, set)):
            obj_id = id(v)
            if obj_id in visited:
                return "<circular reference>"
            visited.add(obj_id)
            try:
                next_depth = depth + 1
                return [_deep_copy_object(x, next_depth) for x in v]
            finally:
                visited.remove(obj_id)

        if isinstance(v, Mapping):
            obj_id = id(v)
            if obj_id in visited:
                return "<circular reference>"
            visited.add(obj_id)
            try:
                result = {}
                next_depth = depth + 1
                for k, value in v.items():
                    if type(k) is str:
                        key_str = k
                    else:
                        try:
                            key_str = str(k)
                        except Exception:
                            key_str = f"<non-stringifiable-key: {type(k).__name__}>"
                    result[key_str] = _deep_copy_object(value, next_depth)
                return result
            finally:
                visited.remove(obj_id)

        try:
            return _to_bt_safe(v)
        except Exception:
            return f"<non-sanitizable: {type(v).__name__}>"

    return _deep_copy_object(obj)


def _safe_str(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


def _to_json_safe(obj: Any) -> Any:
    """
    Handler for non-JSON-serializable objects. Returns a string representation of the object.
    """
    # avoid circular imports
    from braintrust.logger import BaseAttachment

    try:
        v = _to_bt_safe(obj)

        # JSON-safe representation of Attachment objects are their reference.
        # If we get this object at this point, we have to assume someone has already uploaded the attachment!
        if isinstance(v, BaseAttachment):
            v = v.reference

        return v
    except Exception:
        pass

    # When everything fails, try to return the string representation of the object
    return _safe_str(obj)


class BraintrustJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for standard json library.

    This is used as a fallback when orjson is not available or fails.
    """

    def default(self, o: Any):
        return _to_json_safe(o)


class BraintrustStrEncoder(json.JSONEncoder):
    def default(self, o: Any):
        return _safe_str(o)


class Encoder(NamedTuple):
    native: type[json.JSONEncoder]
    orjson: Callable[[Any], Any]


_json_encoder = Encoder(native=BraintrustJSONEncoder, orjson=_to_json_safe)
_str_encoder = Encoder(native=BraintrustStrEncoder, orjson=_safe_str)


def bt_dumps(obj: Any, encoder: Encoder | None = _json_encoder, **kwargs: Any) -> str:
    """
    Serialize obj to a JSON-formatted string.

    Automatically uses orjson if available for better performance (3-5x faster),
    with fallback to standard json library if orjson is not installed or fails.

    Args:
        obj: Object to serialize
        encoder: Encoder to use, defaults to `_default_encoder`
        **kwargs: Additional arguments (passed to json.dumps in fallback path)

    Returns:
        JSON string representation of obj
    """
    if _HAS_ORJSON:
        # Try orjson first for better performance
        try:
            # pylint: disable=no-member  # orjson is a C extension, pylint can't introspect it
            return orjson.dumps(  # type: ignore[possibly-unbound]
                obj,
                default=encoder.orjson if encoder else None,
                # options match json.dumps behavior for bc
                option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,  # type: ignore[possibly-unbound]
            ).decode("utf-8")
        except Exception:
            # If orjson fails, fall back to standard json
            pass

    # Use standard json (either orjson not available or it failed)
    # Use sort_keys=True for deterministic output (matches orjson OPT_SORT_KEYS)
    return json.dumps(obj, cls=encoder.native if encoder else None, allow_nan=False, sort_keys=True, **kwargs)


def bt_loads(s: str, **kwargs) -> Any:
    """
    Deserialize s (a str containing a JSON document) to a Python object.

    Automatically uses orjson if available for better performance (2-3x faster),
    with fallback to standard json library if orjson is not installed or fails.

    Args:
        s: JSON string to deserialize
        **kwargs: Additional arguments (passed to json.loads in fallback path)

    Returns:
        Python object representation of JSON string
    """
    if _HAS_ORJSON:
        # Try orjson first for better performance
        try:
            # pylint: disable=no-member  # orjson is a C extension, pylint can't introspect it
            return orjson.loads(s)  # type: ignore[possibly-unbound]
        except Exception:
            # If orjson fails, fall back to standard json
            pass

    # Use standard json (either orjson not available or it failed)
    return json.loads(s, **kwargs)
