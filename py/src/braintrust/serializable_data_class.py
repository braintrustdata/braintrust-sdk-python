import copy
import dataclasses
import json
import types
from typing import Any, Union, get_origin


_EXPLICITLY_SET_FIELDS_ATTR = "_braintrust_explicitly_set_fields"
_INIT_FIELDS_REMAINING_ATTR = "_braintrust_init_fields_remaining"


def _dataclass_fields(cls_or_instance: Any) -> tuple[dataclasses.Field, ...]:
    try:
        return dataclasses.fields(cls_or_instance)
    except TypeError:
        return ()


def _explicit_constructor_fields(cls: type, args: tuple[Any, ...], kwargs: dict[str, Any]) -> set[str]:
    fields = _dataclass_fields(cls)
    positional_fields = [f.name for f in fields if f.init and not getattr(f, "kw_only", False)]
    init_field_names = {f.name for f in fields if f.init}

    explicit_fields = set(positional_fields[: len(args)])
    explicit_fields.update(k for k in kwargs if k in init_field_names)
    return explicit_fields


def _field_names(cls_or_instance: Any) -> set[str]:
    return {f.name for f in _dataclass_fields(cls_or_instance)}


def _init_field_names(cls_or_instance: Any) -> set[str]:
    return {f.name for f in _dataclass_fields(cls_or_instance) if f.init}


def _clear_init_tracking(obj: Any) -> None:
    if hasattr(obj, _INIT_FIELDS_REMAINING_ATTR):
        object.__delattr__(obj, _INIT_FIELDS_REMAINING_ATTR)


class SerializableDataClass:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        init_fields = _init_field_names(cls)
        if init_fields:
            object.__setattr__(instance, _INIT_FIELDS_REMAINING_ATTR, init_fields)
        object.__setattr__(instance, _EXPLICITLY_SET_FIELDS_ATTR, _explicit_constructor_fields(cls, args, kwargs))
        return instance

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        if name not in _field_names(type(self)):
            return

        init_fields_remaining = getattr(self, _INIT_FIELDS_REMAINING_ATTR, None)
        if init_fields_remaining is not None:
            init_fields_remaining.discard(name)
            if not init_fields_remaining:
                _clear_init_tracking(self)
            return

        getattr(self, _EXPLICITLY_SET_FIELDS_ATTR).add(name)

    def __post_init__(self) -> None:
        _clear_init_tracking(self)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # copy/deepcopy and pickle rebuild via __new__ without running dataclass __init__,
        # so do not persist the temporary constructor-assignment tracking marker.
        state.pop(_INIT_FIELDS_REMAINING_ATTR, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        explicitly_set_fields = getattr(self, _EXPLICITLY_SET_FIELDS_ATTR, None)
        if explicitly_set_fields is not None:
            object.__setattr__(self, _EXPLICITLY_SET_FIELDS_ATTR, set(explicitly_set_fields))
        _clear_init_tracking(self)

    def as_dict(self, exclude_unset: bool = False):
        """Serialize the object to a dictionary."""
        if not exclude_unset:
            return dataclasses.asdict(self)

        explicitly_set_fields = getattr(self, _EXPLICITLY_SET_FIELDS_ATTR, None)
        if explicitly_set_fields is None:
            return dataclasses.asdict(self)
        explicitly_set_fields = set(explicitly_set_fields)

        return {
            f.name: _as_dict_value(getattr(self, f.name), exclude_unset=exclude_unset)
            for f in dataclasses.fields(self)
            if f.name in explicitly_set_fields or getattr(self, f.name) is not None
        }

    def as_json(self, exclude_unset: bool = False, **kwargs):
        """Serialize the object to JSON."""
        if exclude_unset:
            return json.dumps(self.as_dict(exclude_unset=True), **kwargs)
        return json.dumps(self.as_dict(), **kwargs)

    def __getitem__(self, item: str):
        return getattr(self, item)

    @classmethod
    def from_dict(cls, d: dict):
        """Deserialize the object from a dictionary. This method
        is shallow and will not call from_dict() on nested objects."""
        fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in fields}
        return cls(**filtered)

    @classmethod
    def from_dict_deep(cls, d: dict):
        """Deserialize the object from a dictionary. This method
        is deep and will call from_dict_deep() on nested objects."""
        fields = {f.name: f for f in dataclasses.fields(cls)}
        filtered = {}
        for k, v in d.items():
            if k not in fields:
                continue

            if (
                isinstance(v, dict)
                and isinstance(fields[k].type, type)
                and issubclass(fields[k].type, SerializableDataClass)
            ):
                filtered[k] = fields[k].type.from_dict_deep(v)
            elif get_origin(fields[k].type) is Union or isinstance(fields[k].type, types.UnionType):
                for t in fields[k].type.__args__:
                    if t == type(None) and v is None:
                        filtered[k] = None
                        break
                    if isinstance(t, type) and issubclass(t, SerializableDataClass) and v is not None:
                        try:
                            filtered[k] = t.from_dict_deep(v)
                            break
                        except TypeError:
                            pass
                else:
                    filtered[k] = v
            elif (
                isinstance(v, list)
                and get_origin(fields[k].type) == list
                and len(fields[k].type.__args__) == 1
                and isinstance(fields[k].type.__args__[0], type)
                and issubclass(fields[k].type.__args__[0], SerializableDataClass)
            ):
                filtered[k] = [fields[k].type.__args__[0].from_dict_deep(i) for i in v]
            else:
                filtered[k] = v
        return cls(**filtered)


def _as_dict_value(value: Any, exclude_unset: bool) -> Any:
    if isinstance(value, SerializableDataClass):
        return value.as_dict(exclude_unset=exclude_unset)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [_as_dict_value(v, exclude_unset=exclude_unset) for v in value]
    if isinstance(value, tuple):
        return tuple(_as_dict_value(v, exclude_unset=exclude_unset) for v in value)
    if isinstance(value, dict):
        return {copy.deepcopy(k): _as_dict_value(v, exclude_unset=exclude_unset) for k, v in value.items()}
    return copy.deepcopy(value)
