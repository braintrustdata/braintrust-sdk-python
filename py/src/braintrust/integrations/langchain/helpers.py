from typing import Any, Sequence
from unittest.mock import ANY


# Base types that can appear in values
PrimitiveValue = str | int | float | bool | None
RecursiveValue = PrimitiveValue | dict[str, Any] | Sequence[Any]


def deep_hashable_dict(d: RecursiveValue):
    """Recursively convert a dictionary into a hashable representation, handling nested values."""
    if isinstance(d, dict):
        return frozenset((k, deep_hashable_dict(v)) for k, v in d.items())
    elif isinstance(d, Sequence) and not isinstance(d, str):
        return frozenset(deep_hashable_dict(x) for x in d)
    else:
        return d


def assert_matches_object(
    actual: RecursiveValue,
    expected: RecursiveValue,
    ignore_order: bool = False,
) -> None:
    """Assert that actual contains all key-value pairs from expected.

    For lists, each item in expected must match the corresponding item in actual.
    For dicts, all key-value pairs in expected must exist in actual.

    Args:
        actual: The actual value to check
        expected: The expected value to match against

    Raises:
        AssertionError: If the actual value doesn't match the expected value
    """
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), f"Expected sequence but got {type(actual)}"
        assert len(actual) >= len(expected), (
            f"Expected sequence of length >= {len(expected)} but got length {len(actual)}"
        )
        if not ignore_order:
            for i, expected_item in enumerate(expected):
                assert_matches_object(actual[i], expected_item)
        else:
            for expected_item in expected:
                matched = False
                for actual_item in actual:
                    try:
                        assert_matches_object(actual_item, expected_item)
                        matched = True
                    except Exception:
                        pass

                assert matched, f"Expected {expected_item} in unordered sequence but couldn't find match in {actual}"

    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected dict but got {type(actual)}"
        actual_dict: dict[str, Any] = actual
        for k, v in expected.items():
            assert k in actual_dict, f"Missing key {k}"
            if v is ANY:
                continue  # ANY matches anything
            if isinstance(v, (dict, list, tuple)):
                assert_matches_object(actual_dict[k], v)
            else:
                assert actual_dict[k] == v, f"Key {k}: expected {v} but got {actual_dict[k]}"
    else:
        assert actual == expected, f"Expected {expected} but got {actual}"


def find_spans_by_attributes(spans: list[Any], **attributes: Any) -> list[Any]:
    """Find all spans that match the given attributes."""
    matching_spans: list[Any] = []
    for span in spans:
        matches = True
        if "span_attributes" not in span:
            matches = False
            continue
        span_attrs = span.get("span_attributes") or {}
        for key, value in attributes.items():
            if key not in span_attrs or span_attrs.get(key) != value:
                matches = False
                break
        if matches:
            matching_spans.append(span)
    return matching_spans
