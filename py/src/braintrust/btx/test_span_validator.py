import pytest

from .span_validator import _validate_value
from .spec_loader import FnMatcher, OrMatcher


@pytest.mark.parametrize(
    "actual, expected, valid",
    [
        ({}, FnMatcher("undefined_or_null"), True),
        ({"metric": None}, FnMatcher("undefined_or_null"), True),
        ({"metric": 0}, FnMatcher("undefined_or_null"), False),
        ({"metric": 42}, FnMatcher("undefined_or_null"), False),
        ({}, FnMatcher("is_non_negative_number"), False),
        ({}, 0, False),
        ({}, OrMatcher([FnMatcher("is_positive_number"), FnMatcher("undefined_or_null")]), True),
        ({"metric": 0}, OrMatcher([FnMatcher("is_positive_number"), FnMatcher("undefined_or_null")]), False),
    ],
)
def test_absent_metric_matcher(actual, expected, valid):
    errors = []
    _validate_value(actual, {"metric": expected}, "metrics", errors)
    if valid:
        assert errors == []
    else:
        assert errors and all("metrics.metric" in error for error in errors)
