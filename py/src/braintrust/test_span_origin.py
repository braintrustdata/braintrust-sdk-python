from unittest.mock import patch

from braintrust import span_origin


def test_sdk_version_is_cached():
    span_origin._sdk_version.cache_clear()
    try:
        with patch("braintrust.span_origin.importlib.metadata.version", return_value="1.2.3") as version:
            assert span_origin.merge_span_origin_context({}, "test", None)["span_origin"]["version"] == "1.2.3"
            assert span_origin.merge_span_origin_context({}, "test", None)["span_origin"]["version"] == "1.2.3"

        version.assert_called_once_with("braintrust")
    finally:
        span_origin._sdk_version.cache_clear()
