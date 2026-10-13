import importlib.metadata
import sys
from unittest.mock import patch

import pytest
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


@pytest.mark.skipif(sys.version_info < (3, 15), reason="importlib.metadata.MetadataNotFound is new in Python 3.15")
def test_sdk_version_tolerates_dist_info_without_metadata():
    # On 3.15, a lone .dist-info without METADATA (e.g. an interrupted install)
    # raises MetadataNotFound, a FileNotFoundError rather than a
    # PackageNotFoundError. The SDK version lookup runs on every span.
    span_origin._sdk_version.cache_clear()
    try:
        with patch(
            "braintrust.span_origin.importlib.metadata.version",
            side_effect=importlib.metadata.MetadataNotFound(  # pylint: disable=no-member
                "No package metadata was found."
            ),
        ):
            assert span_origin.merge_span_origin_context({}, "test", None)["span_origin"]["version"] == "unknown"
    finally:
        span_origin._sdk_version.cache_clear()
