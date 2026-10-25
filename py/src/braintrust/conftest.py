import os
from unittest.mock import MagicMock

import pytest
from braintrust import _test_cassette_usage
from braintrust.framework2 import ProjectIdCache


# Record cassette reads when check-unused-cassettes.py is collecting usage.
_test_cassette_usage.install()


def _skip_reason(report) -> str:
    longrepr = report.longrepr
    return longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and not hasattr(report, "wasxfail"):
        _test_cassette_usage.record_skip(item.path, item.nodeid, _skip_reason(report))


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    # Whole modules skipped at collection (e.g. importorskip) never reach makereport.
    outcome = yield
    report = outcome.get_result()
    if report.skipped:
        _test_cassette_usage.record_skip(collector.path, report.nodeid, _skip_reason(report))


def _patch_vcr_aiohttp_stubs():
    """Patch VCR.py's aiohttp stubs to decompress gzip response bodies.

    Cassettes store gzip-compressed response bodies (from Accept-Encoding: gzip).
    VCR's httpx stubs handle decompression, but the aiohttp stubs return raw gzip
    bytes, causing UnicodeDecodeError. Fix: decompress gzip in text() and read().
    """
    try:
        from vcr.stubs import aiohttp_stubs
    except ImportError:
        return

    if getattr(aiohttp_stubs.MockClientResponse, "_bt_patched", False):
        return

    import gzip

    def _decompress_body(body):
        """Decompress gzip body if needed."""
        if body and body[:2] == b"\x1f\x8b":
            return gzip.decompress(body)
        return body

    async def patched_text(self, encoding="utf-8", errors="strict"):
        return _decompress_body(self._body).decode(encoding, errors=errors)

    aiohttp_stubs.MockClientResponse.text = patched_text

    async def patched_read(self):
        return _decompress_body(self._body)

    aiohttp_stubs.MockClientResponse.read = patched_read
    aiohttp_stubs.MockClientResponse._bt_patched = True


_patch_vcr_aiohttp_stubs()


@pytest.fixture
def mock_project_ids():
    mock = MagicMock(spec=ProjectIdCache)
    mock.get.return_value = "project-123"
    mock.get_by_name.return_value = "project-123"
    return mock


@pytest.fixture(autouse=True)
def override_app_url_for_tests():
    """
    Temporarily override BRAINTRUST_APP_URL to production URL for consistent test behavior.

    This fixture ensures that tests always use the production URL (https://www.braintrust.dev)
    regardless of the local development environment settings. This prevents test failures
    when BRAINTRUST_APP_URL is set to localhost for development.
    """
    original_app_url = os.environ.get("BRAINTRUST_APP_URL")
    original_app_public_url = os.environ.get("BRAINTRUST_APP_PUBLIC_URL")

    # Set to production URL for consistent test behavior
    os.environ["BRAINTRUST_APP_URL"] = "https://www.braintrust.dev"
    if "BRAINTRUST_APP_PUBLIC_URL" in os.environ:
        del os.environ["BRAINTRUST_APP_PUBLIC_URL"]

    try:
        yield
    finally:
        # Restore original environment variables
        if original_app_url is not None:
            os.environ["BRAINTRUST_APP_URL"] = original_app_url
        elif "BRAINTRUST_APP_URL" in os.environ:
            del os.environ["BRAINTRUST_APP_URL"]

        if original_app_public_url is not None:
            os.environ["BRAINTRUST_APP_PUBLIC_URL"] = original_app_public_url


@pytest.fixture(autouse=True)
def setup_braintrust():
    os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "your_google_api_key_here"))
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-api-key-for-vcr-tests")
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-dummy-api-key-for-vcr-tests")
    os.environ.setdefault("MISTRAL_API_KEY", "mistral-test-dummy-api-key-for-vcr-tests")
    os.environ.setdefault("TYPESAFE_API_KEY", "typesafe-test-dummy-api-key-for-vcr-tests")
    os.environ.setdefault("CO_API_KEY", os.getenv("COHERE_API_KEY", "co-test-dummy-api-key-for-vcr-tests"))
    os.environ.setdefault("COHERE_API_KEY", os.getenv("CO_API_KEY", "co-test-dummy-api-key-for-vcr-tests"))


@pytest.fixture(autouse=True)
def reset_braintrust_state():
    """Reset all Braintrust global state after each test."""
    yield
    from braintrust import logger

    logger._state = logger.BraintrustState()


_SENSITIVE_RESPONSE_HEADERS = {
    "openai-organization",
    "openai-project",
    "set-cookie",
}


def _scrub_sensitive_response_headers(response):
    """Remove sensitive provider headers before writing a cassette."""
    response["headers"] = {
        name: value
        for name, value in response.get("headers", {}).items()
        if name.lower() not in _SENSITIVE_RESPONSE_HEADERS
    }
    return response


def get_vcr_config():
    """
    Get VCR configuration for recording/playing back HTTP interactions.

    In CI, use "none" to fail if cassette is missing.
    Locally, use "once" to record new cassettes if they don't exist.
    """
    record_mode = "none" if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) else "once"
    return {
        "record_mode": record_mode,
        "decode_compressed_response": True,
        "filter_headers": [
            "authorization",
            "Authorization",
            "openai-organization",
            "openai-project",
            "cookie",
            "Cookie",
            "x-api-key",
            "api-key",
            "openai-api-key",
            "x-goog-api-key",
            "x-bt-auth-token",
            "x-amz-security-token",
            "x-amz-date",
            "x-amz-content-sha256",
            "amz-sdk-invocation-id",
            "amz-sdk-request",
            "x-amzn-bedrock-api-key",
        ],
        "before_record_response": _scrub_sensitive_response_headers,
    }


@pytest.fixture(scope="session")
def vcr_config():
    """Pytest fixture wrapper for get_vcr_config()."""
    return get_vcr_config()


@pytest.fixture
def vcr_cassette_name(request):
    """Name of the VCR cassette, overridable with ``@pytest.mark.vcr(cassette_name=...)``.

    pytest-vcr forwards marker kwargs to vcrpy as config and ignores
    ``cassette_name``, so without this override tests that share a cassette
    silently record and replay their own.
    """
    marker = request.node.get_closest_marker("vcr")
    if marker and "cassette_name" in marker.kwargs:
        return marker.kwargs["cassette_name"]
    if request.cls:
        return f"{request.cls.__name__}.{request.node.name}"
    return request.node.name
