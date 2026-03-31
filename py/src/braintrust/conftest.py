import os
from unittest.mock import MagicMock

import pytest
from braintrust.framework2 import ProjectIdCache


def _patch_vcr_aiohttp_stubs():
    """Patch VCR.py's aiohttp stubs to fix bugs with google-genai >= 1.64.0 and litellm.

    Problems fixed:
    1. Infinite loop: VCR's MockClientResponse.content is a @property that creates
       a new MockStream on every access. google-genai reads streaming responses via
       `while True: await response.content.readline()`, creating a fresh stream each
       iteration that never reaches EOF. Fix: cache the stream per instance.

    2. Gzip decoding: Cassettes store gzip-compressed response bodies (from
       Accept-Encoding: gzip). VCR's httpx stubs handle decompression, but the
       aiohttp stubs return raw gzip bytes, causing UnicodeDecodeError.
       Fix: decompress gzip in text(), read(), and the content stream.

    3. set_exception: aiohttp's close() sets a ClientConnectionError on the content
       stream, which then raises on subsequent reads. Fix: no-op set_exception on
       MockStream.

    4. Body consumed during recording: VCR's record_response() calls
       `await response.read()` which consumes the real response body. When
       litellm's aiohttp transport then tries to stream the response, the body
       is empty. Fix: after reading the body for recording, reset the response's
       content stream so it can be read again by the caller.

    See: https://github.com/kevin1024/vcrpy/issues/927
    """
    try:
        from vcr.stubs import aiohttp_stubs
    except ImportError:
        return

    if getattr(aiohttp_stubs.MockClientResponse, "_bt_patched", False):
        return

    import asyncio
    import gzip

    from aiohttp import ClientConnectionError, streams

    def _decompress_body(body):
        """Decompress gzip body if needed."""
        if body and body[:2] == b"\x1f\x8b":
            return gzip.decompress(body)
        return body

    aiohttp_stubs.MockStream.set_exception = lambda self, exc: None

    async def patched_text(self, encoding="utf-8", errors="strict"):
        return _decompress_body(self._body).decode(encoding, errors=errors)

    aiohttp_stubs.MockClientResponse.text = patched_text

    async def patched_read(self):
        return _decompress_body(self._body)

    aiohttp_stubs.MockClientResponse.read = patched_read

    @property
    def cached_content(self):
        if not hasattr(self, "_cached_content"):
            stream = aiohttp_stubs.MockStream()
            stream.feed_data(_decompress_body(self._body))
            stream.feed_eof()
            self._cached_content = stream
        return self._cached_content

    aiohttp_stubs.MockClientResponse.content = cached_content
    aiohttp_stubs.MockClientResponse._bt_patched = True

    # Patch record_response to not consume the response body. VCR's original
    # implementation calls `await response.read()` which exhausts the body,
    # making it unavailable for the actual caller (e.g. litellm's aiohttp
    # transport). We read the body, record it, then reset the content stream
    # so downstream consumers can still read it.
    _original_record_response = aiohttp_stubs.record_response

    async def _patched_record_response(cassette, vcr_request, response):
        try:
            body_bytes = await response.read()
        except ClientConnectionError:
            body_bytes = b""

        vcr_response = {
            "status": {"code": response.status, "message": response.reason},
            "headers": aiohttp_stubs._serialize_headers(response.headers),
            "body": {"string": body_bytes} if body_bytes else {},
        }
        cassette.append(vcr_request, vcr_response)

        # Reset the response content stream so the caller can still read it.
        # aiohttp's ClientResponse stores the payload in response.content which
        # is a StreamReader. After read() exhausts it, we replace it with a new
        # stream containing the same data.
        new_stream = streams.StreamReader(response._protocol, 2**16, loop=asyncio.get_event_loop())
        new_stream.feed_data(body_bytes)
        new_stream.feed_eof()
        response.content = new_stream

    aiohttp_stubs.record_response = _patched_record_response


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


@pytest.fixture(autouse=True)
def reset_braintrust_state():
    """Reset all Braintrust global state after each test."""
    yield
    from braintrust import logger

    logger._state = logger.BraintrustState()


@pytest.fixture(autouse=True)
def skip_vcr_tests_in_wheel_mode(request):
    """Skip VCR tests when running from an installed wheel.

    Wheel mode (BRAINTRUST_TESTING_WHEEL=1) is a pre-release sanity check
    that verifies the built package installs and runs correctly. It's not
    intended to be a full test suite - VCR cassettes are not included in
    the wheel, so we skip those tests here. The full test suite with VCR
    tests runs against source code during normal CI.
    """
    if os.environ.get("BRAINTRUST_TESTING_WHEEL") == "1":
        if request.node.get_closest_marker("vcr"):
            pytest.skip("VCR tests skipped in wheel mode (pre-release sanity check only)")


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
            "x-api-key",
            "api-key",
            "openai-api-key",
            "x-goog-api-key",
            "x-bt-auth-token",
        ],
    }


@pytest.fixture(scope="session")
def vcr_config():
    """Pytest fixture wrapper for get_vcr_config()."""
    return get_vcr_config()
