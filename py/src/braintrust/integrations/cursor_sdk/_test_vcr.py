"""Cursor bridge request scrubbing for pytest-vcr cassettes."""

import json
import re
import struct
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

from braintrust.conftest import get_vcr_config


# `cursor_sdk._tool_callback.TOOL_CALLBACK_SERVICE`, inlined so this module stays
# importable without the SDK installed.
_TOOL_CALLBACK_PATH_PREFIX = "/sdk.v1.SdkCustomToolCallbackService/"
_SENSITIVE_KEY_RE = re.compile(r"(?:api.?key|authorization|auth.?token|secret|password|env.?vars|headers)$", re.I)
_PATH_KEY_RE = re.compile(r"(?:cwd|workspace|state.?root)$", re.I)
_IDEMPOTENCY_KEY_RE = re.compile(r"idempotency.?key$", re.I)


def _scrub_value(value, *, key=""):
    if key == "data" and isinstance(value, str) and len(value) >= 20:
        return "<BASE64_DATA>"
    if _SENSITIVE_KEY_RE.search(key):
        return "<REDACTED>"
    if _PATH_KEY_RE.search(key):
        return "<WORKSPACE>"
    if _IDEMPOTENCY_KEY_RE.search(key):
        return "<IDEMPOTENCY_KEY>"
    if isinstance(value, Mapping):
        return {str(item_key): _scrub_value(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_value(item, key=key) for item in value]
    return value


def scrub_cursor_bridge_request(request):
    """Redact secrets/paths and normalize the bridge's dynamic loopback URL."""
    parts = urlsplit(request.uri)
    # Let shutdown reach the real test bridge. Replaying this request would
    # leave the bridge's child Node process running after the test exits.
    if parts.path.endswith("/Shutdown"):
        return None
    # Custom tools arrive as a request *into* the SDK's loopback callback
    # server, which VCR cannot record. Let the test drive that server for real
    # instead of matching these against the cassette.
    if parts.path.startswith(_TOOL_CALLBACK_PATH_PREFIX):
        return None
    request.uri = urlunsplit((parts.scheme, "cursor-sdk-bridge", parts.path, parts.query, parts.fragment))
    body = request.body
    if not isinstance(body, bytes) or not body:
        return request
    is_connect_stream = request.headers.get("Content-Type") == "application/connect+json"
    prefix = body[:5] if is_connect_stream else b""
    payload = body[5:] if is_connect_stream else body
    try:
        scrubbed = json.dumps(_scrub_value(json.loads(payload.decode("utf-8")))).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return request
    if is_connect_stream:
        prefix = bytes([prefix[0]]) + struct.pack(">I", len(scrubbed))
    request.body = prefix + scrubbed
    return request


def cursor_vcr_config():
    """Return standard VCR configuration specialized for Cursor bridge RPCs."""
    config = get_vcr_config()
    config["before_record_request"] = scrub_cursor_bridge_request
    config["filter_headers"] = [*config["filter_headers"], "host"]
    config["match_on"] = ["method", "path", "body"]
    return config
