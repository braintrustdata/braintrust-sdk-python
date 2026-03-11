"""Test-only cassette transport for Claude Agent SDK subprocess traffic."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import anyio

try:
    import claude_agent_sdk
    from claude_agent_sdk._internal.transport import Transport
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    _CLAUDE_AGENT_SDK_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    claude_agent_sdk = None
    Transport = object
    SubprocessCLITransport = None
    _CLAUDE_AGENT_SDK_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from claude_agent_sdk.types import ClaudeAgentOptions

CASSETTES_DIR = Path(
    os.environ.get(
        "BRAINTRUST_CLAUDE_AGENT_SDK_CASSETTES_DIR",
        Path(__file__).resolve().parent / "cassettes",
    )
)


def get_record_mode() -> str:
    """Match VCR-like defaults for subprocess transport cassettes."""
    mode = os.environ.get("BRAINTRUST_CLAUDE_AGENT_SDK_RECORD_MODE")
    if mode:
        return mode
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return "none"
    return "once"


def _version_suffix() -> str:
    version = getattr(claude_agent_sdk, "__version__", "unknown")
    return version.replace(".", "_")


def _require_sdk() -> None:
    if _CLAUDE_AGENT_SDK_IMPORT_ERROR is not None:
        raise ImportError(
            "claude_agent_sdk is required to use braintrust.wrappers.claude_agent_sdk._test_transport"
        ) from _CLAUDE_AGENT_SDK_IMPORT_ERROR


def cassette_path(name: str) -> Path:
    return CASSETTES_DIR / f"{name}__sdk_{_version_suffix()}.json"


def _normalize_write(data: str, *, sanitize: bool = False) -> dict[str, Any]:
    payload = data.rstrip("\n")
    try:
        value = json.loads(payload)
        if sanitize:
            value = _normalize_for_match(value)
        return {"kind": "json", "value": value}
    except json.JSONDecodeError:
        return {"kind": "raw", "value": payload}


async def _empty_stream():
    return
    yield {}  # type: ignore[unreachable]


def _normalize_for_match(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_for_match(item) for item in value]
    if isinstance(value, dict):
        normalized = {key: _normalize_for_match(item) for key, item in value.items()}
        if normalized.get("type") == "control_request" and isinstance(normalized.get("request_id"), str):
            normalized["request_id"] = "CONTROL_REQUEST_ID"
        return normalized
    return value


def _sanitize_json_for_storage(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize_json_for_storage(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_field_for_storage(key, item) for key, item in value.items()}
    if isinstance(value, str):
        return _sanitize_string_for_storage(value)
    return value


def _sanitize_field_for_storage(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return _sanitize_json_for_storage(value)
    if key == "cwd":
        return "<REDACTED_CWD>"
    if key == "signature":
        return "<REDACTED_SIGNATURE>"
    if key.lower() in _SENSITIVE_FIELDS:
        return "<REDACTED>"
    return _sanitize_string_for_storage(value)


def _sanitize_string_for_storage(value: str) -> str:
    sanitized = _PATH_RE.sub("<REDACTED_PATH>", value)
    sanitized = _sanitize_url_string(sanitized)
    sanitized = _AUTH_BEARER_RE.sub("Bearer <REDACTED>", sanitized)
    sanitized = _API_KEY_RE.sub("sk-<REDACTED>", sanitized)
    return sanitized


def _sanitize_url_string(value: str) -> str:
    if "?" not in value or "://" not in value:
        return value

    try:
        parts = urlsplit(value)
    except ValueError:
        return value

    if not parts.query:
        return value

    query = []
    changed = False
    for key, item in parse_qsl(parts.query, keep_blank_values=True):
        if _SENSITIVE_QUERY_PARAM_RE.search(key):
            query.append((key, "<REDACTED>"))
            changed = True
        else:
            query.append((key, item))

    if not changed:
        return value

    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment))


_PATH_RE = re.compile(r"(?:(?:/Users|/home)/[^\s\"']+|[A-Za-z]:\\\\[^\s\"']+)")
_AUTH_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._-]+")
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]+\b")
_SENSITIVE_FIELDS = {
    "authorization",
    "openai-organization",
    "x-api-key",
    "api-key",
    "openai-api-key",
    "x-goog-api-key",
    "x-bt-auth-token",
}
_SENSITIVE_QUERY_PARAM_RE = re.compile(
    r"(?:^|[_-])(token|api[_-]?key|auth|signature|sig|key)(?:$|[_-])",
    re.IGNORECASE,
)


class ClaudeAgentSdkCassetteTransport(Transport):
    """Record or replay the SDK<->CLI JSON protocol at the transport layer."""

    def __init__(
        self,
        *,
        cassette_name: str,
        prompt: str | Any,
        options: ClaudeAgentOptions,
        record_mode: str | None = None,
    ) -> None:
        _require_sdk()
        self._cassette_name = cassette_name
        self._cassette_path = cassette_path(cassette_name)
        self._prompt = prompt
        self._options = options
        self._record_mode = record_mode or get_record_mode()
        self._delegate: SubprocessCLITransport | None = None
        self._recording = False
        self._events: list[dict[str, Any]] = []
        self._cursor = 0
        self._ready = False
        self._cursor_lock = anyio.Lock()
        self._cursor_changed = anyio.Event()
        self._control_request_ids: dict[str, str] = {}

    async def connect(self) -> None:
        if self._ready:
            return

        if self._should_replay():
            cassette = json.loads(self._cassette_path.read_text(encoding="utf-8"))
            self._events = cassette.get("events", [])
            self._ready = True
            return

        if self._record_mode not in {"once", "all"}:
            raise FileNotFoundError(
                f"Cassette missing for {self._cassette_name}: {self._cassette_path}"
            )

        self._cassette_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = _empty_stream() if self._prompt == "" else self._prompt
        assert SubprocessCLITransport is not None
        self._delegate = SubprocessCLITransport(prompt=prompt, options=self._options)
        await self._delegate.connect()
        self._recording = True
        self._ready = True

    async def write(self, data: str) -> None:
        if self._recording:
            assert self._delegate is not None
            await self._delegate.write(data)
            self._events.append({"op": "write", "payload": _normalize_write(data)})
            return

        recorded = await self._wait_for_event("write")
        actual_raw = _normalize_write(data)
        actual = _normalize_write(data, sanitize=True)
        expected = _normalize_for_match(recorded["payload"])
        self._maybe_remap_control_request_id(recorded["payload"], actual_raw)
        if expected != actual:
            raise AssertionError(
                f"Write mismatch for {self._cassette_name}\nexpected: {expected}\nactual: {actual}"
            )

    def read_messages(self):
        return self._read_messages_impl()

    async def _read_messages_impl(self):
        if self._recording:
            assert self._delegate is not None
            async for message in self._delegate.read_messages():
                self._events.append({"op": "read", "payload": message})
                yield message
            return

        while True:
            event = await self._wait_for_event("read", allow_eof=True)
            if event is None:
                return
            yield self._remap_read_message(event["payload"])

    async def close(self) -> None:
        self._ready = False
        if self._recording:
            assert self._delegate is not None
            await self._delegate.close()
            payload = {
                "sdk_version": getattr(claude_agent_sdk, "__version__", "unknown"),
                "cassette_name": self._cassette_name,
                "events": _sanitize_json_for_storage(self._events),
            }
            self._cassette_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        if self._recording:
            assert self._delegate is not None
            await self._delegate.end_input()

    def _should_replay(self) -> bool:
        if self._record_mode == "all":
            return False
        if self._record_mode in {"once", "none"} and self._cassette_path.exists():
            return True
        return False

    async def _wait_for_event(
        self, op: str, *, allow_eof: bool = False
    ) -> dict[str, Any] | None:
        while True:
            async with self._cursor_lock:
                if self._cursor >= len(self._events):
                    if allow_eof:
                        return None
                    raise AssertionError(
                        f"Replay for {self._cassette_name} exhausted before expected {op}"
                    )

                event = self._events[self._cursor]
                if event["op"] == op:
                    self._cursor += 1
                    old_event = self._cursor_changed
                    self._cursor_changed = anyio.Event()
                    old_event.set()
                    return event

                waiter = self._cursor_changed

            await waiter.wait()

    def _maybe_remap_control_request_id(
        self, recorded_payload: dict[str, Any], actual_payload: dict[str, Any]
    ) -> None:
        if recorded_payload.get("kind") != "json" or actual_payload.get("kind") != "json":
            return

        recorded_value = recorded_payload.get("value")
        actual_value = actual_payload.get("value")
        if not isinstance(recorded_value, dict) or not isinstance(actual_value, dict):
            return
        if recorded_value.get("type") != "control_request" or actual_value.get("type") != "control_request":
            return

        recorded_id = recorded_value.get("request_id")
        actual_id = actual_value.get("request_id")
        if isinstance(recorded_id, str) and isinstance(actual_id, str):
            self._control_request_ids[recorded_id] = actual_id

    def _remap_read_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("type") != "control_response":
            return payload

        response = payload.get("response")
        if not isinstance(response, dict):
            return payload

        recorded_id = response.get("request_id")
        if not isinstance(recorded_id, str):
            return payload

        actual_id = self._control_request_ids.get(recorded_id)
        if not actual_id:
            return payload

        return {
            **payload,
            "response": {
                **response,
                "request_id": actual_id,
            },
        }


def make_cassette_transport(
    *,
    cassette_name: str,
    prompt: str | Any,
    options: "ClaudeAgentOptions",
    record_mode: str | None = None,
) -> ClaudeAgentSdkCassetteTransport:
    return ClaudeAgentSdkCassetteTransport(
        cassette_name=cassette_name,
        prompt=prompt,
        options=options,
        record_mode=record_mode,
    )
