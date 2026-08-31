"""Deterministic identity and privacy-safe normalization helpers."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid5


PLUGIN_NAMESPACE = UUID("67ea9f8a-e42a-5f31-96d8-85bcf27ca4c9")
_SECRET_KEY = re.compile(r"(?:api[_-]?key|token|secret|password|credential|authorization|cookie)", re.IGNORECASE)
_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_TEMPLATE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}$")
_KEY_SEGMENT = re.compile(r"[^0-9A-Za-z]+|(?<=[a-z0-9])(?=[A-Z])")
# Keys built from a counting or budgeting word are measurements, not credentials.
# "max_tokens" and "total_tokens" match the secret pattern as substrings, and
# redacting them both destroys usage metadata and collapses agent configurations
# that differ only by a token budget into one partition.
_COUNTER_SEGMENTS = frozenset(
    {
        "average",
        "avg",
        "budget",
        "cache",
        "cached",
        "completion",
        "count",
        "counts",
        "input",
        "limit",
        "max",
        "maximum",
        "min",
        "minimum",
        "num",
        "output",
        "per",
        "prompt",
        "reasoning",
        "remaining",
        "size",
        "sum",
        "total",
        "usage",
        "used",
        "window",
    }
)


@dataclass(frozen=True)
class NormalizedValue:
    value: Any
    warnings: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Report whether the value survived normalization intact.

        Every warning this module emits names data it dropped, truncated, or
        replaced, so the absence of warnings is the completeness signal.
        """
        return not self.warnings


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_id(scope: str, value: str) -> str:
    return str(uuid5(PLUGIN_NAMESPACE, f"{scope}:{value}"))


def dataset_record_id(dataset_scope: str, logical_task_key: str) -> str:
    return deterministic_id("dataset-record", f"{dataset_scope}:{logical_task_key}")


def child_span_id(trial_id: str, semantic_path: str) -> str:
    try:
        namespace = UUID(str(trial_id))
    except ValueError:
        namespace = uuid5(PLUGIN_NAMESPACE, str(trial_id))
    return str(uuid5(namespace, semantic_path))


def _is_absolute_path(value: str) -> bool:
    return value.startswith(("/", "~/", "file://")) or bool(_ABSOLUTE_WINDOWS_PATH.match(value))


def _key_segments(key: str) -> set[str]:
    return {segment.lower() for segment in _KEY_SEGMENT.split(key) if segment}


def _is_secret_key(key: str, value: Any) -> bool:
    """Report whether a key names a credential whose value must not be logged."""
    if value is None or isinstance(value, (bool, int, float)):
        # A number is never a credential, so redacting it can only lose data.
        return False
    if not _SECRET_KEY.search(key):
        return False
    return _key_segments(key).isdisjoint(_COUNTER_SEGMENTS)


def try_parse_json(data: bytes) -> tuple[Any, bool]:
    """Parse a task-controlled document, reporting failure rather than raising.

    json.loads raises RecursionError, not JSONDecodeError, for a deeply nested
    document, and a task can write one to any file this package reads.
    """
    try:
        return json.loads(data), True
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None, False


def _json_size(value: Any) -> int:
    try:
        return len(canonical_json(value).encode("utf-8"))
    except (TypeError, ValueError):
        return 10**18


def normalize_json(
    value: Any,
    *,
    max_bytes: int | None,
    redact_patterns: tuple[str, ...] = (),
    max_depth: int = 8,
    redact_absolute_paths: bool = True,
) -> NormalizedValue:
    """Normalize untrusted metadata while preserving JSON types and nulls.

    Set ``redact_absolute_paths=False`` for payloads produced inside the task
    sandbox: their absolute paths are container paths the agent actually operated
    on, so redacting them erases the substance of filesystem tool calls.

    ``max_bytes=None`` redacts without truncating, for payloads bound for an
    attachment rather than a span field.
    """
    warnings: list[str] = []
    compiled_patterns = tuple(re.compile(pattern) for pattern in redact_patterns)

    def walk(item: Any, path: str, depth: int, key: str | None = None) -> Any:
        if depth > max_depth:
            warnings.append(f"dropped {path}: depth limit")
            return "[DROPPED: depth limit]"
        if key is not None and _is_secret_key(key, item):
            if isinstance(item, str) and _TEMPLATE.match(item):
                return item
            warnings.append(f"redacted {path}: sensitive key")
            return "[REDACTED]"
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            if redact_absolute_paths and _is_absolute_path(item):
                warnings.append(f"dropped {path}: absolute path")
                return "[REDACTED PATH]"
            result = item
            for pattern in compiled_patterns:
                result = pattern.sub("[REDACTED]", result)
            return result
        if isinstance(item, PurePath):
            raw = str(item)
            if redact_absolute_paths and (item.is_absolute() or _is_absolute_path(raw)):
                warnings.append(f"dropped {path}: absolute path")
                return "[REDACTED PATH]"
            return item.as_posix()
        if isinstance(item, dict):
            normalized: dict[str, Any] = {}
            for raw_key, child in item.items():
                child_key = str(raw_key)
                child_path = f"{path}.{child_key}" if path else child_key
                normalized[child_key] = walk(child, child_path, depth + 1, child_key)
            return normalized
        if isinstance(item, (list, tuple)):
            return [walk(child, f"{path}[{index}]", depth + 1) for index, child in enumerate(item)]
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            try:
                return walk(model_dump(mode="json", exclude_none=False), path, depth)
            except Exception:
                pass
        warnings.append(f"dropped {path}: unsupported type {type(item).__name__}")
        return f"[DROPPED: {type(item).__name__}]"

    normalized = walk(value, "", 0)
    if max_bytes is None or _json_size(normalized) <= max_bytes:
        return NormalizedValue(normalized, tuple(warnings))

    # Fitting a container to a byte budget needs each entry's serialized size, not
    # a fresh serialization of every prefix: the payload is serialized again on its
    # way to Braintrust, so measuring it here more than once is pure overhead.
    # Canonical JSON adds two braces or brackets plus one separator between
    # entries, and one colon per object key, so entry sizes accumulate exactly.
    warnings.append(f"truncated value: exceeded {max_bytes} bytes")
    if isinstance(normalized, dict):
        bounded: dict[str, Any] = {}
        total = 2
        for key in sorted(normalized):
            entry = _json_size(key) + 1 + _json_size(normalized[key]) + (1 if bounded else 0)
            if total + entry > max_bytes:
                warnings.append(f"dropped {key}: size limit")
                continue
            total += entry
            bounded[key] = normalized[key]
        normalized = bounded
    elif isinstance(normalized, str):
        normalized = normalized.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    elif isinstance(normalized, list):
        # Keep the leading elements so an oversized list stays a list. Replacing
        # the whole value with a placeholder string would change its JSON type.
        total = 2
        kept = 0
        for index, item in enumerate(normalized):
            total += _json_size(item) + (1 if index else 0)
            if total > max_bytes:
                warnings.append(f"dropped [{index}:]: size limit")
                break
            kept = index + 1
        normalized = normalized[:kept]
    else:
        normalized = "[DROPPED: size limit]"
    return NormalizedValue(normalized, tuple(warnings))


def safe_git_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path.rstrip("/"), "", ""))


def logical_task_key(task_config: Any, task_lock: Any | None = None) -> str:
    """Choose a stable logical task identity without exposing local paths."""
    task = getattr(task_config, "task", None)
    if task is not None:
        name = getattr(task, "name", None)
        ref = getattr(task, "ref", None)
        if name:
            return f"package:{name}@{ref or 'default'}"

    git_url = getattr(task, "git_url", None)
    path = getattr(task, "path", None)
    if git_url:
        relative = Path(path).as_posix().lstrip("/") if path is not None else ""
        return f"git:{safe_git_url(git_url)}#{relative}"

    lock_name = getattr(getattr(task_lock, "task", task_lock), "name", None)
    source = getattr(task, "source", None) or getattr(getattr(task_lock, "task", task_lock), "source", None)
    if lock_name:
        return f"harbor:{source or 'adhoc'}:{lock_name}"
    name = Path(path).name if path is not None else "task"
    return f"local:{source or 'adhoc'}:{name}"


def dataset_scope(source: str) -> str:
    """Scope a dataset by task source alone.

    The resolved task subset is deliberately excluded. It changes whenever a job
    runs a subset of the source's tasks, or whenever backfill cannot read one
    trial's result, and because this scope also feeds record IDs, partition keys,
    and the experiment name, including it would fork a new dataset and experiment
    instead of reconciling the existing ones. Records are keyed per logical task,
    so a narrower run upserts a subset of rows.
    """
    return f"{source}:tasks"


def dataset_display_name(source: str, prefix: str = "harbor") -> str:
    return f"{prefix} · {source}"


def semantic_agent_config(agent: Any, skills: list[Any]) -> dict[str, Any]:
    def safe_env(raw: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in (raw or {}).items():
            value = str(value)
            if _is_secret_key(str(key), value):
                result[str(key)] = value if _TEMPLATE.match(value) else f"${{{key}}}"
            else:
                result[str(key)] = value
        return result

    raw = {
        "name": getattr(agent, "name", None),
        "import_path": getattr(agent, "import_path", None),
        "model": getattr(agent, "model_name", None),
        "kwargs": getattr(agent, "kwargs", None) or {},
        "env": safe_env(getattr(agent, "env", None)),
        "mcp_servers": getattr(agent, "mcp_servers", None) or [],
        "resume_trajectory": bool(getattr(agent, "resume_trajectory", False)),
        "load_trajectory": getattr(agent, "load_trajectory", None),
        "skills": sorted(
            [
                {
                    "name": getattr(skill, "name", None),
                    "digest": getattr(skill, "digest", None),
                    "git_url": safe_git_url(str(getattr(skill, "git_url")))
                    if getattr(skill, "git_url", None)
                    else None,
                    "git_commit_id": getattr(skill, "git_commit_id", None),
                }
                for skill in skills
            ],
            key=canonical_json,
        ),
    }
    return normalize_json(raw, max_bytes=200_000).value


def partition_key(dataset_key: str, agent_config: dict[str, Any]) -> str:
    return stable_hash({"dataset": dataset_key, "agent": agent_config})
