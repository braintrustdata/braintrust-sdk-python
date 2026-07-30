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


@dataclass(frozen=True)
class NormalizedValue:
    value: Any
    warnings: tuple[str, ...] = ()


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


def _json_size(value: Any) -> int:
    try:
        return len(canonical_json(value).encode("utf-8"))
    except (TypeError, ValueError):
        return 10**18


def normalize_json(
    value: Any,
    *,
    max_bytes: int,
    redact_patterns: tuple[str, ...] = (),
    max_depth: int = 8,
) -> NormalizedValue:
    """Normalize untrusted metadata while preserving JSON types and nulls."""
    warnings: list[str] = []
    compiled_patterns = tuple(re.compile(pattern) for pattern in redact_patterns)

    def walk(item: Any, path: str, depth: int, key: str | None = None) -> Any:
        if depth > max_depth:
            warnings.append(f"dropped {path}: depth limit")
            return "[DROPPED: depth limit]"
        if key is not None and _SECRET_KEY.search(key):
            if isinstance(item, str) and _TEMPLATE.match(item):
                return item
            return "[REDACTED]"
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            if _is_absolute_path(item):
                warnings.append(f"dropped {path}: absolute path")
                return "[REDACTED PATH]"
            result = item
            for pattern in compiled_patterns:
                result = pattern.sub("[REDACTED]", result)
            return result
        if isinstance(item, PurePath):
            raw = str(item)
            if item.is_absolute() or _is_absolute_path(raw):
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
    if _json_size(normalized) <= max_bytes:
        return NormalizedValue(normalized, tuple(warnings))

    warnings.append(f"dropped metadata: exceeded {max_bytes} bytes")
    if isinstance(normalized, dict):
        bounded: dict[str, Any] = {}
        for key in sorted(normalized):
            candidate = {**bounded, key: normalized[key]}
            if _json_size(candidate) > max_bytes:
                warnings.append(f"dropped {key}: size limit")
                continue
            bounded[key] = normalized[key]
        normalized = bounded
    elif isinstance(normalized, str):
        normalized = normalized.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
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


def dataset_scope(source: str, logical_keys: list[str]) -> str:
    key_hash = stable_hash(sorted(logical_keys))[:8]
    return f"{source}:tasks-{key_hash}"


def dataset_display_name(source: str, logical_keys: list[str]) -> str:
    scope = dataset_scope(source, logical_keys)
    return f"harbor · {source} · {scope.rsplit(':', 1)[-1]}"


def semantic_agent_config(agent: Any, skills: list[Any]) -> dict[str, Any]:
    def safe_env(raw: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in (raw or {}).items():
            value = str(value)
            if _SECRET_KEY.search(str(key)):
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
