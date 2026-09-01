"""Configuration for the Harbor job plugin."""

import fnmatch
import json
import math
import os
from dataclasses import dataclass, field, fields
from typing import Any


_UNSET = object()
_PREFIX = "HARBOR_BRAINTRUST_"


def _environment_value(name: str, default: Any) -> Any:
    names = [f"{_PREFIX}{name.upper()}"]
    if name == "project_name":
        names.append(f"{_PREFIX}PROJECT")
    for environment_name in names:
        value = os.environ.get(environment_name)
        if value is not None:
            return value
    return default


def _resolve(value: Any, name: str, default: Any) -> Any:
    return _environment_value(name, default) if value is _UNSET else value


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _parse_json(value: Any, name: str, expected_type: type) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"{name} must be a JSON {expected_type.__name__}")
    return value


def _parse_patterns(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    value = _parse_json(value, name, list) if isinstance(value, str) else value
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a JSON array of non-empty strings")
    return tuple(value)


def _patterns_overlap(left: str, right: str) -> bool:
    # Exact equality and either pattern matching the other catch all useful,
    # deterministic overlap cases without pretending to solve glob intersection.
    return left == right or fnmatch.fnmatchcase(left, right) or fnmatch.fnmatchcase(right, left)


@dataclass(frozen=True)
class PluginConfig:
    project_name: str | None = None
    project_id: str | None = None
    experiment_prefix: str | None = None
    base_experiment_name: str | None = None
    base_experiment_id: str | None = None
    dataset_mode: str = "sync"
    dataset_name: str | None = None
    trajectory_mode: str = "atif"
    content_mode: str = "messages"
    include_custom_metadata: bool = True
    max_custom_metadata_bytes: int = 100_000
    score_keys: tuple[str, ...] = ()
    metric_keys: tuple[str, ...] = ()
    reward_rules: dict[str, dict[str, Any]] = field(default_factory=dict)
    classifier_rules: dict[str, str] = field(default_factory=dict)
    invalid_score_policy: str = "metric"
    include_tracebacks: bool = False
    attachments: str = "all"
    artifact_include: tuple[str, ...] = ()
    max_content_bytes: int = 20_000
    max_trajectory_bytes: int = 20_000_000
    log_retry_attempts: bool = False
    strict: bool = False
    redact_patterns: tuple[str, ...] = ()

    @classmethod
    def from_options(cls, **options: Any) -> "PluginConfig":
        defaults = cls()
        values: dict[str, Any] = {}
        for config_field in fields(defaults):
            name = config_field.name
            values[name] = _resolve(options.get(name, _UNSET), name, getattr(defaults, name))

        for name in (
            "include_custom_metadata",
            "include_tracebacks",
            "log_retry_attempts",
            "strict",
        ):
            values[name] = _parse_bool(values[name], name)
        for name in (
            "max_custom_metadata_bytes",
            "max_content_bytes",
            "max_trajectory_bytes",
        ):
            values[name] = _parse_int(values[name], name)
        for name in ("score_keys", "metric_keys", "artifact_include", "redact_patterns"):
            values[name] = _parse_patterns(values[name], name)
        for name in ("reward_rules", "classifier_rules"):
            values[name] = _parse_json(values[name], name, dict) or {}

        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.project_name and self.project_id:
            raise ValueError("project_name and project_id are mutually exclusive")
        if self.base_experiment_name and self.base_experiment_id:
            raise ValueError("base_experiment_name and base_experiment_id are mutually exclusive")
        if self.dataset_mode not in {"sync", "none"}:
            raise ValueError("dataset_mode must be 'sync' or 'none'; 'existing' is not supported yet")
        if self.dataset_name and self.dataset_mode != "sync":
            raise ValueError("dataset_name requires dataset_mode='sync'")
        if self.trajectory_mode not in {"atif", "summary", "native"}:
            raise ValueError("trajectory_mode must be 'atif', 'summary', or 'native'")
        # 'full' is 'messages' plus fields the instrumentation contract explicitly
        # allows. No such field exists yet, so the two capture the same payload.
        if self.content_mode not in {"metadata", "messages", "full"}:
            raise ValueError("content_mode must be 'metadata', 'messages', or 'full'")
        if self.log_retry_attempts:
            raise ValueError("log_retry_attempts=True is not implemented; only the final attempt is logged")
        if self.invalid_score_policy not in {"metric", "drop", "error"}:
            raise ValueError("invalid_score_policy must be 'metric', 'drop', or 'error'")
        if self.attachments not in {"none", "structured", "all"}:
            raise ValueError("attachments must be 'none', 'structured', or 'all'")
        if self.artifact_include and self.attachments != "all":
            raise ValueError("artifact_include requires attachments='all'")

        for score_pattern in self.score_keys:
            for metric_pattern in self.metric_keys:
                if _patterns_overlap(score_pattern, metric_pattern):
                    raise ValueError(f"score_keys and metric_keys overlap: {score_pattern!r}, {metric_pattern!r}")

        for key, rule in self.reward_rules.items():
            if not isinstance(key, str) or not key or not isinstance(rule, dict):
                raise ValueError("reward_rules must map non-empty strings to objects")
            rule_type = rule.get("type")
            if rule_type not in {"score", "metric"}:
                raise ValueError(f"reward_rules[{key!r}].type must be 'score' or 'metric'")
            if rule_type == "metric" and any(field in rule for field in ("direction", "min", "max", "score_name")):
                raise ValueError(f"metric reward rule {key!r} cannot define score normalization")
            if rule_type == "score":
                direction = rule.get("direction", "maximize")
                if direction not in {"maximize", "minimize"}:
                    raise ValueError(f"reward_rules[{key!r}].direction must be 'maximize' or 'minimize'")
                has_min, has_max = "min" in rule, "max" in rule
                if has_min != has_max:
                    raise ValueError(f"reward_rules[{key!r}] must define both min and max")
                if has_min:
                    minimum, maximum = rule["min"], rule["max"]
                    if (
                        isinstance(minimum, bool)
                        or isinstance(maximum, bool)
                        or not isinstance(minimum, (int, float))
                        or not isinstance(maximum, (int, float))
                        or not math.isfinite(float(minimum))
                        or not math.isfinite(float(maximum))
                        or minimum >= maximum
                    ):
                        raise ValueError(f"reward_rules[{key!r}] requires finite min < max")

        if not all(
            isinstance(name, str) and name and isinstance(path, str) and path
            for name, path in self.classifier_rules.items()
        ):
            raise ValueError("classifier_rules must map non-empty names to non-empty JSON paths")
