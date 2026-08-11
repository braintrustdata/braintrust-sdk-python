"""Harbor reward and classifier conversion."""

import fnmatch
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from .config import PluginConfig


@dataclass(frozen=True)
class ScoreValue:
    name: str
    value: float
    source_key: str
    raw_value: int | float
    transformed: bool = False


@dataclass(frozen=True)
class RewardConversion:
    scores: tuple[ScoreValue, ...] = ()
    metrics: dict[str, int | float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _numeric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _matches(key: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)


def _metric_name(key: str) -> str:
    # These keys have standardized Braintrust meanings. Harbor rewards are not
    # presumed to share them, so retain an explicit integration namespace.
    standard = {
        "start",
        "end",
        "duration",
        "tokens",
        "prompt_tokens",
        "completion_tokens",
        "estimated_cost",
        "time_to_first_token",
    }
    return f"harbor_reward.{key}" if key in standard else key


def classify_rewards(rewards: dict[str, Any] | None, config: PluginConfig) -> RewardConversion:
    scores: list[ScoreValue] = []
    metrics: dict[str, int | float] = {}
    warnings: list[str] = []
    if not rewards:
        return RewardConversion()

    for key, raw in rewards.items():
        if not _numeric(raw):
            warnings.append(f"reward {key!r} is not a finite number and was omitted")
            continue
        value = float(raw)
        rule = config.reward_rules.get(key)
        requested_type: str | None = rule.get("type") if rule else None
        if requested_type is None:
            if _matches(key, config.score_keys):
                requested_type = "score"
            elif _matches(key, config.metric_keys):
                requested_type = "metric"
            elif key == "reward" and 0 <= value <= 1:
                requested_type = "score"
            else:
                requested_type = "metric"

        if requested_type == "metric":
            metrics[_metric_name(key)] = raw
            continue

        score_name = str(rule.get("score_name", key)) if rule else key
        transformed = False
        normalized = value
        if rule and "min" in rule and "max" in rule:
            minimum, maximum = float(rule["min"]), float(rule["max"])
            if minimum <= value <= maximum:
                direction = rule.get("direction", "maximize")
                normalized = (value - minimum) / (maximum - minimum)
                if direction == "minimize":
                    normalized = (maximum - value) / (maximum - minimum)
                transformed = True
            else:
                normalized = float("nan")

        if not math.isfinite(normalized) or not 0 <= normalized <= 1:
            warning = f"configured score {key!r} has invalid value {raw!r}"
            if config.invalid_score_policy == "error":
                raise ValueError(warning)
            warnings.append(warning)
            if config.invalid_score_policy == "metric":
                metrics[_metric_name(key)] = raw
            continue

        scores.append(
            ScoreValue(
                name=score_name,
                value=normalized,
                source_key=key,
                raw_value=raw,
                transformed=transformed,
            )
        )
        if transformed:
            metrics[f"harbor_reward.raw.{key}"] = raw

    return RewardConversion(tuple(scores), metrics, tuple(warnings))


def extract_json_path(value: Any, path: str) -> Any:
    """Extract a documented dotted/JSON-pointer-like path from a model or mapping."""
    parts = [part for part in path.replace("/", ".").split(".") if part]
    current = value
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(path)
            current = current[part]
        elif isinstance(current, (list, tuple)):
            current = current[int(part)]
        else:
            model_dump = getattr(current, "model_dump", None)
            if callable(model_dump):
                current = model_dump(mode="python", exclude_none=False)
                if part not in current:
                    raise KeyError(path)
                current = current[part]
            else:
                raise KeyError(path)
    return current


def validate_classifications(value: Any) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else [value]
    validated: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("classification items require a non-empty string id")
        if "label" in item and item["label"] is not None and not isinstance(item["label"], str):
            raise ValueError("classification item label must be a string or null")
        if "metadata" in item and item["metadata"] is not None and not isinstance(item["metadata"], dict):
            raise ValueError("classification item metadata must be a JSON object or null")
        validated.append({key: item[key] for key in ("id", "label", "metadata") if key in item})
    return validated
