"""Host-side ATIF to Braintrust span conversion."""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from braintrust.logger import Attachment

from .config import PluginConfig
from .identity import child_span_id, normalize_json


_INSTRUMENTATION = "braintrust.plugin.harbor"


@dataclass(frozen=True)
class ATIFImportResult:
    final_message: Any = None
    schema_version: str | None = None
    root_extra: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    repairs: tuple[str, ...] = ()
    imported_llm_spans: int = 0
    imported_tool_spans: int = 0


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, OverflowError):
        return None


def _step_times(steps: list[dict[str, Any]], start: float, end: float) -> tuple[list[float], list[str]]:
    if end < start:
        end = start
    repairs: list[str] = []
    parsed = [_timestamp(step.get("timestamp")) for step in steps]
    count = max(len(steps), 1)
    result: list[float] = []
    previous = start
    for index, value in enumerate(parsed):
        if value is None:
            value = start + (end - start) * index / count
            repairs.append(f"step {index + 1}: interpolated missing timestamp")
        clamped = min(max(value, start), end)
        if clamped != value:
            repairs.append(f"step {index + 1}: clamped timestamp to agent phase")
        if clamped < previous:
            clamped = previous
            repairs.append(f"step {index + 1}: repaired non-monotonic timestamp")
        result.append(clamped)
        previous = clamped
    return result, repairs


def _provider(model: str | None) -> tuple[str | None, str | None]:
    if not model:
        return None, None
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider.lower(), model_name
    return "unknown", model


def _valid_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _valid_cost(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
        else None
    )


def _usage_metrics(raw: Any) -> dict[str, int | float]:
    if not isinstance(raw, dict):
        return {}
    prompt = _valid_count(raw.get("prompt_tokens"))
    completion = _valid_count(raw.get("completion_tokens"))
    cached = _valid_count(raw.get("cached_tokens"))
    cost = _valid_cost(raw.get("cost_usd"))
    metrics: dict[str, int | float] = {}
    if prompt is not None:
        metrics["prompt_tokens"] = prompt
    if completion is not None:
        metrics["completion_tokens"] = completion
    if prompt is not None and completion is not None:
        metrics["tokens"] = prompt + completion
    if cached is not None:
        metrics["prompt_cached_tokens"] = cached
    if cost is not None:
        metrics["estimated_cost"] = cost
    extra = raw.get("extra")
    if isinstance(extra, dict):
        reasoning = _valid_count(extra.get("reasoning_tokens"))
        first_token_ms = _valid_cost(extra.get("time_to_first_token_ms"))
        cache_write = _valid_count(extra.get("cache_write_tokens"))
        if reasoning is not None:
            metrics["completion_reasoning_tokens"] = reasoning
        if first_token_ms is not None:
            metrics["time_to_first_token"] = first_token_ms / 1000
        if cache_write is not None:
            metrics["prompt_cache_creation_tokens"] = cache_write
    return metrics


def _bounded(value: Any, config: PluginConfig) -> Any:
    return normalize_json(
        value,
        max_bytes=config.max_content_bytes,
        redact_patterns=config.redact_patterns,
        max_depth=10,
    ).value


def _content(value: Any, trajectory_dir: Path, config: PluginConfig) -> tuple[Any, bool]:
    if isinstance(value, str) or value is None:
        return _bounded(value, config), True
    if not isinstance(value, list):
        return _bounded(value, config), False
    result: list[Any] = []
    complete = True
    for part in value:
        if not isinstance(part, dict):
            complete = False
            result.append(_bounded(part, config))
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            result.append({"type": "text", "text": _bounded(part["text"], config)})
            continue
        source = part.get("source")
        if part.get("type") == "image" and isinstance(source, dict) and isinstance(source.get("path"), str):
            raw_path = Path(source["path"])
            if raw_path.is_absolute():
                complete = False
                result.append({"type": "text", "text": "[image omitted: absolute path]"})
                continue
            path = (trajectory_dir / raw_path).resolve()
            try:
                path.relative_to(trajectory_dir.resolve())
                data = path.read_bytes()
            except (OSError, ValueError):
                complete = False
                result.append(_bounded(part, config))
                continue
            if len(data) > config.max_attachment_bytes:
                complete = False
                result.append({"type": "text", "text": "[image omitted: size limit]"})
                continue
            attachment = Attachment(
                data=data,
                filename=path.name,
                content_type=source.get("media_type", "application/octet-stream"),
            )
            result.append({"type": "image_url", "image_url": {"url": attachment}})
            continue
        complete = False
        result.append(_bounded(part, config))
    return result, complete


def _end_time(times: list[float], index: int, phase_end: float) -> float:
    if index + 1 < len(times):
        return max(times[index], times[index + 1])
    return max(times[index], phase_end)


def summarize_trajectory(trajectory_path: Path, config: PluginConfig) -> ATIFImportResult:
    """Read bounded trajectory summary data without creating detailed leaves."""
    try:
        if trajectory_path.stat().st_size > config.max_total_attachment_bytes:
            return ATIFImportResult(warnings=("trajectory omitted: size limit",))
        trajectory = json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return ATIFImportResult(warnings=(f"trajectory unavailable or malformed: {exc}",))
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
        return ATIFImportResult(warnings=("trajectory malformed: steps must be an array",))
    final_message = None
    for step in trajectory["steps"]:
        if isinstance(step, dict) and step.get("source") == "agent" and not step.get("is_copied_context"):
            final_message = _bounded(step.get("message"), config)
    extra = trajectory.get("extra") if isinstance(trajectory.get("extra"), dict) else None
    final_metrics = trajectory.get("final_metrics")
    root_extra = dict(extra or {})
    if isinstance(final_metrics, dict):
        root_extra["final_metrics"] = _bounded(final_metrics, config)
    return ATIFImportResult(
        final_message=final_message,
        schema_version=(
            trajectory.get("schema_version") if isinstance(trajectory.get("schema_version"), str) else None
        ),
        root_extra=root_extra or None,
    )


def import_trajectory(
    parent: Any,
    trajectory_path: Path,
    *,
    trial_id: str,
    semantic_prefix: str,
    phase_start: float,
    phase_end: float,
    config: PluginConfig,
    _trajectory_data: dict[str, Any] | None = None,
) -> ATIFImportResult:
    warnings: list[str] = []
    if _trajectory_data is not None:
        trajectory = _trajectory_data
    else:
        try:
            if trajectory_path.stat().st_size > config.max_total_attachment_bytes:
                return ATIFImportResult(warnings=("trajectory omitted: size limit",))
            trajectory = json.loads(trajectory_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return ATIFImportResult(warnings=(f"trajectory unavailable or malformed: {exc}",))
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
        return ATIFImportResult(warnings=("trajectory malformed: steps must be an array",))

    steps = [step for step in trajectory["steps"] if isinstance(step, dict)]
    times, repairs = _step_times(steps, phase_start, phase_end)
    agent = trajectory.get("agent") if isinstance(trajectory.get("agent"), dict) else {}
    default_model = agent.get("model_name")
    tools = agent.get("tool_definitions") if isinstance(agent.get("tool_definitions"), list) else None
    messages: list[dict[str, Any]] = []
    final_message: Any = None
    llm_count = 0
    tool_count = 0
    observations: dict[str, Any] = {}
    for step in steps:
        observation = step.get("observation")
        if isinstance(observation, dict) and isinstance(observation.get("results"), list):
            for result in observation["results"]:
                if isinstance(result, dict) and isinstance(result.get("source_call_id"), str):
                    observations[result["source_call_id"]] = result

    for index, step in enumerate(steps):
        source = step.get("source")
        content, content_complete = _content(step.get("message"), trajectory_path.parent, config)
        if source in {"system", "user"}:
            if config.content_mode != "metadata":
                messages.append({"role": source, "content": content})
            continue
        if source != "agent":
            warnings.append(f"step {index + 1}: unknown source")
            continue

        tool_calls = step.get("tool_calls") if isinstance(step.get("tool_calls"), list) else []
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        normalized_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id, name, arguments = call.get("tool_call_id"), call.get("function_name"), call.get("arguments")
            if isinstance(call_id, str) and isinstance(name, str) and isinstance(arguments, dict):
                normalized_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments, sort_keys=True)},
                    }
                )
        if normalized_calls:
            assistant_message["tool_calls"] = normalized_calls

        llm_call_count = step.get("llm_call_count")
        metrics = _usage_metrics(step.get("metrics"))
        provider, model = _provider(step.get("model_name") or default_model)
        can_be_llm = (
            config.content_mode != "metadata"
            and llm_call_count == 1
            and content_complete
            and provider is not None
            and model is not None
            and "tokens" in metrics
        )
        path = f"{semantic_prefix}/turn/{step.get('step_id', index + 1)}"
        if can_be_llm:
            metadata: dict[str, Any] = {"provider": provider, "model": model}
            if tools:
                metadata["tools"] = _bounded(tools, config)
            llm_span = parent.start_span(
                name="chat.completions.create",
                type="llm",
                id=child_span_id(trial_id, f"{path}/llm"),
                start_time=times[index],
                set_current=False,
                input=list(messages),
                metadata=metadata,
                internal={"instrumentation": _INSTRUMENTATION},
            )
            llm_span.log(output=assistant_message, metrics=metrics)
            llm_span.end(end_time=_end_time(times, index, phase_end))
            llm_count += 1
        else:
            reason = "not exactly one conforming model call"
            if llm_call_count == 1 and "tokens" not in metrics:
                reason = "model call missing token usage"
            warnings.append(f"step {index + 1}: downgraded to task ({reason})")
            summary_span = parent.start_span(
                name=f"trajectory.step.{step.get('step_id', index + 1)}",
                type="task",
                id=child_span_id(trial_id, f"{path}/summary"),
                start_time=times[index],
                set_current=False,
                input={"source": source},
                internal={"instrumentation": _INSTRUMENTATION},
            )
            summary_span.log(output={"message": content, "tool_call_count": len(normalized_calls)})
            summary_span.end(end_time=_end_time(times, index, phase_end))

        if config.content_mode != "metadata":
            messages.append(assistant_message)
        if not step.get("is_copied_context"):
            final_message = assistant_message

        for call_index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            call_id, name, arguments = call.get("tool_call_id"), call.get("function_name"), call.get("arguments")
            result = observations.get(call_id) if isinstance(call_id, str) else None
            tool_path = f"{semantic_prefix}/tool/{call_id or call_index}"
            if (
                isinstance(call_id, str)
                and isinstance(name, str)
                and isinstance(arguments, dict)
                and isinstance(result, dict)
            ):
                tool_output, tool_complete = _content(result.get("content"), trajectory_path.parent, config)
                result_extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
                tool_error = result_extra.get("error") if isinstance(result_extra.get("error"), str) else None
                has_result = result.get("content") is not None or tool_error is not None
                if config.content_mode != "metadata" and tool_complete and has_result:
                    tool_span = parent.start_span(
                        name=name,
                        type="tool",
                        id=child_span_id(trial_id, tool_path),
                        start_time=times[index],
                        set_current=False,
                        input=_bounded(arguments, config),
                        metadata={"tool_call_id": call_id},
                        internal={"instrumentation": _INSTRUMENTATION},
                    )
                    if tool_error is not None:
                        tool_span.log(error=tool_error)
                    else:
                        tool_span.log(output=tool_output)
                    tool_span.end(end_time=_end_time(times, index, phase_end))
                    tool_count += 1
                else:
                    warnings.append(f"tool {call_id}: downgraded because payload is incomplete")
                if config.content_mode != "metadata":
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_output})
            else:
                warnings.append(f"tool {call_id or call_index}: missing correlated arguments or result")

    # Preserve subagents as explicit nested task trees. Their detailed leaves use
    # the same conformance gate recursively.
    for sub_index, subagent in enumerate(trajectory.get("subagent_trajectories") or []):
        if not isinstance(subagent, dict):
            continue
        sub_parent = parent.start_span(
            name=f"subagent:{(subagent.get('agent') or {}).get('name', sub_index)}",
            type="task",
            id=child_span_id(trial_id, f"{semantic_prefix}/subagent/{sub_index}"),
            start_time=phase_start,
            set_current=False,
            internal={"instrumentation": _INSTRUMENTATION},
        )
        imported = import_trajectory(
            sub_parent,
            trajectory_path,
            trial_id=trial_id,
            semantic_prefix=f"{semantic_prefix}/subagent/{sub_index}",
            phase_start=phase_start,
            phase_end=phase_end,
            config=config,
            _trajectory_data=subagent,
        )
        sub_parent.end(end_time=phase_end)
        warnings.extend(imported.warnings)
        repairs.extend(imported.repairs)
        llm_count += imported.imported_llm_spans
        tool_count += imported.imported_tool_spans

    extra = trajectory.get("extra") if isinstance(trajectory.get("extra"), dict) else None
    root_extra = dict(extra or {})
    if isinstance(trajectory.get("final_metrics"), dict):
        root_extra["final_metrics"] = _bounded(trajectory["final_metrics"], config)
    return ATIFImportResult(
        final_message=final_message,
        schema_version=trajectory.get("schema_version") if isinstance(trajectory.get("schema_version"), str) else None,
        root_extra=root_extra or None,
        warnings=tuple(warnings),
        repairs=tuple(repairs),
        imported_llm_spans=llm_count,
        imported_tool_spans=tool_count,
    )
