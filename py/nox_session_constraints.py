"""Python and platform constraints shared by nox and its CI sharder."""

import re


CONSTRAINTS = {
    "test_ai_sdk": {"min_python": "3.12"},
    "test_litellm": {"versions": {"latest": {"min_python": "3.11"}}},
    "test_livekit_agents": {"max_python_exclusive": "3.14", "ci_platforms": ["linux"]},
    "test_pipecat": {"min_python": "3.11", "max_python_exclusive": "3.14"},
    "test_agentscope": {"versions": {"latest": {"min_python": "3.11"}}},
    "test_dspy": {"versions": {"latest": {"min_python": "3.11"}}},
    "test_crewai": {"max_python_exclusive": "3.14"},
    "test_deepagents": {"min_python": "3.11"},
    "test_transformers": {"versions": {"not_latest": {"max_python_exclusive": "3.13"}}},
    "test_harbor": {"min_python": "3.12"},
    "test_otel": {"versions": {"not_latest_below": {"version": "1.28.0", "skip_from_python": "3.14"}}},
}


def _version_key(value: str) -> tuple[int, ...]:
    """Compare the numeric release versions used by this session matrix."""
    return tuple(int(component) for component in re.findall(r"\d+", value))


def _normalize_platform(operating_system: str) -> str:
    platform_name = operating_system.lower()
    if platform_name.startswith("windows"):
        return "windows"
    if platform_name.startswith(("ubuntu", "linux")):
        return "linux"
    if platform_name.startswith(("macos", "darwin")):
        return "darwin"
    return platform_name


def _session_rules(session_id: str) -> tuple[dict, list[dict]]:
    match = re.fullmatch(r"([^()]+)(?:\(([^()]*)\))?", session_id)
    if not match:
        return {}, []
    name, version = match.groups()
    constraint = CONSTRAINTS.get(name)
    if not constraint:
        return {}, []

    version_rules = constraint.get("versions", {})
    selected = []
    if version is not None:
        if version == "latest":
            selected.append(version_rules.get("latest", {}))
        else:
            selected.append(version_rules.get("not_latest", {}))
            threshold = version_rules.get("not_latest_below")
            if threshold and _version_key(version) < _version_key(threshold["version"]):
                selected.append(threshold)
    return constraint, selected


def incompatibility_reason(
    session_id: str,
    python_version: str,
    operating_system: str,
    *,
    check_ci_platforms: bool = False,
) -> str | None:
    """Return the declared reason a session is incompatible, if any."""
    python = _version_key(python_version)
    platform_name = _normalize_platform(operating_system)
    constraint, version_rules = _session_rules(session_id)
    for rule in [constraint, *version_rules]:
        if check_ci_platforms and "ci_platforms" in rule and platform_name not in rule["ci_platforms"]:
            return f"not supported on {operating_system}"
        if "min_python" in rule and python < _version_key(rule["min_python"]):
            return f"requires Python {rule['min_python']}+"
        if "max_python_exclusive" in rule and python >= _version_key(rule["max_python_exclusive"]):
            return f"does not support Python {rule['max_python_exclusive']}+"
        if "skip_from_python" in rule and python >= _version_key(rule["skip_from_python"]):
            return f"skips on Python {rule['skip_from_python']}+"
    return None


def session_is_compatible(
    session_id: str,
    python_version: str,
    operating_system: str,
    *,
    check_ci_platforms: bool = False,
) -> bool:
    """Return whether the requested nox session has work for this target."""
    return (
        incompatibility_reason(
            session_id,
            python_version,
            operating_system,
            check_ci_platforms=check_ci_platforms,
        )
        is None
    )
