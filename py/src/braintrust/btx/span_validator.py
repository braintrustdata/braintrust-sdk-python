"""Validate in-memory Braintrust spans against BTX YAML spec expectations.

The validation is recursive and collects *all* failures before raising,
so a single test run shows every mismatch at once.

Matchers (from spec_loader):
  FnMatcher      — named predicate or Python lambda expression
  StartsWithMatcher — string prefix check
  OrMatcher      — at-least-one-of

Semantics:
  - dict keys in ``expected`` that are absent from ``actual`` → failure
  - extra keys in ``actual`` are ignored (lenient)
  - lists: len(actual) >= len(expected); first N elements validated pairwise
  - scalar None in ``expected`` → "don't care" (always passes)
  - scalars compared with ==
"""

from __future__ import annotations

import json
from typing import Any

from braintrust.logger import Attachment

from .spec_loader import FnMatcher, LlmSpanSpec, OrMatcher, StartsWithMatcher


# ---------------------------------------------------------------------------
# Named predicates (mirror of is_* functions in framework.py / SpanValidator.java)
# ---------------------------------------------------------------------------


def _is_non_negative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def _is_reasoning_message(value: Any) -> bool:
    """A list of {type: summary_text, text: <non-empty>} entries (may be empty)."""
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        return True  # Empty reasoning list is allowed
    for item in value:
        if not isinstance(item, dict):
            return False
        if item.get("type") != "summary_text":
            return False
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            return False
    return True


_NAMED_MATCHERS: dict[str, Any] = {
    "is_non_negative_number": _is_non_negative_number,
    "is_non_empty_string": _is_non_empty_string,
    "is_reasoning_message": _is_reasoning_message,
}


def _resolve_fn_matcher(matcher: FnMatcher) -> Any:
    """Return a callable for this FnMatcher.

    For well-known names, return the dedicated function.
    For anything else, eval() the expression — since this is Python we can
    actually execute arbitrary lambda strings from the spec.
    """
    if matcher.expr in _NAMED_MATCHERS:
        return _NAMED_MATCHERS[matcher.expr]
    # Arbitrary Python expression (e.g. "lambda value: \"440\" in value")
    try:
        func = eval(matcher.expr)  # noqa: S307 — intentional, internal test framework
        if not callable(func):
            raise ValueError(f"!fn expression did not evaluate to a callable: {matcher.expr}")
        return func
    except Exception as exc:
        raise ValueError(f"Failed to evaluate !fn expression {matcher.expr!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Core recursive validator
# ---------------------------------------------------------------------------


def _normalize_actual(actual: Any) -> Any:
    """Normalize values that have a richer in-memory representation.

    Specifically: ``Attachment`` objects are replaced with their ``.reference``
    dict (``{type: braintrust_attachment, filename, content_type, key}``),
    which is what the spec's ``expected_brainstore_spans`` asserts against.
    """
    if isinstance(actual, Attachment):
        return actual.reference
    return actual


def _validate_value(actual: Any, expected: Any, path: str, errors: list[str]) -> None:
    """Recursively validate ``actual`` against ``expected``, appending to ``errors``."""
    actual = _normalize_actual(actual)

    # --- OrMatcher: try each alternative, succeed if any passes ---
    if isinstance(expected, OrMatcher):
        or_errors: list[str] = []
        for i, alt in enumerate(expected.alternatives):
            alt_errors: list[str] = []
            _validate_value(actual, alt, path, alt_errors)
            if not alt_errors:
                return  # This alternative matched
            or_errors.append(f"  alternative[{i}]: " + "; ".join(alt_errors))
        errors.append(
            f"{path}: none of {len(expected.alternatives)} OR alternatives matched:\n" + "\n".join(or_errors)
        )
        return

    # --- FnMatcher ---
    if isinstance(expected, FnMatcher):
        fn = _resolve_fn_matcher(expected)
        try:
            result = fn(actual)
        except Exception as exc:
            errors.append(f"{path}: validator raised {type(exc).__name__}: {exc} (actual={actual!r})")
            return
        if not result:
            errors.append(f"{path}: validator {expected.expr!r} returned False for actual={actual!r}")
        return

    # --- StartsWithMatcher ---
    if isinstance(expected, StartsWithMatcher):
        if not isinstance(actual, str) or not actual.startswith(expected.prefix):
            errors.append(f"{path}: expected string starting with {expected.prefix!r}, got {actual!r}")
        return

    # --- None expected → don't care ---
    if expected is None:
        return

    # --- dict: recurse into keys ---
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}: expected dict, got {type(actual).__name__} ({actual!r})")
            return
        for key, exp_val in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: key not found in actual span")
            else:
                _validate_value(actual[key], exp_val, f"{path}.{key}", errors)
        return

    # --- list: lenient length check, validate first N elements ---
    if isinstance(expected, list):
        if not isinstance(actual, list):
            # Special case: if expected has exactly one dict element and actual is a dict,
            # treat it as a single-element list (mirrors Java SpanValidator behaviour)
            if len(expected) == 1 and isinstance(expected[0], dict) and isinstance(actual, dict):
                _validate_value(actual, expected[0], f"{path}[0]", errors)
                return
            errors.append(f"{path}: expected list, got {type(actual).__name__} ({actual!r})")
            return
        if len(actual) < len(expected):
            errors.append(f"{path}: list too short — expected at least {len(expected)} elements, got {len(actual)}")
            return
        for i, exp_item in enumerate(expected):
            _validate_value(actual[i], exp_item, f"{path}[{i}]", errors)
        return

    # --- scalar: exact equality ---
    if actual != expected:
        errors.append(f"{path}: expected={expected!r}, actual={actual!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_spans(actual_spans: list[dict[str, Any]], spec: LlmSpanSpec) -> None:
    """Assert that *actual_spans* match the spec's ``expected_brainstore_spans``.

    ``actual_spans`` are the plain dicts from ``memory_logger.pop()``.
    They are already in brainstore format (the same payload that would be
    sent to the Braintrust API), so no conversion is needed.

    Spans are sorted by ``span_attributes.exec_counter`` before comparison
    so that multi-span specs are matched in creation order regardless of
    flush ordering.

    Raises:
        AssertionError: with a full description of every mismatch found.
    """
    expected_spans = spec.expected_brainstore_spans

    # Filter to LLM spans only (type == "llm") — mirrors the scorer-span
    # filtering in the original btx framework
    llm_spans = [s for s in actual_spans if s.get("span_attributes", {}).get("type") == "llm"]

    # Sort by exec_counter for deterministic ordering
    llm_spans.sort(key=lambda s: s.get("span_attributes", {}).get("exec_counter", 0))

    if len(llm_spans) < len(expected_spans):
        raise AssertionError(
            f"{spec.display_name}: expected at least {len(expected_spans)} LLM span(s), "
            f"got {len(llm_spans)}.\n"
            f"All captured spans:\n{json.dumps(actual_spans, indent=2, default=str)}"
        )

    all_errors: list[str] = []

    for i, (actual_span, expected_span) in enumerate(zip(llm_spans, expected_spans)):
        span_errors: list[str] = []
        for key, exp_val in expected_span.items():
            if key not in actual_span:
                span_errors.append(f"  span[{i}].{key}: key not found in actual span")
            else:
                _validate_value(actual_span[key], exp_val, f"span[{i}].{key}", span_errors)

        if span_errors:
            all_errors.append(
                f"\n--- Span {i} ({actual_span.get('span_attributes', {}).get('name', '?')}) ---\n"
                + "\n".join(span_errors)
                + f"\n\nFull span JSON:\n{json.dumps(actual_span, indent=2, default=str)}"
            )

    if all_errors:
        raise AssertionError(f"{spec.display_name}: span validation failed:\n" + "\n".join(all_errors))
