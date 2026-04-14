"""Benchmarks for merge_dicts and merge_dicts_with_paths.

merge_dicts is called on every span log update and during row merging,
making it one of the most frequently executed SDK functions.

Note: merge_dicts mutates merge_into, so each benchmark wrapper creates a
fresh copy of the target dict before calling. This means each bench_func
measures a shallow/deep copy plus the merge itself — the copy cost is
intentionally kept proportional to the input size so relative comparisons
remain valid.
"""

import copy
import pathlib
import sys
from typing import Any

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.util import merge_dicts

from benchmarks._utils import disable_pyperf_psutil
from benchmarks.fixtures import make_large_payload, make_medium_payload, make_small_payload


# Updates are pre-built once; only merge_into is copied per iteration.
_SMALL_UPDATE: dict[str, Any] = {
    "metadata": {"extra_key": "extra_value"},
    "scores": {"relevance": 0.8},
    "tags": ["new_tag"],
}

_MEDIUM_UPDATE: dict[str, Any] = {
    "metadata": {"workspace_id": "workspace-789", "new_flag": True},
    "metrics": {"cached_tokens": 64},
    "tags": ["updated", "benchmark"],
}

_LARGE_UPDATE: dict[str, Any] = {
    "metadata": {"routing": {"tier": "standard"}, "extra": "value"},
    "metrics": {"cached_tokens": 512},
    "tags": ["updated"],
    "output": {"summary": "revised"},
}

# Pre-built base payloads (copied per iteration, not mutated at module level).
_SMALL_BASE = make_small_payload()
_MEDIUM_BASE = make_medium_payload()
_LARGE_BASE = make_large_payload()

_NESTED_BASE: dict[str, Any] = {
    "a": {"b": {"c": {"d": 1, "e": 2}, "f": 3}, "g": 4},
    "h": {"i": {"j": {"k": 5}}},
}
_NESTED_UPDATE: dict[str, Any] = {
    "a": {"b": {"c": {"d": 99}, "new": "value"}, "g": 99},
    "h": {"i": {"j": {"new_key": "hello"}}},
}

# Tags set-union: top-level "tags" field uses set-union semantics in merge_dicts.
_TAGS_UPDATE: dict[str, Any] = {"tags": ["c", "d", "e"]}


def _bench_small() -> None:
    merge_dicts(dict(_SMALL_BASE), _SMALL_UPDATE)


def _bench_medium() -> None:
    # Shallow copy is enough: _MEDIUM_UPDATE only touches top-level dict values.
    merge_dicts(dict(_MEDIUM_BASE), _MEDIUM_UPDATE)


def _bench_large() -> None:
    merge_dicts(dict(_LARGE_BASE), _LARGE_UPDATE)


def _bench_nested() -> None:
    # Deep copy required because the update recurses into nested dicts.
    merge_dicts(copy.deepcopy(_NESTED_BASE), _NESTED_UPDATE)


def _bench_tags_union() -> None:
    # Tags list grows on each call, so start from a fresh copy every time.
    merge_dicts({"tags": ["a", "b"], "value": 1}, _TAGS_UPDATE)


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_func("merge_dicts[small]", _bench_small)
    runner.bench_func("merge_dicts[medium]", _bench_medium)
    runner.bench_func("merge_dicts[large]", _bench_large)
    runner.bench_func("merge_dicts[nested-deep]", _bench_nested)
    runner.bench_func("merge_dicts[tags-union]", _bench_tags_union)


if __name__ == "__main__":
    main()
