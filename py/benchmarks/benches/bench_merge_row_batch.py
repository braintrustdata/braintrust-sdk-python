"""Benchmarks for merge_row_batch and batch_items.

merge_row_batch is called before every flush to the Braintrust API to
de-duplicate and merge rows in a pending batch.  batch_items is used to
split the resulting rows into API-request-sized chunks.

Both functions mutate their inputs, so each benchmark wrapper builds fresh
row lists per iteration.
"""

import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.db_fields import IS_MERGE_FIELD
from braintrust.merge_row_batch import batch_items, merge_row_batch

from benchmarks._utils import disable_pyperf_psutil


# ---------------------------------------------------------------------------
# Row factories — called inside each benchmark wrapper to get fresh dicts.
# ---------------------------------------------------------------------------


def _unique_rows(n: int) -> list[dict]:
    """n rows, all distinct IDs — no merging needed."""
    return [{"id": f"row-{i}", "project_id": "proj-1", "value": i} for i in range(n)]


def _merge_rows(n: int) -> list[dict]:
    """n rows forming n//2 pairs: first is a base, second is an IS_MERGE update."""
    rows = []
    for i in range(n // 2):
        rows.append({"id": f"row-{i}", "project_id": "proj-1", "payload": {"a": i}})
        rows.append(
            {
                "id": f"row-{i}",
                "project_id": "proj-1",
                "payload": {"b": i + 100},
                IS_MERGE_FIELD: True,
            }
        )
    return rows


def _mixed_rows(n: int) -> list[dict]:
    """Mix of unique rows and merge pairs (roughly half each)."""
    rows = []
    for i in range(n // 4):
        # pair that will be merged
        rows.append({"id": f"merge-{i}", "project_id": "proj-1", "payload": {"a": i}})
        rows.append(
            {
                "id": f"merge-{i}",
                "project_id": "proj-1",
                "payload": {"b": i + 100},
                IS_MERGE_FIELD: True,
            }
        )
    for i in range(n // 2):
        rows.append({"id": f"unique-{i}", "project_id": "proj-1", "value": i})
    return rows


# ---------------------------------------------------------------------------
# Benchmark wrappers
# ---------------------------------------------------------------------------

_SMALL_N = 10
_MEDIUM_N = 50
_LARGE_N = 200


def _bench_no_conflict_small() -> None:
    merge_row_batch(_unique_rows(_SMALL_N))


def _bench_no_conflict_medium() -> None:
    merge_row_batch(_unique_rows(_MEDIUM_N))


def _bench_no_conflict_large() -> None:
    merge_row_batch(_unique_rows(_LARGE_N))


def _bench_all_merge_small() -> None:
    merge_row_batch(_merge_rows(_SMALL_N))


def _bench_all_merge_medium() -> None:
    merge_row_batch(_merge_rows(_MEDIUM_N))


def _bench_mixed_medium() -> None:
    merge_row_batch(_mixed_rows(_MEDIUM_N))


# batch_items: split a list of strings by item-count and byte-count limits.
_BATCH_STRINGS = [f"item-payload-{i:04d}" * 4 for i in range(200)]
_ITEM_SIZE = len(_BATCH_STRINGS[0].encode())


def _bench_batch_items_count_limit() -> None:
    batch_items(_BATCH_STRINGS, batch_max_num_items=20)


def _bench_batch_items_byte_limit() -> None:
    batch_items(
        _BATCH_STRINGS,
        batch_max_num_bytes=_ITEM_SIZE * 15,
        get_byte_size=lambda s: len(s.encode()),
    )


def _bench_batch_items_both_limits() -> None:
    batch_items(
        _BATCH_STRINGS,
        batch_max_num_items=20,
        batch_max_num_bytes=_ITEM_SIZE * 15,
        get_byte_size=lambda s: len(s.encode()),
    )


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_func("merge_row_batch[no-conflict-small]", _bench_no_conflict_small)
    runner.bench_func("merge_row_batch[no-conflict-medium]", _bench_no_conflict_medium)
    runner.bench_func("merge_row_batch[no-conflict-large]", _bench_no_conflict_large)
    runner.bench_func("merge_row_batch[all-merge-small]", _bench_all_merge_small)
    runner.bench_func("merge_row_batch[all-merge-medium]", _bench_all_merge_medium)
    runner.bench_func("merge_row_batch[mixed-medium]", _bench_mixed_medium)

    runner.bench_func("batch_items[count-limit]", _bench_batch_items_count_limit)
    runner.bench_func("batch_items[byte-limit]", _bench_batch_items_byte_limit)
    runner.bench_func("batch_items[both-limits]", _bench_batch_items_both_limits)


if __name__ == "__main__":
    main()
