from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from .db_fields import IS_MERGE_FIELD
from .span_customizer import _StrippedSpanExportData


T = TypeVar("T")
from .util import merge_dicts


_MergedRowKey = tuple[Any | None, ...]


def _generate_merged_row_key(row: dict[str, Any]) -> _MergedRowKey:
    return tuple(
        row.get(k)
        for k in [
            "org_id",
            "project_id",
            "experiment_id",
            "dataset_id",
            "prompt_session_id",
            "log_id",
            "id",
        ]
    )


# These fields will be retained as-is when merging rows.
MERGE_ROW_SKIP_FIELDS = [
    "created",
    "span_id",
    "root_span_id",
    "span_parents",
    "_parent_id",
    # TODO: handle merge paths.
]


def _pop_merge_row_skip_fields(row: dict[str, Any]) -> dict[str, Any]:
    popped = {}
    for field in MERGE_ROW_SKIP_FIELDS:
        if field in row:
            popped[field] = row.pop(field)
    return popped


def _restore_merge_row_skip_fields(row: dict[str, Any], skip_fields: dict[str, Any]):
    for field in MERGE_ROW_SKIP_FIELDS:
        row.pop(field, None)
        if field in skip_fields:
            row[field] = skip_fields[field]


def merge_row_batch(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Like `merge_row_batch_waves`, flattened into one ordered list of rows."""
    return [row for wave in merge_row_batch_waves(rows) for row in wave]


def merge_row_batch_waves(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Given a batch of rows, merges conflicting rows together to end up with a
    set of rows to insert. Returns de-conflicted rows split into ordered waves.

    Note that the returned rows will be the same objects as the input `rows`,
    meaning they are mutated in place.
    SDK-stripped customizer failures are exceptions: they are never mutated and
    act as merge barriers for their own row only. Rows for that key before the
    barrier, the barrier itself, and rows after it land in successive waves.
    A wave never contains two rows with the same key, and without barriers there
    is exactly one wave. Exporters must send waves in order to preserve same-ID
    update ordering; rows within a wave are independent.

    There are a few important considerations for the merge procedure:

    - Ensuring we only log one version of each row to the DB:

    Imagine we have several rows in the batch with the same ID:

        [{"make_object_id(..)": "xyz", "id": 1, "value": 1},
         {"make_object_id(...)": "xyz", "id": 1, "value": 2}]

    If we log both rows and assign them both the same transaction ID, future
    queries will not be able to disamgiguate ordering here (i.e that `value: 2`
    is the "later" value). So we must consolidate these rows into one before
    logging.

    - Merging rows with IS_MERGE_FIELD == True:

    Rows can either be incrementally updated or replaced entirely. For a
    particular row, we use the IS_MERGE_FIELD to determine whether we merge or
    replace. In case there are several incremental updates to the same row
    within the batch, we merge them into one incremental update here, so that we
    only need to do one merge in the DB.

    We need to be careful to preserve the correct value of IS_MERGE_FIELD with
    respect to the DB. For instance, if we have one batch of rows:

        [{"make_object_id(...)": "xyz", "id": 1, "value": {"a": 12}},
         {"make_object_id(...)": "xyz", "id": 1, "value": {"b": 13},
          IS_MERGE_FIELD: True}]

    We need to make sure the row inserted into the DB has IS_MERGE_FIELD == False,
    otherwise we might merge it with a previous existing version of the row.
    """

    # Check that no row is missing an ID.
    for row in rows:
        if row.get("id") is None:
            raise Exception(
                "Logged row is missing an id. This is an internal braintrust error. Please contact us at info@braintrust.dev for help"
            )

    waves: list[dict[_MergedRowKey, dict[str, Any]]] = []
    # The earliest wave each key may still be written to.
    next_wave: dict[_MergedRowKey, int] = {}
    for row in rows:
        key = _generate_merged_row_key(row)
        wave_index = next_wave.get(key, 0)
        if wave_index == len(waves):
            waves.append({})
        existing_row = waves[wave_index].get(key)
        if isinstance(row, _StrippedSpanExportData):
            # Never mutate a memoized failure with neighboring record content.
            if existing_row is not None:
                wave_index += 1
                if wave_index == len(waves):
                    waves.append({})
            waves[wave_index][key] = row
            next_wave[key] = wave_index + 1
            continue
        # If there is an existing row and the new row has the IS_MERGE_FIELD ==
        # True property, we merge it with the existing row. Otherwise we can
        # replace it.
        if existing_row is not None and row.get(IS_MERGE_FIELD):
            skip_fields = _pop_merge_row_skip_fields(existing_row)
            # Preserve IS_MERGE_FIELD == False if the existing_row had it set to
            # false.
            preserve_nomerge = not existing_row.get(IS_MERGE_FIELD)
            merge_dicts(existing_row, row)
            _restore_merge_row_skip_fields(existing_row, skip_fields)
            if preserve_nomerge:
                del existing_row[IS_MERGE_FIELD]
        else:
            waves[wave_index][key] = row

    return [list(wave.values()) for wave in waves]


def batch_items(
    items: list[T],
    batch_max_num_items: int | None = None,
    batch_max_num_bytes: int | None = None,
    get_byte_size: Callable[[T], int] | None = None,
) -> list[list[T]]:
    """Repartition the given list of items into batches.

    Arguments:

    - `items` is a list of items to batch.

    - `batch_max_num_items` is the maximum number of items in each batch.
      If not provided, there is no limit on the number of items.

    - `batch_max_num_bytes` is the maximum number of bytes in each batch.
      If an individual item exceeds `batch_max_num_bytes` in size, we
      will place it in its own batch. If not provided, there is no limit on
      the number of bytes.

    - `get_byte_size` is a function that returns the byte size of an item.
      If not provided, defaults to `len(item)` (works for strings).
    """

    if batch_max_num_items is not None and batch_max_num_items <= 0:
        raise ValueError(f"batch_max_num_items must be positive; got {batch_max_num_items}")
    if batch_max_num_bytes is not None and batch_max_num_bytes < 0:
        raise ValueError(f"batch_max_num_bytes must be nonnegative; got {batch_max_num_bytes}")

    if get_byte_size is None:

        def get_byte_size(item: T) -> int:
            return len(item)  # type: ignore[arg-type]

    output: list[list[T]] = []
    batch: list[T] = []
    batch_len = 0

    def flush_batch() -> None:
        nonlocal batch, batch_len
        output.append(batch)
        batch = []
        batch_len = 0

    for item in items:
        item_size = get_byte_size(item)
        if len(batch) > 0 and not (
            (batch_max_num_bytes is None or item_size + batch_len <= batch_max_num_bytes)
            and (batch_max_num_items is None or len(batch) < batch_max_num_items)
        ):
            flush_batch()
        batch.append(item)
        batch_len += item_size

    if len(batch) > 0:
        flush_batch()

    return output
