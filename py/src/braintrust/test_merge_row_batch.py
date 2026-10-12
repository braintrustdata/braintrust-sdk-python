import unittest

from braintrust.db_fields import IS_MERGE_FIELD
from braintrust.merge_row_batch import batch_items, merge_row_batch, merge_row_batch_waves
from braintrust.span_customizer import _StrippedSpanExportData


class MergeRowBatchTest(unittest.TestCase):
    def test_basic(self):
        rows = [
            # These rows should get merged together, ending up as a merge.
            dict(
                experiment_id="e0",
                id="x",
                inputs=dict(a=12),
                **{IS_MERGE_FIELD: True},
            ),
            dict(
                experiment_id="e0",
                id="x",
                inputs=dict(b=10),
                **{IS_MERGE_FIELD: True},
            ),
            dict(
                experiment_id="e0",
                id="x",
                inputs=dict(c="hello"),
                **{IS_MERGE_FIELD: True},
            ),
            # The first row should be clobbered by the second, but the third
            # merged with the second, ending up as a replacement.
            dict(
                experiment_id="e0",
                id="y",
                inputs=dict(a="hello"),
            ),
            dict(
                experiment_id="e0",
                id="y",
                inputs=dict(b=10),
            ),
            dict(
                experiment_id="e0",
                id="y",
                inputs=dict(c=12),
                **{IS_MERGE_FIELD: True},
            ),
            # These rows should be clobbered separately from the last batch.
            dict(
                dataset_id="d0",
                id="y",
                inputs=dict(a="hello"),
            ),
            dict(
                dataset_id="d0",
                id="y",
                inputs=dict(b=10),
            ),
            dict(
                dataset_id="d0",
                id="y",
                inputs=dict(c=12),
            ),
        ]

        merged_rows = merge_row_batch(rows)
        key_to_rows = {(row.get("experiment_id"), row.get("dataset_id"), row.get("id")): row for row in merged_rows}
        self.assertEqual(
            {
                ("e0", None, "x"): dict(
                    experiment_id="e0",
                    id="x",
                    inputs=dict(a=12, b=10, c="hello"),
                    **{IS_MERGE_FIELD: True},
                ),
                ("e0", None, "y"): dict(
                    experiment_id="e0",
                    id="y",
                    inputs=dict(b=10, c=12),
                ),
                (None, "d0", "y"): dict(
                    dataset_id="d0",
                    id="y",
                    inputs=dict(c=12),
                ),
            },
            key_to_rows,
        )

    def test_skip_fields(self):
        rows = [
            # These rows should get merged together, ending up as a merge. But
            # the original fields should be retained, regardless of whether we
            # populated them or not.
            dict(
                experiment_id="e0",
                id="x",
                inputs=dict(a=12),
                **{IS_MERGE_FIELD: True},
                created=123,
                root_span_id="abc",
                _parent_id="baz",
                span_parents=["foo", "bar"],
            ),
            dict(
                experiment_id="e0",
                id="x",
                inputs=dict(b=10),
                **{IS_MERGE_FIELD: True},
                created=456,
                span_id="foo",
                root_span_id="bar",
                _parent_id="boop",
                span_parents=[],
            ),
        ]

        merged_rows = merge_row_batch(rows)
        self.assertEqual(
            merged_rows,
            [
                dict(
                    experiment_id="e0",
                    id="x",
                    inputs=dict(a=12, b=10),
                    **{IS_MERGE_FIELD: True},
                    created=123,
                    root_span_id="abc",
                    _parent_id="baz",
                    span_parents=["foo", "bar"],
                ),
            ],
        )


def _row(row_id, **fields):
    return dict(experiment_id="e0", id=row_id, **fields)


def _stripped(row_id, **fields):
    return _StrippedSpanExportData(_row(row_id, context={"customizer_error": True}, **fields))


class MergeRowBatchWavesTest(unittest.TestCase):
    def test_no_barriers_is_one_wave(self):
        rows = [_row("x", a=1), _row("y", a=2), _row("x", b=3, **{IS_MERGE_FIELD: True})]
        self.assertEqual(merge_row_batch_waves(rows), [[_row("x", a=1, b=3), _row("y", a=2)]])
        self.assertEqual(merge_row_batch_waves([]), [])

    def test_barrier_only_splits_its_own_row(self):
        barrier = _stripped("x", **{IS_MERGE_FIELD: True})
        rows = [
            _row("x", a=1),
            _row("y", a=1),
            barrier,
            _row("y", b=2, **{IS_MERGE_FIELD: True}),
            _row("x", c=3, **{IS_MERGE_FIELD: True}),
            _row("z", a=1),
            _row("x", d=4, **{IS_MERGE_FIELD: True}),
        ]
        waves = merge_row_batch_waves(rows)
        self.assertEqual(
            waves,
            [
                [_row("x", a=1), _row("y", a=1, b=2), _row("z", a=1)],
                [barrier],
                [_row("x", c=3, d=4, **{IS_MERGE_FIELD: True})],
            ],
        )
        self.assertIs(waves[1][0], barrier)
        self.assertEqual(barrier, _stripped("x", **{IS_MERGE_FIELD: True}))

    def test_leading_and_consecutive_barriers(self):
        # A barrier with no earlier rows for its key needs no extra wave, but
        # consecutive barriers for one key must not share a wave.
        first = _stripped("x", n=1)
        second = _stripped("x", n=2, **{IS_MERGE_FIELD: True})
        rows = [first, _row("y", a=1), second, _row("x", a=1, **{IS_MERGE_FIELD: True}), _stripped("y")]
        self.assertEqual(
            merge_row_batch_waves(rows),
            [
                [first, _row("y", a=1)],
                [second, _stripped("y")],
                [_row("x", a=1, **{IS_MERGE_FIELD: True})],
            ],
        )

    def test_waves_never_repeat_keys_and_flatten_in_order(self):
        rows = [
            _row("x", n=0),
            _stripped("x"),
            _row("y", n=0),
            _stripped("y"),
            _row("x", n=1, **{IS_MERGE_FIELD: True}),
            _stripped("x"),
            _row("y", n=1, **{IS_MERGE_FIELD: True}),
        ]
        waves = merge_row_batch_waves(rows)
        for wave in waves:
            self.assertEqual(len({row["id"] for row in wave}), len(wave))
        self.assertEqual(merge_row_batch(list(rows)), [row for wave in waves for row in wave])
        self.assertEqual(len(waves), 4)


class BatchItemsTest(unittest.TestCase):
    def test_exact_byte_limit(self):
        self.assertEqual(batch_items(["aa", "bb", "c"], batch_max_num_bytes=4), [["aa", "bb"], ["c"]])

    def test_basic(self):
        a = "x" * 1
        b = "x" * 2
        c = "x" * 4
        d = "y" * 1
        e = "y" * 2
        f = "y" * 4

        items = [a, b, c, f, e, d]

        # No limits.
        output = batch_items(items)
        self.assertEqual(output, [[a, b, c, f, e, d]])

        # Num items limit.
        output = batch_items(items, batch_max_num_items=2)
        self.assertEqual(output, [[a, b], [c, f], [e, d]])

        # Num bytes limit.
        output = batch_items(items, batch_max_num_bytes=2)
        self.assertEqual(output, [[a], [b], [c], [f], [e], [d]])

        # Both items and num bytes limit.
        output = batch_items(items, batch_max_num_items=2, batch_max_num_bytes=5)
        self.assertEqual(output, [[a, b], [c], [f], [e, d]])
