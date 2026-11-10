import pytest
from braintrust import logger
from braintrust._audio.schema import LINKS, set_links, span_id
from braintrust._audio.types import selection
from braintrust.logger import NOOP_SPAN
from braintrust.test_helpers import init_test_logger


@pytest.fixture(autouse=True)
def _restore_logger_metadata(monkeypatch):
    # init_test_logger replaces this global; restore it after each test.
    monkeypatch.setattr(logger, "_compute_logger_metadata", logger._compute_logger_metadata)


def test_links_are_typed_deduplicated_and_replaced_per_call():
    init_test_logger("audio-tests")
    with logger._internal_with_memory_background_logger() as memory:
        source, target = logger.start_span(name="source"), logger.start_span(name="target")
        set_links(source, [("uses_context", NOOP_SPAN)])
        # Each call replaces the span's links; duplicates within one are merged.
        set_links(source, [("generated_by", target), ("generated_by", target), ("uses_context", NOOP_SPAN)])
        source.end()
        target.end()
        row = next(row for row in memory.pop() if row["span_id"] == span_id(source))
    assert row["metadata"][LINKS] == [
        {
            "context": {"traceId": target.root_span_id, "spanId": span_id(target), "traceFlags": 1},
            "attributes": {"braintrust.relationship.type": "generated_by"},
        }
    ]


def test_selection_rounds_offsets_and_never_ends_before_it_starts():
    assert selection(1000.4, 2000.6) == {"start_offset_ms": 1000, "end_offset_ms": 2001}
    assert selection(500, 100) == {"start_offset_ms": 500, "end_offset_ms": 500}


def test_each_call_replaces_the_previous_links():
    init_test_logger("audio-tests")
    with logger._internal_with_memory_background_logger() as memory:
        source, first, second = (logger.start_span(name=name) for name in ("source", "first", "second"))
        set_links(source, [("generated_by", first)])
        set_links(source, [("generated_by", second)])
        for span in (source, first, second):
            span.end()
        row = next(row for row in memory.pop() if row["span_id"] == span_id(source))
    assert [link["context"]["spanId"] for link in row["metadata"][LINKS]] == [span_id(second)]


@pytest.mark.parametrize("replacement", ["empty", "only_noop_targets"])
def test_replacing_with_no_links_clears_existing_links(replacement):
    init_test_logger("audio-tests")
    with logger._internal_with_memory_background_logger() as memory:
        source, target = logger.start_span(name="source"), logger.start_span(name="target")
        set_links(source, [("generated_by", target)])
        set_links(source, [] if replacement == "empty" else [("uses_context", NOOP_SPAN)])
        source.end()
        target.end()
        row = next(row for row in memory.pop() if row["span_id"] == span_id(source))
    assert row["metadata"][LINKS] == []
