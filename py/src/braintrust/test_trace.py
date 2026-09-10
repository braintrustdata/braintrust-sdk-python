"""Tests for Trace functionality."""

import os

import braintrust
import pytest
from braintrust.git_fields import GitMetadataSettings
from braintrust.logger import DATA_API_VERSION, BraintrustState
from braintrust.span_cache import CachedSpan
from braintrust.trace import (
    CachedSpanFetcher,
    LocalTrace,
    SpanData,
    SpanFetcher,
    _matches_span_filters,
    _normalize_span_filters,
)


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
@pytest.mark.asyncio
async def test_span_filters_backend_parity(vcr_cassette):
    state = BraintrustState()
    experiment = braintrust.init(
        project="python-sdk-vcr-tests",
        experiment="span-filters-backend-parity-v2",
        update=True,
        api_key=os.environ.get("BRAINTRUST_API_KEY", "sk-dummy-for-vcr-replay"),
        git_metadata_settings=GitMetadataSettings(collect="none"),
        state=state,
        set_current=False,
    )
    experiment._get_state()
    root = "span-filters-root"
    spans = [
        SpanData(span_id=root, span_attributes={"name": "root", "type": "task"}),
        SpanData(
            span_id="search",
            span_attributes={"name": "search", "type": "tool"},
            metrics={"start": 100, "end": 102},
            metadata={"request": {"region": "us", "model": None}, "flag": True},
        ),
        SpanData(
            span_id="failed",
            span_attributes={"name": "search", "type": "tool"},
            error="failed",
            metrics={"start": 100, "end": 105},
            metadata={"request": {"region": "eu", "model": "test"}, "flag": 1},
        ),
        SpanData(
            span_id="lookup",
            span_attributes={"name": "lookup", "type": "llm"},
            error="",
            metrics={"start": 100, "end": 100.5},
            metadata={"request": {}},
        ),
        SpanData(span_id="open", span_attributes={"name": "open", "type": "tool"}, metrics={"start": 100}),
        SpanData(
            span_id="scorer",
            span_attributes={"name": "search", "type": "score", "purpose": "scorer"},
            metrics={"start": 100, "end": 102},
        ),
    ]
    rows = [
        dict(
            span.to_dict(),
            id=span.span_id,
            root_span_id=root,
            experiment_id=experiment.id,
            span_parents=[] if span.span_id == root else [root],
        )
        for span in spans
    ]
    state.api_conn().post("/logs3", json={"rows": rows, "api_version": DATA_API_VERSION}).raise_for_status()

    async def get_state():
        return state

    remote = CachedSpanFetcher(
        object_type="experiment", object_id=experiment.id, root_span_id=root, get_state=get_state
    )
    cases = [
        ({"span_type": ["tool"]}, {"search", "failed", "open"}),
        ({"name": ["search", "lookup"]}, {"search", "failed", "lookup"}),
        ({"has_error": True}, {"failed", "lookup"}),
        ({"has_error": False}, {root, "search", "open"}),
        ({"metadata": {"request": {"region": "us"}}}, {"search"}),
        ({"metadata": {"request": {"model": None}}}, {root, "search", "lookup", "open"}),
        ({"metadata": {"flag": True}}, {"search"}),
        ({"metadata": {"flag": 1}}, {"failed"}),
        ({"duration": {"min": 2, "max": 5}}, {"search", "failed"}),
        ({"duration": {"max": 0.5}}, {"lookup"}),
        ({"name": ["search"], "has_error": False, "duration": {"min": 2, "max": 2}}, {"search"}),
        ({"name": []}, set()),
        ({"span_type": []}, set()),
        ({"metadata": {}}, {root, "search", "failed", "lookup", "open"}),
        ({"metadata": {"request": {}}}, {root, "search", "failed", "lookup", "open"}),
        ({"duration": {}}, {root, "search", "failed", "lookup", "open"}),
        ({"duration": {"min": -1}}, {"search", "failed", "lookup"}),
        ({"duration": {"min": 5, "max": 2}}, set()),
    ]
    # Fetch each filter before populating the complete remote cache.
    backend_results = [await remote.get_spans(filters=filters) for filters, _ in cases]
    await remote.get_spans()
    local = LocalTrace("experiment", experiment.id, root, None, state)
    state.span_cache.start()
    try:
        for span in spans:
            state.span_cache.queue_write(root, span.span_id, CachedSpan.from_dict(span.to_dict()))

        request_count = (len(vcr_cassette.requests), vcr_cassette.play_count)
        for (filters, expected), backend in zip(cases, backend_results):
            assert {span.span_id for span in backend} == expected, filters
            assert {span.span_id for span in await remote.get_spans(filters=filters)} == expected, filters
            assert {span.span_id for span in await local.get_spans(filters=filters)} == expected, filters
        assert {
            span.span_id for span in await local.get_spans(filters={"name": ["search"]}, include_scorers=True)
        } == {"search", "failed", "scorer"}
        assert (len(vcr_cassette.requests), vcr_cassette.play_count) == request_count
    finally:
        state.span_cache.stop()
        state.span_cache.dispose()
    assert {span.span_id for span in await remote.get_spans(filters={"name": ["search"]}, include_scorers=True)} == {
        "search",
        "failed",
        "scorer",
    }


# Helper to create span data
def make_span(span_id: str, span_type: str, *, name: str | None = None, **extra) -> SpanData:
    span_attributes = {"type": span_type}
    if name is not None:
        span_attributes["name"] = name
    return SpanData(
        span_id=span_id,
        input={"text": f"input-{span_id}"},
        output={"text": f"output-{span_id}"},
        span_attributes=span_attributes,
        **extra,
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, True),
        ({}, True),
        ({"request": None}, True),
        ({"request": {}}, True),
        ({"request": {"model": None}}, True),
        ({"request": {"model": "gpt-5"}}, False),
    ],
)
def test_null_metadata_filter_matches_missing_paths(metadata, expected):
    filters = _normalize_span_filters({"metadata": {"request": {"model": None}}})
    assert _matches_span_filters(SpanData(metadata=metadata), filters) is expected
    assert not _matches_span_filters(SpanData(metadata=metadata), {"metadata": {"request": {"model": "other"}}})


@pytest.mark.parametrize("actual, expected", [(True, 1), (1, True), ([True], [1]), ([{"flag": True}], [{"flag": 1}])])
def test_metadata_filters_do_not_coerce_booleans(actual, expected):
    assert not _matches_span_filters(SpanData(metadata={"value": actual}), {"metadata": {"value": expected}})
    assert _matches_span_filters(SpanData(metadata={"value": actual}), {"metadata": {"value": actual}})


class TestCachedSpanFetcher:
    """Test CachedSpanFetcher caching behavior."""

    @pytest.mark.asyncio
    async def test_fetch_all_spans_without_filter(self):
        """Test fetching all spans when no filter specified."""
        mock_spans = [
            make_span("span-1", "llm"),
            make_span("span-2", "function"),
            make_span("span-3", "llm"),
        ]

        call_count = 0

        async def fetch_fn(filters):
            nonlocal call_count
            call_count += 1
            return mock_spans

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)
        result = await fetcher.get_spans()

        assert call_count == 1
        assert len(result) == 3
        assert {s.span_id for s in result} == {"span-1", "span-2", "span-3"}

    @pytest.mark.asyncio
    async def test_fetch_all_after_typed_fetch_has_no_duplicates(self):
        """A typed fetch followed by a full fetch must not duplicate spans."""
        all_spans = [
            make_span("fn-1", "function"),
            make_span("llm-1", "llm"),
            make_span("llm-2", "llm"),
        ]

        async def fetch_fn(filters):
            span_type = filters.get("span_type")
            if span_type:
                return [s for s in all_spans if s.span_attributes["type"] in span_type]
            return all_spans

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)
        await fetcher.get_spans(filters={"span_type": ["llm"]})
        result = await fetcher.get_spans()

        span_ids = [s.span_id for s in result]
        assert sorted(span_ids) == ["fn-1", "llm-1", "llm-2"]
        assert len(span_ids) == len(set(span_ids)), f"duplicate spans: {span_ids}"

    def test_span_data_roundtrip(self):
        row = {
            "span_id": "tool-span",
            "expected": {"answer": "ok"},
            "error": "boom",
            "metrics": {"start": 1, "end": 2},
            "scores": {"quality": 0},
            "tags": ["debug"],
        }
        assert SpanData.from_dict(row).to_dict() == row

    @pytest.mark.asyncio
    async def test_fetch_specific_span_types(self):
        """Test fetching specific span types when filter specified."""
        llm_spans = [make_span("span-1", "llm"), make_span("span-2", "llm")]

        call_count = 0

        async def fetch_fn(filters):
            nonlocal call_count
            call_count += 1
            assert filters == {"span_type": ["llm"]}
            return llm_spans

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)
        result = await fetcher.get_spans(filters={"span_type": ["llm"]})

        assert call_count == 1
        assert len(result) == 2

    @pytest.mark.parametrize(
        ("span_type", "expected_ids"),
        [
            (None, ["span-1", "span-2", "span-3", "span-4"]),
            (["llm"], ["span-1", "span-4"]),
            (["llm", "tool"], ["span-1", "span-3", "span-4"]),
            (["nonexistent"], []),
        ],
    )
    @pytest.mark.asyncio
    async def test_full_cache_answers_any_span_type_query(self, span_type, expected_ids):
        """One unfiltered fetch makes the cache authoritative for every span type.

        Including types that turn out to be absent: an empty result is a real answer here,
        not a cache miss to be retried against the server.
        """
        all_spans = [
            make_span("span-1", "llm"),
            make_span("span-2", "function"),
            make_span("span-3", "tool"),
            make_span("span-4", "llm"),
        ]
        call_count = 0

        async def fetch_fn(filters):
            nonlocal call_count
            call_count += 1
            return all_spans

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)
        await fetcher.get_spans()

        result = await fetcher.get_spans(filters={"span_type": span_type} if span_type else None)

        assert call_count == 1
        assert sorted(span.span_id for span in result) == expected_ids

    @pytest.mark.asyncio
    async def test_partial_cache_fetches_only_missing_types(self):
        """A type already in the cache is never re-requested, only the types missing from it."""
        by_type = {"llm": [make_span("span-1", "llm")], "function": [make_span("span-2", "function")]}
        requested = []

        async def fetch_fn(filters):
            requested.append(filters["span_type"])
            return [span for t in filters["span_type"] for span in by_type.get(t, [])]

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)

        assert [s.span_id for s in await fetcher.get_spans(filters={"span_type": ["llm"]})] == ["span-1"]
        assert [s.span_id for s in await fetcher.get_spans(filters={"span_type": ["llm"]})] == ["span-1"]
        result = await fetcher.get_spans(filters={"span_type": ["llm", "function"]})

        assert sorted(span.span_id for span in result) == ["span-1", "span-2"]
        # The second call was served from cache; the third asked only for what it lacked.
        assert requested == [["llm"], ["function"]]

    @pytest.mark.asyncio
    async def test_handle_spans_with_no_type(self):
        """Test handling spans without type (empty string type)."""
        spans = [
            make_span("span-1", "llm"),
            SpanData(span_id="span-2", input={}, span_attributes={}),  # No type
            SpanData(span_id="span-3", input={}),  # No span_attributes
        ]

        async def fetch_fn(filters):
            return spans

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)

        # Fetch all
        result = await fetcher.get_spans()
        assert len(result) == 3

        # Spans without type go into "" bucket
        no_type_result = await fetcher.get_spans(filters={"span_type": [""]})
        assert len(no_type_result) == 2

    @pytest.mark.parametrize("filters", [None, {"span_type": ["llm"]}])
    @pytest.mark.asyncio
    async def test_empty_results_are_not_cached(self, filters):
        """An empty fetch caches nothing, so spans logged later are still picked up.

        The cache records which types it holds by the spans it saw, so a fetch that returned
        nothing leaves no trace and the next call goes back to the server.
        """
        call_count = 0

        async def fetch_fn(_filters):
            nonlocal call_count
            call_count += 1
            return [] if call_count == 1 else [make_span("span-1", "llm")]

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)

        assert await fetcher.get_spans(filters=filters) == []
        assert [span.span_id for span in await fetcher.get_spans(filters=filters)] == ["span-1"]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_advanced_filters_are_pushed_down_and_never_cached(self):
        """Filters the cache cannot reason about go to the fetcher whole, every time.

        The cache is partitioned by span type alone, so it cannot tell whether it holds
        every span matching some other field. Rather than guess, these queries are pushed
        down in full and their results are used once and discarded.
        """
        spans = [
            make_span("errored", "tool", name="search", error={"message": "boom"}),
            make_span("successful", "tool", name="search"),
        ]
        received = []

        async def fetch_fn(filters):
            received.append(filters)
            return [span for span in spans if _matches_span_filters(span, filters)]

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)
        filters = {"span_type": ["tool"], "has_error": True}

        first = await fetcher.get_spans(filters=filters)
        second = await fetcher.get_spans(filters=filters)

        # Handed down whole, returned unchanged (no second, client-side filtering pass),
        # and re-fetched rather than served from the first call's results.
        assert received == [filters, filters]
        assert [span.span_id for span in first] == ["errored"]
        assert [span.span_id for span in second] == ["errored"]

    @pytest.mark.parametrize(
        ("filters", "message"),
        [
            ({"span_type": "tool"}, "span_type"),
            ({"name": [1]}, "name"),
            ({"has_error": "yes"}, "has_error"),
            ({"metadata": []}, "metadata"),
            ({"metadata": {1: "value"}}, "metadata"),
            ({"duration": {"min": "slow"}}, "duration"),
            ({"duration": {"min": float("nan")}}, "duration"),
            ({"duration": {"minimum": 1}}, "duration"),
            ({"unknown": True}, "Unsupported"),
        ],
    )
    def test_rejects_invalid_advanced_filters(self, filters, message):
        with pytest.raises(ValueError, match=message):
            _normalize_span_filters(filters)

    @pytest.mark.parametrize(
        ("brainstore_realtime", "expected"),
        [
            (None, True),
            (False, False),
        ],
    )
    def test_span_fetcher_threads_realtime_setting(self, brainstore_realtime, expected):
        calls = []
        state = _DummyState(calls)
        kwargs = dict(
            object_type="project_logs",
            object_id="project-1",
            root_span_id="root-1",
            state=state,
        )
        if brainstore_realtime is not None:
            kwargs["brainstore_realtime"] = brainstore_realtime
        fetcher = SpanFetcher(**kwargs)

        assert list(fetcher.fetch()) == []
        assert calls[0]["json"]["brainstore_realtime"] is expected

    @pytest.mark.asyncio
    async def test_cached_span_fetcher_threads_realtime_setting(self):
        calls = []
        state = _DummyState(calls)

        async def get_state():
            return state

        fetcher = CachedSpanFetcher(
            object_type="project_logs",
            object_id="project-1",
            root_span_id="root-1",
            get_state=get_state,
            brainstore_realtime=False,
        )

        assert await fetcher.get_spans() == []
        assert calls[0]["json"]["brainstore_realtime"] is False


@pytest.mark.asyncio
@pytest.mark.filterwarnings("error::DeprecationWarning")
async def test_span_type_argument_compatibility():
    state = BraintrustState()
    state.span_cache.start()
    try:
        for span_id, span_type in (("tool", "tool"), ("llm", "llm")):
            state.span_cache.queue_write(
                "root", span_id, CachedSpan(span_id=span_id, span_attributes={"type": span_type})
            )
        trace = LocalTrace("experiment", "experiment", "root", None, state)
        for filters in (None, {}):
            assert {span.span_id for span in await trace.get_spans(filters=filters)} == {"tool", "llm"}
        assert {span.span_id for span in await trace.get_spans(span_type=[])} == {"tool", "llm"}
        assert [span.span_id for span in await trace.get_spans(["tool"])] == ["tool"]
        assert [span.span_id for span in await trace.get_spans(span_type=["llm"], filters={"has_error": False})] == [
            "llm"
        ]
        assert await trace.get_spans(filters={"span_type": []}) == []
        with pytest.raises(ValueError, match="span_type"):
            await trace.get_spans(["tool"], filters={"span_type": ["llm"]})
    finally:
        state.span_cache.stop()
        state.span_cache.dispose()


class _DummyState:
    def __init__(self, api_calls=None):
        self.api_calls = api_calls

    def login(self):
        return None

    def api_conn(self):
        return _DummyApiConn(self.api_calls)


class _DummyResponse:
    text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": []}


class _DummyApiConn:
    def __init__(self, calls):
        self.calls = calls

    def post(self, path, *args, **kwargs):
        if self.calls is not None:
            self.calls.append({"path": path, "args": args, **kwargs})
        return _DummyResponse()


class TestLocalTraceGetThread:
    @pytest.mark.asyncio
    async def test_calls_invoke_with_correct_parameters(self, monkeypatch):
        mock_thread = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        calls = []

        def fake_invoke(**kwargs):
            calls.append(kwargs)
            return mock_thread

        monkeypatch.setattr("braintrust.trace.invoke", fake_invoke)

        trace = LocalTrace(
            object_type="experiment",
            object_id="exp-123",
            root_span_id="root-456",
            ensure_spans_flushed=None,
            state=_DummyState(),
        )

        result = await trace.get_thread()

        assert len(calls) == 1
        assert calls[0]["global_function"] == "project_default"
        assert calls[0]["function_type"] == "preprocessor"
        assert calls[0]["mode"] == "json"
        assert calls[0]["input"] == {
            "trace_ref": {
                "object_type": "experiment",
                "object_id": "exp-123",
                "root_span_id": "root-456",
            }
        }
        assert result == mock_thread

    @pytest.mark.asyncio
    async def test_uses_custom_preprocessor(self, monkeypatch):
        calls = []

        def fake_invoke(**kwargs):
            calls.append(kwargs)
            return [{"role": "user", "content": "Test"}]

        monkeypatch.setattr("braintrust.trace.invoke", fake_invoke)

        trace = LocalTrace(
            object_type="project_logs",
            object_id="proj-789",
            root_span_id="root-abc",
            ensure_spans_flushed=None,
            state=_DummyState(),
        )

        await trace.get_thread(options={"preprocessor": "custom_preprocessor"})
        assert calls[0]["global_function"] == "custom_preprocessor"
        assert calls[0]["function_type"] == "preprocessor"

    @pytest.mark.asyncio
    async def test_caches_by_preprocessor(self, monkeypatch):
        call_count = 0

        def fake_invoke(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["global_function"] == "project_default":
                return [{"role": "user", "content": "Default"}]
            return [{"role": "user", "content": "Custom"}]

        monkeypatch.setattr("braintrust.trace.invoke", fake_invoke)

        trace = LocalTrace(
            object_type="experiment",
            object_id="exp-123",
            root_span_id="root-456",
            ensure_spans_flushed=None,
            state=_DummyState(),
        )

        result1 = await trace.get_thread()
        result2 = await trace.get_thread()
        result3 = await trace.get_thread(options={"preprocessor": "custom"})
        result4 = await trace.get_thread()

        assert result1 == [{"role": "user", "content": "Default"}]
        assert result2 == [{"role": "user", "content": "Default"}]
        assert result3 == [{"role": "user", "content": "Custom"}]
        assert result4 == [{"role": "user", "content": "Default"}]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_returns_empty_array_for_non_array_invoke_result(self, monkeypatch):
        def fake_invoke(**kwargs):
            return "not-an-array"

        monkeypatch.setattr("braintrust.trace.invoke", fake_invoke)

        trace = LocalTrace(
            object_type="experiment",
            object_id="exp-123",
            root_span_id="root-456",
            ensure_spans_flushed=None,
            state=_DummyState(),
        )

        result = await trace.get_thread()
        assert result == []
