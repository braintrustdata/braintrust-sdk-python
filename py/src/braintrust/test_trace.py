"""Tests for Trace functionality."""

import pytest
from braintrust.span_cache import CachedSpan
from braintrust.trace import (
    _FILTER_SPECS,
    CachedSpanFetcher,
    LocalTrace,
    SpanData,
    SpanFetcher,
    SpanFilters,
    _matches_span_filters,
)


# Helper to create mock spans
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


def test_every_span_filter_field_has_a_spec():
    """SpanFilters is the public contract; _FILTER_SPECS is what implements it."""
    assert set(_FILTER_SPECS) == set(SpanFilters.__annotations__)
    # Registration order fixes BTQL child order, so it must track the declaration order.
    assert list(_FILTER_SPECS) == list(SpanFilters.__annotations__)


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

    @pytest.mark.asyncio
    async def test_fetch_preserves_span_result_fields(self):
        """Test that fetched spans preserve fields needed for full trace attachments."""
        mock_spans = [
            make_span(
                "span-1",
                "tool",
                expected={"answer": "ok"},
                error={"message": "boom"},
                metrics={"start": 1, "end": 2},
                scores={"quality": 0},
                tags=["debug"],
            )
        ]

        async def fetch_fn(filters):
            del filters
            return mock_spans

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)
        result = await fetcher.get_spans()

        assert result[0].expected == {"answer": "ok"}
        assert result[0].error == {"message": "boom"}
        assert result[0].metrics == {"start": 1, "end": 2}
        assert result[0].scores == {"quality": 0}
        assert result[0].tags == ["debug"]
        assert result[0].to_dict()["error"] == {"message": "boom"}

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

    def test_span_fetcher_builds_advanced_filter(self):
        calls = []
        state = _DummyState(calls)
        fetcher = SpanFetcher(
            object_type="project_logs",
            object_id="project-1",
            root_span_id="root-1",
            state=state,
            filters={
                "span_type": ["tool"],
                "name": ["search", "lookup"],
                "has_error": False,
                "metadata": {"model": "gpt-5", "optional": None, "request": {"region": "us-east-1"}},
                "duration": {"min": 0.5, "max": 10},
            },
        )

        assert list(fetcher.fetch()) == []

        def comparison(op, path, value):
            return {
                "op": op,
                "left": {"op": "ident", "name": path},
                "right": {"op": "literal", "value": value},
            }

        def isnull(path):
            return {"op": "isnull", "expr": {"op": "ident", "name": path}}

        duration = {
            "op": "sub",
            "left": {"op": "ident", "name": ["metrics", "end"]},
            "right": {"op": "ident", "name": ["metrics", "start"]},
        }
        purpose = ["span_attributes", "purpose"]

        # Asserting the whole expression pins the child order too, so adding a filter
        # cannot silently reorder or drop one.
        assert calls[0]["json"]["query"]["filter"] == {
            "op": "and",
            "children": [
                comparison("eq", ["root_span_id"], "root-1"),
                {"op": "or", "children": [isnull(purpose), comparison("ne", purpose, "scorer")]},
                comparison("in", ["span_attributes", "type"], ["tool"]),
                comparison("in", ["span_attributes", "name"], ["search", "lookup"]),
                isnull(["error"]),
                comparison("eq", ["metadata", "model"], "gpt-5"),
                isnull(["metadata", "optional"]),
                comparison("eq", ["metadata", "request", "region"], "us-east-1"),
                {"op": "ge", "left": duration, "right": {"op": "literal", "value": 0.5}},
                {"op": "le", "left": duration, "right": {"op": "literal", "value": 10}},
            ],
        }

    @pytest.mark.asyncio
    async def test_advanced_filters_use_full_cache_when_available(self):
        spans = [
            make_span(
                "matching",
                "tool",
                name="search",
                error={"message": "boom"},
                metadata={"model": "gpt-5", "request": {"region": "us-east-1", "id": 1}},
                metrics={"start": 1, "end": 4},
            ),
            make_span(
                "too-fast",
                "tool",
                name="search",
                error={"message": "boom"},
                metadata={"model": "gpt-5"},
                metrics={"start": 1, "end": 1.1},
            ),
            make_span("successful", "tool", name="search"),
        ]
        call_count = 0

        async def fetch_fn(filters):
            nonlocal call_count
            del filters
            call_count += 1
            return spans

        fetcher = CachedSpanFetcher(fetch_fn=fetch_fn)
        await fetcher.get_spans()
        result = await fetcher.get_spans(
            filters={
                "span_type": ["tool"],
                "name": ["search"],
                "has_error": True,
                "metadata": {"request": {"region": "us-east-1"}},
                "duration": {"min": 2},
            }
        )

        assert call_count == 1
        assert [span.span_id for span in result] == ["matching"]

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
            ({"span_type": []}, "span_type"),
            ({"name": []}, "name"),
            ({"duration": {"min": -1}}, "duration.min"),
            ({"duration": {"min": 2, "max": 1}}, "duration.min"),
            # A field that is present must constrain something, at any nesting depth.
            ({"metadata": {}}, "filters.metadata must not be empty"),
            ({"metadata": {"a": {}}}, "filters.metadata.a must not be empty"),
            ({"metadata": {"a": {"b": {}}}}, "filters.metadata.a.b must not be empty"),
            ({"duration": {}}, "filters.duration must specify at least one of: min, max"),
        ],
    )
    @pytest.mark.asyncio
    async def test_rejects_invalid_advanced_filters(self, filters, message):
        fetcher = CachedSpanFetcher(fetch_fn=lambda filters: None)

        with pytest.raises(ValueError, match=message):
            await fetcher.get_spans(filters=filters)

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


class TestLocalTraceGetSpans:
    @pytest.mark.asyncio
    async def test_applies_advanced_filters_to_local_spans(self):
        def cached_span(span_id, name, **attributes):
            return CachedSpan(
                span_id=span_id,
                input={"query": "weather"},
                output={"result": "sunny"},
                error={"message": "retryable"},
                metrics={"start": 1, "end": 4},
                metadata={"request": {"region": "us-east-1", "id": 1}},
                span_parents=[],
                span_attributes={"type": "tool", "name": name, **attributes},
            )

        spans = [
            cached_span("matching", "search"),
            cached_span("wrong-name", "lookup"),
            cached_span("scorer", "search", purpose="scorer"),
        ]
        trace = LocalTrace(
            object_type="project_logs",
            object_id="project-1",
            root_span_id="root-1",
            ensure_spans_flushed=None,
            state=_DummyState(spans=spans),
        )

        result = await trace.get_spans(
            filters={
                "span_type": ["tool"],
                "name": ["search"],
                "has_error": True,
                "metadata": {"request": {"region": "us-east-1"}},
                "duration": {"min": 2, "max": 5},
            }
        )

        assert [span.span_id for span in result] == ["matching"]

        with pytest.warns(DeprecationWarning, match="span_type argument is deprecated"):
            legacy_result = await trace.get_spans(span_type=["tool"])
        assert [span.span_id for span in legacy_result] == ["matching", "wrong-name"]

        with pytest.warns(DeprecationWarning, match="span_type argument is deprecated"):
            with pytest.raises(ValueError, match="span_type"):
                await trace.get_spans(["llm"], filters={"span_type": ["tool"]})

    @pytest.mark.asyncio
    async def test_empty_filters_object_is_not_a_filter(self):
        """filters={} constrains nothing, unlike a field that is present but empty."""
        spans = [CachedSpan(span_id="tool-span", span_attributes={"type": "tool"})]
        trace = LocalTrace(
            object_type="project_logs",
            object_id="project-1",
            root_span_id="root-1",
            ensure_spans_flushed=None,
            state=_DummyState(spans=spans),
        )

        assert [span.span_id for span in await trace.get_spans(filters={})] == ["tool-span"]
        assert [span.span_id for span in await trace.get_spans()] == ["tool-span"]

    @pytest.mark.asyncio
    async def test_empty_span_type_only_bypasses_filtering_on_the_deprecated_argument(self):
        """span_type=[] keeps its legacy "no filter" meaning; filters={"span_type": []} does not."""
        spans = [
            CachedSpan(span_id="tool-span", span_attributes={"type": "tool"}),
            CachedSpan(span_id="llm-span", span_attributes={"type": "llm"}),
        ]
        trace = LocalTrace(
            object_type="project_logs",
            object_id="project-1",
            root_span_id="root-1",
            ensure_spans_flushed=None,
            state=_DummyState(spans=spans),
        )

        with pytest.warns(DeprecationWarning, match="span_type argument is deprecated"):
            unfiltered = await trace.get_spans(span_type=[])
        assert [span.span_id for span in unfiltered] == ["tool-span", "llm-span"]

        with pytest.warns(DeprecationWarning, match="span_type argument is deprecated"):
            filtered = await trace.get_spans(span_type=["tool"])
        assert [span.span_id for span in filtered] == ["tool-span"]

        with pytest.raises(ValueError, match="span_type"):
            await trace.get_spans(filters={"span_type": []})


class _DummySpanCache:
    def __init__(self, spans=None):
        self.spans = spans

    def get_by_root_span_id(self, root_span_id: str):
        return self.spans


class _DummyState:
    def __init__(self, api_calls=None, spans=None):
        self.span_cache = _DummySpanCache(spans)
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
