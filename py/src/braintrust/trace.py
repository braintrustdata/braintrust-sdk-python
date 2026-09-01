"""
Trace objects for accessing spans in evaluations.

This module provides the LocalTrace class which allows scorers to access
spans from the current evaluation task without making server round-trips.
"""

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, TypedDict, cast

from braintrust.functions.invoke import invoke
from braintrust.logger import BraintrustState, ObjectFetcher
from braintrust.types import Metadata
from braintrust.util import clean_nones


class SpanDurationFilter(TypedDict, total=False):
    """Inclusive duration bounds, in seconds."""

    min: float
    """Minimum value of metrics.end - metrics.start."""
    max: float
    """Maximum value of metrics.end - metrics.start."""


class SpanFilters(TypedDict, total=False):
    """Filters supported by Trace.get_spans(). Different fields combine with AND.

    Empty name/span_type lists match no spans. Empty metadata/duration objects
    add no constraints. Omit a field to leave it unfiltered.
    """

    span_type: list[str]
    """Match spans whose span_attributes.type equals any of these."""
    name: list[str]
    """Match spans whose span_attributes.name equals any of these."""
    has_error: bool
    """True to keep only spans that recorded an error, False to keep only those that did not."""
    metadata: dict[str, Any]
    """Match named metadata keys at any depth without type coercion. None matches null or missing paths."""
    duration: SpanDurationFilter
    """Bound how long the span took, inclusive, in seconds."""


def _metadata_leaves(metadata: Mapping[str, Any], path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Flatten a partial metadata object into paths shared by local and BTQL matching."""
    leaves = []
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("filters.metadata keys must be strings")
        child_path = (*path, key)
        if isinstance(value, Mapping):
            leaves.extend(_metadata_leaves(value, child_path))
        else:
            leaves.append((child_path, value))
    return leaves


def _normalize_span_filters(filters: Any, span_type: list[str] | None = None) -> SpanFilters:
    """Check shapes needed by both execution paths and fold in the top-level span_type."""
    if filters is not None and not isinstance(filters, Mapping):
        raise ValueError("filters must be an object")
    values = dict(filters or {})
    if span_type is not None:
        if "span_type" in values:
            raise ValueError("span_type cannot be provided both directly and in filters")
        # Preserve the original API's span_type=[] meaning of no constraint.
        if span_type:
            values["span_type"] = span_type
    if set(values) - SpanFilters.__annotations__.keys():
        raise ValueError("Unsupported span filter fields")
    for field in ("span_type", "name"):
        if field in values:
            items = values[field]
            if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
                raise ValueError(f"filters.{field} must be a list of strings")
    if "has_error" in values and not isinstance(values["has_error"], bool):
        raise ValueError("filters.has_error must be a boolean")
    if "metadata" in values:
        if not isinstance(values["metadata"], Mapping):
            raise ValueError("filters.metadata must be an object")
        _metadata_leaves(values["metadata"])
    if "duration" in values:
        bounds = values["duration"]
        if not isinstance(bounds, Mapping) or set(bounds) - {"min", "max"}:
            raise ValueError("filters.duration must be an object with min and/or max")
        if any(not _is_finite_number(value) for value in bounds.values()):
            raise ValueError("filters.duration bounds must be finite numbers")
    return cast(SpanFilters, values)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metadata_equal(actual: Any, expected: Any) -> bool:
    """JSON equality without Python's bool/number coercion, including inside arrays."""
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_metadata_equal(a, e) for a, e in zip(actual, expected))
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and actual.keys() == expected.keys()
            and all(_metadata_equal(actual[key], value) for key, value in expected.items())
        )
    return actual == expected


def _matches_span_filters(span: Any, filters: SpanFilters) -> bool:
    attributes = span.span_attributes or {}
    for field, attribute in (("span_type", "type"), ("name", "name")):
        if field in filters and attributes.get(attribute) not in filters[field]:
            return False
    if "has_error" in filters and (span.error is not None) != filters["has_error"]:
        return False
    for path, expected in _metadata_leaves(filters.get("metadata", {})):
        actual = span.metadata
        for key in path:
            actual = actual.get(key) if isinstance(actual, Mapping) else None
        if not _metadata_equal(actual, expected):
            return False
    if bounds := filters.get("duration"):
        metrics = span.metrics or {}
        start, end = metrics.get("start"), metrics.get("end")
        if not _is_finite_number(start) or not _is_finite_number(end):
            return False
        elapsed = end - start
        if "min" in bounds and elapsed < bounds["min"]:
            return False
        if "max" in bounds and elapsed > bounds["max"]:
            return False
    return True


def _btql_cmp(op: str, name: list[str], value: Any) -> dict[str, Any]:
    return {"op": op, "left": {"op": "ident", "name": name}, "right": {"op": "literal", "value": value}}


def _btql_null_check(op: str, name: list[str]) -> dict[str, Any]:
    return {"op": op, "expr": {"op": "ident", "name": name}}


def _span_filter_clauses(filters: SpanFilters) -> list[dict[str, Any]]:
    children = []
    for field, attribute in (("span_type", "type"), ("name", "name")):
        if field in filters:
            # BTQL rejects IN []; an empty set of alternatives is always false.
            children.append(
                _btql_cmp("in", ["span_attributes", attribute], filters[field])
                if filters[field]
                else {"op": "literal", "value": False}
            )
    if "has_error" in filters:
        children.append(_btql_null_check("isnotnull" if filters["has_error"] else "isnull", ["error"]))
    for path, value in _metadata_leaves(filters.get("metadata", {})):
        name = ["metadata", *path]
        children.append(_btql_null_check("isnull", name) if value is None else _btql_cmp("eq", name, value))
    elapsed = {
        "op": "sub",
        "left": {"op": "ident", "name": ["metrics", "end"]},
        "right": {"op": "ident", "name": ["metrics", "start"]},
    }
    bounds = filters.get("duration", {})
    for bound, op in (("min", "ge"), ("max", "le")):
        if bound in bounds:
            children.append({"op": op, "left": elapsed, "right": {"op": "literal", "value": bounds[bound]}})
    return children


class SpanData:
    """One span, as returned by get_spans().

    Fields mirror the span columns; anything the server sends that is not named explicitly
    is still kept, as an attribute, so a newer backend does not lose data on the way through.
    """

    def __init__(
        self,
        input: Any | None = None,
        output: Any | None = None,
        metadata: Metadata | None = None,
        expected: Any | None = None,
        error: Any | None = None,
        scores: Any | None = None,
        metrics: Any | None = None,
        span_id: str | None = None,
        span_parents: list[str] | None = None,
        span_attributes: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ):
        self.input = input
        self.output = output
        self.metadata = metadata
        self.expected = expected
        self.error = error
        self.scores = scores
        self.metrics = metrics
        self.span_id = span_id
        self.span_parents = span_parents
        self.span_attributes = span_attributes
        self.tags = tags
        # Store any additional fields
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpanData":
        """Build a span from a row, keeping columns this class does not name."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return the span's set fields, dropping those left as None."""
        return clean_nones(self.__dict__)


class SpanFetcher(ObjectFetcher[dict[str, Any]]):
    """
    Fetcher for spans by root_span_id, using the ObjectFetcher pattern.
    Handles pagination automatically via cursor-based iteration.
    """

    def __init__(
        self,
        object_type: str,  # Literal["experiment", "project_logs", "playground_logs"]
        object_id: str,
        root_span_id: str,
        state: BraintrustState,
        include_scorers: bool = False,
        brainstore_realtime: bool = True,
        filters: SpanFilters | None = None,
    ):
        # `filters` is expected to already be normalized by _normalize_span_filters.
        filter_expr = self._build_filter(root_span_id, filters, include_scorers)

        super().__init__(
            object_type=object_type,
            _internal_btql={"filter": filter_expr},
            _internal_brainstore_realtime=brainstore_realtime,
        )
        self._object_id = object_id
        self._state = state

    @staticmethod
    def _build_filter(
        root_span_id: str,
        filters: SpanFilters | None = None,
        include_scorers: bool = False,
    ) -> dict[str, Any]:
        """Combine trace identity, scorer exclusion, and span filters with AND."""
        # Scorer exclusion is a fetch mode rather than a SpanFilters field, so it stays here.
        purpose = ["span_attributes", "purpose"]
        children: list[dict[str, Any]] = [_btql_cmp("eq", ["root_span_id"], root_span_id)]

        if not include_scorers:
            children.append(
                {
                    "op": "or",
                    "children": [
                        _btql_null_check("isnull", purpose),
                        _btql_cmp("ne", purpose, "scorer"),
                    ],
                }
            )

        children.extend(_span_filter_clauses(filters or {}))

        return {"op": "and", "children": children}

    @property
    def id(self) -> str:
        return self._object_id

    def _get_state(self) -> BraintrustState:
        return self._state


SpanFetchFn = Callable[[SpanFilters], Awaitable[list[SpanData]]]
SpanFetchWithOptionsFn = Callable[[SpanFilters, bool], Awaitable[list[SpanData]]]


class GetThreadOptions(TypedDict, total=False):
    preprocessor: str


class CachedSpanFetcher:
    """
    Fetches spans for one root span, reusing what it has already seen.

    The cache is keyed by span type, plus a flag for whether an unfiltered fetch has
    happened. That shape is what makes it useful and also what bounds it: it can answer a
    span_type query offline, because it knows it holds every span of the types it has
    fetched, but it cannot answer a query on any other field, because a partial result set
    says nothing about the spans it never asked for. Those queries go to the server every
    time and their results are used once rather than cached.
    """

    def __init__(
        self,
        object_type: str | None = None,  # Literal["experiment", "project_logs", "playground_logs"]
        object_id: str | None = None,
        root_span_id: str | None = None,
        get_state: Callable[[], Awaitable[BraintrustState]] | None = None,
        fetch_fn: SpanFetchFn | None = None,
        brainstore_realtime: bool = True,
    ):
        self._span_cache: dict[str, list[SpanData]] = {}
        self._all_fetched = False

        if fetch_fn is not None:
            # Direct fetch function injection (for testing). Like the server, the injected
            # function is responsible for honoring every filter it is given.
            async def _fetch_fn(
                filters: SpanFilters,
                include_scorers: bool = False,
            ) -> list[SpanData]:
                del include_scorers
                return await fetch_fn(filters)

            self._fetch_fn: SpanFetchWithOptionsFn = _fetch_fn
        else:
            # Standard constructor with SpanFetcher
            if object_type is None or object_id is None or root_span_id is None or get_state is None:
                raise ValueError(
                    "Must provide either fetch_fn or all of object_type, object_id, root_span_id, get_state"
                )

            async def _fetch_fn(
                filters: SpanFilters,
                include_scorers: bool = False,
            ) -> list[SpanData]:
                state = await get_state()
                fetcher = SpanFetcher(
                    object_type=object_type,
                    object_id=object_id,
                    root_span_id=root_span_id,
                    state=state,
                    include_scorers=include_scorers,
                    brainstore_realtime=brainstore_realtime,
                    filters=filters,
                )
                spans = [SpanData.from_dict(row) for row in fetcher.fetch()]
                # Backend comparisons can coerce metadata types. Keep the same exact
                # matching as the local cache while still pushing filters down.
                if filters.get("metadata"):
                    spans = [span for span in spans if _matches_span_filters(span, filters)]
                return spans

            self._fetch_fn = _fetch_fn

    async def get_spans(
        self,
        *,
        filters: SpanFilters | None = None,
        include_scorers: bool = False,
    ) -> list[SpanData]:
        """
        Get spans, using the cache where it can answer the query.

        Args:
            filters: Optional filters for span type, name, error state, metadata, and duration
            include_scorers: Include spans with span_attributes.purpose = "scorer"

        Returns:
            List of matching spans
        """
        filters = _normalize_span_filters(filters)
        span_type = filters.get("span_type")
        # A partial cache is only authoritative for the fields it partitions on.
        has_advanced_filters = any(field != "span_type" for field in filters)
        if span_type == []:
            return []

        if include_scorers:
            return await self._fetch_fn(filters, True)

        # A complete cache can answer every supported filter locally.
        if self._all_fetched:
            spans = self._get_from_cache(span_type)
            return [span for span in spans if _matches_span_filters(span, filters)] if has_advanced_filters else spans

        # Arbitrary filtered results are not authoritative for their span type.
        if has_advanced_filters:
            return await self._fetch_fn(filters, False)

        # If no filter requested, fetch everything.
        if not span_type:
            # A full fetch is authoritative; reset the per-type cache first so a
            # prior typed fetch's spans are not duplicated by re-fetching them
            # (_fetch_spans appends).
            self._span_cache = {}
            await self._fetch_spans(None)
            if self._span_cache:  # Only cache if we got results
                self._all_fetched = True
            return self._get_from_cache(None)

        # Find which span types we don't have in cache yet.
        missing_types = [t for t in span_type if t not in self._span_cache]
        if missing_types:
            await self._fetch_spans(missing_types)
        return self._get_from_cache(span_type)

    async def _fetch_spans(self, span_type: list[str] | None) -> None:
        """Fetch spans and file them into the cache under their own type.

        Spans are filed by the type they report, not the type that was asked for, so a
        requested type that yields nothing leaves no entry and will be asked for again.
        """
        spans = await self._fetch_fn({"span_type": span_type} if span_type else {}, False)

        for span in spans:
            span_attrs = span.span_attributes or {}
            span_type_str = span_attrs.get("type", "")
            if span_type_str not in self._span_cache:
                self._span_cache[span_type_str] = []
            self._span_cache[span_type_str].append(span)

    def _get_from_cache(self, span_type: list[str] | None) -> list[SpanData]:
        """Read spans back out of the cache, optionally narrowing to some types.

        Assumes the caller has established that the cache holds what is being asked for;
        types with no entry are simply absent from the result, not fetched.
        """
        if not span_type or len(span_type) == 0:
            # Return all spans
            result = []
            for spans in self._span_cache.values():
                result.extend(spans)
            return result

        # Return only requested types
        result = []
        for type_str in span_type:
            if type_str in self._span_cache:
                result.extend(self._span_cache[type_str])
        return result


class Trace(Protocol):
    """
    Interface for trace objects that can be used by scorers.
    Both the SDK's LocalTrace class and the API wrapper's WrapperTrace implement this.
    """

    def get_configuration(self) -> dict[str, str]:
        """Get the trace configuration (object_type, object_id, root_span_id)."""
        ...

    async def get_spans(
        self,
        span_type: list[str] | None = None,
        *,
        filters: SpanFilters | None = None,
        include_scorers: bool = False,
    ) -> list[SpanData]:
        """
        Fetch all spans for this root span.

        Args:
            span_type: Optional span types; may also be provided in filters, but not both
            filters: Optional filters for span type, name, error state, metadata, and duration
            include_scorers: Include spans with span_attributes.purpose = "scorer"

        Returns:
            List of matching spans
        """
        ...

    async def get_thread(self, options: GetThreadOptions | None = None) -> list[Any]:
        """
        Get the thread (preprocessed messages) for this trace.

        Args:
            options: Optional options object. Supports "preprocessor".

        Returns:
            The preprocessed thread as an array of messages.
        """
        ...


class LocalTrace(dict[str, Any]):
    """
    SDK implementation of Trace that uses local span cache and falls back to BTQL.
    Carries identifying information about the evaluation so scorers can perform
    richer logging or side effects.

    Inherits from dict so that it serializes to {"trace_ref": {...}} when passed
    to json.dumps(). This allows LocalTrace to be transparently serialized when
    passed through invoke() or other JSON-serializing code paths.
    """

    def __init__(
        self,
        object_type: str,  # Literal["experiment", "project_logs", "playground_logs"]
        object_id: str,
        root_span_id: str,
        ensure_spans_flushed: Callable[[], Awaitable[None]] | None,
        state: BraintrustState,
    ):
        # Initialize dict with trace_ref for JSON serialization
        super().__init__(
            {
                "trace_ref": {
                    "object_type": object_type,
                    "object_id": object_id,
                    "root_span_id": root_span_id,
                }
            }
        )

        self._object_type = object_type
        self._object_id = object_id
        self._root_span_id = root_span_id
        self._ensure_spans_flushed = ensure_spans_flushed
        self._state = state
        self._spans_flushed = False
        self._spans_flush_promise: asyncio.Task[None] | None = None
        self._thread_cache: dict[str, asyncio.Task[list[Any]]] = {}

        async def get_state() -> BraintrustState:
            await self._ensure_spans_ready()
            # Ensure state is logged in
            await asyncio.get_event_loop().run_in_executor(None, lambda: state.login())
            return state

        self._cached_fetcher = CachedSpanFetcher(
            object_type=object_type,
            object_id=object_id,
            root_span_id=root_span_id,
            get_state=get_state,
        )

    def get_configuration(self) -> dict[str, str]:
        """Get the trace configuration."""
        return {
            "object_type": self._object_type,
            "object_id": self._object_id,
            "root_span_id": self._root_span_id,
        }

    async def get_spans(
        self,
        span_type: list[str] | None = None,
        *,
        filters: SpanFilters | None = None,
        include_scorers: bool = False,
    ) -> list[SpanData]:
        """
        Fetch all rows for this root span from its parent object (experiment or project logs).
        First checks the local span cache for recently logged spans, then falls
        back to CachedSpanFetcher which handles BTQL fetching and caching.

        Args:
            span_type: Optional span types; may also be provided in filters, but not both
            filters: Optional filters for span type, name, error state, metadata, and duration
            include_scorers: Include spans with span_attributes.purpose = "scorer"

        Returns:
            List of matching spans
        """
        normalized_filters = _normalize_span_filters(filters, span_type)

        # Try local span cache first (for recently logged spans not yet flushed)
        cached_spans = self._state.span_cache.get_by_root_span_id(self._root_span_id)
        if cached_spans and len(cached_spans) > 0:
            spans = [
                span
                for span in cached_spans
                if (include_scorers or not (span.span_attributes or {}).get("purpose") == "scorer")
                and _matches_span_filters(span, normalized_filters)
            ]

            return [SpanData.from_dict(span.to_dict()) for span in spans]

        # Fall back to CachedSpanFetcher for BTQL fetching with caching.
        return await self._cached_fetcher.get_spans(filters=normalized_filters, include_scorers=include_scorers)

    async def get_thread(self, options: GetThreadOptions | None = None) -> list[Any]:
        """
        Get the thread (preprocessed messages) for this trace.
        Uses the project default preprocessor, falling back to global "thread".
        """
        preprocessor = options.get("preprocessor") if options and options.get("preprocessor") else None
        cache_key = preprocessor or "project_default"
        if cache_key not in self._thread_cache:
            self._thread_cache[cache_key] = asyncio.create_task(self._fetch_thread(options))
        return await self._thread_cache[cache_key]

    async def _fetch_thread(self, options: GetThreadOptions | None = None) -> list[Any]:
        """Fetch thread messages via preprocessor invocation."""
        await self._ensure_spans_ready()
        await asyncio.get_event_loop().run_in_executor(None, lambda: self._state.login())
        preprocessor = options.get("preprocessor") if options and options.get("preprocessor") else None

        result: Any = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: invoke(
                global_function=preprocessor or "project_default",
                function_type="preprocessor",
                mode="json",
                input={
                    "trace_ref": {
                        "object_type": self._object_type,
                        "object_id": self._object_id,
                        "root_span_id": self._root_span_id,
                    }
                },
            ),
        )

        return result if isinstance(result, list) else []

    async def _ensure_spans_ready(self) -> None:
        """Flush pending spans so a fetch sees them, at most once per trace.

        Concurrent scorers share one in-flight flush rather than each triggering their own.
        A failed flush clears that shared handle so the next caller can retry.
        """
        ensure_spans_flushed = self._ensure_spans_flushed
        if self._spans_flushed or ensure_spans_flushed is None:
            return

        if self._spans_flush_promise is None:

            async def flush_and_mark() -> None:
                try:
                    await ensure_spans_flushed()
                    self._spans_flushed = True
                except Exception as err:
                    self._spans_flush_promise = None
                    raise err

            self._spans_flush_promise = asyncio.create_task(flush_and_mark())

        await self._spans_flush_promise
