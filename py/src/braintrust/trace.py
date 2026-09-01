"""
Trace objects for accessing spans in evaluations.

This module provides the LocalTrace class which allows scorers to access
spans from the current evaluation task without making server round-trips.
"""

import asyncio
import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ClassVar, Protocol, TypedDict, cast

from braintrust.functions.invoke import invoke
from braintrust.logger import BraintrustState, ObjectFetcher
from braintrust.types import Metadata
from braintrust.util import clean_nones, is_numeric


class SpanDurationFilter(TypedDict, total=False):
    """Inclusive duration bounds, in seconds."""

    min: float
    """Minimum value of metrics.end - metrics.start."""
    max: float
    """Maximum value of metrics.end - metrics.start."""


class SpanFilters(TypedDict, total=False):
    """Filters supported by Trace.get_spans(). Different fields combine with AND.

    Every field provided must constrain something: empty lists and empty objects are
    rejected rather than ignored, so a filter that comes out empty fails loudly instead
    of silently matching every span. Omit a field to leave it unfiltered.
    """

    span_type: list[str]
    """Match spans whose span_attributes.type equals any of these."""
    name: list[str]
    """Match spans whose span_attributes.name equals any of these."""
    has_error: bool
    """True to keep only spans that recorded an error, False to keep only those that did not."""
    metadata: dict[str, Any]
    """Match the named metadata keys, at any depth, leaving the rest of the object free."""
    duration: SpanDurationFilter
    """Bound how long the span took, inclusive, in seconds."""


_DURATION_FILTER_FIELDS = ("min", "max")


def _merge_deprecated_span_type(span_type: list[str] | None, filters: Any) -> Any:
    """Fold the deprecated positional span_type argument into a filters object.

    Passing it both ways is an error rather than a merge, since there is no sensible way to
    combine two lists that were each meant to be the whole constraint. span_type=[] keeps
    its historical meaning of "no filter", which is why it cannot simply be copied across:
    inside filters an empty list is rejected.
    """
    if span_type is None:
        return filters
    warnings.warn(
        "The span_type argument is deprecated; use filters={'span_type': [...]} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if filters is not None and not isinstance(filters, Mapping):
        raise ValueError("filters must be an object")
    if filters is not None and "span_type" in filters:
        raise ValueError("span_type cannot be provided both directly and in filters")
    if not span_type:
        # Preserve the legacy behavior where span_type=[] means no filter.
        return filters
    return {**(filters or {}), "span_type": span_type}


def _btql_cmp(op: str, name: list[str], value: Any) -> dict[str, Any]:
    """BTQL comparison between the column at path `name` and a literal value."""
    return {"op": op, "left": {"op": "ident", "name": name}, "right": {"op": "literal", "value": value}}


def _btql_null_check(op: str, name: list[str]) -> dict[str, Any]:
    """BTQL "isnull" or "isnotnull" test on the column at path `name`."""
    return {"op": op, "expr": {"op": "ident", "name": name}}


class _FilterSpec(ABC):
    """One field of SpanFilters, and everything the SDK knows about it.

    Each filter has to be understood twice: the server evaluates it as BTQL, and the SDK
    evaluates it directly against spans it already holds. Those two readings have to agree,
    so they sit on one class instead of in per-field branches spread across three distant
    functions where they can quietly drift apart.

    Adding a filter means a new subclass plus a SpanFilters entry;
    test_every_span_filter_field_has_a_spec is what keeps that pairing honest.
    """

    field: ClassVar[str]
    """The SpanFilters key this class implements."""
    cacheable: ClassVar[bool] = False
    """Whether CachedSpanFetcher's cache can answer this filter without asking the server.

    That cache is partitioned by span type and knows only that it holds every span of a
    given type. For any other field it cannot tell a genuine empty result from a gap in
    what it has fetched, so the query has to go out.
    """

    @abstractmethod
    def validate(self, value: Any) -> Any:
        """Check a raw filter value and return it in the form the other two methods expect.

        Raises ValueError if the value constrains nothing. An empty list or object is read
        as a mistake rather than as a request for everything, so that a filter built up
        programmatically fails loudly instead of quietly matching the whole trace.
        """

    @abstractmethod
    def matches(self, span: Any, value: Any) -> bool:
        """Evaluate this filter locally, against a SpanData or CachedSpan.

        `value` has already been through validate(). The result must agree with to_btql():
        a span the server would have returned is one this returns True for.
        """

    @abstractmethod
    def to_btql(self, value: Any) -> list[dict[str, Any]]:
        """Compile this filter into BTQL clauses, ANDed with the rest of the query.

        `value` has already been through validate(). Returning several clauses is normal --
        a tag filter emits one per tag -- but returning none would mean the filter
        constrains nothing, which validate() is responsible for having rejected.
        """


class _SpanAttributeFilter(_FilterSpec):
    """Base for filters that match one span_attributes key against a list of values.

    Matching is exact and case-sensitive, and the list is a set of alternatives: a span
    matches if the attribute equals any entry.
    """

    attribute: ClassVar[str]
    """The key to read out of span_attributes."""
    missing: ClassVar[Any] = None
    """Stands in for the attribute when a span does not carry it at all."""

    def validate(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
            raise ValueError(f"filters.{self.field} must be a non-empty list of strings")
        return list(value)

    def matches(self, span: Any, value: Any) -> bool:
        return (getattr(span, "span_attributes", None) or {}).get(self.attribute, self.missing) in value

    def to_btql(self, value: Any) -> list[dict[str, Any]]:
        return [_btql_cmp("in", ["span_attributes", self.attribute], value)]


class _SpanTypeFilter(_SpanAttributeFilter):
    """Filter on span_attributes.type, the one field the span cache is partitioned by."""

    field = "span_type"
    attribute = "type"
    # CachedSpanFetcher files typeless spans under "", so an explicit [""] query finds them.
    missing = ""
    cacheable = True


class _NameFilter(_SpanAttributeFilter):
    """Filter on span_attributes.name. A span with no name matches no name filter."""

    field = "name"
    attribute = "name"


class _HasErrorFilter(_FilterSpec):
    """Filter on whether the span recorded an error, testing only for presence.

    Any non-null error counts, whatever shape it has.
    """

    field = "has_error"

    def validate(self, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("filters.has_error must be a boolean")
        return value

    def matches(self, span: Any, value: Any) -> bool:
        return (getattr(span, "error", None) is not None) == value

    def to_btql(self, value: Any) -> list[dict[str, Any]]:
        return [_btql_null_check("isnotnull" if value else "isnull", ["error"])]


class _MetadataFilter(_FilterSpec):
    """Filter on metadata by deep partial match.

    Only the keys named are compared, at any depth, so this narrows without having to
    describe the whole metadata object. A None leaf matches a key whose value is null.
    """

    field = "metadata"

    def validate(self, value: Any, path: str = "filters.metadata") -> dict[str, Any]:
        """Copy the filter, checking it at every depth.

        Empty objects are rejected wherever they appear, since they constrain nothing.
        `path` follows the recursion so the error names the offending key rather than the
        filter as a whole.
        """
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be an object")
        if not value:
            raise ValueError(f"{path} must not be empty")

        result = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            result[key] = self.validate(child, f"{path}.{key}") if isinstance(child, Mapping) else child
        return result

    def matches(self, span: Any, value: Any) -> bool:
        return self._contains(getattr(span, "metadata", None), value)

    def _contains(self, actual: Any, expected: Mapping[str, Any]) -> bool:
        """Whether `actual` carries every leaf of `expected`.

        Keys `actual` has and `expected` does not are ignored, so {"a": 1} matches a span
        whose metadata is {"a": 1, "b": 2}.
        """
        if not isinstance(actual, Mapping):
            return False
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if isinstance(expected_value, Mapping):
                if not self._contains(actual[key], expected_value):
                    return False
            elif actual[key] != expected_value:
                return False
        return True

    def to_btql(self, value: Any, path: tuple[str, ...] = ("metadata",)) -> list[dict[str, Any]]:
        """Flatten the filter into one comparison per leaf.

        BTQL cannot match an object partially, so {"a": {"b": 1}} has to compile to a single
        comparison against the column metadata.a.b. `path` accumulates that column path as
        the recursion descends.
        """
        children: list[dict[str, Any]] = []
        for key, child in value.items():
            child_path = (*path, key)
            if isinstance(child, Mapping):
                children.extend(self.to_btql(child, child_path))
            elif child is None:
                children.append(_btql_null_check("isnull", list(child_path)))
            else:
                children.append(_btql_cmp("eq", list(child_path), child))
        return children


class _DurationFilter(_FilterSpec):
    """Filter on elapsed wall-clock time, metrics.end - metrics.start, in seconds.

    Both bounds are inclusive. A span that is still open, or whose start/end metrics are
    missing or non-numeric, has no duration and so matches no duration filter.
    """

    field = "duration"

    def validate(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("filters.duration must be an object")
        unknown = set(value) - set(_DURATION_FILTER_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported filters.duration fields: {', '.join(sorted(unknown))}")
        # Iterating the constant rather than `value` keeps the bounds, and the BTQL built
        # from them, in a fixed order.
        bounds = {bound: self._bound(value[bound], bound) for bound in _DURATION_FILTER_FIELDS if bound in value}
        if not bounds:
            raise ValueError(f"filters.duration must specify at least one of: {', '.join(_DURATION_FILTER_FIELDS)}")
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise ValueError("filters.duration.min must be less than or equal to filters.duration.max")
        return bounds

    def _bound(self, value: Any, name: str) -> float:
        """Check one end of the range.

        NaN and infinities are rejected because they make a bound that can never be
        satisfied or never be violated, and negatives because elapsed time cannot be
        negative.
        """
        if not self._is_real_number(value) or not math.isfinite(value) or value < 0:
            raise ValueError(f"filters.duration.{name} must be a finite, non-negative number")
        return value

    def matches(self, span: Any, value: Any) -> bool:
        metrics = getattr(span, "metrics", None)
        if not isinstance(metrics, Mapping):
            return False
        start, end = metrics.get("start"), metrics.get("end")
        if not self._is_real_number(start) or not self._is_real_number(end):
            return False
        elapsed = end - start
        if "min" in value and elapsed < value["min"]:
            return False
        if "max" in value and elapsed > value["max"]:
            return False
        return True

    @staticmethod
    def _is_real_number(value: Any) -> bool:
        """Whether `value` is a number that orders against a duration bound.

        is_numeric already rules out bool; complex is numeric but has no ordering.
        """
        return is_numeric(value) and not isinstance(value, complex)

    def to_btql(self, value: Any) -> list[dict[str, Any]]:
        elapsed = {
            "op": "sub",
            "left": {"op": "ident", "name": ["metrics", "end"]},
            "right": {"op": "ident", "name": ["metrics", "start"]},
        }
        children = []
        if "min" in value:
            children.append({"op": "ge", "left": elapsed, "right": {"op": "literal", "value": value["min"]}})
        if "max" in value:
            children.append({"op": "le", "left": elapsed, "right": {"op": "literal", "value": value["max"]}})
        return children


# Registration order fixes the order of BTQL `and` children; keep it aligned with SpanFilters.
_FILTER_SPECS: dict[str, _FilterSpec] = {
    spec.field: spec
    for spec in (
        _SpanTypeFilter(),
        _NameFilter(),
        _HasErrorFilter(),
        _MetadataFilter(),
        _DurationFilter(),
    )
}
_FILTER_FIELDS = frozenset(_FILTER_SPECS)


def _normalize_span_filters(filters: Any) -> SpanFilters:
    """Validate user-supplied filters into the form the rest of this module assumes.

    Every consumer downstream -- the local matcher, SpanFetcher's BTQL, CachedSpanFetcher's
    reasoning about what its cache can answer -- takes filters as already normalized, so
    this is the one place bad input has to be caught. Unknown or vacuous fields raise
    rather than being dropped, since a silently ignored filter returns too many spans and
    looks like a bug elsewhere.
    """
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise ValueError("filters must be an object")

    unknown_fields = set(filters) - _FILTER_FIELDS
    if unknown_fields:
        raise ValueError(f"Unsupported span filter fields: {', '.join(sorted(unknown_fields))}")

    return cast(
        SpanFilters,
        {field: spec.validate(filters[field]) for field, spec in _FILTER_SPECS.items() if field in filters},
    )


def _matches_span_filters(span: Any, filters: SpanFilters) -> bool:
    """Evaluate normalized `filters` against a span the SDK already holds.

    This is the path taken for spans served out of a cache instead of fetched, so it has to
    agree with the BTQL SpanFetcher would have sent for the same filters.
    """
    if not filters:
        return True
    return all(_FILTER_SPECS[field].matches(span, value) for field, value in filters.items())


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
        """Compile the fetch into a single BTQL filter expression.

        Children are ANDed. Their order follows _FILTER_SPECS registration order and is
        pinned by tests, so a new filter cannot silently reshuffle the query.
        """
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

        filter_values: Mapping[str, Any] = filters or {}
        for field, spec in _FILTER_SPECS.items():
            if field in filter_values:
                children.extend(spec.to_btql(filter_values[field]))

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
                rows = list(fetcher.fetch())
                return [
                    SpanData(
                        input=row.get("input"),
                        output=row.get("output"),
                        expected=row.get("expected"),
                        error=row.get("error"),
                        scores=row.get("scores"),
                        metrics=row.get("metrics"),
                        metadata=row.get("metadata"),
                        span_id=row.get("span_id"),
                        span_parents=row.get("span_parents"),
                        span_attributes=row.get("span_attributes"),
                        id=row.get("id"),
                        _xact_id=row.get("_xact_id"),
                        _pagination_key=row.get("_pagination_key"),
                        root_span_id=row.get("root_span_id"),
                        is_root=row.get("is_root"),
                        created=row.get("created"),
                        tags=row.get("tags"),
                    )
                    for row in rows
                ]

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
        has_advanced_filters = any(not _FILTER_SPECS[field].cacheable for field in filters)

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
            span_type: Deprecated; use filters["span_type"] instead
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
            span_type: Deprecated; use filters["span_type"] instead
            filters: Optional filters for span type, name, error state, metadata, and duration
            include_scorers: Include spans with span_attributes.purpose = "scorer"

        Returns:
            List of matching spans
        """
        normalized_filters = _normalize_span_filters(_merge_deprecated_span_type(span_type, filters))

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
