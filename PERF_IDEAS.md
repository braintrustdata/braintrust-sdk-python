# Tracing Performance Optimization Ideas

Baseline: **967 us/req** user thread, **39 us/item** flush (5000 reqs e2e)

## User Thread Bottlenecks (profiled with cProfile, 3000 reqs)

Total: 10.88s for 3000 requests = 3627 us/req

### 1. `_to_bt_safe` primitives-last (1.63s / 15%)

`_to_bt_safe` is called for every leaf value in `_deep_copy_object`. It checks
Span/Experiment/Dataset/Logger isinstance, dataclasses, Pydantic model_dump
(with warnings.catch_warnings + filterwarnings!), and Pydantic v1 `.dict()` --
all before checking if the value is a simple int/str/float/bool/None.

**Fix**: Move primitive checks (`type(v) is int/str/float`) to the top of
`_to_bt_safe`. Guard Pydantic attempts with `hasattr(v, "model_dump")`.

**Impact**: Eliminates ~5s of isinstance/warnings/regex overhead. Estimated 3-5x
improvement on user thread.

### 2. `_deep_copy_object` uses `isinstance(v, Mapping)` (0.62s + isinstance overhead)

Every dict goes through `isinstance(v, (Mapping, list, tuple, set))` which is
slow for abstract types from `collections.abc`. Then a second
`isinstance(v, Mapping)` check.

**Fix**: Use `type(v) is dict` / `type(v) is list` for the common fast path.
Also inline the primitive check at the top of `_deep_copy_object` to skip
calling `_to_bt_safe` entirely for leaf values.

**Impact**: Combined with #1, reduces `_deep_copy_object` from ~9.2s to ~0.3s.

### 3. `warnings.catch_warnings` + `filterwarnings` in `_to_bt_safe` (0.51s + 0.49s)

Every call to `_to_bt_safe` on a non-primitive does
`warnings.catch_warnings()` + `warnings.filterwarnings(...)` which involves
regex compilation (`re.compile`), list manipulation, and lock acquisition.
Called 195k times for 3000 requests.

**Fix**: Already fixed by #1 (primitives skip this entirely). Additionally,
guard with `hasattr(v, "model_dump")` so only actual Pydantic models pay
the cost.

### 4. `get_caller_location()` always called (visible in __init__)

`get_caller_location()` walks the stack with `inspect.currentframe()` on every
span creation, even when the caller provides an explicit `name=`.

**Fix**: Only call `get_caller_location()` when `name is None`.

**Impact**: Small but free (~5us per span).

### 5. `bt_safe_deep_copy` called on internal-only data (end/set_attributes)

`end()` calls `log_internal(internal_data={metrics: {end: time}})` which goes
through the full `bt_safe_deep_copy`. This data is all primitives -- no user
object references to break.

**Fix**: Skip `bt_safe_deep_copy` when `event` is None/empty (internal-only).

**Impact**: Saves ~15-20us per `end()` call.

### 6. `_strip_nones` recurses unnecessarily (0.11s)

Called with `deep=True` on internal_data, recurses into every nested dict even
when there are no Nones. Also always creates a new dict even when no Nones.

**Fix**: Fast-path: check if any values are None before copying. Use
`type(d) is dict` instead of `isinstance`. Skip recursion when no nested dicts.

### 7. `split_logging_data` does redundant work for empty event/internal_data

When `event=None` (from `end()`), it still calls `_validate_and_sanitize({})`,
`_strip_nones({})`, and `merge_dicts({}, ...)`.

**Fix**: Short-circuit when one side is empty. Add early return to
`_validate_and_sanitize` for empty events.

### 8. `_EXEC_COUNTER` uses threading.Lock (small)

Global counter protected by a lock. Under CPython GIL, `itertools.count()` with
`next()` is atomic and lock-free.

**Fix**: Replace `threading.Lock` + global int with `itertools.count(1)`.

### 9. `merge_dicts` path tracking overhead (small)

`merge_dicts` delegates to `merge_dicts_with_paths(... (), set())` which creates
tuples for every key path. The simple `merge_dicts` call never uses merge_paths.

**Fix**: Inline the simple merge logic in `merge_dicts` without path tracking.

## Flush Thread Bottlenecks (profiled with cProfile, 3000 reqs)

Total: 0.756s for 6000 items = 126 us/item (includes merge of 18000 -> 6000)

### 10. `_get_exporter` calls `os.getenv` every time (0.048s)

Called 18000 times in flush. Does `os.getenv("BRAINTRUST_OTEL_COMPAT")` +
`.lower()` comparison each time.

**Fix**: Cache the result in a module-level variable. Add `_reset_cached_exporter()`
for tests. Also reuse in `export()` which has a duplicate env var check.

### 11. `compute_record` creates SpanComponentsV3 per item (in _get_exporter cost)

Each queued item's `compute_record()` closure calls `_get_exporter()(object_type=...,
object_id=...).object_id_fields()`, creating a new dataclass with `__post_init__`
assertions and then a small dict. This is constant per span.

**Fix**: Cache `object_id_fields` result per span in a `LazyValue`, reuse across
all `compute_record` closures from the same span.

### 12. `merge_row_batch` with merge_dicts_with_paths (0.134s)

The merge step uses the full `merge_dicts_with_paths` with tuple path tracking.
Also `_pop_merge_row_skip_fields` / `_restore_merge_row_skip_fields` do field-by-field
dict manipulation.

**Fix**: Already partially addressed by #9 (merge_dicts fast path). Further
optimization possible but lower priority since flush is already fast.

## Implementation Priority

High impact (implement first):
1. `_to_bt_safe` primitives-first + hasattr guards (#1, #3)
2. `_deep_copy_object` type-identity fast paths (#2)
3. Skip deep copy for internal-only data (#5)
4. Lazy `get_caller_location` (#4)

Medium impact:
5. `_strip_nones` / `split_logging_data` / `_validate_and_sanitize` fast paths (#6, #7)
6. `merge_dicts` inline fast path (#9)
7. Cache `_get_exporter` (#10)
8. Cache `object_id_fields` per span (#11)

Low impact:
9. `itertools.count` for exec counter (#8)

## Expected Combined Result

Based on isolated testing of each change:
- User thread: ~967 us/req -> ~200 us/req (4-5x improvement)
- Flush: ~39 us/item -> ~25 us/item (1.5x improvement, more with orjson)
