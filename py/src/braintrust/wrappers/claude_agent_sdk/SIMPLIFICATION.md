# Simplification Analysis: Claude Agent SDK Instrumentation

This document analyses the current three-tracker architecture and proposes concrete
simplifications that reduce the number of trackers, eliminate redundant state, and
make context routing explicit.

---

## 0. The Wrapper Layer

The monkeypatch installs three wrappers, but they serve two completely different jobs:

| Wrapper | Job |
|---------|-----|
| `WrappedClaudeSDKClient` | Stream processing — observes every SDK message, creates TASK/LLM/TOOL spans, drives all three trackers |
| `WrappedSdkMcpTool` / `wrapped_tool_fn` | Handler activation — wraps tool handlers at registration time so they re-enter the pre-created TOOL span when the SDK calls them |

The handler wrappers (`SdkMcpTool` and `tool`) are a bridge between two execution
contexts: span *creation* happens on the stream side (controlled by Braintrust) and
span *activation* happens on the handler side (called by the Claude SDK). See
`INSTRUMENTATION.md § 1b` for the full two-phase handoff diagram.

### 0a. `wrapped_tool_fn` is redundant and can be removed

`claude_agent_sdk.tool()` is not an independent code path. Its entire body is:

```python
def decorator(handler) -> SdkMcpTool[Any]:
    return SdkMcpTool(name=name, description=description, input_schema=input_schema, handler=handler, ...)
return decorator
```

The `SdkMcpTool` name inside that function is resolved through `tool.__globals__`,
which is `claude_agent_sdk.__dict__`. Patching `claude_agent_sdk.SdkMcpTool =
WrappedSdkMcpTool` is therefore sufficient — every `tool()` call already routes
through `WrappedSdkMcpTool.__init__`, which wraps the handler via
`_wrap_tool_handler`. No separate `tool` patch is needed.

This holds even for the `from claude_agent_sdk import tool` pre-import case that the
`sys.modules` sweep was designed to handle: because `tool.__globals__ is
claude_agent_sdk.__dict__`, the function always looks up `SdkMcpTool` from the
module it was *defined* in, not from the importing module.

The one real obstacle is that `tool()`'s inner `decorator` function has a
`-> SdkMcpTool[Any]` return annotation that Python evaluates eagerly. This calls
`__class_getitem__` on whatever `SdkMcpTool` currently is, which would raise
`TypeError` on a plain subclass. The `__class_getitem__` override already present on
`WrappedSdkMcpTool` handles this:

```python
__class_getitem__ = classmethod(lambda cls, params: cls)
```

**What can be removed:**

| Location | What to remove |
|----------|----------------|
| `_wrapper.py` | `_wrap_tool_factory` function entirely |
| `__init__.py` | `_wrap_tool_factory` import |
| `__init__.py` | `original_tool_fn` / `wrapped_tool_fn` block and its `sys.modules` sweep |

`WrappedSdkMcpTool` and its `__class_getitem__` override stay exactly as-is.

The rest of this document focuses on the three tracker objects that live inside
`receive_response()`.

---

## 1. Current Architecture: Three Trackers, Many Interactions

The current implementation uses three distinct tracker objects that collaborate via
method calls and shared references:

```
receive_response()
  │
  ├── LLMSpanTracker        — per-subagent-context LLM span lifecycle
  ├── ToolSpanTracker       — live tool spans, dispatch queues, pending-task IDs
  └── TaskEventSpanTracker  — TASK spans for subagents, needs a ref to ToolSpanTracker
```

They interact with each other in non-obvious ways:

| Caller | Callee | Why |
|--------|--------|-----|
| `TaskEventSpanTracker.__init__` | receives `ToolSpanTracker` | needs `get_span_export()` to set task span parent |
| `TaskEventSpanTracker.process` | `tool_tracker.mark_task_started()` | removes tool_use_id from `_pending_task_link_tool_use_ids` |
| `receive_response` | `task_event_span_tracker.active_tool_use_ids` + `tool_tracker.pending_task_link_tool_use_ids` | builds combined exclusion set for cleanup |
| `receive_response` | `task_event_span_tracker.parent_export_for_message()` | gets LLM span parent before calling `llm_tracker.start_llm_span()` |
| `receive_response` | `llm_tracker.current_span_export` → passed to `tool_tracker.start_tool_spans()` | chains LLM export to tool parent |

Five cross-tracker interactions in a hot loop. Every time a new subagent feature needs
a change, the developer has to reason about all three trackers simultaneously.

---

## 2. Redundant and Duplicated State

### 2a. Two half-pictures of the same "Agent tool call" lifecycle

`ToolSpanTracker._pending_task_link_tool_use_ids` and
`TaskEventSpanTracker._task_span_by_tool_use_id` together track the full lifecycle
of an `Agent` tool call:

```
State              Stored in                     Description
──────             ─────────                     ───────────
Pending            ToolSpanTracker               Agent span created, TaskStarted not yet seen
Linked             TaskEventSpanTracker           TaskStarted arrived, task_span_by_tool_use_id set
Ended              (both remove the entry)        TaskNotification arrived
```

These two dictionaries key on `agent_tool_use_id` and always move in lockstep:
`pending → linked` happens atomically in `process()` via `mark_task_started()`.
The consumer in `receive_response` always reads *both*:

```python
active_subagent_tool_use_ids = (
    task_event_span_tracker.active_tool_use_ids          # linked
    | tool_tracker.pending_task_link_tool_use_ids         # pending
)
```

This set union reconstructs information that was always a single set of "live agent
tool calls". Splitting it between two trackers is unnecessary.

### 2b. `LLMSpanTracker` and `TaskEventSpanTracker` share the same routing key

Both trackers key their primary state on `parent_tool_use_id` (the agent tool call
that spawned a subagent). The connection is direct:

- `LLMSpanTracker._states[parent_tool_use_id]` → a subagent's LLM span state
- `TaskEventSpanTracker._task_span_by_tool_use_id[parent_tool_use_id]` → a subagent's TASK span

A subagent has exactly one TASK span and a sequence of LLM spans, all keyed by the
same `parent_tool_use_id`. Keeping them in two different tracker objects means every
subagent-related operation must touch two places.

### 2c. `_active_context` is an implicit, mutable cursor

`LLMSpanTracker._active_context` is set via `set_context()` before any method that
should route to a specific subagent. The sentinel `_UNSET_PARENT = object()` then
distinguishes "use active context" from "use orchestrator (None)".

This makes it easy to introduce bugs where `set_context()` is forgotten or called
out of order. The `mark_next_llm_start` method has an entire special-case block to
compensate for `UserMessage`s that arrive with `parent_tool_use_id=None` while the
active context is set to a subagent:

```python
def mark_next_llm_start(self, parent_tool_use_id=_UNSET_PARENT):
    if parent_tool_use_id is None and self._active_context is not None:
        parent_tool_use_id = _UNSET_PARENT   # fall back to active context
    self._get_state(parent_tool_use_id).next_start_time = time.time()
```

This implicit fallback would be unnecessary if context routing were always explicit.

### 2d. `cleanup()` has three orthogonal filter modes in one method

```python
def cleanup(
    self,
    end_time: float | None = None,
    exclude_tool_use_ids: frozenset[str] | None = None,
    only_parent_tool_use_id: Any = _UNSET_PARENT,   # sentinel again
) -> None:
```

Three call sites, each using a different combination of parameters. This is a sign
the method is doing three different jobs:

1. **End-of-stream**: called with no filters — close everything.
2. **Pre-LLM cleanup within a context**: called with `only_parent_tool_use_id` + `exclude_tool_use_ids` — close dangling tool spans scoped to one subagent, but skip live Agent spans.
3. **Dangling-span cleanup**: called from tests with just `end_time` or no args.

A simpler API would expose these three intents as distinct methods or with clearer
parameter names that do not require a sentinel object.

---

## 3. What Is Genuinely Irreducible

Not all complexity can be removed. The following pieces are load-bearing:

### 3a. Per-subagent-context state

Concurrent subagents interleave on a single message stream. Each subagent needs its
own LLM span sequence and TASK span. Keying state on `parent_tool_use_id` (or `None`
for the orchestrator) is the correct abstraction.

### 3b. Dispatch queues in `ToolSpanTracker`

When two subagents call the same tool with identical arguments, the handler receives
only `(tool_name, args)` — not a `tool_use_id`. The FIFO dispatch queue maps the
handler invocation order to the span creation order, which matches the Claude SDK's
own execution order. This is necessary and correct.

### 3c. Thread-local for handler-to-span bridging

Tool handlers are called by the Claude SDK without any Braintrust context. A
thread-local is the only way to bridge the active stream session to the handler.
This cannot be removed without changing the SDK's calling convention.

### 3d. `next_start_time` for non-overlapping sequential spans

Stamping the time when a `UserMessage` with tool results arrives, then using that
stamp as both the end time of the previous LLM span and the start time of the next
one, is necessary to produce accurate, non-overlapping span timelines. This logic
must live somewhere.

---

## 4. Proposed Simplifications

### 4a. Merge `LLMSpanTracker` and `TaskEventSpanTracker` into `ContextTracker`

Since both trackers key on `parent_tool_use_id`, merge them into a single object
with one state record per subagent context:

```python
@dataclasses.dataclass
class _AgentContext:
    # LLM state (from LLMSpanTracker._SubagentState)
    llm_span: Any | None = None
    llm_span_export: str | None = None
    llm_parent_export: str | None = None
    llm_output: list | None = None
    next_llm_start: float | None = None
    # Task state (from TaskEventSpanTracker._task_span_by_tool_use_id)
    task_span: Any | None = None
    task_id: str | None = None

class ContextTracker:
    def __init__(self, root_span_export: str, query_start_time: float | None = None):
        self._root_span_export = root_span_export
        # parent_tool_use_id (or None for orchestrator) → _AgentContext
        self._contexts: dict[str | None, _AgentContext] = {
            None: _AgentContext(next_llm_start=query_start_time)
        }
        self._active_key: str | None = None  # still needed as a cursor, see 4b
        self._task_order: list[str] = []     # for fallback parent resolution

    def set_active(self, parent_tool_use_id: str | None) -> None: ...
    def start_llm_span(self, message, prompt, history, parent_export) -> ...: ...
    def mark_next_llm_start(self, parent_tool_use_id: str | None) -> None: ...
    def process_task_event(self, message) -> None: ...  # replaces TaskEventSpanTracker.process
    def llm_parent_export_for_message(self, message) -> str: ...
    def log_usage(self, metrics) -> None: ...
    def cleanup(self) -> None: ...
```

**What this removes:**
- `TaskEventSpanTracker` as a separate class (≈ 100 lines of code).
- The `ToolSpanTracker` constructor argument `tool_tracker` from `TaskEventSpanTracker`.
- The `_task_span_by_tool_use_id` dict — it becomes `_contexts[tool_use_id].task_span`.
- The `_active_task_order` list can stay on `ContextTracker` as `_task_order` for
  the same fallback-parent purpose.

**The two remaining `ToolSpanTracker` cross-calls** become:
- `mark_task_started(tool_use_id)` → `ContextTracker.process_task_event` already knows
  this; `ToolSpanTracker` can expose a simple `unlink_agent_span(tool_use_id)` or the
  pending-ID set can move into `ContextTracker` entirely (see 4b).
- `get_span_export(tool_use_id)` → `ContextTracker._contexts[tool_use_id].task_span.export()`

### 4b. Move the "pending Agent spans" set into `ContextTracker`

`ToolSpanTracker._pending_task_link_tool_use_ids` exists solely to tell `cleanup()`
"don't close this Agent tool span, its TaskStarted hasn't arrived yet". The decision
of whether an Agent span is pending or linked is owned by the task event lifecycle,
which will live in `ContextTracker` after 4a. So the set belongs there.

`ContextTracker` would track whether a context has been confirmed by `TaskStarted`
as a boolean flag on `_AgentContext`:

```python
@dataclasses.dataclass
class _AgentContext:
    ...
    task_confirmed: bool = False  # True after TaskStarted received
```

`ToolSpanTracker.cleanup()` would receive the full set of "live agent tool_use_ids"
(both confirmed and unconfirmed) from `ContextTracker.live_agent_tool_use_ids` —
a single property, not two properties unioned by the caller.

### 4c. Make context routing explicit, remove the `_UNSET_PARENT` sentinel

The `_UNSET_PARENT = object()` sentinel is a code smell — it is a non-serializable
runtime object used as a dict key guard. The need for it arises because
`mark_next_llm_start` has an implicit fallback: "if you passed `None` but there's
an active subagent, use the active subagent instead."

Replace the implicit fallback with explicit routing at the call site in
`receive_response`, where the `UserMessage`'s `parent_tool_use_id` is already being
read:

```python
# Before (implicit fallback inside LLMSpanTracker):
llm_tracker.mark_next_llm_start(user_parent)

# After (caller resolves the context before calling):
resolved_context = user_parent if user_parent is not None else self._active_context
context_tracker.mark_next_llm_start(resolved_context)
```

With this change, `_UNSET_PARENT` can be deleted along with the fallback branch
inside `mark_next_llm_start`. The tracker method signature becomes simply
`mark_next_llm_start(context_key: str | None)`.

### 4d. Simplify `ToolSpanTracker.cleanup()` into two focused methods

Replace the three-mode method with two explicit ones:

```python
def cleanup_context(self, parent_tool_use_id: str | None, *, end_time: float | None = None, exclude_ids: frozenset[str] = frozenset()) -> None:
    """Close all active tool spans belonging to a specific subagent context,
    optionally skipping Agent spans that are still live."""

def cleanup_all(self, end_time: float | None = None) -> None:
    """Close all remaining active spans. Called at end-of-stream."""
```

The three call sites in `receive_response` and tests map cleanly:
- Pre-LLM cleanup → `cleanup_context(incoming_parent, end_time=..., exclude_ids=live_agent_ids)`
- End-of-stream → `cleanup_all()`
- Test helpers → `cleanup_all()` or `cleanup_context(...)`

No sentinel needed; the filter intent is expressed in the method name.

---

## 5. Summary of Changes

| Change | Effect |
|--------|--------|
| Merge `LLMSpanTracker` + `TaskEventSpanTracker` → `ContextTracker` | −1 tracker class, eliminates constructor coupling, unifies per-subagent state |
| Move `_pending_task_link_tool_use_ids` into `ContextTracker` | Eliminates two-property union at call site, single source of truth for Agent span liveness |
| Remove `_UNSET_PARENT` sentinel | Eliminates implicit fallback, makes `receive_response` loop more readable |
| Split `cleanup()` into `cleanup_context()` + `cleanup_all()` | Clarifies intent at each call site, removes three-mode parameter combination |

**Trackers before:** 3 (`ToolSpanTracker`, `LLMSpanTracker`, `TaskEventSpanTracker`)
**Trackers after:** 2 (`ToolSpanTracker`, `ContextTracker`)

**Cross-tracker interactions before:** 5 (see §1 table)
**Cross-tracker interactions after:** 2 (ContextTracker gives ToolSpanTracker the live-agent-id set for cleanup; ToolSpanTracker gives ContextTracker a task span parent export via `get_span_export`)

---

## 6. What Does Not Change

- **`WrappedSdkMcpTool`** — the handler-side wrapper is a separate concern (span
  activation, not span creation) and is entirely unaffected. See
  `INSTRUMENTATION.md § 1b`. `wrapped_tool_fn` is removed as part of § 0a above.
- The `_dispatch_queues` FIFO mechanism in `ToolSpanTracker` — still required.
- The thread-local for handler bridging — still required. The handler wrappers read
  it to find the active `ToolSpanTracker`; after this refactor they would read it to
  find the active `ToolSpanTracker` inside `ContextTracker` (or a direct reference
  to the same object — the public API is unchanged).
- The `next_llm_start` stamping logic — still required, just moves into `_AgentContext`.
- The `_active_context` / `set_active()` cursor on `ContextTracker` — still needed
  because `AssistantMessage` arrives with a `parent_tool_use_id` that sets routing
  for the rest of that message's processing. The cursor avoids threading it through
  every call signature inside the message loop.
- The test surface — all existing unit and integration tests remain valid; only
  the internal class and method names change.
