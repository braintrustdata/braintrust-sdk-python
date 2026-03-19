# Simplification Plan: Claude Agent SDK Instrumentation

Replace `LLMSpanTracker` + `TaskEventSpanTracker` with a single `ContextTracker`
class that consumes the raw SDK message stream and owns all span bookkeeping.
See `SIMPLIFICATION.md` for the full rationale.

**Unchanged:** `WrappedSdkMcpTool`, `_wrap_tool_handler`,
`_activate_tool_span_for_handler`, `_thread_local`, `_dispatch_queues`,
`next_llm_start` stamping, test cassettes.

---

## Target Design

### `_AgentContext`

One instance per subagent context, keyed by `parent_tool_use_id` (`None` =
orchestrator).

```python
@dataclasses.dataclass
class _AgentContext:
    llm_span: Any | None = None          # current open LLM span
    llm_parent_export: str | None = None # parent of current LLM span (merge guard)
    llm_output: list[dict[str, Any]] | None = None  # accumulated output for merge path
    next_llm_start: float | None = None  # timestamp from tool results
    task_span: Any | None = None         # TASK span for this subagent
    task_confirmed: bool = False         # True after TaskStartedMessage
```

Two fields dropped vs the old trackers:
- `llm_span_export` → derived: `ctx.llm_span.export() if ctx.llm_span else None`
- `task_id` → written to metadata at creation, never read back

`llm_parent_export` was retained (originally planned for removal) because it
guards against incorrect merges when a subagent `AssistantMessage` with
`parent_tool_use_id=None` follows an orchestrator `AssistantMessage` — the
resolved parent changes but `next_llm_start` is still `None`.

### `ContextTracker` — public API

```python
class ContextTracker:
    def __init__(self, root_span, prompt, query_start_time=None):
        self._root_span = root_span
        self._root_span_export = root_span.export()
        self._prompt = prompt
        self._tool_tracker = ToolSpanTracker()        # private, also set on _thread_local
        self._contexts: dict[str | None, _AgentContext] = {
            None: _AgentContext(next_llm_start=query_start_time)
        }
        self._active_key: str | None = None           # most recent parent_tool_use_id
        self._task_order: list[str | None] = []       # insertion-order for parent fallback
        self._final_results: list[dict[str, Any]] = []
        self._task_events: list[dict[str, Any]] = []
        _thread_local.tool_span_tracker = self._tool_tracker

    def add(self, message) -> None:
        """Dispatch one SDK message to the appropriate handler."""
        message_type = type(message).__name__
        if message_type == MessageClassName.ASSISTANT:
            self._handle_assistant(message)
        elif message_type == MessageClassName.USER:
            self._handle_user(message)
        elif message_type == MessageClassName.RESULT:
            self._handle_result(message)
        elif message_type in SYSTEM_MESSAGE_TYPES:
            self._handle_system(message)

    def log_output(self) -> None:
        if self._final_results:
            self._root_span.log(output=self._final_results[-1])

    def log_tasks(self) -> None:
        if self._task_events:
            self._root_span.log(metadata={"task_events": self._task_events})

    def cleanup(self) -> None:
        for ctx in self._contexts.values():
            if ctx.llm_span:
                ctx.llm_span.end()
                ctx.llm_span = None
            if ctx.task_span:
                ctx.task_span.end()
                ctx.task_span = None
        self._task_order.clear()
        self._tool_tracker.cleanup_all()
        if hasattr(_thread_local, "tool_span_tracker"):
            delattr(_thread_local, "tool_span_tracker")
```

### `ContextTracker` — internal handlers

#### `_handle_assistant`

Called on each `AssistantMessage`. This is the most complex handler because it
orchestrates tool cleanup, LLM span creation/merge, tool span creation, and
agent context pre-registration — all scoped to the correct subagent context.

Corresponds to the `AssistantMessage` branch of the current `receive_response`
loop, which coordinates across all three old trackers.

```python
def _handle_assistant(self, message: Any) -> None:
    incoming_parent = getattr(message, "parent_tool_use_id", None)
    self._active_key = incoming_parent
    ctx = self._get_context(incoming_parent)

    # 1. Close dangling tool spans from the previous turn in this context.
    #    Skip Agent tool spans that are still live (pending or task running).
    #    Replaces: tool_tracker.cleanup(end_time=..., exclude_tool_use_ids=...,
    #              only_parent_tool_use_id=...)
    if ctx.llm_span and self._tool_tracker.has_active_spans:
        self._tool_tracker.cleanup_context(
            incoming_parent,
            end_time=ctx.next_llm_start or time.time(),
            exclude_ids=self._live_agent_tool_use_ids(),
        )

    # 2. Resolve LLM span parent, then create or merge.
    #    Replaces: task_event_span_tracker.parent_export_for_message(...)
    #              + llm_tracker.start_llm_span(...)
    parent_export = self._llm_parent_for_message(message)
    final_content, extended = self._start_or_merge_llm_span(message, parent_export, ctx)

    # 3. Open TOOL spans for tool calls in this message (parent = LLM span).
    #    Replaces: tool_tracker.start_tool_spans(message, llm_tracker.current_span_export)
    llm_export = ctx.llm_span.export() if ctx.llm_span else None
    self._tool_tracker.start_tool_spans(message, llm_export)

    # 4. Pre-create contexts for Agent tool calls so cleanup_context will
    #    skip them before their TaskStartedMessage arrives.
    #    Replaces: tool_tracker._pending_task_link_tool_use_ids.add(...)
    self._register_pending_agent_contexts(message)

    # 5. Accumulate conversation history.
    if final_content:
        if (extended
                and self._final_results
                and self._final_results[-1].get("role") == "assistant"):
            self._final_results[-1] = final_content
        else:
            self._final_results.append(final_content)
```

#### `_handle_user`

Called on each `UserMessage`. Finishes tool spans that have results, serializes
content for conversation history, and stamps `next_llm_start` on the correct
context.

The context resolution here replaces the `_UNSET_PARENT` sentinel: if the
`UserMessage` has no `parent_tool_use_id`, we use `_active_key` (the most
recently seen `AssistantMessage`'s context) instead of falling back inside the
tracker.

```python
def _handle_user(self, message: Any) -> None:
    self._tool_tracker.finish_tool_spans(message)
    has_tool_results = False
    if hasattr(message, "content"):
        has_tool_results = any(
            type(b).__name__ == BlockClassName.TOOL_RESULT for b in message.content
        )
        content = _serialize_content_blocks(message.content)
        self._final_results.append({"content": content, "role": "user"})
    if has_tool_results:
        user_parent = getattr(message, "parent_tool_use_id", None)
        resolved_key = user_parent if user_parent is not None else self._active_key
        self._get_context(resolved_key).next_llm_start = time.time()
```

#### `_handle_result`

Called on `ResultMessage` (end of stream). Logs usage metrics to the
orchestrator's LLM span and session metadata to the root span.

```python
def _handle_result(self, message: Any) -> None:
    self._active_key = None
    if hasattr(message, "usage"):
        usage_metrics = _extract_usage_from_result_message(message)
        ctx = self._get_context(None)
        if ctx.llm_span and usage_metrics:
            ctx.llm_span.log(metrics=usage_metrics)
    result_metadata = {
        k: v for k, v in {
            "num_turns": getattr(message, "num_turns", None),
            "session_id": getattr(message, "session_id", None),
        }.items() if v is not None
    }
    if result_metadata:
        self._root_span.log(metadata=result_metadata)
```

#### `_handle_system`

Called on `SystemMessage` subtypes (TaskStarted, TaskProgress,
TaskNotification). Resolves the Agent tool span export from `ToolSpanTracker`,
then delegates to `_process_task_event`.

This keeps `ContextTracker` and `ToolSpanTracker` loosely coupled:
`ContextTracker` asks for the export string; `ToolSpanTracker` doesn't need a
back-reference.

```python
def _handle_system(self, message: Any) -> None:
    agent_span_export = self._tool_tracker.get_span_export(
        getattr(message, "tool_use_id", None)
    )
    self._process_task_event(message, agent_span_export)
    self._task_events.append(_serialize_system_message(message))
```

### `ContextTracker` — internal helpers

#### `_get_context`

Lazy-create `_AgentContext` instances on demand.

```python
def _get_context(self, key: str | None) -> _AgentContext:
    ctx = self._contexts.get(key)
    if ctx is None:
        ctx = _AgentContext()
        self._contexts[key] = ctx
    return ctx
```

#### `_register_pending_agent_contexts`

Pre-create an `_AgentContext` (with `task_confirmed=False`) for each Agent tool
call in an `AssistantMessage`. This ensures `_live_agent_tool_use_ids` will
include them, preventing `cleanup_context` from closing the Agent tool span
before its `TaskStartedMessage` arrives.

Replaces `ToolSpanTracker._pending_task_link_tool_use_ids.add()`.

```python
def _register_pending_agent_contexts(self, message: Any) -> None:
    if not hasattr(message, "content"):
        return
    for block in message.content:
        if (type(block).__name__ == BlockClassName.TOOL_USE
                and getattr(block, "name", None) == "Agent"):
            tool_use_id = getattr(block, "id", None)
            if tool_use_id:
                self._get_context(str(tool_use_id))
```

#### `_live_agent_tool_use_ids`

Returns tool_use_ids of Agent spans that must not be closed yet. Includes both
unconfirmed contexts (pending) and confirmed contexts whose task span is still
open.

Replaces the union of `task_event_span_tracker.active_tool_use_ids |
tool_tracker.pending_task_link_tool_use_ids` in the old `receive_response`.

```python
def _live_agent_tool_use_ids(self) -> frozenset[str]:
    result: set[str] = set()
    for key, ctx in self._contexts.items():
        if key is None:
            continue
        if not ctx.task_confirmed or ctx.task_span is not None:
            result.add(key)
    return frozenset(result)
```

#### `_llm_parent_for_message`

Determines the parent span export for an incoming `AssistantMessage`.

Replaces `TaskEventSpanTracker.parent_export_for_message()`. The logic is the
same but reads directly from `_contexts` instead of a separate
`_task_span_by_tool_use_id` dict.

```python
def _llm_parent_for_message(self, message: Any) -> str:
    parent_tool_use_id = getattr(message, "parent_tool_use_id", None)

    # 1. Subagent message → use that subagent's task span.
    if parent_tool_use_id is not None:
        ctx = self._contexts.get(str(parent_tool_use_id))
        if ctx is not None and ctx.task_span is not None:
            return ctx.task_span.export()

    # 2. Orchestrator launching Agent tools → root span (not a task span).
    if _message_starts_subagent_tool(message):
        return self._root_span_export

    # 3. Fallback: most recently opened task span (orchestrator messages
    #    that arrive while a subagent task is running).
    for key in reversed(self._task_order):
        ctx = self._contexts.get(key)
        if ctx is not None and ctx.task_span is not None:
            return ctx.task_span.export()

    # 4. Root span.
    return self._root_span_export
```

#### `_start_or_merge_llm_span`

Starts a new LLM span or extends the existing one via merge.

**Merge path:** consecutive `AssistantMessage`s in the same context with no tool
results between them (`ctx.next_llm_start is None`). This happens in the
orchestrator context when the model emits a thinking block then a tool-call
block as two separate messages. Returns `(merged_content, True)`.

**New span path:** ends the previous span at `resolved_start`, opens a fresh
one. Returns `(final_content, False)`.

The `llm_parent_export` guard from `LLMSpanTracker` is dropped — see
SIMPLIFICATION.md §3b for why it's always true in practice.

```python
def _start_or_merge_llm_span(
    self, message: Any, parent_export: str | None, ctx: _AgentContext,
) -> tuple[dict[str, Any] | None, bool]:
    current_message = _serialize_assistant_message(message)

    # Merge path.
    if ctx.llm_span and ctx.next_llm_start is None and current_message is not None:
        merged = _merge_assistant_messages(
            ctx.llm_output[0] if ctx.llm_output else None,
            current_message,
        )
        if merged is not None:
            ctx.llm_output = [merged]
            ctx.llm_span.log(output=ctx.llm_output)
        return merged, True

    # New span path.
    resolved_start = ctx.next_llm_start or time.time()
    first_token_time = time.time()

    if ctx.llm_span:
        ctx.llm_span.end(end_time=resolved_start)

    final_content, span = _create_llm_span_for_messages(
        [message], self._prompt, self._final_results,
        parent=parent_export, start_time=resolved_start,
    )
    if span is not None:
        span.log(metrics={"time_to_first_token": max(0.0, first_token_time - resolved_start)})
    ctx.llm_span = span
    ctx.llm_output = [final_content] if final_content is not None else None
    ctx.next_llm_start = None
    return final_content, False
```

#### `_process_task_event`

Handles TaskStarted / TaskProgress / TaskNotification system messages.

Key difference from `TaskEventSpanTracker.process()`: contexts are keyed by
`tool_use_id` (not `task_id`), because that's the same key used everywhere else
in `ContextTracker`. The old tracker maintained two parallel dicts
(`_active_spans` keyed by `task_id` and `_task_span_by_tool_use_id` keyed by
`tool_use_id`); this merges them.

```python
def _process_task_event(self, message: Any, agent_span_export: str | None) -> None:
    task_id = getattr(message, "task_id", None)
    if task_id is None:
        return
    task_id = str(task_id)
    tool_use_id = getattr(message, "tool_use_id", None)
    tool_use_id_str = str(tool_use_id) if tool_use_id is not None else None
    ctx = self._get_context(tool_use_id_str)
    message_type = type(message).__name__

    if ctx.task_span is None:
        # TaskStartedMessage — open the TASK span.
        ctx.task_span = start_span(
            name=_task_span_name(message, task_id),
            span_attributes={"type": SpanTypeAttribute.TASK},
            metadata=_task_metadata(message),
            parent=agent_span_export or self._root_span_export,
        )
        ctx.task_confirmed = True
        self._task_order.append(tool_use_id_str)
    else:
        # TaskProgressMessage — update existing task span.
        update: dict[str, Any] = {}
        metadata = _task_metadata(message)
        if metadata:
            update["metadata"] = metadata
        output = _task_output(message)
        if output is not None:
            update["output"] = output
        if update:
            ctx.task_span.log(**update)

    if message_type == MessageClassName.TASK_NOTIFICATION:
        ctx.task_span.end()
        ctx.task_span = None
        self._task_order = [k for k in self._task_order if k != tool_use_id_str]
```

### `ToolSpanTracker` — new methods

These are added alongside the existing `cleanup()`, which stays untouched until
Step 5 deletes it.

#### `cleanup_context`

Closes tool spans belonging to one subagent context. Called by
`ContextTracker._handle_assistant` before starting a new LLM span for that
context. Skips any span whose `tool_use_id` is in `exclude_ids` (live Agent
spans).

Replaces the mid-stream `cleanup(end_time=..., exclude_tool_use_ids=...,
only_parent_tool_use_id=...)` call.

```python
def cleanup_context(
    self,
    parent_tool_use_id: str | None,
    *,
    end_time: float | None = None,
    exclude_ids: frozenset[str] = frozenset(),
) -> None:
    for tool_use_id in list(self._active_spans):
        if tool_use_id in exclude_ids:
            continue
        if self._active_spans[tool_use_id].parent_tool_use_id != parent_tool_use_id:
            continue
        self._end_tool_span(tool_use_id, end_time=end_time)
```

#### `cleanup_all`

Closes all remaining active spans. Called at end-of-stream by
`ContextTracker.cleanup()`.

Replaces the no-args `cleanup()` call in `finally:`.

```python
def cleanup_all(self, end_time: float | None = None) -> None:
    for tool_use_id in list(self._active_spans):
        self._end_tool_span(tool_use_id, end_time=end_time)
```

### Module-level helpers (extracted from `TaskEventSpanTracker`)

```python
def _task_span_name(message: Any, task_id: str) -> str:
    return (getattr(message, "description", None)
            or getattr(message, "task_type", None)
            or f"Task {task_id}")

def _task_metadata(message: Any) -> dict[str, Any]:
    return {k: v for k, v in {
        "task_id":       getattr(message, "task_id", None),
        "session_id":    getattr(message, "session_id", None),
        "tool_use_id":   getattr(message, "tool_use_id", None),
        "task_type":     getattr(message, "task_type", None),
        "status":        getattr(message, "status", None),
        "last_tool_name":getattr(message, "last_tool_name", None),
        "usage":         getattr(message, "usage", None),
    }.items() if v is not None}

def _task_output(message: Any) -> dict[str, Any] | None:
    summary = getattr(message, "summary", None)
    output_file = getattr(message, "output_file", None)
    if summary is None and output_file is None:
        return None
    return {k: v for k, v in {"summary": summary, "output_file": output_file}.items()
            if v is not None}
```

### `receive_response` (final form)

```python
async def receive_response(self) -> AsyncGenerator[Any, None]:
    generator = self.__client.receive_response()
    with start_span(
        name=CLAUDE_AGENT_TASK_SPAN_NAME,
        span_attributes={"type": SpanTypeAttribute.TASK},
        input=self.__last_prompt or None,
    ) as span:
        input_needs_update = self.__captured_messages is not None
        tracker = ContextTracker(span, self.__last_prompt, self.__query_start_time)
        try:
            async for message in generator:
                if input_needs_update:
                    captured = self.__captured_messages or []
                    if captured:
                        span.log(input=captured)
                    input_needs_update = False
                tracker.add(message)
                yield message
        except asyncio.CancelledError:
            tracker.log_output()
        else:
            tracker.log_output()
        finally:
            tracker.log_tasks()
            tracker.cleanup()
```

### Span parentage

| Span type | Parent |
|---|---|
| Root TASK (`"Claude Agent"`) | Ambient caller context |
| Subagent TASK | Agent tool span → fallback: root TASK |
| LLM (orchestrator) | Root TASK, or latest active subagent TASK (`_task_order` fallback) |
| LLM (subagent) | That subagent's TASK span |
| TOOL | LLM span of the `AssistantMessage` containing the tool call |
| Nested user span in tool handler | TOOL span (via `set_current()`) |

---

## Implementation Order

Each step ends with a green `nox -s "test_claude_agent_sdk(latest)"` run.

### Step 0 ✅ — Remove `_wrap_tool_factory`

Done. Deleted the redundant `tool()` patch from `_wrapper.py` and `__init__.py`.

### Step 1 ✅ — Extract task-event helpers to module-level functions

Done. Added `_task_span_name()`, `_task_metadata()`, `_task_output()` as
module-level functions. `TaskEventSpanTracker._span_name` / `._metadata` /
`._output` now delegate to them.

### Step 2 ✅ — Add `cleanup_context` / `cleanup_all` to `ToolSpanTracker`

Done. Added both methods. Existing `cleanup()` left untouched.

### Step 3 ✅ — Migrate mid-stream cleanup call in `receive_response`

Done. Mid-stream call now uses `cleanup_context()`. Old `cleanup()` delegates
to `cleanup_all()`. Two unit tests updated to use `cleanup_context()` directly.

### Step 4 ✅ — Add `_AgentContext` and `ContextTracker`

Done. Full `ContextTracker` class implemented (dead code — not wired in yet).

### Step 5 ✅ — Wire `ContextTracker` into `receive_response`; delete old classes

Done. Rewrote `receive_response` to use `ContextTracker`. Deleted
`LLMSpanTracker`, `TaskEventSpanTracker`, `_UNSET_PARENT`. Cleaned up
`ToolSpanTracker` (removed pending-task-link bookkeeping and old `cleanup()`).

**Implementation note:** `llm_parent_export` was retained on `_AgentContext`
(contrary to the original plan's §1b which proposed dropping it). Testing
revealed it's needed when a subagent `AssistantMessage` arrives with
`parent_tool_use_id=None` right after an orchestrator `AssistantMessage` — the
parent export changes (root → task span) but `next_llm_start` is still `None`,
so without the guard the two messages would incorrectly merge.

---

All steps complete. The three-tracker architecture (`LLMSpanTracker` +
`TaskEventSpanTracker` + `ToolSpanTracker`) has been replaced with two
(`ContextTracker` + `ToolSpanTracker`), with `ContextTracker` owning the
`ToolSpanTracker` as a private component.
