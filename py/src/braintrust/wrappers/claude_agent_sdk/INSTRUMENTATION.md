# Claude Agent SDK Instrumentation — Deep Dive

## Overview

This document explains how the Braintrust wrapper instruments the Claude Agent SDK: how the monkeypatch works, what data structures are used, and how they collaborate to produce a correct span tree even when multiple subagents run concurrently on a single interleaved message stream.

---

## 1. The Monkeypatch

`setup_claude_agent_sdk()` (in `__init__.py`) patches three things in the `claude_agent_sdk` module **and** in every already-imported module in `sys.modules`:

```
claude_agent_sdk.ClaudeSDKClient   →  WrappedClaudeSDKClient   (via _create_client_wrapper_class)
claude_agent_sdk.SdkMcpTool        →  WrappedSdkMcpTool        (via _create_tool_wrapper_class)
claude_agent_sdk.tool              →  wrapped_tool_fn           (via _wrap_tool_factory)
```

All three wrappers are **generated at call time** via factory functions — they dynamically create new classes/functions that subclass or close over the originals. The `sys.modules` sweep handles the case where user code has already done `from claude_agent_sdk import ClaudeSDKClient` before calling `setup_claude_agent_sdk`.

```
User Code                       Braintrust Wrapper              Original SDK
─────────                       ──────────────────              ────────────
ClaudeSDKClient(...)       →    WrappedClaudeSDKClient(...)  →  original.__init__(...)
  client.query(...)         →    captures prompt + start_time →  original.query(...)
  client.receive_response() →    starts TASK span             →  original.receive_response()
                                 processes every message
                                 creates LLM/TOOL spans
                                 yields message to user
```

`WrappedClaudeSDKClient` extends `Wrapper` (a base that proxies attribute access to the inner client), so any attributes the user accesses that aren't explicitly overridden fall through transparently to the original.

### 1b. Why `SdkMcpTool` and `tool` are wrapped separately from `ClaudeSDKClient`

`ClaudeSDKClient` is responsible for the **stream side**: it observes every message,
creates TOOL spans for each `ToolUseBlock`, and stores them in `ToolSpanTracker`.
At that point the spans exist but are **not yet the active context** — the tool
handler hasn't run yet.

`SdkMcpTool` and `tool` are responsible for the **handler side**: they intercept
tool handler registration at decoration/instantiation time and wrap every handler
via `_wrap_tool_handler`. When the Claude SDK later calls the handler (through its
own internal machinery, not Braintrust code), the wrapper fires first:

```python
async def wrapped_handler(args):
    active_tool_span = _activate_tool_span_for_handler(tool_name, args)

    if not active_tool_span.has_span:
        # No stream active — create a standalone TOOL span as a fallback
        with start_span(name=str(tool_name), type=TOOL, input=args) as span:
            result = await handler(args)
            span.log(output=result)
            return result

    try:
        return await handler(args)     # ← user code runs here, under the span
    except Exception as exc:
        active_tool_span.log_error(exc)
        raise
    finally:
        active_tool_span.release()     # span.unset_current()
```

`_activate_tool_span_for_handler` reads the thread-local `ToolSpanTracker`, finds
the pre-created span by `(tool_name, args)`, and calls `span.set_current()` —
making that span the active context for the duration of the call. Any span the user
creates *inside* their handler therefore nests under the correct TOOL span
automatically.

**The two-phase handoff in full:**

```
receive_response() — Braintrust controls          Claude SDK internals — Braintrust does NOT control
──────────────────────────────────────            ──────────────────────────────────────────────────
AssistantMessage arrives
  → start_tool_spans()
    → create TOOL span            ─── stored in ToolSpanTracker via thread-local ──→
    → store in _active_spans

                                                  SDK calls tool.handler(args)
                                                    → _wrap_tool_handler fires
                                                    → reads thread-local tracker
                                                    → acquires + activates TOOL span
                                                    → user handler runs nested under it
                                                    → span released (unset_current)

UserMessage arrives (ToolResultBlock)
  → finish_tool_spans()
    → log output + end span
```

**Without the `SdkMcpTool`/`tool` wrappers**, step 2 never happens. The pre-created
spans sit in the tracker with their context never activated, and any spans created
inside user handler code have no TOOL span parent — they would float up to the TASK
span or be rootless.

**The fallback path** (no stream active) covers two practical cases:
- A tool handler called directly in a unit test.
- A tool handler invoked before or after a `receive_response()` session.

In both cases `_activate_tool_span_for_handler` finds no `ToolSpanTracker` on the
thread-local and returns `_NOOP_ACTIVE_TOOL_SPAN`, triggering the `with start_span`
fallback branch which creates and closes a standalone TOOL span for that single
invocation.

---

## 2. The SDK Message Stream

The Claude Agent SDK streams messages from a subprocess over a JSON protocol. Every message is surfaced on a single `async for message in client.receive_response()` iterator. When subagents run concurrently, their messages **interleave** on this one stream:

```
─────── Single stream (time flows down) ────────────────────────────────────────────────
 AssistantMessage  (orchestrator: calls Agent A and Agent B)
 SystemMessage     (TaskStarted for task A)
 SystemMessage     (TaskStarted for task B)
 AssistantMessage  (subagent A's LLM turn: calls Bash)    ← parent_tool_use_id = "call-A"
 AssistantMessage  (subagent B's LLM turn: calls Read)    ← parent_tool_use_id = "call-B"
 UserMessage       (Bash result for A)                    ← parent_tool_use_id = "call-A"
 UserMessage       (Read result for B)                    ← parent_tool_use_id = "call-B"
 SystemMessage     (TaskNotification for task A — done)
 SystemMessage     (TaskNotification for task B — done)
 ResultMessage     (final usage)
────────────────────────────────────────────────────────────────────────────────────────
```

The key field is `parent_tool_use_id`: every message from a subagent carries the `tool_use_id` of the `Agent` tool call that spawned it. Orchestrator messages have `parent_tool_use_id = None`.

---

## 3. The Span Hierarchy Being Built

```
Claude Agent  [TASK]
├── anthropic.messages.create  [LLM]   ← orchestrator's turn
│   ├── Agent  [TOOL]                  ← "Agent" tool call → spawns subagent A
│   └── Agent  [TOOL]                  ← "Agent" tool call → spawns subagent B
├── Task A  [TASK]
│   ├── anthropic.messages.create  [LLM]   ← subagent A turn 1
│   │   └── Bash  [TOOL]
│   └── anthropic.messages.create  [LLM]   ← subagent A turn 2
│       └── Read  [TOOL]
└── Task B  [TASK]
    ├── anthropic.messages.create  [LLM]   ← subagent B turn 1
    │   └── Bash  [TOOL]
    └── anthropic.messages.create  [LLM]   ← subagent B turn 2
        └── Read  [TOOL]
```

Three independent trackers collaborate to build this tree. They are described below.

---

## 4. Data Structures

### 4a. `ParsedToolName` (frozen dataclass)

```python
@dataclasses.dataclass(frozen=True)
class ParsedToolName:
    raw_name: str        # "mcp__server__remote_tool"
    display_name: str    # "remote_tool"   (or same as raw_name for non-MCP)
    is_mcp: bool         # True
    mcp_server: str|None # "server"
```

MCP tools from the Claude SDK have names like `mcp__myserver__some_tool`. `_parse_tool_name()` splits on `__` to extract `server` and `some_tool`, giving the span a clean display name and storing MCP metadata.

---

### 4b. `_ActiveToolSpan` (dataclass)

One instance per live tool call. Lives in `ToolSpanTracker._active_spans` keyed by `tool_use_id`.

```
_ActiveToolSpan
┌─────────────────────────────────────────────────────┐
│ span              : the Braintrust span object       │
│ raw_name          : "mcp__server__tool"              │
│ display_name      : "tool"                           │
│ input             : {"arg": "val"}  ← from SDK block │
│ tool_use_id       : "toolu_abc123"                   │
│ parent_tool_use_id: "toolu_agent_a" ← which subagent │
│ handler_active    : False ← True while handler runs  │
└─────────────────────────────────────────────────────┘
```

`activate()` sets `handler_active=True` and calls `span.set_current()` — making the Braintrust span the active context so any `start_span()` inside a tool handler automatically nests under it. `release()` undoes this.

There is also `_NoopActiveToolSpan` — a sentinel used when no matching span is found. It has the same interface but does nothing, so `_wrap_tool_handler` can call `.activate()` / `.release()` unconditionally without null checks.

---

### 4c. `ToolSpanTracker`

This is the most complex tracker. It manages all live tool spans across all subagent contexts.

```
ToolSpanTracker
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  _active_spans: dict[tool_use_id → _ActiveToolSpan]                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                        │
│  │ "toolu_a1"   │  │ "toolu_b1"   │  │ "toolu_c1"   │  ...                   │
│  │ Bash         │  │ Bash         │  │ remote_tool  │                        │
│  │ parent=A     │  │ parent=B     │  │ parent=C     │                        │
│  └──────────────┘  └──────────────┘  └──────────────┘                        │
│                                                                               │
│  _dispatch_queues: dict[(tool_name, input_sig) → deque[tool_use_id]]         │
│  ┌──────────────────────────────────────────────────┐                        │
│  │ ("Bash", '{"cmd":"echo"}') → deque["a1", "b1"]   │  ← FIFO               │
│  │ ("Read", '{"path":"/f"}')  → deque["a2", "b2"]   │                        │
│  └──────────────────────────────────────────────────┘                        │
│                                                                               │
│  _pending_task_link_tool_use_ids: set[tool_use_id]                           │
│  { "toolu_agent_a", "toolu_agent_b" }  ← "Agent" calls awaiting TaskStarted  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Lifecycle of a tool span through `ToolSpanTracker`:**

```
AssistantMessage arrives with ToolUseBlock
        │
        ▼
  start_tool_spans()
  ├── creates span with parent = current LLM span export
  ├── inserts into _active_spans[tool_use_id]
  ├── enqueues tool_use_id into _dispatch_queues[(name, input)]
  └── if name == "Agent": adds to _pending_task_link_tool_use_ids

Tool handler is called (by Claude SDK)
        │
        ▼
  _activate_tool_span_for_handler()
  ├── reads _thread_local.tool_span_tracker
  └── calls tracker.acquire_span_for_handler(name, args)
      ├── find candidates: active spans with matching name, not handler_active
      ├── _match_via_dispatch_queue()   ← try FIFO first
      │   └── pop from deque, return matching candidate
      ├── fallback: _match_tool_span_for_handler()  ← exact input match
      └── matched_span.activate()  → handler_active=True, set_current()

Tool handler finishes / UserMessage with ToolResultBlock arrives
        │
        ▼
  finish_tool_spans()
  └── _end_tool_span(tool_use_id, tool_result_block=block)
      ├── pop from _active_spans
      ├── remove from _dispatch_queues
      ├── log output from ToolResultBlock
      └── span.end()
```

**`_dispatch_queues` — the FIFO disambiguator:**

When subagent A and subagent B both call `Bash` with `{"cmd": "echo hi"}`, two identical `_ActiveToolSpan` entries exist. Without disambiguation, `acquire_span_for_handler` can't tell which handler invocation should own which span. The dispatch queue solves this by recording creation order:

```
Creation order:         Queue state:
  span "bash-A" added   →  ("Bash", '{"cmd":"echo hi"}')  deque: ["bash-A"]
  span "bash-B" added   →  ("Bash", '{"cmd":"echo hi"}')  deque: ["bash-A", "bash-B"]

Handler for A fires:    pop "bash-A"  →  give it bash-A span  ✓
Handler for B fires:    pop "bash-B"  →  give it bash-B span  ✓
```

**`cleanup()` — the scoped closer:**

```python
def cleanup(self, end_time=None, exclude_tool_use_ids=None, only_parent_tool_use_id=_UNSET_PARENT)
```

Three filter modes:
- No filters → close all active spans (called at the very end of `receive_response`).
- `exclude_tool_use_ids` → skip "Agent" spans still waiting for their `TaskStarted` event.
- `only_parent_tool_use_id` → **only** close spans belonging to a specific subagent context. This is called every time an `AssistantMessage` arrives, scoped to that message's `parent_tool_use_id`, so it never accidentally closes another subagent's still-open tool spans.

---

### 4d. `LLMSpanTracker._SubagentState` (inner dataclass)

One per subagent context. `None` key = orchestrator.

```
_SubagentState
┌──────────────────────────────────────────────────────────────────┐
│ current_span        : the open LLM span (or None)                │
│ current_span_export : span.export() string for use as parent ref  │
│ current_parent_export: parent export used when span was created   │
│ current_output      : [{"role":"assistant","content":[...]}]      │
│                       accumulated output so streaming chunks merge│
│ next_start_time     : float timestamp — when the next LLM call    │
│                       will start (set after tool results arrive)  │
└──────────────────────────────────────────────────────────────────┘
```

`next_start_time` is the key to non-overlapping sequential spans within one subagent. The sequence is:

```
UserMessage (tool results arrive)
    → mark_next_llm_start()  ← stamps the time NOW

AssistantMessage (next LLM response)
    → start_llm_span()
        → resolved_start_time = next_start_time  (the stamp from above)
        → current_span.end(end_time=resolved_start_time)  ← previous span ends HERE
        → create new span with start = resolved_start_time
        → next_start_time = None
```

This ensures the outgoing LLM span ends exactly when the next one begins — no gap, no overlap — even though the Python code observing the stream sees them arrive sequentially.

---

### 4e. `LLMSpanTracker`

Manages a `_SubagentState` for every subagent context, plus an `_active_context` pointer that says "which state should the next operation touch":

```
LLMSpanTracker
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  _active_context: "call-A"   ← set by set_context() on each AssistantMessage │
│                                                                               │
│  _states: dict[parent_tool_use_id → _SubagentState]                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐            │
│  │ None (orchestr.) │  │ "call-A"         │  │ "call-B"         │            │
│  │ next_start=t0    │  │ current_span=s1  │  │ current_span=s2  │            │
│  │ current_span=s0  │  │ next_start=None  │  │ next_start=t1    │            │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘            │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Context routing via `_get_state`:**

```python
def _get_state(self, parent_tool_use_id=_UNSET_PARENT):
    key = self._active_context if parent_tool_use_id is _UNSET_PARENT else parent_tool_use_id
    ...
```

- Called with `_UNSET_PARENT` (the default) → uses `_active_context`, whichever subagent was most recently set via `set_context()`.
- Called with an explicit value (e.g. from `mark_next_llm_start(user_parent)`) → routes directly to that subagent's state regardless of `_active_context`.

This is why `_UNSET_PARENT = object()` exists — it is a sentinel that can be distinguished from `None`, which is a valid key meaning "orchestrator".

**`mark_next_llm_start()` edge case:**

UserMessages from the Claude SDK sometimes don't carry `parent_tool_use_id` even when they belong to a subagent context. The special-case logic handles this:

```python
def mark_next_llm_start(self, parent_tool_use_id=_UNSET_PARENT):
    if parent_tool_use_id is None and self._active_context is not None:
        parent_tool_use_id = _UNSET_PARENT  # fall back to active context
    self._get_state(parent_tool_use_id).next_start_time = time.time()
```

If the UserMessage says `parent_tool_use_id=None` (field absent or None) but `_active_context` is set (we are processing a subagent's turn), treat it as "active context" rather than routing to the orchestrator state.

---

### 4f. `TaskEventSpanTracker`

Manages TASK spans for subagent tasks, driven by `SystemMessage` subtypes.

```
TaskEventSpanTracker
┌───────────────────────────────────────────────────────────────────────────────┐
│ _root_span_export        : export of the top-level "Claude Agent" TASK span  │
│ _tool_tracker            : ref to ToolSpanTracker (to get Agent span export) │
│                                                                               │
│ _active_spans            : dict[task_id → span]                              │
│ ┌──────────────┐  ┌──────────────┐                                           │
│ │ "task_a"     │  │ "task_b"     │  ...                                      │
│ │ span=...     │  │ span=...     │                                            │
│ └──────────────┘  └──────────────┘                                           │
│                                                                               │
│ _task_span_by_tool_use_id: dict[agent_tool_use_id → span]                    │
│ ┌─────────────────────────────────────────────────────┐                      │
│ │ "toolu_agent_a" → task-A span                       │                      │
│ │ "toolu_agent_b" → task-B span                       │                      │
│ └─────────────────────────────────────────────────────┘                      │
│                                                                               │
│ _active_task_order       : ["task_a", "task_b"]  ← insertion order          │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Lifecycle:**

- `TaskStartedMessage` → create a TASK span. Parent is the `Agent` tool span for this task (looked up via `_tool_tracker.get_span_export(message.tool_use_id)`), falling back to the root span. Also calls `_tool_tracker.mark_task_started(tool_use_id)`, removing the agent tool_use_id from `_pending_task_link_tool_use_ids`, which tells `cleanup()` it is now safe to close that `Agent` span.
- `TaskProgressMessage` → log metadata/output updates to the existing TASK span.
- `TaskNotificationMessage` → end the TASK span and remove it from both dicts.

**`parent_export_for_message()`** finds the right parent for a subagent's LLM span given an `AssistantMessage`:

1. If `parent_tool_use_id` is set, look up `_task_span_by_tool_use_id[parent_tool_use_id]` — return that task span as parent. ✓
2. Else if the message itself contains an `Agent` ToolUseBlock (orchestrator calling a subagent), use the top-level span as parent (not the most recently opened task).
3. Else fall back to the latest open task span in `_active_task_order`.

---

## 5. Thread-Local: Bridging the Stream to Tool Handlers

The trickiest part is that tool handlers are called **by the Claude SDK** — not directly by Braintrust code. There is no way to pass context as a function argument. The solution is a thread-local:

```python
_thread_local = threading.local()
```

At the start of `receive_response()`:
```python
_thread_local.tool_span_tracker = tool_tracker
```

Inside every wrapped tool handler:
```python
def _activate_tool_span_for_handler(tool_name, args):
    tool_span_tracker = getattr(_thread_local, "tool_span_tracker", None)
    if tool_span_tracker is None:
        return _NOOP_ACTIVE_TOOL_SPAN  # no tracing session active
    return tool_span_tracker.acquire_span_for_handler(tool_name, args) or _NOOP_ACTIVE_TOOL_SPAN
```

This means:
- One `receive_response()` session running on a thread → that thread's tool handlers find their tracker.
- If a tool is called outside of a `receive_response()` session → returns `_NOOP_ACTIVE_TOOL_SPAN`, tracing is skipped gracefully.
- The thread-local is cleaned up in the `finally` block of `receive_response()`.

```
Thread T1: receive_response() starts
  _thread_local.tool_span_tracker = tracker_T1

  Claude SDK calls tool handler "Bash" on T1
    → _activate_tool_span_for_handler reads _thread_local.tool_span_tracker
    → gets tracker_T1 → acquires correct span → handler runs nested under it

  receive_response() finally:
    del _thread_local.tool_span_tracker
```

---

## 6. Full Message Loop

How every message type affects each tracker:

```
Message arrives from SDK
│
├── AssistantMessage  (parent_tool_use_id = X)
│   ├── llm_tracker.set_context(X)              → route all LLM ops to subagent X's state
│   ├── if current LLM span + active tool spans:
│   │   tool_tracker.cleanup(                   → close only X's dangling tool spans
│   │       end_time=next_start_time,           → timed to the gap before this LLM span
│   │       exclude=active_subagent_ids,        → leave "Agent" spans still open
│   │       only_parent=X)                      → don't touch other subagents' tool spans
│   ├── task_event_span_tracker
│   │       .parent_export_for_message()        → find which TASK span is the parent
│   ├── llm_tracker.start_llm_span(...)         → end previous span for X; start new one
│   └── tool_tracker.start_tool_spans(...)      → open tool spans for any ToolUseBlocks
│
├── UserMessage  (parent_tool_use_id = X)
│   ├── tool_tracker.finish_tool_spans(...)     → close tool spans with output from ToolResultBlocks
│   └── if has_tool_results:
│       llm_tracker.mark_next_llm_start(X)     → stamp "next LLM for X starts now"
│
├── ResultMessage
│   ├── llm_tracker.set_context(None)           → route to orchestrator state
│   └── llm_tracker.log_usage(...)              → attach token usage to orchestrator LLM span
│
└── SystemMessage / TaskStarted / TaskProgress / TaskNotification
    └── task_event_span_tracker.process(...)    → create / update / end TASK spans
```

---

## 7. End-to-End Example: Two Concurrent Subagents

Walkthrough of exactly what the three trackers look like at each step for the `test_interleaved_subagent_tool_output_preserved` scenario:

```
Stream event                         ToolSpanTracker._active_spans    LLMTracker._states           TaskEventSpanTracker
────────────────────────────────     ────────────────────────────     ─────────────────────        ────────────────────
[1] AssistantMessage(parent=None)    {}                               {None: {span=LLM-0}}         {}
    orchestrator calls Agent(α), Agent(β)
    after start_tool_spans:
                                     {"call-α": Agent-span-α,
                                      "call-β": Agent-span-β}
                                     pending: {"call-α", "call-β"}

[2] TaskStartedMessage(task=alpha)   pending: {"call-β"}             (unchanged)                  {"alpha": Task-α (parent=Agent-α)}
[3] TaskStartedMessage(task=beta)    pending: {}                     (unchanged)                  {"alpha": Task-α, "beta": Task-β}

[4] AssistantMessage(parent=call-α)  (subagent alpha's LLM turn: Bash call)
    set_context("call-α")
    cleanup(only_parent="call-α")    → closes nothing (α has no old tool spans)
    start_llm_span                   (unchanged)                     {"call-α": {span=LLM-α}}
    start_tool_spans("bash-1")       {"call-α": Bash-span (parent=LLM-α),
                                      "call-β": Agent-span-β}

[5] AssistantMessage(parent=call-β)  (subagent beta's LLM turn: Read call)
    set_context("call-β")
    cleanup(only_parent="call-β")    → closes nothing (β has no old tool spans)
                                     Bash-span is NOT closed ← key fix
    start_llm_span                   (unchanged)                     {"call-β": {span=LLM-β}}
    start_tool_spans("read-1")       {"call-α": Bash-span (still open!),
                                      "call-β": Read-span (parent=LLM-β)}

[6] UserMessage(ToolResult bash-1="alpha_file_contents", parent=call-α)
    finish_tool_spans                Bash-span.log(output), Bash-span.end()
    mark_next_llm_start("call-α")                                    {call-α: {next_start=now}}

[7] UserMessage(ToolResult read-1="beta_file_contents", parent=call-β)
    finish_tool_spans                Read-span.log(output), Read-span.end()
    mark_next_llm_start("call-β")                                    {call-β: {next_start=now}}

[8] ResultMessage
    set_context(None)
    log_usage                                                         LLM-0.log(tokens)

finally:
    task_event_span_tracker.cleanup()  → end Task-α, Task-β
    tool_tracker.cleanup()             → end Agent-α, Agent-β (if still open)
    llm_tracker.cleanup()              → end LLM-α, LLM-β, LLM-0
```

At step [5], the old code called `cleanup()` globally, ending Bash-span before step [6] could record its output. The `only_parent_tool_use_id="call-β"` filter introduced by the fix prevents that — Bash-span survives to receive its result.
