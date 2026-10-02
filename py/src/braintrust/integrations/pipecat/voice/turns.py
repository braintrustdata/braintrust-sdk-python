"""Correlate native aggregator lifecycles without changing pipeline behavior."""

from collections import deque


class Turns:
    def __init__(self, root, hooks=None):
        from .hooks import Hooks

        self.hooks = hooks or Hooks()
        self.root = root
        self.user = None
        self.assistant = None
        self.last_user = None
        self.pending = {"user": deque(), "assistant": deque()}
        self.states = []
        self.messages = {}
        self.on_completed = None

    def start(self, role, trigger, reply_to=None):
        # First-push observation precedes the aggregator's asynchronous callback.
        current = getattr(self, role)
        if current is not None:
            return current
        state = {
            "span": self.root.start_span(
                name=f"{role}_turn",
                type="task",
                set_current=False,
                internal={"instrumentation": "pipecat-auto"},
                metadata={"pipecat.turn.start_observation": trigger},
            ),
            "role": role,
            "confirmed": False,
            "ended": False,
        }
        state["metadata"] = {"turn.id": state["span"].span_id}
        if role == "assistant" and reply_to:
            state["metadata"]["turn.reply_to"] = reply_to
        state["span"].log(metadata=state["metadata"])
        setattr(self, role, state)
        self.pending[role].append(state)
        self.states.append(state)
        if role == "user":
            self.last_user = state
        return state

    def metadata(self, state):
        return dict(state["metadata"]) if state else {}

    def log_message(self, state, content):
        state["content"] = content
        message = {"role": state["role"], "content": content or ""}
        if state.get("tool_calls"):
            message["tool_calls"] = state["tool_calls"]
        if state["role"] == "user":
            state["span"].log(input={"messages": [message]})
        else:
            state["span"].log(output=[message])

    def install(self, user, assistant):
        async def user_started(aggregator, strategy):
            self.confirm("user", type(strategy).__name__)

        async def user_stopped(aggregator, strategy, message):
            self.stop("user", message, type(strategy).__name__)

        async def user_message(aggregator, message):
            state = self.messages.get(("user", message.timestamp))
            if state:
                self.log_message(state, message.content)
                state["span"].log(
                    metadata={
                        "pipecat.content": message.content,
                        "pipecat.timestamp": message.timestamp,
                        "pipecat.user_id": message.user_id,
                    }
                )

        async def assistant_started(aggregator):
            self.confirm("assistant")

        async def assistant_stopped(aggregator, message):
            self.stop("assistant", message)

        self.hooks.event(user, "on_user_turn_started", user_started)
        self.hooks.event(user, "on_user_turn_stopped", user_stopped)
        self.hooks.event(user, "on_user_turn_message_added", user_message)
        self.hooks.event(assistant, "on_assistant_turn_started", assistant_started)
        self.hooks.event(assistant, "on_assistant_turn_stopped", assistant_stopped)

    def confirm(self, role, strategy=None):
        state = next((s for s in self.pending[role] if not s["confirmed"]), None)
        if state is None:
            state = self.start(role, f"on_{role}_turn_started")
        state["confirmed"] = True
        metadata = {"pipecat.turn.start_event": f"on_{role}_turn_started"}
        if strategy:
            metadata["pipecat.turn.start_strategy"] = strategy
        state["span"].log(metadata=metadata)

    def stop(self, role, message, strategy=None):
        if not self.pending[role]:
            return
        state = self.pending[role].popleft()
        metadata = {
            "pipecat.turn.stop_event": f"on_{role}_turn_stopped",
            "pipecat.content": message.content,
            "pipecat.timestamp": message.timestamp,
        }
        for field in ("user_id", "interrupted"):
            if hasattr(message, field):
                metadata[f"pipecat.{field}"] = getattr(message, field)
        if strategy:
            metadata["pipecat.turn.stop_strategy"] = strategy
        state["span"].log(metadata=metadata)
        self.log_message(state, message.content)
        state["span"].end()
        state["ended"] = True
        if self.on_completed:
            self.on_completed(state)
        self.messages[(role, message.timestamp)] = state
        if len(self.messages) > 512:
            self.messages.pop(next(iter(self.messages)))
        self.states = [pending for pending in self.states if pending is not state]
        if getattr(self, role) is state:
            setattr(self, role, None)

    def finish(self):
        for state in self.states:
            if not state["ended"]:
                state["span"].log(metadata={"turn.incomplete": True})
                state["span"].end()
                state["ended"] = True
