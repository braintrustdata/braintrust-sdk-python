"""Correlation between model requests and the operations whose output they use."""

from collections import OrderedDict, deque
from typing import Any

from braintrust._audio.schema import Relationship
from braintrust.integrations.utils import _is_not_given, _normalize_chat_messages
from braintrust.logger import Span

from ._common import LOG, MAX_PENDING, message_text, tools_schema_to_openai_tools


class ContextLinker:
    """Tracks the latest model context and what its new messages consume.

    Transcripts and tool results wait here until a model request includes
    them. Only messages added since the previous request are linked: the user
    aggregator may join several transcripts into one message, so transcripts
    match in arrival order as substrings. Unmatched transcripts older than a
    match were dropped from context and are retired; newer ones stay pending.
    """

    def __init__(self) -> None:
        self.messages: Any = None
        self.metadata: dict[str, Any] = {}
        # Snapshots of submitted native messages detect an unchanged prefix.
        # Applications may edit a message in place, and normalization rebuilds
        # multimodal parts, so neither identity nor normalized output can.
        self._submitted: tuple[Any, ...] = ()
        self._consumed = 0
        self._transcripts: deque[tuple[str, Span]] = deque(maxlen=MAX_PENDING)
        self._tool_results: OrderedDict[str, Span] = OrderedDict()
        self._warned: set[str] = set()

    def submit(self, context: Any) -> dict[str, Any] | None:
        """Record a submitted context; return its latest message if from the user."""
        metadata: dict[str, Any] = {}
        tools = getattr(context, "tools", None)
        if tools is not None and not _is_not_given(tools):
            converted = tools_schema_to_openai_tools(tools)
            if converted:
                metadata["tools"] = converted
        tool_choice = getattr(context, "tool_choice", None)
        if tool_choice is not None and not _is_not_given(tool_choice):
            metadata["tool_choice"] = tool_choice
        raw = list(context.get_messages() or [])
        submitted = tuple(_snapshot(message) for message in raw)
        # Context can append, truncate, replace or edit messages. Retain the
        # cursor only through the prefix that is unchanged.
        consumed = 0
        for old, new in zip(self._submitted, submitted):
            if consumed >= self._consumed or old != new:
                break
            consumed += 1
        self._submitted, self._consumed = submitted, consumed
        self.messages, self.metadata = _normalize_chat_messages(raw), metadata
        last = self.messages[-1] if self.messages else None
        return last if isinstance(last, dict) and last.get("role") == "user" else None

    def add_transcript(self, text: str, span: Span) -> None:
        if len(self._transcripts) == MAX_PENDING:
            self._warn("transcript")
        self._transcripts.append((text, span))

    def add_tool_result(self, tool_call_id: str, span: Span) -> None:
        self._tool_results[tool_call_id] = span
        if len(self._tool_results) > MAX_PENDING:
            self._warn("tool result")
            self._tool_results.popitem(last=False)

    def consume(self) -> list[tuple[Relationship, Span]]:
        """Link the messages added since the previous request, and retire them."""
        links: list[tuple[Relationship, Span]] = []
        messages = self.messages if isinstance(self.messages, list) else []
        pending = list(self._transcripts)
        matched = -1
        for message in messages[self._consumed :]:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                text, position = message_text(message.get("content")), 0
                for index in range(matched + 1, len(pending)):
                    transcript = pending[index][0].strip()
                    found = text.find(transcript, position) if transcript else -1
                    if found >= 0:
                        links.append(("uses_context", pending[index][1]))
                        matched, position = index, found + len(transcript)
            elif role == "tool" and message.get("tool_call_id") in self._tool_results:
                links.append(("consumes_tool_result", self._tool_results.pop(message["tool_call_id"])))
        if matched >= 0:
            self._transcripts = deque(pending[matched + 1 :], maxlen=MAX_PENDING)
        self._consumed = len(messages)
        return links

    def _warn(self, kind: str) -> None:
        if kind not in self._warned:
            self._warned.add(kind)
            LOG.warning("Pipecat pending %s links exceeded the retention limit; retiring the oldest links", kind)


def _snapshot(value: Any) -> Any:
    """Freeze message structure so later in-place edits compare unequal.

    Containers are copied; leaves (strings, bytes, provider objects) are
    shared, so a snapshot costs one pass over the message structure.
    """
    if isinstance(value, dict):
        return tuple((key, _snapshot(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value
