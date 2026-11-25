"""Correlation between model requests and the operations whose output they use."""

from collections import OrderedDict
from typing import Any

from braintrust._audio.schema import Relationship
from braintrust.integrations.utils import _is_not_given, _normalize_chat_messages
from braintrust.logger import Span

from ._common import LOG, MAX_PENDING, tools_schema_to_openai_tools


class ContextLinker:
    """Correlate native message identities and tool IDs, never transcript text."""

    def __init__(self) -> None:
        self.messages: Any = None
        self.metadata: dict[str, Any] = {}
        # Snapshots of submitted native messages detect an unchanged prefix.
        # Applications may edit a message in place, and normalization rebuilds
        # multimodal parts, so neither identity nor normalized output can.
        self._submitted: tuple[Any, ...] = ()
        self._consumed = 0
        self._transcripts: OrderedDict[int, tuple[Span, Span | None]] = OrderedDict()
        self._provenance: OrderedDict[int, tuple[Any, Any, tuple[int, ...]]] = OrderedDict()
        self._raw: list[Any] = []
        self.reply_to: Span | None = None
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
        self._raw = raw
        self.messages, self.metadata = _normalize_chat_messages(raw), metadata
        last = self.messages[-1] if self.messages else None
        return last if isinstance(last, dict) and last.get("role") == "user" else None

    def add_transcript(self, frame_id: int, span: Span, turn: Span | None = None) -> None:
        self._transcripts[frame_id] = (span, turn)
        if len(self._transcripts) > MAX_PENDING:
            self._transcripts.popitem(last=False)

    def associate(self, message: Any, frame_ids: tuple[int, ...]) -> None:
        # Holding the message prevents id reuse. An application edit invalidates
        # this evidence; native transformations made before insertion retain it.
        self._provenance[id(message)] = (message, _snapshot(message), frame_ids)
        if len(self._provenance) > MAX_PENDING:
            self._provenance.popitem(last=False)

    def add_tool_result(self, tool_call_id: str, span: Span) -> None:
        self._tool_results[tool_call_id] = span
        if len(self._tool_results) > MAX_PENDING:
            self._warn("tool result")
            self._tool_results.popitem(last=False)

    def consume(self) -> list[tuple[Relationship, Span]]:
        """Link the messages added since the previous request, and retire them."""
        links: list[tuple[Relationship, Span]] = []
        messages = self.messages if isinstance(self.messages, list) else []
        turns: dict[str, Span] = {}
        unknown_turn = False
        for raw, message in zip(self._raw[self._consumed :], messages[self._consumed :]):
            if not isinstance(message, dict):
                continue
            evidence = self._provenance.get(id(raw))
            if evidence is not None and evidence[0] is raw and evidence[1] == _snapshot(raw):
                for frame_id in evidence[2]:
                    transcript = self._transcripts.pop(frame_id, None)
                    if transcript is not None:
                        source, turn = transcript
                        links.append(("uses_context", source))
                        if turn is not None:
                            turns[turn.id] = turn
                        else:
                            unknown_turn = True
                    else:
                        unknown_turn = True
            if message.get("role") == "tool" and message.get("tool_call_id") in self._tool_results:
                links.append(("consumes_tool_result", self._tool_results.pop(message["tool_call_id"])))
        self.reply_to = next(iter(turns.values())) if len(turns) == 1 and not unknown_turn else None
        self._consumed = len(messages)
        return links

    def _warn(self, kind: str) -> None:
        if kind not in self._warned:
            self._warned.add(kind)
            LOG.warning("Pipecat pending %s links exceeded the retention limit; retiring the oldest links", kind)

    def clear(self) -> None:
        """Release routing payloads after all session work has been exported."""
        self.messages = None
        self.metadata.clear()
        self._submitted = ()
        self._consumed = 0
        self._raw.clear()
        self._transcripts.clear()
        self._provenance.clear()
        self._tool_results.clear()
        self._warned.clear()
        self.reply_to = None


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
