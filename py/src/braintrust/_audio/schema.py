"""Audio v3 trace schema: shared span metadata keys and cross-span links.

Voice integrations emit the same session, turn, speech and recording shapes,
consistent with each other and with the JavaScript SDK.
"""

from typing import Literal

from braintrust.logger import NOOP_SPAN, Span


SCHEMA_VERSION = "3.0.0"

# A turn records its own ID; every operation within it records the turn's ID.
TURN_ID = "turn.id"
TURN_REPLY_TO = "turn.reply_to"
TURN_INCOMPLETE = "turn.incomplete"
SOURCE_ID = "audio.source_id"
PARTICIPANT_IDS = "audio.participant_ids"
SPEAKER_IDS = "audio.speaker_ids"
SELECTIONS = "audio.selections"
LINKS = "links"

Relationship = Literal["transcribes", "uses_context", "generated_by", "requested_by", "consumes_tool_result"]


def span_id(span: Span) -> str:
    identifier = getattr(span, "span_id", "")
    return identifier if isinstance(identifier, str) else ""


def set_links(span: Span, relationships: list[tuple[Relationship, Span]]) -> None:
    """Set the typed links from ``span`` to other spans.

    Links are a single metadata value, so each call *replaces* the span's
    links: pass its complete set. An empty set clears them. Duplicate
    (relationship, target) pairs within a call are logged once; no-op targets
    are skipped.
    """
    # Match the JS integrations: native rows currently carry links in metadata.
    links = {}
    for relationship, target in relationships:
        if target == NOOP_SPAN:
            continue
        links[(relationship, span_id(target))] = {
            "context": {
                "traceId": getattr(target, "root_span_id", ""),
                "spanId": span_id(target),
                "traceFlags": 1,
            },
            "attributes": {"braintrust.relationship.type": relationship},
        }
    span.log(metadata={LINKS: list(links.values())})
