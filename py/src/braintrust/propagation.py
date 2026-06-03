"""Native W3C Trace Context propagation for Braintrust.

This module implements the propagation wire format described in the Braintrust
distributed-tracing spec, using pure Python with no dependency on
``opentelemetry``. It parses and serializes the W3C ``traceparent`` and
``baggage`` headers and the Braintrust ``braintrust.parent`` baggage entry.

Trace identity (trace id + parent span id) is carried in ``traceparent``; the
Braintrust container the trace belongs to (project/experiment) is carried in
``baggage`` under the ``braintrust.parent`` key.
"""

import logging
import re
from typing import NamedTuple
from urllib.parse import quote, unquote


__all__ = [
    "TRACEPARENT_HEADER",
    "TRACESTATE_HEADER",
    "BAGGAGE_HEADER",
    "BRAINTRUST_PARENT_KEY",
    "DEFAULT_TRACE_FLAGS",
    "ParsedTraceparent",
    "PropagatedState",
    "parse_traceparent",
    "format_traceparent",
    "parse_baggage",
    "merge_baggage",
    "get_header",
]

log = logging.getLogger(__name__)

TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"
BAGGAGE_HEADER = "baggage"
BRAINTRUST_PARENT_KEY = "braintrust.parent"

# Trace-flags byte we emit for traces we originate: sampled (low bit set).
DEFAULT_TRACE_FLAGS = "01"


class ParsedTraceparent(NamedTuple):
    """Parsed W3C ``traceparent`` fields.

    Tuple-compatible: unpacks as ``(trace_id, span_id, trace_flags)``.
    ``trace_flags`` is the raw 2-hex trace-flags byte (e.g. ``"01"`` sampled,
    ``"00"`` not sampled), kept raw so any future flag bits survive a
    parse -> format round trip without per-bit handling.
    """

    trace_id: str
    span_id: str
    trace_flags: str


class PropagatedState(NamedTuple):
    """Inbound W3C trace-context state that Braintrust forwards but never interprets.

    Captured at the span created from inbound headers (via
    ``extract_trace_context``) and inherited by every subspan, so that any
    ``inject()`` within the trace re-emits the upstream state unchanged, per the
    W3C Trace Context spec.

    - ``tracestate``: the W3C ``tracestate`` header (opaque vendor state).
    - ``trace_flags``: the raw 2-hex ``traceparent`` trace-flags byte. Stored raw
      so future flag bits are preserved without per-bit handling.
    """

    tracestate: str | None = None
    trace_flags: str | None = None


# W3C traceparent: version-traceid-parentid-flags, version 00, lowercase hex.
_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_ZERO_TRACE_ID = "0" * 32
_ZERO_SPAN_ID = "0" * 16

# W3C Baggage limits (§3.3.2): a conformant baggage-string must satisfy *both*
# of these conditions. https://www.w3.org/TR/baggage/#limits
#   - Condition 1: at most 64 list-members.
#   - Condition 2: at most 8192 bytes total.
#
# We reuse these as a defensive bound on parsing/relaying untrusted inbound
# headers: the header arrives from the network and is attacker-controllable, so
# we never split or decode an unbounded string. When an inbound header exceeds
# either limit we drop trailing list-members rather than truncate one mid-value:
# the spec says a platform that cannot propagate all list-members "MUST NOT
# propagate any partial list-members", so we keep only the leading whole members
# that fit within both limits.
_MAX_BAGGAGE_LENGTH = 8192
_MAX_BAGGAGE_MEMBERS = 64


def _cap_baggage_to_member_boundary(value):
    """Return ``value`` bounded to the W3C limits, never splitting a list-member
    mid-value.

    Enforces both §3.3.2 limits: at most ``_MAX_BAGGAGE_MEMBERS`` list-members
    and at most ``_MAX_BAGGAGE_LENGTH`` UTF-8 bytes (the spec limit is a byte
    count, not code points). If the header is within both limits it is returned
    unchanged. Otherwise we keep the leading whole members that fit and drop the
    rest -- a trailing member that would be partial is never kept. If even the
    first member exceeds the byte limit there is no complete member to keep, so
    we return an empty string.
    """
    encoded = value.encode("utf-8")
    within_bytes = len(encoded) <= _MAX_BAGGAGE_LENGTH
    # Cheap structural cap on member count: actual members are <= comma count + 1.
    within_members = encoded.count(b",") < _MAX_BAGGAGE_MEMBERS
    if within_bytes and within_members:
        return value

    # Walk members in order, keeping whole ones until either limit is reached.
    # We accumulate on the encoded bytes so the byte budget is exact and we only
    # ever cut on comma (an ASCII byte), so partial code points are never split.
    kept = []
    length = 0
    for raw_member in encoded.split(b","):
        if len(kept) >= _MAX_BAGGAGE_MEMBERS:
            break
        cost = len(raw_member) + (1 if kept else 0)
        if length + cost > _MAX_BAGGAGE_LENGTH:
            break
        kept.append(raw_member)
        length += cost
    if not kept:
        # The first member alone already exceeds the byte limit.
        return ""
    return b",".join(kept).decode("utf-8", errors="ignore")


def get_header(headers, name):
    """Case-insensitive header lookup.

    Some frameworks normalize header names to title case (e.g. ``Traceparent``)
    while the W3C keys are lowercase. Returns the first matching value or None.
    """
    if not headers:
        return None
    # Fast path: exact (lowercase) match.
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    for key, val in headers.items():
        if isinstance(key, str) and key.lower() == lowered:
            return val
    return None


def parse_traceparent(value):
    """Parse a W3C ``traceparent`` value into a :class:`ParsedTraceparent`.

    Returns a ``(trace_id, span_id, trace_flags)`` named tuple, where
    ``trace_flags`` is the raw 2-hex trace-flags byte. Returns None for any
    malformed value (bad version, wrong length, non-hex, or all-zero ids). Never
    raises.
    """
    if not value or not isinstance(value, str):
        return None
    match = _TRACEPARENT_RE.match(value.strip())
    if not match:
        return None
    trace_id, span_id, flags = match.group(1), match.group(2), match.group(3)
    if trace_id == _ZERO_TRACE_ID or span_id == _ZERO_SPAN_ID:
        return None
    return ParsedTraceparent(trace_id, span_id, flags)


def format_traceparent(trace_id, span_id, trace_flags=DEFAULT_TRACE_FLAGS):
    """Serialize a W3C ``traceparent`` value from hex trace/span ids.

    ``trace_flags`` is the raw 2-hex trace-flags byte to emit; it is forwarded
    verbatim so any upstream/future flag bits survive. Falls back to
    ``DEFAULT_TRACE_FLAGS`` (sampled) when not a valid 2-hex byte. Returns None
    if the ids are not valid W3C-shaped hex (so callers can omit the header
    rather than emit something malformed).
    """
    if not _is_hex(trace_id, 32) or trace_id == _ZERO_TRACE_ID:
        return None
    if not _is_hex(span_id, 16) or span_id == _ZERO_SPAN_ID:
        return None
    flags = trace_flags if _is_hex(trace_flags, 2) else DEFAULT_TRACE_FLAGS
    return f"00-{trace_id}-{span_id}-{flags}"


def parse_baggage(value):
    """Parse a W3C ``baggage`` header into an ordered dict of key -> value.

    Tolerates malformed/oversized input by skipping bad entries; never raises.
    Property metadata (after ';') is ignored. Keys and values are percent-decoded.
    """
    result = {}
    if not value or not isinstance(value, str):
        return result
    # Oversized header: bound the work to whole list-members (never mid-value).
    value = _cap_baggage_to_member_boundary(value)
    for member in value.split(","):
        member = member.strip()
        if not member or "=" not in member:
            continue
        # Strip any ';'-delimited properties.
        member = member.split(";", 1)[0]
        key, _, val = member.partition("=")
        key = _percent_decode(key.strip())
        if not key:
            continue
        result[key] = _percent_decode(val.strip())
    return result


def merge_baggage(existing, braintrust_parent):
    """Merge a ``braintrust.parent`` value into an existing ``baggage`` header.

    This preserves every other vendor's baggage member byte-for-byte: their raw
    ``key=value`` substrings (properties included) are forwarded exactly as
    received rather than decoded and re-encoded. Decoding then re-encoding would
    silently rewrite another vendor's percent-encoding (e.g. ``path=a%2Fb`` ->
    ``path=a/b``), so we keep Braintrust a transparent relay. Whitespace around
    list members is insignificant per W3C and is trimmed.

    Only the ``braintrust.parent`` member is (re)serialized, by us, from the
    ``braintrust_parent`` argument. Any pre-existing ``braintrust.parent``
    member in ``existing`` is dropped in favor of the supplied value.

    The result is bounded to both W3C limits (§3.3.2): at most 64 list-members
    and at most 8192 bytes. Our own ``braintrust.parent`` member is prioritized:
    its byte cost and one member slot are reserved first, then relayed members
    are appended in order until either budget is exhausted, always on
    whole-member boundaries (never a partial list-member). Relayed members that
    do not fit are dropped.

    Returns the merged header value, or None if there is nothing to emit (so
    callers omit the header rather than emit an empty one).
    """
    bt_member = None
    if braintrust_parent:
        encoded_key = _percent_encode(BRAINTRUST_PARENT_KEY)
        encoded_val = _percent_encode(str(braintrust_parent))
        bt_member = f"{encoded_key}={encoded_val}"

    # Reserve both budgets for our own member first so it always survives;
    # relayed members fill whatever space remains.
    byte_budget = _MAX_BAGGAGE_LENGTH
    member_budget = _MAX_BAGGAGE_MEMBERS
    if bt_member is not None:
        # +1 for the comma joining our member to any preceding relayed member.
        byte_budget -= len(bt_member.encode("utf-8")) + 1
        member_budget -= 1

    relayed = []
    length = 0
    if existing and isinstance(existing, str):
        for raw_member in existing.split(","):
            member = raw_member.strip()
            if not member or "=" not in member:
                continue
            # Identify the key (ignoring ';'-delimited properties) only to skip
            # any inbound braintrust.parent; everything else is forwarded raw.
            key_part = member.split(";", 1)[0].partition("=")[0]
            key = _percent_decode(key_part.strip())
            if key == BRAINTRUST_PARENT_KEY:
                continue
            # Stop at whole-member boundaries once either budget is exhausted; we
            # never forward a partial member (W3C §3.3.2).
            if len(relayed) >= member_budget:
                break
            cost = len(member.encode("utf-8")) + (1 if relayed else 0)
            if length + cost > byte_budget:
                break
            relayed.append(member)
            length += cost

    members = relayed
    if bt_member is not None:
        members = relayed + [bt_member]

    if not members:
        return None
    return ",".join(members)


def _is_hex(value, length):
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return all(c in "0123456789abcdef" for c in value)


# Per W3C Baggage (§3.3.1.3), a value's unencoded bytes are restricted to the
# ``baggage-octet`` set:
#
#   baggage-octet = %x21 / %x23-2B / %x2D-3A / %x3C-5B / %x5D-7E
#
# i.e. US-ASCII excluding CTLs, whitespace, DQUOTE, comma, semicolon, and
# backslash; the percent sign MUST be encoded; and any non-ASCII code point MUST
# be percent-encoded (as UTF-8 octets). We only ever encode our own
# ``braintrust.parent`` member, whose value embeds an arbitrary, user-controlled
# project/experiment name -- so it can contain any of those characters.
#
# ``quote(value, safe="")`` percent-encodes every byte outside the RFC 3986
# unreserved set (``A-Z a-z 0-9 - _ . ~``). That set is a strict subset of
# ``baggage-octet``, so the result is always spec-compliant: we may over-encode
# some characters that are technically legal unencoded (the spec explicitly
# permits this), but we never emit a byte that violates the grammar. Space is
# emitted as ``%20`` (per the W3C example), not form-urlencoded ``+``.
#
# On receive we decode with ``unquote``, the exact inverse of ``quote`` (``%20``
# -> space, multi-byte UTF-8 reassembled). We intentionally do not use
# ``unquote_plus``: ``+`` is the ``application/x-www-form-urlencoded`` space
# convention, not W3C Baggage, so a literal ``+`` in a value must stay a ``+``.
#
# Byte-for-byte pass-through of *other* vendors' baggage is handled separately by
# :func:`merge_baggage`, which forwards their raw member strings unchanged rather
# than round-tripping them through this codec.


def _percent_encode(value):
    return quote(str(value), safe="")


def _percent_decode(value):
    if "%" not in value:
        return value
    try:
        return unquote(value)
    except Exception:
        return value
