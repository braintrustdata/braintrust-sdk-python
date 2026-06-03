"""Tests for native W3C trace-context propagation.

These tests cover the Braintrust distributed-tracing spec's test matrix using
the pure-Python propagation path (no opentelemetry dependency), so they run in
the `test_core` nox session.
"""

import re

import pytest
from braintrust.logger import (
    _internal_with_memory_background_logger,
    extract_trace_context,
    inject_trace_context,
)
from braintrust.propagation import (
    BAGGAGE_HEADER,
    BRAINTRUST_PARENT_KEY,
    TRACEPARENT_HEADER,
    TRACESTATE_HEADER,
    format_traceparent,
    get_header,
    merge_baggage,
    parse_baggage,
    parse_traceparent,
)
from braintrust.span_identifier_v4 import SpanObjectTypeV3
from braintrust.test_helpers import init_test_logger


TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")

VALID_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
VALID_SPAN_ID = "00f067aa0ba902b7"
VALID_TRACEPARENT = f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01"


# --------------------------------------------------------------------------- #
# Primitives: traceparent / baggage parse + format
# --------------------------------------------------------------------------- #


class TestTraceparent:
    def test_parse_valid(self):
        assert parse_traceparent(VALID_TRACEPARENT) == (VALID_TRACE_ID, VALID_SPAN_ID, "01")

    def test_parse_strips_whitespace(self):
        assert parse_traceparent(f"  {VALID_TRACEPARENT}  ") == (VALID_TRACE_ID, VALID_SPAN_ID, "01")

    @pytest.mark.parametrize(
        "value",
        [
            "",
            None,
            "invalid",
            "00-tooshort-00f067aa0ba902b7-01",
            f"00-{VALID_TRACE_ID}-00f067aa-01",  # short span id
            f"99-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01",  # bad version
            f"00-{'0' * 32}-{VALID_SPAN_ID}-01",  # zero trace id
            f"00-{VALID_TRACE_ID}-{'0' * 16}-01",  # zero span id
            f"00-{VALID_TRACE_ID.upper()}-{VALID_SPAN_ID}-01",  # uppercase hex
        ],
    )
    def test_parse_invalid(self, value):
        assert parse_traceparent(value) is None

    def test_format_round_trip(self):
        tp = format_traceparent(VALID_TRACE_ID, VALID_SPAN_ID)
        assert TRACEPARENT_RE.match(tp)
        assert parse_traceparent(tp) == (VALID_TRACE_ID, VALID_SPAN_ID, "01")

    def test_format_rejects_non_hex(self):
        assert format_traceparent("not-hex", VALID_SPAN_ID) is None
        assert format_traceparent(VALID_TRACE_ID, "00000000-0000") is None
        assert format_traceparent("0" * 32, VALID_SPAN_ID) is None

    def test_parse_reports_trace_flags(self):
        # The raw trace-flags byte must be recoverable so it can be carried
        # through extract -> inject. A not-sampled (`-00`) inbound trace must not
        # be silently upgraded to sampled.
        assert parse_traceparent(f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01").trace_flags == "01"
        assert parse_traceparent(f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-00").trace_flags == "00"

    def test_format_preserves_flags_round_trip(self):
        # Re-emitting an inbound not-sampled trace must stay `-00`, not flip to
        # `-01`. Without carrying the flags through, a mid-chain service would
        # override the upstream sampling decision.
        parsed = parse_traceparent(f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-00")
        tp = format_traceparent(VALID_TRACE_ID, VALID_SPAN_ID, parsed.trace_flags)
        assert tp.endswith("-00")
        assert parse_traceparent(tp).trace_flags == "00"

    def test_format_defaults_to_sampled(self):
        # With no flags supplied (a trace we originate), default to sampled.
        assert format_traceparent(VALID_TRACE_ID, VALID_SPAN_ID).endswith("-01")

    def test_format_falls_back_on_bad_flags(self):
        # A malformed flags value falls back to the sampled default rather than
        # emitting a malformed traceparent.
        assert format_traceparent(VALID_TRACE_ID, VALID_SPAN_ID, "zz").endswith("-01")


class TestBaggage:
    def test_parse_simple(self):
        assert parse_baggage("braintrust.parent=project_id:abc") == {"braintrust.parent": "project_id:abc"}

    def test_parse_preserves_unrelated_keys(self):
        parsed = parse_baggage("foo=bar,braintrust.parent=project_id:abc,baz=qux")
        assert parsed["foo"] == "bar"
        assert parsed["baz"] == "qux"
        assert parsed["braintrust.parent"] == "project_id:abc"

    def test_parse_ignores_properties(self):
        assert parse_baggage("k=v;prop=1") == {"k": "v"}

    @pytest.mark.parametrize("value", ["", None, "no-equals", ",,,"])
    def test_parse_malformed_does_not_raise(self, value):
        assert parse_baggage(value) == {}

    def test_parse_oversized_does_not_raise(self):
        big = "x=" + ("a" * 100000)
        # Must not raise; returns something bounded. A single member larger than
        # the limit has no complete member to keep, so it is dropped entirely
        # (we never decode a partial value).
        assert parse_baggage(big) == {}

    def test_parse_oversized_keeps_whole_members_only(self):
        # An oversized header is cut back to the last complete list-member; the
        # member straddling the limit is dropped rather than parsed mid-value.
        # Build members that comfortably exceed 8192 bytes total.
        members = [f"k{i}={'v' * 100}" for i in range(200)]
        header = ",".join(members)
        parsed = parse_baggage(header)
        # Some leading members survive, and every surviving value is intact
        # (length 100), i.e. nothing was truncated mid-value.
        assert parsed
        assert all(v == "v" * 100 for v in parsed.values())
        # The kept members are a prefix of the input, never a partial tail one.
        kept_keys = list(parsed.keys())
        assert kept_keys == [f"k{i}" for i in range(len(kept_keys))]

    def test_parse_caps_member_count(self):
        # W3C §3.3.2 Condition 1: at most 64 list-members. A header with many
        # small members (well under the byte limit) is still capped to 64,
        # keeping the leading prefix.
        header = ",".join(f"k{i}=v" for i in range(200))
        parsed = parse_baggage(header)
        assert len(parsed) == 64
        assert list(parsed.keys()) == [f"k{i}" for i in range(64)]

    @pytest.mark.parametrize("count,expected", [(64, 64), (65, 64), (10, 10)])
    def test_parse_member_count_boundary(self, count, expected):
        header = ",".join(f"k{i}=v" for i in range(count))
        assert len(parse_baggage(header)) == expected

    def test_parse_decodes_standard_encoder_values(self):
        # Per the W3C Baggage spec, values are percent-encoded, and standard
        # encoders (e.g. OpenTelemetry's propagator) percent-encode `:` as
        # `%3A`. We must fully decode inbound values to interoperate, otherwise
        # `braintrust.parent=project_id%3Aabc` is not recognized downstream.
        assert parse_baggage(f"{BRAINTRUST_PARENT_KEY}=project_id%3Aabc123") == {
            BRAINTRUST_PARENT_KEY: "project_id:abc123"
        }


class TestMergeBaggage:
    def test_adds_braintrust_parent_when_no_existing(self):
        merged = merge_baggage(None, "project_id:abc")
        assert parse_baggage(merged) == {BRAINTRUST_PARENT_KEY: "project_id:abc"}

    def test_none_parent_and_no_existing_returns_none(self):
        assert merge_baggage(None, None) is None
        assert merge_baggage("", None) is None

    def test_preserves_unrelated_baggage_byte_for_byte(self):
        # An upstream vendor may percent-encode octets outside the small set
        # Braintrust itself encodes (e.g. `/` as `%2F`). Forwarding such baggage
        # must be a transparent relay: the raw member is forwarded unchanged, so
        # we never silently rewrite another vendor's value (`path=a%2Fb` must not
        # become `path=a/b`).
        merged = merge_baggage("path=a%2Fb,user=alice", "project_id:abc")
        assert "path=a%2Fb" in merged
        assert "user=alice" in merged
        # Our own value is percent-encoded for spec compliance (`:` -> `%3A`).
        assert f"{BRAINTRUST_PARENT_KEY}=project_id%3Aabc" in merged
        # The encoded value still decodes back to the original on receive.
        assert parse_baggage(merged)[BRAINTRUST_PARENT_KEY] == "project_id:abc"

    def test_does_not_decode_unowned_percent_sequences(self):
        # `%41` is the percent-encoding of `A`. A transparent relay must not
        # collapse `a%41b` to `aAb`; the inbound wire form is forwarded unchanged.
        merged = merge_baggage("k=a%41b", None)
        assert merged == "k=a%41b"

    @pytest.mark.parametrize(
        "value",
        [
            "a%2Fb",  # `/` outside our encode set
            "x%3Ay",  # `:` (what OTel encodes)
            "c%2Cd",  # encoded comma
            "a%3Db",  # encoded `=`
            "%C3%A9",  # multi-byte UTF-8 (é) already percent-encoded
            "%2520",  # a literal `%20` the upstream double-encoded
        ],
    )
    def test_unowned_value_encodings_pass_through_verbatim(self, value):
        # Whatever percent-encoding an upstream vendor chose for its own value
        # must survive byte-for-byte; we are a relay for keys we do not own.
        merged = merge_baggage(f"vendor={value}", None)
        assert merged == f"vendor={value}"

    def test_multiple_unowned_members_pass_through_verbatim(self):
        # Several unrelated members, each with a different encoding, are all
        # forwarded unchanged and in order, with braintrust.parent appended.
        inbound = "p1=a%2Fb,p2=x%3Ay,p3=c%2Cd"
        merged = merge_baggage(inbound, "project_id:p")
        # Unowned members forward verbatim; our own value is percent-encoded.
        assert merged == f"{inbound},{BRAINTRUST_PARENT_KEY}=project_id%3Ap"

    def test_preserves_member_properties(self):
        # W3C baggage members may carry `;`-delimited properties. Unlike
        # parse_baggage (which drops them), the relay forwards the full member
        # verbatim, properties included.
        merged = merge_baggage("k=v;meta=1;ttl=30,vendor=y", None)
        assert merged == "k=v;meta=1;ttl=30,vendor=y"

    def test_empty_value_member_is_preserved(self):
        merged = merge_baggage("k=,vendor=y", None)
        assert merged == "k=,vendor=y"

    def test_optional_whitespace_is_trimmed(self):
        # W3C treats whitespace around list members as insignificant, so
        # trimming it on forward is lossless. Pin the behavior so it is not
        # mistaken for a pass-through bug.
        merged = merge_baggage(" a=1 , b=2 ", "project_id:p")
        assert merged == f"a=1,b=2,{BRAINTRUST_PARENT_KEY}=project_id%3Ap"

    def test_replaces_existing_braintrust_parent(self):
        # A stale inbound braintrust.parent must be dropped in favor of the
        # value we supply, not duplicated.
        merged = merge_baggage(
            f"{BRAINTRUST_PARENT_KEY}=project_id:old,vendor=x",
            "project_id:new",
        )
        parsed = parse_baggage(merged)
        assert parsed[BRAINTRUST_PARENT_KEY] == "project_id:new"
        assert parsed["vendor"] == "x"
        # Only one braintrust.parent member is emitted.
        assert merged.count(f"{BRAINTRUST_PARENT_KEY}=") == 1

    def test_drops_existing_braintrust_parent_when_no_new_value(self):
        # If we have no braintrust.parent to add, an inbound one is still
        # consumed (it is ours to own) rather than forwarded raw.
        merged = merge_baggage(f"{BRAINTRUST_PARENT_KEY}=project_id:old,vendor=x", None)
        assert merged == "vendor=x"

    def test_encodes_braintrust_parent_with_reserved_chars(self):
        # Braintrust owns its braintrust.parent value, which can carry an
        # arbitrary project_name containing reserved characters. That value must
        # be encoded on emit and decode back cleanly.
        merged = merge_baggage(None, "project_name:a,b c")
        assert parse_baggage(merged) == {BRAINTRUST_PARENT_KEY: "project_name:a,b c"}

    # The braintrust.parent value embeds a user-controlled project/experiment
    # name, so it can contain any character. Per W3C Baggage §3.3.1.3, a value's
    # unencoded bytes are restricted to:
    #   baggage-octet = %x21 / %x23-2B / %x2D-3A / %x3C-5B / %x5D-7E
    # Everything else (DQUOTE, backslash, comma, semicolon, space, controls, the
    # percent sign, and all non-ASCII) MUST be percent-encoded. These cases run
    # through the real SDK encode (merge_baggage) and decode (parse_baggage)
    # paths and assert both a clean round trip and that the emitted member
    # carries no raw baggage-octet violator on the wire.
    @pytest.mark.parametrize(
        "name",
        [
            'a"b',  # DQUOTE (%x22) - outside baggage-octet
            "a\\b",  # backslash (%x5C) - outside baggage-octet
            "a\tb",  # tab - a control character
            "a\nb",  # newline - a control character
            "abcd\u00e9",  # non-ASCII (e) -> multi-byte UTF-8
            "emoji\U0001f600",  # non-ASCII astral plane -> multi-byte UTF-8
            "a+b",  # literal plus must stay a plus (not become space)
            "a%b",  # literal percent (%x25) MUST be encoded
            "a,b c",  # comma + space (the pre-existing covered set)
            "a;b=c",  # semicolon + equals
        ],
    )
    def test_braintrust_parent_value_round_trips_spec_compliant(self, name):
        value = f"project_name:{name}"
        merged = merge_baggage(None, value)

        # Round-trip through the same decode path the SDK uses on receive.
        assert parse_baggage(merged) == {BRAINTRUST_PARENT_KEY: value}

        # The emitted header must be valid on the wire: ASCII only, and the
        # braintrust.parent member must contain no raw baggage-octet violators.
        assert merged.isascii()
        member = next(m for m in merged.split(",") if m.startswith(f"{BRAINTRUST_PARENT_KEY}="))
        encoded_val = member.partition("=")[2]
        for ch in encoded_val:
            codepoint = ord(ch)
            allowed = (
                codepoint == 0x21
                or 0x23 <= codepoint <= 0x2B
                or 0x2D <= codepoint <= 0x3A
                or 0x3C <= codepoint <= 0x5B
                or 0x5D <= codepoint <= 0x7E
            )
            assert allowed, f"emitted byte {ch!r} (U+{codepoint:04X}) is not a valid baggage-octet"

    def test_skips_malformed_existing_members(self):
        merged = merge_baggage("garbage,,k=v,no-equals", "project_id:abc")
        parsed = parse_baggage(merged)
        assert parsed["k"] == "v"
        assert parsed[BRAINTRUST_PARENT_KEY] == "project_id:abc"

    def test_oversized_existing_relays_whole_members_only(self):
        # An oversized inbound header must not forward a partial list-member
        # (W3C §3.3.2). The relayed members are cut back to the last complete
        # one, and our braintrust.parent is still appended.
        members = [f"k{i}={'v' * 100}" for i in range(200)]
        existing = ",".join(members)
        merged = merge_baggage(existing, "project_id:abc")
        parsed = parse_baggage(merged)
        # The emitted header stays within the W3C 8192-byte limit.
        assert len(merged.encode("utf-8")) <= 8192
        # Our value is prioritized and always present...
        assert parsed[BRAINTRUST_PARENT_KEY] == "project_id:abc"
        # ...and every relayed member kept its full, intact value.
        relayed = {k: v for k, v in parsed.items() if k != BRAINTRUST_PARENT_KEY}
        assert relayed
        assert all(v == "v" * 100 for v in relayed.values())
        # No raw member in the emitted header is a partial (mid-value) cut: each
        # forwarded member round-trips to a length-100 value.
        for member in merged.split(","):
            if member.startswith(f"{BRAINTRUST_PARENT_KEY}="):
                continue
            assert member.endswith("v" * 100)

    def test_caps_member_count_and_reserves_slot_for_braintrust_parent(self):
        # W3C §3.3.2 Condition 1: at most 64 list-members. With many small
        # inbound members (well under the byte limit), the merged result must
        # still be <= 64 members total, and our braintrust.parent reserves one
        # of those slots so it always survives (63 relayed + 1 ours).
        existing = ",".join(f"k{i}=v" for i in range(200))
        merged = merge_baggage(existing, "project_id:abc")
        parsed = parse_baggage(merged)
        assert merged.count(",") + 1 == 64
        assert len(parsed) == 64
        assert parsed[BRAINTRUST_PARENT_KEY] == "project_id:abc"
        # The relayed members are the leading 63, in order.
        relayed_keys = [k for k in parsed if k != BRAINTRUST_PARENT_KEY]
        assert relayed_keys == [f"k{i}" for i in range(63)]

    def test_member_count_cap_without_braintrust_parent(self):
        # With no braintrust.parent to add, all 64 slots are available for relay.
        existing = ",".join(f"k{i}=v" for i in range(200))
        merged = merge_baggage(existing, None)
        parsed = parse_baggage(merged)
        assert len(parsed) == 64
        assert list(parsed.keys()) == [f"k{i}" for i in range(64)]

    def test_under_member_limit_keeps_all(self):
        # A modest inbound list (under both limits) is relayed in full, plus ours.
        existing = ",".join(f"k{i}=v" for i in range(10))
        merged = merge_baggage(existing, "project_id:abc")
        parsed = parse_baggage(merged)
        assert len(parsed) == 11
        assert parsed[BRAINTRUST_PARENT_KEY] == "project_id:abc"


def test_get_header_case_insensitive():
    headers = {"TraceParent": VALID_TRACEPARENT, "BAGGAGE": "foo=bar"}
    assert get_header(headers, "traceparent") == VALID_TRACEPARENT
    assert get_header(headers, "baggage") == "foo=bar"
    assert get_header(headers, "missing") is None


# --------------------------------------------------------------------------- #
# Send: header injection
# --------------------------------------------------------------------------- #


@pytest.fixture
def memory_and_logger():
    with _internal_with_memory_background_logger() as mem:
        logger = init_test_logger("propagation-test")
        yield mem, logger


class TestInject:
    def test_traceparent_well_formed_and_matches_span(self, memory_and_logger):
        _mem, logger = memory_and_logger
        with logger.start_span(name="svc_a") as span:
            carrier = span.inject({})

        tp = carrier[TRACEPARENT_HEADER]
        assert TRACEPARENT_RE.match(tp)
        trace_id, span_id, _flags = parse_traceparent(tp)
        assert trace_id == span.root_span_id
        assert span_id == span.span_id

    def test_baggage_contains_braintrust_parent(self, memory_and_logger):
        _mem, logger = memory_and_logger
        with logger.start_span(name="svc_a") as span:
            carrier = span.inject({})

        parsed = parse_baggage(carrier[BAGGAGE_HEADER])
        assert parsed[BRAINTRUST_PARENT_KEY] == "project_name:propagation-test"

    def test_preexisting_baggage_preserved(self, memory_and_logger):
        _mem, logger = memory_and_logger
        with logger.start_span(name="svc_a") as span:
            carrier = span.inject({BAGGAGE_HEADER: "user=alice,team=eng"})

        parsed = parse_baggage(carrier[BAGGAGE_HEADER])
        assert parsed["user"] == "alice"
        assert parsed["team"] == "eng"
        assert parsed[BRAINTRUST_PARENT_KEY] == "project_name:propagation-test"

    def test_title_cased_baggage_emits_single_lowercase_header(self, memory_and_logger):
        # Per W3C (§3.3.1) the header name SHOULD be sent lowercase. A carrier
        # that arrives with a title-cased `Baggage` (e.g. from a framework that
        # normalizes header casing) must be rewritten to a single lowercase
        # `baggage` key, not left with two conflicting case-variants.
        _mem, logger = memory_and_logger
        with logger.start_span(name="svc_a") as span:
            carrier = span.inject({"Baggage": "user=alice"})

        baggage_keys = [k for k in carrier if k.lower() == BAGGAGE_HEADER]
        assert baggage_keys == [BAGGAGE_HEADER]
        parsed = parse_baggage(carrier[BAGGAGE_HEADER])
        assert parsed["user"] == "alice"
        assert parsed[BRAINTRUST_PARENT_KEY] == "project_name:propagation-test"

    def test_title_cased_traceparent_emits_single_lowercase_header(self, memory_and_logger):
        # A pre-existing title-cased `Traceparent` must be replaced by a single
        # lowercase `traceparent` (W3C §3.2.1), with no stale variant remaining.
        _mem, logger = memory_and_logger
        with logger.start_span(name="svc_a") as span:
            carrier = span.inject({"Traceparent": "stale"})

        traceparent_keys = [k for k in carrier if k.lower() == TRACEPARENT_HEADER]
        assert traceparent_keys == [TRACEPARENT_HEADER]
        tp = parse_traceparent(carrier[TRACEPARENT_HEADER])
        assert (tp.trace_id, tp.span_id) == (span.root_span_id, span.span_id)

    def test_never_emits_x_bt_parent(self, memory_and_logger):
        _mem, logger = memory_and_logger
        with logger.start_span(name="svc_a") as span:
            carrier = span.inject({})
        assert "x-bt-parent" not in {k.lower() for k in carrier}

    def test_no_braintrust_parent_injects_traceparent_without_baggage(self):
        # When the Braintrust parent is unknown but the span has W3C-shaped hex
        # ids, traceparent is still injected and braintrust.parent is absent from
        # baggage (rather than emitted empty).
        from braintrust.logger import _inject_into_carrier

        carrier = {}
        _inject_into_carrier(
            carrier,
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
            braintrust_parent=None,
        )
        assert TRACEPARENT_RE.match(carrier[TRACEPARENT_HEADER])
        # No braintrust parent and no pre-existing baggage -> no baggage header.
        assert BAGGAGE_HEADER not in carrier

    def test_no_braintrust_parent_preserves_existing_baggage_without_bt_key(self):
        # With an unknown Braintrust parent, pre-existing baggage is preserved
        # but no (empty) braintrust.parent entry is added.
        from braintrust.logger import _inject_into_carrier

        carrier = {BAGGAGE_HEADER: "user=alice"}
        _inject_into_carrier(
            carrier,
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
            braintrust_parent=None,
        )
        parsed = parse_baggage(carrier[BAGGAGE_HEADER])
        assert parsed["user"] == "alice"
        assert BRAINTRUST_PARENT_KEY not in parsed

    def test_inject_trace_context_free_function(self, memory_and_logger):
        _mem, logger = memory_and_logger
        with logger.start_span(name="svc_a") as span:
            carrier = inject_trace_context()
            tp = parse_traceparent(carrier[TRACEPARENT_HEADER])
            assert (tp.trace_id, tp.span_id) == (span.root_span_id, span.span_id)

    def test_inject_no_current_span_is_safe(self):
        # No active span -> NOOP span -> no traceparent, no raise.
        carrier = inject_trace_context({})
        assert TRACEPARENT_HEADER not in carrier


# --------------------------------------------------------------------------- #
# Receive: header extraction
# --------------------------------------------------------------------------- #


class TestExtract:
    # extract_trace_context returns an opaque W3C-headers dict; behavior is
    # asserted by passing it into start_span and inspecting the resulting span.

    def test_traceparent_with_baggage_parent(self, memory_and_logger):
        _mem, logger = memory_and_logger
        ctx = extract_trace_context(
            {
                "traceparent": VALID_TRACEPARENT,
                "baggage": f"{BRAINTRUST_PARENT_KEY}=project_id:abc123",
            }
        )
        with logger.start_span(name="h", parent=ctx) as span:
            assert span.root_span_id == VALID_TRACE_ID
            assert span.span_parents == [VALID_SPAN_ID]

    def test_traceparent_baggage_with_unrelated_keys(self, memory_and_logger):
        _mem, logger = memory_and_logger
        ctx = extract_trace_context(
            {
                "traceparent": VALID_TRACEPARENT,
                "baggage": f"user=alice,{BRAINTRUST_PARENT_KEY}=project_id:abc,team=eng",
            }
        )
        with logger.start_span(name="h", parent=ctx) as span:
            assert span.root_span_id == VALID_TRACE_ID
            assert span.span_parents == [VALID_SPAN_ID]

    def test_traceparent_no_baggage_uses_current_logger(self, memory_and_logger):
        # No braintrust.parent in baggage -> route under the active logger.
        _mem, logger = memory_and_logger
        ctx = extract_trace_context({"traceparent": VALID_TRACEPARENT})
        assert ctx is not None  # context is still produced; routing resolved at start_span
        with logger.start_span(name="h", parent=ctx) as span:
            assert span.root_span_id == VALID_TRACE_ID
            assert span.span_parents == [VALID_SPAN_ID]

    def test_no_headers_returns_none(self):
        assert extract_trace_context({}) is None
        assert extract_trace_context(None) is None

    def test_malformed_traceparent_returns_none(self):
        # No valid traceparent -> no context (fresh root downstream).
        assert (
            extract_trace_context(
                {
                    "traceparent": "garbage",
                    "baggage": f"{BRAINTRUST_PARENT_KEY}=project_id:abc",
                }
            )
            is None
        )

    def test_case_insensitive_headers(self, memory_and_logger):
        _mem, logger = memory_and_logger
        ctx = extract_trace_context(
            {
                "TraceParent": VALID_TRACEPARENT,
                "Baggage": f"{BRAINTRUST_PARENT_KEY}=project_id:abc",
            }
        )
        with logger.start_span(name="h", parent=ctx) as span:
            assert span.root_span_id == VALID_TRACE_ID
            assert span.span_parents == [VALID_SPAN_ID]

    def test_extract_returns_opaque_dict(self):
        ctx = extract_trace_context(
            {
                "traceparent": VALID_TRACEPARENT,
                "baggage": f"{BRAINTRUST_PARENT_KEY}=project_id:abc",
                "tracestate": "congo=t61",
            }
        )
        assert isinstance(ctx, dict)
        assert ctx[TRACEPARENT_HEADER] == VALID_TRACEPARENT

    def test_no_parent_and_no_logger_starts_fresh_root(self, memory_and_logger):
        # Valid traceparent but no braintrust.parent and (here) routing falls to
        # the active logger; with linkage when a logger is present we still link.
        # When neither baggage nor logger can route, start_span must not raise.
        ctx = extract_trace_context({"traceparent": VALID_TRACEPARENT})
        # No active logger context here would yield a fresh root; with the
        # fixture's logger active, it links. Either way, no exception.
        _mem, logger = memory_and_logger
        with logger.start_span(name="h", parent=ctx) as span:
            assert span.span_id is not None


# --------------------------------------------------------------------------- #
# Round trip + cross-service linking
# --------------------------------------------------------------------------- #


def test_round_trip_inject_extract(memory_and_logger):
    _mem, logger = memory_and_logger
    with logger.start_span(name="svc_a") as span_a:
        carrier = span_a.inject({})
        a_root, a_span = span_a.root_span_id, span_a.span_id

    parent = extract_trace_context(carrier)
    with logger.start_span(name="svc_b", parent=parent) as span_b:
        assert span_b.root_span_id == a_root
        assert span_b.span_parents == [a_span]


def test_cross_service_linking(memory_and_logger):
    _mem, logger = memory_and_logger
    with logger.start_span(name="svc_a") as span_a:
        carrier = span_a.inject({})
        a_root, a_span = span_a.root_span_id, span_a.span_id

    parent = extract_trace_context(carrier)
    with logger.start_span(name="svc_b", parent=parent) as span_b:
        assert span_b.root_span_id == a_root
        assert span_b.span_parents == [a_span]


def test_inject_does_not_break_span_emission_without_parent():
    # Inject with an unknown braintrust parent must not drop the span.
    with _internal_with_memory_background_logger() as mem:
        logger = init_test_logger("emit-test")
        with logger.start_span(name="svc_a") as span:
            span.inject({})
            span.log(output="hello")
        logger.flush()
        spans = mem.pop()
        assert any(s.get("output") == "hello" for s in spans)


def test_legacy_export_slug_round_trips_with_hex_ids(memory_and_logger):
    # The pre-existing span.export() + start_span(parent=slug) pattern must keep
    # working with the new default hex IDs (8-byte span id, 16-byte trace id).
    _mem, logger = memory_and_logger
    with logger.start_span(name="parent") as parent:
        slug = parent.export()
        p_root, p_span = parent.root_span_id, parent.span_id

    # Sanity: ids are OTEL-shaped hex.
    assert len(p_span) == 16
    assert len(p_root) == 32

    with logger.start_span(name="child", parent=slug) as child:
        assert child.root_span_id == p_root
        assert child.span_parents == [p_span]


def test_inject_noops_in_legacy_uuid_mode():
    # In legacy UUID mode, span ids aren't W3C-shaped, so inject must not write
    # traceparent/baggage and must leave pre-existing headers untouched.
    import os

    from braintrust.test_helpers import preserve_env_vars

    with preserve_env_vars("BRAINTRUST_OTEL_COMPAT", "BRAINTRUST_LEGACY_IDS"):
        os.environ.pop("BRAINTRUST_OTEL_COMPAT", None)
        os.environ["BRAINTRUST_LEGACY_IDS"] = "true"
        with _internal_with_memory_background_logger():
            logger = init_test_logger("legacy-inject")
            with logger.start_span(name="p") as span:
                # Legacy spans use UUID ids (share root == span).
                assert len(span.span_id) == 36
                carrier = span.inject({"existing": "header"})

    assert carrier == {"existing": "header"}
    assert TRACEPARENT_HEADER not in carrier
    assert BAGGAGE_HEADER not in carrier


def test_legacy_parent_slug_linked_in_hex_mode():
    # Back-compat: in hex mode (default), a parent slug carrying legacy UUID ids
    # is still a usable parent. The child links to the slug's span/root ids even
    # though they are a different format from the child's own (hex) span id. This
    # keeps `start_span(parent=<slug>)` working across an SDK upgrade where an
    # older (UUID) sender's slug reaches a newer (hex) receiver.
    import uuid

    from braintrust.span_identifier_v3 import SpanComponentsV3

    p_span = str(uuid.uuid4())
    p_root = str(uuid.uuid4())
    legacy_slug = SpanComponentsV3(
        object_type=SpanObjectTypeV3.PROJECT_LOGS,
        object_id="legacy-proj",
        row_id=str(uuid.uuid4()),
        span_id=p_span,
        root_span_id=p_root,
    ).to_str()

    with _internal_with_memory_background_logger():
        logger = init_test_logger("legacy-proj")
        with logger.start_span(name="child", parent=legacy_slug) as child:
            # Links to the slug's UUID ids; the child's own span id stays hex.
            assert child.root_span_id == p_root
            assert child.span_parents == [p_span]
            assert len(child.span_id) == 16


def test_legacy_parent_slug_linked_in_hex_mode_toplevel_start_span():
    # Same as above, but through the module-level `start_span`, which resolves
    # the parent slug independently of `Logger.start_span`.
    import uuid

    from braintrust.logger import start_span
    from braintrust.span_identifier_v3 import SpanComponentsV3

    p_span = str(uuid.uuid4())
    p_root = str(uuid.uuid4())
    legacy_slug = SpanComponentsV3(
        object_type=SpanObjectTypeV3.PROJECT_LOGS,
        object_id="legacy-proj",
        row_id=str(uuid.uuid4()),
        span_id=p_span,
        root_span_id=p_root,
    ).to_str()

    with _internal_with_memory_background_logger():
        init_test_logger("legacy-proj")
        with start_span(name="child", parent=legacy_slug) as child:
            assert child.root_span_id == p_root
            assert child.span_parents == [p_span]
            assert len(child.span_id) == 16


def test_legacy_parent_slug_linked_in_legacy_mode():
    # In legacy UUID mode, a parent slug carrying UUID ids links to the slug's
    # span/root ids (same format).
    import os
    import uuid

    from braintrust.span_identifier_v3 import SpanComponentsV3
    from braintrust.test_helpers import preserve_env_vars

    p_span = str(uuid.uuid4())
    p_root = str(uuid.uuid4())
    legacy_slug = SpanComponentsV3(
        object_type=SpanObjectTypeV3.PROJECT_LOGS,
        object_id="legacy-proj",
        row_id=str(uuid.uuid4()),
        span_id=p_span,
        root_span_id=p_root,
    ).to_str()

    with preserve_env_vars("BRAINTRUST_OTEL_COMPAT", "BRAINTRUST_LEGACY_IDS"):
        os.environ.pop("BRAINTRUST_OTEL_COMPAT", None)
        os.environ["BRAINTRUST_LEGACY_IDS"] = "true"
        with _internal_with_memory_background_logger():
            logger = init_test_logger("legacy-proj")
            with logger.start_span(name="child", parent=legacy_slug) as child:
                assert child.root_span_id == p_root
                assert child.span_parents == [p_span]


def test_hex_parent_slug_linked_in_legacy_mode():
    # Back-compat (reverse direction): in legacy UUID mode, a parent slug
    # carrying hex ids still links to the slug's span/root ids. The child's own
    # span id stays in the active (UUID) format.
    import os

    from braintrust.span_identifier_v4 import SpanComponentsV4
    from braintrust.test_helpers import preserve_env_vars

    p_span = "00f067aa0ba902b7"  # 8-byte hex
    p_root = "4bf92f3577b34da6a3ce929d0e0e4736"  # 16-byte hex
    hex_slug = SpanComponentsV4(
        object_type=SpanObjectTypeV3.PROJECT_LOGS,
        object_id="legacy-proj",
        row_id="bt-row",
        span_id=p_span,
        root_span_id=p_root,
    ).to_str()

    with preserve_env_vars("BRAINTRUST_OTEL_COMPAT", "BRAINTRUST_LEGACY_IDS"):
        os.environ.pop("BRAINTRUST_OTEL_COMPAT", None)
        os.environ["BRAINTRUST_LEGACY_IDS"] = "true"
        with _internal_with_memory_background_logger():
            logger = init_test_logger("legacy-proj")
            with logger.start_span(name="child", parent=hex_slug) as child:
                # Links to the slug's hex ids; the child's own span id stays UUID.
                assert child.root_span_id == p_root
                assert child.span_parents == [p_span]
                assert len(child.span_id) == 36


# --------------------------------------------------------------------------- #
# tracestate pass-through (W3C: forward upstream vendor state)
# --------------------------------------------------------------------------- #

UPSTREAM_TRACESTATE = "congo=t61rcWkgMzE,rojo=00f067aa0ba902b7"


class TestTracestate:
    def test_extract_then_inject_forwards_tracestate(self, memory_and_logger):
        # Pass-through: a span started from inbound headers carrying tracestate
        # forwards that tracestate unchanged when it later injects.
        _mem, logger = memory_and_logger
        inbound = {
            "traceparent": VALID_TRACEPARENT,
            "baggage": f"{BRAINTRUST_PARENT_KEY}=project_id:abc",
            "tracestate": UPSTREAM_TRACESTATE,
        }
        parent = extract_trace_context(inbound)
        with logger.start_span(name="mid", parent=parent) as span:
            outbound = span.inject({})
        assert outbound[TRACESTATE_HEADER] == UPSTREAM_TRACESTATE
        assert TRACEPARENT_RE.match(outbound[TRACEPARENT_HEADER])

    def test_no_tracestate_emitted_when_none_inbound(self, memory_and_logger):
        # A trace we originate (or that arrives without tracestate) emits none.
        _mem, logger = memory_and_logger
        with logger.start_span(name="root") as span:
            outbound = span.inject({})
        assert TRACESTATE_HEADER not in outbound

    def test_extract_then_inject_preserves_unsampled_flag(self, memory_and_logger):
        # A span started from inbound headers carrying a not-sampled (`-00`)
        # traceparent must re-emit `-00` when it injects, not upgrade it to `-01`.
        # Otherwise a mid-chain Braintrust service overrides the upstream
        # sampling decision for everything downstream.
        _mem, logger = memory_and_logger
        unsampled = f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-00"
        inbound = {
            "traceparent": unsampled,
            "baggage": f"{BRAINTRUST_PARENT_KEY}=project_id:abc",
        }
        parent = extract_trace_context(inbound)
        with logger.start_span(name="mid", parent=parent) as span:
            outbound = span.inject({})
        assert outbound[TRACEPARENT_HEADER].endswith("-00")

    def test_extract_then_inject_preserves_sampled_flag(self, memory_and_logger):
        # The sampled (`-01`) case must also round-trip unchanged.
        _mem, logger = memory_and_logger
        inbound = {
            "traceparent": VALID_TRACEPARENT,  # ...-01
            "baggage": f"{BRAINTRUST_PARENT_KEY}=project_id:abc",
        }
        parent = extract_trace_context(inbound)
        with logger.start_span(name="mid", parent=parent) as span:
            outbound = span.inject({})
        assert outbound[TRACEPARENT_HEADER].endswith("-01")
