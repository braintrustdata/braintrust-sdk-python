import os
import uuid

import pytest
from braintrust import id_gen


@pytest.fixture(autouse=True)
def reset_id_generator_state():
    """Reset ID generator state and environment variables before each test"""
    original_otel = os.getenv("BRAINTRUST_OTEL_COMPAT")
    original_legacy = os.getenv("BRAINTRUST_LEGACY_IDS")

    try:
        yield
    finally:
        os.environ.pop("BRAINTRUST_OTEL_COMPAT", None)
        os.environ.pop("BRAINTRUST_LEGACY_IDS", None)
        if original_otel:
            os.environ["BRAINTRUST_OTEL_COMPAT"] = original_otel
        if original_legacy:
            os.environ["BRAINTRUST_LEGACY_IDS"] = original_legacy


def test_uuid_generator():
    """Test that UUIDGenerator implements IDGenerator interface and generates valid UUIDs"""
    # Test interface implementation
    generator = id_gen.UUIDGenerator()

    # Test that UUID generators should share root_span_id for backwards compatibility
    assert generator.share_root_span_id() == True

    for gen_func in [generator.get_span_id, generator.get_trace_id]:
        ids = gen_func(), gen_func()
        assert ids[0] != ids[1]
        assert all(isinstance(id_val, str) for id_val in ids)
        assert all(uuid.UUID(id_val) for id_val in ids)


def test_otel_id_generator():
    generator = id_gen.OTELIDGenerator()

    # Test that OTEL generators should not share root_span_id
    assert generator.share_root_span_id() == False

    # Test ID generation with regular loops
    test_cases = [
        (generator.get_span_id, 16),
        (generator.get_trace_id, 32),
    ]
    for gen_func, expected_length in test_cases:
        id1 = gen_func()
        id2 = gen_func()
        # Test uniqueness, type, length, and hex format
        assert id1 != id2
        assert len(id1) == len(id2) == expected_length
        assert _is_hex(id1)
        assert _is_hex(id2)


def test_id_get_env_var(reset_id_generator_state):
    # By default (no env vars) the generator produces OTEL-compatible hex IDs.
    # BRAINTRUST_OTEL_COMPAT always implies hex; BRAINTRUST_LEGACY_IDS opts
    # back into UUIDs (only when OTEL_COMPAT is off).
    cases = [
        (None, lambda _id: _assert_is_hex(_id)),
        ("true", lambda _id: _assert_is_hex(_id)),
        ("True", lambda _id: _assert_is_hex(_id)),
        ("TRUE", lambda _id: _assert_is_hex(_id)),
        ("false", lambda _id: _assert_is_hex(_id)),
        ("False", lambda _id: _assert_is_hex(_id)),
    ]

    for env_var_value, assert_func in cases:
        os.environ.pop("BRAINTRUST_OTEL_COMPAT", None)
        os.environ.pop("BRAINTRUST_LEGACY_IDS", None)
        if env_var_value is not None:
            os.environ["BRAINTRUST_OTEL_COMPAT"] = env_var_value
        generator = id_gen.get_id_generator()
        assert_func(generator.get_span_id())
        assert_func(generator.get_trace_id())


def test_id_get_env_var_legacy_uuid(reset_id_generator_state):
    # BRAINTRUST_LEGACY_IDS opts back into UUID IDs.
    cases = [
        ("true", lambda _id: uuid.UUID(_id)),
        ("True", lambda _id: uuid.UUID(_id)),
        ("false", lambda _id: _assert_is_hex(_id)),
    ]

    for env_var_value, assert_func in cases:
        os.environ.pop("BRAINTRUST_OTEL_COMPAT", None)
        os.environ.pop("BRAINTRUST_LEGACY_IDS", None)
        os.environ["BRAINTRUST_LEGACY_IDS"] = env_var_value
        generator = id_gen.get_id_generator()
        assert_func(generator.get_span_id())
        assert_func(generator.get_trace_id())


def test_otel_compat_wins_over_legacy_uuid(reset_id_generator_state):
    # When both are set, OTEL_COMPAT wins: hex IDs are used (no raise).
    os.environ["BRAINTRUST_OTEL_COMPAT"] = "true"
    os.environ["BRAINTRUST_LEGACY_IDS"] = "true"
    generator = id_gen.get_id_generator()
    assert generator.share_root_span_id() is False
    _assert_is_hex(generator.get_span_id())


def _is_hex(s):
    return all(c in "0123456789abcdef" for c in s.lower())


def _assert_is_hex(x):
    assert _is_hex(x)
