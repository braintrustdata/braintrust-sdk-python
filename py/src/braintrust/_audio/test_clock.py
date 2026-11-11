import pytest
from braintrust import logger
from braintrust._audio.clock import SessionClock
from braintrust._audio.session import AudioSessionRecorder
from braintrust.test_helpers import init_test_logger


@pytest.fixture(autouse=True)
def _restore_logger_metadata(monkeypatch):
    # init_test_logger replaces this global; restore it after each test.
    monkeypatch.setattr(logger, "_compute_logger_metadata", logger._compute_logger_metadata)


@pytest.fixture
def clock():
    init_test_logger("audio-tests")
    with logger._internal_with_memory_background_logger():
        yield SessionClock(AudioSessionRecorder(logger.start_span(name="session")))


def test_events_pin_now_to_framework_timestamps_relative_to_the_first(clock):
    with clock.event(5_000_000_000):
        first = clock.now_ms()
    with clock.event(5_250_000_000):
        assert clock.now_ms() == pytest.approx(first + 250)
        # Every reading within one event shares its instant.
        assert clock.now_ms() == clock.now_ms()
        assert clock.wall() - clock.wall(first) == pytest.approx(0.25, abs=1e-6)


def test_nested_events_restore_the_outer_instant(clock):
    with clock.event(1_000_000_000):
        outer = clock.now_ns()
        with clock.event(1_500_000_000):
            assert clock.now_ns() == outer + 500_000_000
        assert clock.now_ns() == outer


def test_wall_conversion_round_trips_and_clamps_to_session_start(clock):
    origin = clock.wall(0)
    assert clock.from_wall_ms(origin + 1.5) == pytest.approx(1500, abs=1e-3)
    assert clock.from_wall_ms(origin - 10) == 0


def test_outside_events_now_is_live(clock):
    before = clock.now_ms()
    assert clock.now_ms() >= before >= 0
