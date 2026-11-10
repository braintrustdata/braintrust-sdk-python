"""Session-relative time for frameworks that stamp events with their own clock."""

from collections.abc import Iterator
from contextlib import contextmanager

from .session import AudioSessionRecorder


class SessionClock:
    """Milliseconds on the session audio timeline.

    Frameworks stamp events with their own monotonic clock. The first stamped
    event fixes the offset between that clock and the session timeline. While
    an event is handled, "now" is pinned to its stamp, so every operation the
    event opens or closes shares one instant. Outside events, "now" is live.

    Events nest: a nested event restores the outer instant on exit. A clock
    tracks one event at a time, so it must not be shared by concurrent tasks;
    give each concurrently handled stream its own clock or serialize events.
    """

    def __init__(self, recorder: AudioSessionRecorder) -> None:
        self._recorder = recorder
        self._offset_ns: int | None = None
        self._event_ns: int | None = None

    @contextmanager
    def event(self, timestamp_ns: int) -> Iterator[None]:
        if self._offset_ns is None:
            self._offset_ns = self._recorder.elapsed_ns() - timestamp_ns
        outer, self._event_ns = self._event_ns, max(0, timestamp_ns + self._offset_ns)
        try:
            yield
        finally:
            self._event_ns = outer

    def now_ns(self) -> int:
        return self._recorder.elapsed_ns() if self._event_ns is None else self._event_ns

    def now_ms(self) -> float:
        return self.now_ns() / 1_000_000

    def wall(self, offset_ms: float | None = None) -> float:
        """Unix seconds for a session offset (default: now)."""
        offset = self.now_ms() if offset_ms is None else offset_ms
        return self._recorder.to_wall(round(offset * 1_000_000))

    def from_wall_ms(self, seconds: float) -> float:
        """Session offset for Unix seconds, clamped to the session start."""
        return max(0, self._recorder.from_wall(seconds) / 1_000_000)
