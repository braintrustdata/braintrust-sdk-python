"""Internal safety bounds; recording policy remains configurable in RecordingOptions."""

# Leave room for output writes that arrive behind the input sample clock.
SEGMENT_HEADROOM_MS = 1000
# One encoding job and one queued segment; a slow exporter cannot grow a backlog.
MAX_SEGMENT_JOBS = 2
# Independent object-count guard for unusually small PCM packets in an active buffer.
MAX_BUFFERED_PACKETS = 100000
# Bound unresolved alignment work, not the lifetime number of turns or frames.
MAX_PENDING_RANGES = 32000
MAX_OUTPUT_CONTEXTS = 16000
