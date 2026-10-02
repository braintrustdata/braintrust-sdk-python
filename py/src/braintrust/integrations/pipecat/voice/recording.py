"""Pipecat transport capture hooks."""

import time


def capture_transport_output(output, recording):
    """Observe only successful writes and preserve the transport's pacing/result."""
    original = output.write_audio_frame

    async def write(frame):
        if recording.reason:
            return await original(frame)
        observed_ns = time.monotonic_ns()
        result = await original(frame)
        if result:
            try:
                recording.capture(1, frame.audio, frame.sample_rate, frame.num_channels, observed_ns=observed_ns)
            except Exception:  # noqa: BLE001 - capture/export failures must not break the call
                recording.omit("capture_error")
        return result

    output.write_audio_frame = write
