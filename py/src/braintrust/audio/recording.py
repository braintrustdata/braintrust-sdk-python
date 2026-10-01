"""Bounded PCM capture; mixing and encoding are called only from worker threads."""

import io
import math
import time
from dataclasses import dataclass

from .budget import source_budget


@dataclass(frozen=True)
class Packet:
    channel: int
    start_ms: float
    pcm: bytes
    sample_rate: int
    channels: int


def encode_audio(chunks, sample_rate, channels, audio_format="ogg"):
    import numpy as np

    started = time.perf_counter()
    cpu_started = time.thread_time()
    pcm = b"".join(chunks)
    samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels)
    result = _encode(samples, sample_rate, audio_format)
    result["processing"] = {
        "wall_ms": (time.perf_counter() - started) * 1000,
        "thread_cpu_ms": (time.thread_time() - cpu_started) * 1000,
        "source_bytes": len(pcm),
        "encoded_bytes": len(result["data"]),
    }
    return result


def _encode(samples, sample_rate, audio_format):
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("Install braintrust[audio] to encode voice recordings") from error

    destination = io.BytesIO()
    if audio_format == "ogg":
        sf.write(destination, samples, sample_rate, format="OGG", subtype="OPUS")
    elif audio_format == "wav":
        sf.write(destination, samples, sample_rate, format="WAV", subtype="PCM_16")
    else:
        raise ValueError("AUDIO_FORMAT must be ogg or wav")
    return {
        "data": destination.getvalue(),
        "duration_ms": len(samples) / sample_rate * 1000,
        "channel_count": samples.shape[1],
        "mime_type": "audio/ogg" if audio_format == "ogg" else "audio/wav",
        "extension": audio_format,
    }


class CallRecording:
    def __init__(
        self,
        *,
        enabled=True,
        max_bytes=8 * 1024 * 1024,
        max_duration_ms=120000,
        max_packets=16000,
    ):
        self.enabled = enabled
        self.sealed = False
        self.max_bytes = max_bytes
        self.max_duration_ms = max_duration_ms
        self.max_packets = max_packets
        self.packets = []
        self.bytes = 0
        self.origin_ns = None
        self.origin_unix_ms = None
        self.ends = [0.0, 0.0]
        self.started = [False, False]
        self.reason = None if enabled else "disabled"

    def __del__(self):
        if getattr(self, "bytes", 0):
            self.clear()

    def seal(self):
        self.sealed = True

    def omit(self, reason):
        self.reason = reason
        self.clear()

    def clear(self):
        source_budget.release(self.bytes)
        self.bytes = 0
        self.packets.clear()

    def capture(self, channel, pcm, sample_rate, channels, *, observed_ns=None, observed_unix_ms=None):
        if self.reason or self.sealed:
            return
        if channel not in (0, 1) or channels not in (1, 2) or not 8000 <= sample_rate <= 48000:
            self.omit("unsupported_audio_format")
            return
        if len(pcm) % (2 * channels):
            self.omit("invalid_pcm_frame")
            return
        if not pcm:
            return
        observed_ns = time.monotonic_ns() if observed_ns is None else observed_ns
        if self.origin_ns is None:
            self.origin_ns = observed_ns
            self.origin_unix_ms = time.time() * 1000 if observed_unix_ms is None else observed_unix_ms
        # This transport delivers continuous input, including silent samples.
        # Anchor its first packet, then use sample duration rather than arrival
        # jitter. Output is intermittent, so preserve pauses between writes.
        start = (
            self.ends[channel]
            if channel == 0 and self.started[channel]
            else max((observed_ns - self.origin_ns) / 1e6, self.ends[channel])
        )
        duration = len(pcm) / 2 / channels / sample_rate * 1000
        if duration > 1000:
            self.omit("frame_size_limit")
        elif start + duration > self.max_duration_ms:
            self.omit("duration_limit")
        elif self.bytes + len(pcm) > self.max_bytes:
            self.omit("capture_byte_limit")
        elif len(self.packets) >= self.max_packets:
            self.omit("packet_limit")
        elif not source_budget.reserve(len(pcm)):
            self.omit("process_capture_byte_limit")
        else:
            self.packets.append(Packet(channel, start, pcm, sample_rate, channels))
            self.bytes += len(pcm)
            self.ends[channel] = start + duration
            self.started[channel] = True
            # Coordinates match encode() placement, including resampling rounding.
            return {
                "start": round(start * 24),
                "end": round(start * 24) + round(duration * 24),
            }

    def encode(self, audio_format="ogg"):
        if self.reason or not self.packets:
            return None

        import numpy as np

        started = time.perf_counter()
        cpu_started = time.thread_time()
        sample_rate = 24000
        sample_count = math.ceil(max(self.ends) * sample_rate / 1000)
        samples = np.zeros((sample_count, 2), dtype=np.int16)
        for packet in self.packets:
            source = np.frombuffer(packet.pcm, dtype="<i2").reshape(-1, packet.channels)
            mono = source[:, 0] if packet.channels == 1 else source.astype(np.float32).mean(axis=1)
            size = round(len(mono) * sample_rate / packet.sample_rate)
            # Resampling happens off the voice event loop.
            converted = np.interp(
                np.arange(size) * packet.sample_rate / sample_rate,
                np.arange(len(mono)),
                mono,
            ).astype(np.int16)
            start = round(packet.start_ms * sample_rate / 1000)
            end = min(start + len(converted), sample_count)
            samples[start:end, packet.channel] = converted[: end - start]
        result = _encode(samples, sample_rate, audio_format)
        result["processing"] = {
            "wall_ms": (time.perf_counter() - started) * 1000,
            "thread_cpu_ms": (time.thread_time() - cpu_started) * 1000,
            "source_bytes": self.bytes,
            "encoded_bytes": len(result["data"]),
        }
        return result
