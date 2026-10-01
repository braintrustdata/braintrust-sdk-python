"""Pipecat output queue provenance hooks."""

from collections import deque
from contextvars import ContextVar


def instrument_output(output, alignment, frame_context=None, hooks=None):
    """Track equal-rate mono PCM through native chunking without copying audio.

    Unsupported resampling/mixing deliberately has no asserted association.
    """
    from .hooks import Hooks

    hooks = hooks or Hooks()
    original_start = hooks.original(output, "start")
    original_write = hooks.original(output, "write_audio_frame")
    frame_parts = {}
    active = ContextVar("pipecat_output_context", default=None)

    async def start(frame):
        result = await original_start(frame)
        for sender in output._media_senders.values():
            install_sender(sender)
        return result

    def install_sender(sender):
        runs, pending = deque(), deque()
        supported = [True]
        handle_audio = hooks.original(sender, "handle_audio_frame")
        handle_stop = hooks.original(sender, "handle_tts_stopped")
        buffer_audio = hooks.original(sender, "_buffer_audio")
        take_chunk = hooks.original(sender, "_take_audio_chunk")
        clear_buffer = hooks.original(sender, "_clear_audio_buffer")
        create_audio_task = hooks.original(sender, "_create_audio_task")

        async def handle(frame):
            context = frame_context(frame) if frame_context else getattr(frame, "context_id", None)
            if frame.sample_rate != sender._sample_rate or frame.num_channels != 1 or sender._mixer:
                context = None
                supported[0] = False
            token = active.set(context)
            try:
                return await handle_audio(frame)
            finally:
                active.reset(token)

        async def stop(frame):
            context = frame_context(frame) if frame_context else frame.context_id
            token = active.set(context if supported[0] else None)
            try:
                return await handle_stop(frame)
            finally:
                active.reset(token)

        def buffer(audio, *, uninterruptible):
            result = buffer_audio(audio, uninterruptible=uninterruptible)
            if audio:
                runs.append([len(audio), active.get()])
            return result

        def take():
            audio, uninterruptible = take_chunk()
            left, offset, parts = len(audio), 0, []
            while left and runs:
                count, context = runs[0]
                amount = min(count, left)
                parts.append((offset, amount, context))
                runs[0][0] -= amount
                if not runs[0][0]:
                    runs.popleft()
                left -= amount
                offset += amount
            pending.append(parts)
            return audio, uninterruptible

        bound_queue = [None]

        def bind_queue():
            if bound_queue[0] is sender._audio_queue:
                return
            bound_queue[0] = sender._audio_queue
            queue_put = hooks.original(sender._audio_queue, "put")

            async def put(frame):
                if hasattr(frame, "audio") and pending:
                    parts = pending.popleft()
                    if len(frame_parts) < 16000:
                        frame_parts[frame.id] = parts
                    else:
                        alignment.omitted += len(parts)
                return await queue_put(frame)

            hooks.set(sender._audio_queue, "put", put)

        def create():
            result = create_audio_task()
            bind_queue()
            return result

        def clear():
            runs.clear()
            pending.clear()
            supported[0] = True
            return clear_buffer()

        hooks.set(sender, "handle_audio_frame", handle)
        hooks.set(sender, "handle_tts_stopped", stop)
        hooks.set(sender, "_buffer_audio", buffer)
        hooks.set(sender, "_take_audio_chunk", take)
        hooks.set(sender, "_clear_audio_buffer", clear)
        hooks.set(sender, "_create_audio_task", create)
        bind_queue()

    async def write(frame):
        import time

        observed = time.monotonic_ns()
        parts = frame_parts.pop(frame.id, [])
        result = await original_write(frame)
        if result:
            try:
                interval = alignment.recording.capture(
                    1, frame.audio, frame.sample_rate, frame.num_channels, observed_ns=observed
                )
            except Exception:  # noqa: BLE001 - instrumentation must preserve transport success
                alignment.recording.omit("capture_error")
                interval = None
            if interval:
                for offset, size, context in parts:
                    # This hook supports only the demo's mono, 24 kHz output.
                    if frame.sample_rate == 24000 and frame.num_channels == 1:
                        ranges = [
                            [
                                interval["start"] + offset // 2,
                                interval["start"] + (offset + size) // 2,
                            ]
                        ]
                        for owner in alignment.contexts.get(context, []):
                            alignment.add(owner, ranges, 1)
        return result

    hooks.set(output, "start", start)
    hooks.set(output, "write_audio_frame", write)
