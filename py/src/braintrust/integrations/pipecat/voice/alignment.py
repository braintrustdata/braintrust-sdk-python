"""Pipecat output queue provenance hooks."""

from collections import deque
from contextvars import ContextVar

from braintrust.audio.timeline import pcm_bytes_to_ms, samples_to_ms


def instrument_output(output, alignment, frame_context=None, hooks=None):
    """Track equal-rate mono PCM through native chunking without copying audio.

    Unsupported resampling/mixing deliberately has no asserted association.
    """
    from .hooks import Hooks

    hooks = hooks or Hooks()
    original_start = hooks.original(output, "start")
    original_write = hooks.original(output, "write_audio_frame")
    token = object()
    source_offsets = {}
    active = ContextVar("pipecat_output_context", default=None)

    async def start(frame):
        result = await original_start(frame)
        for sender in output._media_senders.values():
            install_sender(sender)
        return result

    def install_sender(sender):
        runs, pending = deque(), deque()
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
            start = source_offsets.get(context, 0)
            if context is not None:
                source_offsets[context] = start + len(frame.audio)
            owners = alignment.output_owners(context)
            current = active.set((owners, start))
            try:
                return await handle_audio(frame)
            finally:
                active.reset(current)

        async def stop(frame):
            context = frame_context(frame) if frame_context else frame.context_id
            current = active.set(None)  # Native stop padding is not generated clip audio.
            try:
                return await handle_stop(frame)
            finally:
                active.reset(current)
                source_offsets.pop(context, None)
                alignment.end_output(context)

        def buffer(audio, *, uninterruptible):
            result = buffer_audio(audio, uninterruptible=uninterruptible)
            if audio:
                owners, start = active.get() or ([], 0)
                runs.append([len(audio), owners, start])
            return result

        def take():
            audio, uninterruptible = take_chunk()
            left, offset, parts = len(audio), 0, []
            while left and runs:
                count, owners, start = runs[0]
                amount = min(count, left)
                parts.append((offset, amount, owners, start))
                runs[0][2] += amount
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
            if bound_queue[0] is not None:
                hooks.remove(bound_queue[0], "put")
            bound_queue[0] = sender._audio_queue
            queue_put = hooks.original(sender._audio_queue, "put")

            async def put(frame):
                if hasattr(frame, "audio") and pending:
                    parts = pending.popleft()
                    frame._braintrust_output_parts = (token, parts)
                return await queue_put(frame)

            hooks.set(sender._audio_queue, "put", put)

        def create():
            result = create_audio_task()
            bind_queue()
            return result

        def clear():
            runs.clear()
            pending.clear()
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
        identity = getattr(frame, "_braintrust_output_parts", None)
        parts = identity[1] if identity and identity[0] is token else []
        if identity and identity[0] is token:
            del frame._braintrust_output_parts
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
                for offset, size, owners, start in parts:
                    # This hook supports only the demo's mono, 24 kHz output.
                    if frame.sample_rate == 24000 and frame.num_channels == 1:
                        ranges = [
                            [
                                interval["start"] + offset // 2,
                                interval["start"] + (offset + size) // 2,
                            ]
                        ]
                        if owners:
                            alignment.add_clip_range(
                                owners[0],
                                pcm_bytes_to_ms(start, frame.sample_rate, frame.num_channels),
                                pcm_bytes_to_ms(start + size, frame.sample_rate, frame.num_channels),
                                samples_to_ms(ranges[0][0]),
                                samples_to_ms(ranges[0][1]),
                            )
                        for owner in owners:
                            alignment.add(owner, ranges, 1)
        return result

    hooks.set(output, "start", start)
    hooks.set(output, "write_audio_frame", write)
