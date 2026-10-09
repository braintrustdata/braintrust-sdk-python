"""Carry native synthesis identity across Pipecat's asynchronous frame delivery."""

import asyncio
from contextvars import ContextVar


class TTSRequests:
    def __init__(self, hooks, services):
        self.token = object()
        for service in services:
            self.install(hooks, service)

    def install(self, hooks, service):
        scope = ContextVar("braintrust_tts_request", default=None)
        synthesize = hooks.original(service, "_push_tts_frames")
        push = hooks.original(service, "push_frame")

        async def request(frame, *args, **kwargs):
            token = scope.set((frame, asyncio.current_task()))
            try:
                return await synthesize(frame, *args, **kwargs)
            finally:
                scope.reset(token)

        async def emit(frame, *args, **kwargs):
            active = scope.get()
            if type(frame).__name__ == "MetricsFrame" and active and active[1] is asyncio.current_task():
                context = getattr(active[0], "context_id", None)
                if context:
                    # Private bookkeeping travels with the existing queued frame;
                    # no side table retains frames or audio. Native serialization
                    # ignores private attributes, and the observer consumes it.
                    frame._braintrust_tts_request = (self.token, id(service), context)
            return await push(frame, *args, **kwargs)

        hooks.set(service, "_push_tts_frames", request)
        hooks.set(service, "push_frame", emit)

    def take(self, frame, source):
        identity = getattr(frame, "_braintrust_tts_request", None)
        if identity and identity[:2] == (self.token, id(source)):
            del frame._braintrust_tts_request
            return ("tts", identity[2])
        return None
