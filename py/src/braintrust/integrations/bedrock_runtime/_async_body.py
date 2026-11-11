"""Buffered response bodies, loaded only when an aiobotocore client is used."""

import io

from aiobotocore.response import StreamingBody


class BufferedAsyncBody(StreamingBody):
    """Replay bytes read for tracing through aiobotocore's streaming body API.

    Keep aiobotocore's iteration and line handling, and delegate transport
    cleanup to the original body. The _self_ names also support older releases
    whose StreamingBody inherits from wrapt.ObjectProxy.
    """

    def __init__(self, original, data: bytes):
        raw = getattr(original, "raw_stream", None)
        if raw is None:
            raw = original.__wrapped__
        super().__init__(raw, str(len(data)))
        self._self_buffer = io.BytesIO(data)
        self._self_original = original

    async def read(self, amt=None):
        return self._self_buffer.read(-1 if amt is None else amt)

    async def readinto(self, buffer):
        return self._self_buffer.readinto(buffer)

    def tell(self):
        return self._self_buffer.tell()

    def readable(self):
        return self.tell() < len(self._self_buffer.getbuffer())

    async def __aenter__(self):
        await self._self_original.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._self_original.__aexit__(exc_type, exc_val, exc_tb)

    def close(self):
        return self._self_original.close()

    async def aclose(self):
        close = getattr(self._self_original, "aclose", None)
        if close is not None:
            return await close()
        # Older aiobotocore releases only expose synchronous close().
        return self.close()
