import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import jsonschema

from .. import logger
from ..span_types import SpanTypeAttribute
from ._types import JSONValue, SimulatorOutput, VoiceCall, VoiceCallResult, parse_output


if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


_RATE = 24000
_FRAME_BYTES = 960
_SILENCE = bytes(_FRAME_BYTES)
_CALL_TIMEOUT_SECONDS = 600
_FINALIZE_TIMEOUT = 120
_CALL_SCHEMA = {
    "type": "object",
    "required": list(VoiceCall.__required_keys__),
    "properties": {key: {"type": "string", "minLength": 1} for key in VoiceCall.__required_keys__},
    "additionalProperties": False,
}


class VoiceSession:
    """A connected voice call. Construct through connect(), which scopes its tracing parent."""

    def __init__(self, call: VoiceCall):
        try:
            jsonschema.validate(call, _CALL_SCHEMA)
        except jsonschema.ValidationError:
            raise ValueError("Invalid voice call context; forward the object from voice_task unchanged") from None
        url = urlsplit(call["connection_url"])
        if url.scheme not in ("ws", "wss") or not url.hostname or url.username or url.password or url.fragment:
            raise ValueError("Invalid voice WebSocket connection URL")
        self.call = call
        self._socket: ClientConnection | None = None
        self._incoming: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=500)
        self._outgoing: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        self._started = asyncio.Event()
        self._done = asyncio.Event()
        self._epoch = 0
        self._played = 0
        self._output: SimulatorOutput | None = None
        self._error: Exception | None = None
        self._reader: asyncio.Task[None] | None = None
        self._sender: asyncio.Task[None] | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._deadline: asyncio.Task[None] | None = None
        self._clear: Callable[[], Awaitable[None]] | None = None
        self._on_end: Callable[[], Awaitable[None]] | None = None
        self._write_lock = asyncio.Lock()
        self._remainder = b""
        self._resampler = None
        self._format: tuple[int, int] | None = None
        self._finished = False

    def __repr__(self):
        return f"VoiceSession(session_id={self.call['session_id']!r})"

    async def _open(self):
        from websockets.asyncio.client import connect as websocket_connect
        from websockets.exceptions import InvalidStatus

        expires = datetime.fromisoformat(self.call["expires_at"].replace("Z", "+00:00"))
        remaining = (expires - datetime.now(timezone.utc)).total_seconds()
        deadline = asyncio.get_running_loop().time() + max(0, remaining)
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Voice session connection token expired before connecting")
            try:
                self._socket = await websocket_connect(
                    self.call["connection_url"],
                    additional_headers={"Authorization": f"Bearer {self.call['session_token']}"},
                    open_timeout=min(15, remaining),
                    close_timeout=5,
                    max_size=1024 * 1024,
                )
                break
            except InvalidStatus as error:
                if error.response.status_code != 503:
                    raise
                await asyncio.sleep(min(0.5, max(0, deadline - asyncio.get_running_loop().time())))
        try:
            message = await asyncio.wait_for(self._socket.recv(), min(60, remaining))
            event = json.loads(message)
            if event.get("type") == "error":
                raise RuntimeError(event["message"])
            if (event.get("type"), event.get("sample_rate"), event.get("channels"), event.get("encoding")) != (
                "ready",
                _RATE,
                1,
                "pcm_s16le",
            ):
                raise RuntimeError("Voice session did not provide a supported ready event")
            self._epoch = event["epoch"]
        except BaseException:
            await self._socket.close()
            raise
        self._reader = asyncio.create_task(self._receive())
        self._sender = asyncio.create_task(self._send())
        self._deadline = asyncio.create_task(self._expire())

    async def _expire(self):
        await asyncio.sleep(_CALL_TIMEOUT_SECONDS)
        await self.fail(TimeoutError("Voice call exceeded its configured timeout"))

    async def fail(self, error: Exception):
        """Report a transport failure and stop the agent through the end callback."""
        self._error = self._error or error
        self._done.set()
        if self._on_end:
            await self._on_end()

    async def _receive(self):
        assert self._socket is not None
        try:
            async for message in self._socket:
                if isinstance(message, bytes):
                    if not message or len(message) % 2:
                        raise RuntimeError("Simulator audio must contain complete 16-bit PCM samples")
                    for start in range(0, len(message), _FRAME_BYTES):
                        self._incoming.put_nowait((self._epoch, message[start : start + _FRAME_BYTES]))
                    continue
                event = json.loads(message)
                kind = event.get("type")
                if kind == "audio.clear":
                    epoch = event["epoch"]
                    if not isinstance(epoch, int) or epoch <= self._epoch:
                        raise RuntimeError("Invalid simulator audio epoch")
                    self._epoch, self._played = epoch, 0
                    while not self._incoming.empty():
                        self._incoming.get_nowait()
                    if self._clear:
                        await self._clear()
                elif kind == "ended":
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    output = event.get("output")
                    if not isinstance(output, dict):
                        raise RuntimeError("Voice session ended without output")
                    if output.get("recording_error"):
                        raise RuntimeError(output["recording_error"])
                    if not isinstance(output.get("recording"), dict):
                        raise RuntimeError("Voice session ended without its recording manifest; update the data plane")
                    if output.get("reason") == "Conversation time limit reached":
                        raise TimeoutError("Voice call exceeded its configured timeout")
                    self._output = parse_output(output)
                    self._done.set()
                    if self._on_end:
                        await self._on_end()
                    return
                elif kind == "error":
                    raise RuntimeError(event["message"])
                elif kind != "transcript":
                    raise RuntimeError(f"Unexpected voice session event: {kind!r}")
            raise RuntimeError("Voice WebSocket disconnected without ended")
        except Exception as error:
            await self.fail(error)

    async def _send(self):
        assert self._socket is not None
        try:
            await self._started.wait()
            while not self._done.is_set():
                start = asyncio.get_running_loop().time()
                try:
                    audio = self._outgoing.get_nowait()
                except asyncio.QueueEmpty:
                    audio = None
                await self._socket.send(audio if audio is not None else _SILENCE)
                await asyncio.sleep(max(0, start + 0.02 - asyncio.get_running_loop().time()))
                if audio is not None:
                    self._outgoing.task_done()
        except Exception as error:
            await self.fail(error)

    def set_audio_sink(
        self,
        consume: Callable[[bytes], Awaitable[None]],
        *,
        clear: Callable[[], Awaitable[None]],
        ended: Callable[[], Awaitable[None]],
    ):
        """Attach playback callbacks. consume must await actual audio consumption."""
        if self._consumer:
            raise RuntimeError("A voice session can have only one audio transport")
        self._clear, self._on_end = clear, ended
        self._consumer = asyncio.create_task(self._consume(consume))
        self._started.set()

    async def _consume(self, consume: Callable[[bytes], Awaitable[None]]):
        assert self._socket is not None
        try:
            while not self._done.is_set():
                start = asyncio.get_running_loop().time()
                try:
                    epoch, audio = self._incoming.get_nowait()
                except asyncio.QueueEmpty:
                    epoch, audio = None, _SILENCE
                await consume(audio)
                await asyncio.sleep(max(0, start + len(audio) / (_RATE * 2) - asyncio.get_running_loop().time()))
                if epoch is not None and epoch == self._epoch and not self._done.is_set():
                    self._played += len(audio) // 2
                    await self._socket.send(
                        json.dumps({"type": "playback.ack", "epoch": epoch, "played_samples": self._played})
                    )
        except Exception as error:
            await self.fail(error)

    async def send_audio(self, audio: bytes, sample_rate: int, channels: int):
        """Send agent PCM audio to the simulator, converting and pacing it."""
        if channels not in (1, 2) or sample_rate <= 0:
            raise ValueError("Agent audio must be mono or stereo signed 16-bit PCM")
        if not audio or len(audio) % (2 * channels):
            raise ValueError("Agent audio must contain complete signed 16-bit PCM frames")
        async with self._write_lock:
            if (sample_rate, channels) != (_RATE, 1):
                import av

                if self._resampler is None or self._format != (sample_rate, channels):
                    self._resampler = av.AudioResampler(format="s16", layout="mono", rate=_RATE)
                    self._format = (sample_rate, channels)
                frame = av.AudioFrame(
                    format="s16", layout="mono" if channels == 1 else "stereo", samples=len(audio) // (2 * channels)
                )
                frame.sample_rate = sample_rate
                frame.planes[0].update(audio)
                audio = b"".join(bytes(f.planes[0])[: f.samples * 2] for f in self._resampler.resample(frame))
            audio = self._remainder + audio
            end = len(audio) // _FRAME_BYTES * _FRAME_BYTES
            for start in range(0, end, _FRAME_BYTES):
                if self._done.is_set():
                    self._raise_error()
                    return
                await asyncio.wait_for(self._outgoing.put(audio[start : start + _FRAME_BYTES]), 5)
            self._remainder = audio[end:]
            await asyncio.wait_for(self._outgoing.join(), 5)

    def clear_output(self):
        self._remainder = b""
        self._resampler = None
        self._format = None
        while not self._outgoing.empty():
            self._outgoing.get_nowait()
            self._outgoing.task_done()

    def _raise_error(self):
        if self._error:
            raise self._error

    async def _close(self, failed: bool):
        from websockets.exceptions import ConnectionClosed

        try:
            if self._socket is not None:
                if not self._done.is_set():
                    if not failed:
                        if self._remainder:
                            await asyncio.wait_for(self._outgoing.put(self._remainder.ljust(_FRAME_BYTES, b"\0")), 5)
                            self._remainder = b""
                        await asyncio.wait_for(self._outgoing.join(), 5)
                    try:
                        await self._socket.send(json.dumps({"type": "stop"}))
                    except ConnectionClosed:
                        await asyncio.wait_for(self._done.wait(), 5)
                    await asyncio.wait_for(self._done.wait(), _FINALIZE_TIMEOUT)
                if not failed:
                    self._raise_error()
                    if self._output is None:
                        raise RuntimeError("Voice session did not finish")
                    self._finished = True
        finally:
            for task in [self._reader, self._sender, self._consumer, self._deadline]:
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(t for t in [self._reader, self._sender, self._consumer, self._deadline] if t), return_exceptions=True
            )
            if self._socket is not None:
                await self._socket.close()

    def result(self, *, agent_output: JSONValue = None) -> VoiceCallResult:
        """Return call evidence after leaving connect(), alongside the application's actual result."""
        self._raise_error()
        if not self._finished or self._output is None:
            raise RuntimeError("Call result is available only after connect() has finished")
        output = self._output
        return VoiceCallResult(
            agent_output=agent_output,
            caller_transcript=output["transcript"],
            recording=output["recording"],
            reason=output["reason"],
            session_id=self.call["session_id"],
            span_id=self.call["span_id"],
        )


@asynccontextmanager
async def connect(call: VoiceCall) -> AsyncGenerator[VoiceSession]:
    """Connect in the agent process and activate its parent before constructing the agent.

    Use this context around the entire pipeline run. The application forwards
    the call context unchanged and returns session.result() after context exit.
    """
    session = VoiceSession(call)
    span = logger.start_span(name="Agent session", type=SpanTypeAttribute.TASK, parent=call["parent"])
    try:
        with span:
            await session._open()
            try:
                yield session
            except BaseException:
                try:
                    await session._close(failed=True)
                except Exception:
                    pass
                raise
            else:
                await session._close(failed=False)
    finally:
        await asyncio.to_thread(span.flush)
