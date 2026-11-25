import asyncio
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from ._session import VoiceSession


@dataclass
class _CallerAudio(InputAudioRawFrame):
    consumed: asyncio.Future[None] | None = field(default=None, repr=False)
    epoch: int = 0


class _Input(BaseInputTransport):
    def __init__(self, session: VoiceSession, params: TransportParams):
        super().__init__(params)
        self._voice = session
        self._epoch = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

        async def consume(audio: bytes):
            consumed = asyncio.get_running_loop().create_future()
            await self.push_audio_frame(
                _CallerAudio(audio=audio, sample_rate=24000, num_channels=1, consumed=consumed, epoch=self._epoch)
            )
            await consumed

        async def clear():
            self._epoch += 1

        async def ended():
            await self.push_frame(EndFrame())

        self._voice.set_audio_sink(consume, clear=clear, ended=ended)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if not isinstance(frame, _CallerAudio) or frame.epoch == self._epoch:
            await super().push_frame(frame, direction)
        if isinstance(frame, _CallerAudio) and frame.consumed is not None and not frame.consumed.done():
            frame.consumed.set_result(None)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, ErrorFrame):
            await self._voice.fail(RuntimeError(frame.error))
        await super().process_frame(frame, direction)


class _Output(BaseOutputTransport):
    def __init__(self, session: VoiceSession, params: TransportParams):
        super().__init__(params)
        self._voice = session

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        try:
            await self._voice.send_audio(frame.audio, frame.sample_rate, frame.num_channels)
            return True
        except Exception as error:
            await self._voice.fail(error)
            raise

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            self._voice.clear_output()
        if isinstance(frame, ErrorFrame):
            await self._voice.fail(RuntimeError(frame.error))
        await super().process_frame(frame, direction)


class PipecatTransport(BaseTransport):
    """Native input/output processors for a managed voice eval session."""

    def __init__(self, session: VoiceSession):
        super().__init__()
        params = TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=24000,
            audio_out_sample_rate=24000,
            audio_out_channels=1,
            audio_out_end_silence_secs=0,
        )
        self._input = _Input(session, params)
        self._output = _Output(session, params)

    def input(self) -> BaseInputTransport:
        return self._input

    def output(self) -> BaseOutputTransport:
        return self._output
