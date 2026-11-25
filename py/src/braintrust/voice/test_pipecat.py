import asyncio
import inspect
import json
import struct

import pytest
from braintrust import logger
from braintrust.voice import connect
from braintrust.voice.test_session import simulator, terminal
from braintrust.voice.test_session import tracing as tracing


pytest.importorskip("pipecat")


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_error", [False, True])
async def test_real_pipecat_pipeline_consumes_audio_and_reports_failures(tracing, agent_error):
    # A native pipeline must exchange audio and surface ErrorFrames through the managed session.
    from braintrust.voice.pipecat import PipecatTransport
    from pipecat.frames.frames import ErrorFrame, InputAudioRawFrame, OutputAudioRawFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.workers.runner import WorkerRunner

    wire_audio = []
    acknowledgements = []

    class Echo(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                if agent_error:
                    await self.push_frame(ErrorFrame(error="Agent failed", fatal=True), FrameDirection.UPSTREAM)
                else:
                    await self.push_frame(
                        OutputAudioRawFrame(audio=frame.audio, sample_rate=frame.sample_rate, num_channels=1)
                    )
            else:
                await self.push_frame(frame, direction)

    async def handler(socket):
        await socket.send(struct.pack("<480h", *([100] * 480)))
        if not agent_error:
            await socket.send(json.dumps({"type": "audio.clear", "epoch": 1}))
            await socket.send(struct.pack("<480h", *([200] * 480)))
        finished = False
        async for message in socket:
            if isinstance(message, bytes):
                wire_audio.append(message)
                if any(message) and not finished:
                    finished = True
                    await socket.send(json.dumps(terminal()))
            else:
                event = json.loads(message)
                if event["type"] == "playback.ack":
                    acknowledgements.append(event)
                if event["type"] == "stop":
                    await socket.send(json.dumps(terminal()))
                    return

    async def run(call):
        async with connect(call) as session:
            transport = PipecatTransport(session)
            pipeline = Pipeline([transport.input(), Echo(), transport.output()])
            options = {"enable_turn_tracking": False, "enable_rtvi": False, "check_dangling_tasks": False}
            worker = PipelineWorker(
                pipeline,
                idle_timeout_secs=None,
                **{
                    key: value for key, value in options.items() if key in inspect.signature(PipelineWorker).parameters
                },
            )
            runner = WorkerRunner(
                handle_sigint=False,
                **{
                    key: value
                    for key, value in options.items()
                    if key == "check_dangling_tasks" and key in inspect.signature(WorkerRunner).parameters
                },
            )
            await runner.add_workers(worker)
            await asyncio.wait_for(runner.run(), 5)
        return session.result()

    with logger.start_span(name="Task") as parent:
        async with simulator(handler, parent.export()) as call:
            if agent_error:
                with pytest.raises(RuntimeError, match="Agent failed"):
                    await run(call)
            else:
                result = await run(call)
                assert result["reason"] == "Client stopped session"
                assert any(any(audio) for audio in wire_audio)
                assert all(event["epoch"] == 1 for event in acknowledgements)
