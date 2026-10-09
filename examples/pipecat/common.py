"""Example application plumbing: play a WAV instead of opening a microphone."""

import asyncio
import wave
from pathlib import Path

# Pipecat requires Python 3.11+ and is installed by this example, not the shared lint environment.
# pylint: disable=import-error
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import InputAudioRawFrame, LLMRunFrame
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.workers.runner import WorkerRunner


async def lookup_order(params):
    """A fictional local order database; replace with your application's tool."""
    await params.result_callback({"order_id": params.arguments["order_id"], "delivery": "Friday"})


def order_context():
    return LLMContext(
        tools=ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="lookup_order",
                    description="Look up an order's delivery date",
                    properties={"order_id": {"type": "string"}},
                    required=["order_id"],
                    handler=lookup_order,
                )
            ]
        ),
    )


INSTRUCTIONS = (
    "Speak English. Greet the caller briefly and ask for their order number. "
    "When they supply it, always call lookup_order. "
    "After its result, say only: Your order arrives Friday."
)


class RecordedInput(BaseInputTransport):
    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def play(self, sample_rate):
        with wave.open(str(Path(__file__).with_name("order.wav"))) as audio:
            pcm = audio.readframes(audio.getnframes())
        pcm = await create_stream_resampler().resample(pcm, 16000, sample_rate)
        # Include silence around the request so native VAD determines its boundaries.
        pcm = b"\0" * sample_rate + pcm + b"\0" * (sample_rate * 2)
        frame_bytes = sample_rate // 50 * 2
        for offset in range(0, len(pcm), frame_bytes):
            await self.push_audio_frame(InputAudioRawFrame(pcm[offset : offset + frame_bytes], sample_rate, 1))
            await asyncio.sleep(0.02)


class SilentOutput(BaseOutputTransport):
    """Accept agent audio without requiring an audio device; listen in the trace."""

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        return True


async def run_conversation(pipeline, aggregators, source, *, realtime=False):
    sample_rate = 24000 if realtime else 16000
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=sample_rate,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    greeted, finished = asyncio.Event(), asyncio.Event()

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def assistant_turn(aggregator, message):
        print(f"Agent: {message.content}", flush=True)
        if "friday" in message.content.lower():
            finished.set()
        else:
            greeted.set()

    @aggregators.user().event_handler("on_user_turn_message_added")
    async def user_turn(aggregator, message):
        print(f"Caller: {message.content}", flush=True)

    @worker.event_handler("on_pipeline_started")
    async def started(worker, frame):
        await worker.queue_frame(LLMRunFrame())
        await asyncio.wait_for(greeted.wait(), 30)
        await source.play(sample_rate)

    task = asyncio.create_task(WorkerRunner(handle_sigint=False).run(worker))
    try:
        await asyncio.wait_for(finished.wait(), 60)
        await worker.stop_when_done()
        await task
    finally:
        if not task.done():
            await worker.cancel()
            await task
