"""OpenAI Realtime speech-to-speech with a local tool, traced by Braintrust."""

import asyncio
import os

import braintrust
from braintrust.integrations.pipecat import setup_pipecat
from common import INSTRUCTIONS, RecordedInput, SilentOutput, order_context, run_conversation

# Pipecat requires Python 3.11+ and is installed by this example, not the shared lint environment.
# pylint: disable=import-error
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.base_transport import TransportParams


async def main():
    logger = braintrust.init_logger(project="example-pipecat")
    setup_pipecat(capture_audio_attachments=True)

    transport = TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    source, output = RecordedInput(transport), SilentOutput(transport)
    aggregators = LLMContextAggregatorPair(order_context())
    model = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-realtime",
            system_instruction=INSTRUCTIONS,
            session_properties=events.SessionProperties(
                audio=events.AudioConfiguration(
                    input=events.AudioInput(
                        transcription=events.InputAudioTranscription(model="gpt-4o-mini-transcribe", language="en"),
                        turn_detection=events.TurnDetection(),
                    ),
                    output=events.AudioOutput(voice="alloy"),
                )
            ),
        ),
    )
    pipeline = Pipeline([source, aggregators.user(), model, output, aggregators.assistant()])
    with logger.start_span(name="realtime") as trace:
        await run_conversation(pipeline, aggregators, source, realtime=True)
    await asyncio.to_thread(logger.flush)
    print(f"Trace: {trace.link()}")


if __name__ == "__main__":
    asyncio.run(main())
