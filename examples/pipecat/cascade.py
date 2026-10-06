"""Audio → transcription → model/tool → synthesized speech, traced by Braintrust."""

import asyncio
import os

import braintrust
from braintrust.integrations.pipecat import setup_pipecat
from common import INSTRUCTIONS, RecordedInput, SilentOutput, order_context, run_conversation

# Pipecat requires Python 3.11+ and is installed by this example, not the shared lint environment.
# pylint: disable=import-error
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import TransportParams


async def main():
    logger = braintrust.init_logger(project="example-pipecat")
    setup_pipecat(capture_audio_attachments=True)

    api_key = os.environ["OPENAI_API_KEY"]
    transport = TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    source, output = RecordedInput(transport), SilentOutput(transport)
    aggregators = LLMContextAggregatorPair(
        order_context(),
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
    pipeline = Pipeline(
        [
            source,
            OpenAISTTService(api_key=api_key),
            aggregators.user(),
            OpenAILLMService(
                api_key=api_key,
                settings=OpenAILLMService.Settings(
                    model="gpt-4.1-mini",
                    system_instruction=INSTRUCTIONS,
                ),
            ),
            OpenAITTSService(
                api_key=api_key,
                settings=OpenAITTSService.Settings(
                    model="gpt-4o-mini-tts",
                    voice="alloy",
                ),
            ),
            output,
            aggregators.assistant(),
        ]
    )
    with logger.start_span(name="cascade") as trace:
        await run_conversation(pipeline, aggregators, source)
    await asyncio.to_thread(logger.flush)
    print(f"Trace: {trace.link()}")


if __name__ == "__main__":
    asyncio.run(main())
