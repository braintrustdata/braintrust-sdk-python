"""Run with `uv run voice_agent.py dev` after configuring LiveKit/OpenAI credentials."""

import asyncio
import os

import braintrust
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.openai import setup_openai
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, function_tool
from livekit.plugins import openai, silero
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad


server = AgentServer()


@function_tool
async def lookup_order(order_id: str) -> str:
    """Look up an order's estimated delivery date."""
    return f"Order {order_id} arrives Friday."


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    braintrust.init_logger(project="livekit-cascade")
    setup_livekit_agents()
    setup_openai()
    mode = os.getenv("LIVEKIT_MODE", "cascade")
    if mode == "cascade":
        session = AgentSession(
            vad=silero.VAD.load(),
            stt=openai.STT(model="gpt-4o-mini-transcribe"),
            llm=openai.LLM(model="gpt-4.1-mini"),
            tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
        )
    elif mode in ("realtime", "half-cascade"):
        options = {"modalities": ["text"]} if mode == "half-cascade" else {}
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                model="gpt-realtime",
                voice="marin",
                turn_detection=ServerVad(type="server_vad", silence_duration_ms=300, prefix_padding_ms=200),
                **options,
            ),
            tts=openai.TTS(model="tts-1", voice="alloy") if mode == "half-cascade" else None,
        )
    else:
        raise ValueError("LIVEKIT_MODE must be cascade, realtime, or half-cascade")
    closed = asyncio.Event()
    session.on("close", lambda _: closed.set())
    try:
        await session.start(
            Agent(instructions="Help customers find their order delivery dates.", tools=[lookup_order]), room=ctx.room
        )
        if mode == "cascade":
            await session.say("Hello. How can I help with your order?")
        else:
            session.generate_reply(instructions="Greet the caller and ask how you can help with their order.")
        await closed.wait()
    finally:
        await session.aclose()
        await asyncio.to_thread(braintrust.flush)


if __name__ == "__main__":
    cli.run_app(server)
