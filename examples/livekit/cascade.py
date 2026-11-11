"""Run with `uv run cascade.py dev` after configuring LiveKit/OpenAI credentials."""

import asyncio

import braintrust
from braintrust.integrations.livekit_agents import setup_livekit_agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, function_tool
from livekit.plugins import openai, silero


server = AgentServer()


@function_tool
async def lookup_order(order_id: str) -> str:
    """Look up an order's estimated delivery date."""
    return f"Order {order_id} arrives Friday."


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    braintrust.init_logger(project="livekit-cascade")
    setup_livekit_agents()
    braintrust.setup_openai()
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
    )
    closed = asyncio.Event()
    session.on("close", lambda _: closed.set())
    try:
        await session.start(
            Agent(instructions="Help customers find their order delivery dates.", tools=[lookup_order]), room=ctx.room
        )
        await session.say("Hello. How can I help with your order?")
        await closed.wait()
    finally:
        await session.aclose()
        await asyncio.to_thread(braintrust.flush)


if __name__ == "__main__":
    cli.run_app(server)
