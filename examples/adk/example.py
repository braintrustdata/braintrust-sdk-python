"""Google ADK weather agent traced via braintrust.auto_instrument()."""

import asyncio

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-adk")

from google.adk import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def get_weather(location: str):
    """Get the weather for a location."""
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "sunny",
    }


async def main():
    agent = Agent(
        name="weather_agent",
        model="gemini-2.0-flash",
        instruction="You are a helpful weather assistant.",
        tools=[get_weather],
    )

    session_service = InMemorySessionService()
    await session_service.create_session(app_name="weather_app", user_id="user", session_id="session")

    runner = Runner(agent=agent, app_name="weather_app", session_service=session_service)

    user_msg = types.Content(role="user", parts=[types.Part(text="What's the weather in San Francisco?")])

    async for event in runner.run_async(user_id="user", session_id="session", new_message=user_msg):
        if event.is_final_response():
            print(event.content.parts[0].text)


asyncio.run(main())
