#!/usr/bin/env python
"""CrewAI Agent + Task + Crew traced via braintrust.auto_instrument()."""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-crewai")

from crewai import Agent, Crew, Task


agent = Agent(
    role="Calculator",
    goal="Answer arithmetic questions correctly.",
    backstory="You are a precise mental-math machine.",
    tools=[],
)

task = Task(
    description="What is 12 + 12?",
    expected_output="A single number, with no extra commentary.",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()

print(result)
