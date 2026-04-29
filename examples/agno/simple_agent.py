import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="simple-agent-project")

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools


agent = Agent(
    name="Stock Price Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools()],
    instructions="You are a stock price agent. Answer questions in the style of a stock analyst.",
)

response = agent.run("What is the current price of FIG?")
print(response.content)
