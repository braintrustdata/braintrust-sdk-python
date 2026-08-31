import braintrust


braintrust.auto_instrument()

# An eval run logs to whatever is current. Swap init_logger for
# braintrust.init(project=..., experiment=...) to score it as an experiment row instead.
braintrust.init_logger(project="agno-evals-project")

from agno.agent import Agent
from agno.eval.accuracy import AccuracyEval
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools


agent = Agent(
    name="Stock Price Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools()],
    instructions="You are a stock price agent. Answer with the ticker and nothing else.",
)

evaluation = AccuracyEval(
    name="Ticker Lookup",
    model=OpenAIChat(id="gpt-4o-mini"),
    agent=agent,
    input="Which ticker does Figma trade under?",
    expected_output="FIG",
    num_iterations=2,
)

result = evaluation.run(print_summary=True)
print(f"average score: {result.avg_score}/10")
