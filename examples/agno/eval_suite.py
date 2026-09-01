# agno.eval lazy-imports its submodules through a module-level __getattr__, which
# static analysis cannot see.
# pylint: disable=no-name-in-module

import sys

import braintrust


braintrust.auto_instrument()

# A suite run opens a Braintrust experiment of its own, so each Case lands as a
# scored experiment row. Pass eval_experiments=False to setup_agno() (or set
# BRAINTRUST_AGNO_EVAL_EXPERIMENTS=false) to keep suite runs in logs instead.
braintrust.init_logger(project="agno-evals-project")

from agno.agent import Agent
from agno.eval import Case, cli
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools


agent = Agent(
    id="stock-agent",
    name="Stock Price Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools()],
    instructions="Use your tools for any market data question.",
)

CASES = (
    Case(
        name="looks_up_current_price",
        agent=agent,
        input="What is the current price of FIG?",
        tags=("smoke",),
        criteria="Reports a current share price for Figma.",
        expected_tool_calls=("get_current_stock_price",),
    ),
    Case(
        name="explains_pe_ratio",
        agent=agent,
        input="Explain the P/E ratio in one sentence.",
        criteria="Explains that the P/E ratio compares share price to earnings per share.",
    ),
)

if __name__ == "__main__":
    # python eval_suite.py --tag smoke     # run a tagged subset
    # python eval_suite.py --list          # list cases without running them
    sys.exit(cli(CASES))
