"""Manually patch LangChain with Braintrust tracing.

Usage:
    export BRAINTRUST_API_KEY="your-api-key"
    export OPENAI_API_KEY="your-openai-api-key"
    python manual_patching.py
"""

from braintrust import init_logger
from braintrust.integrations.langchain import BraintrustCallbackHandler, set_global_handler


# Setup LangChain tracing with a specific project
logger = init_logger(project="my-langchain-project")
handler = BraintrustCallbackHandler(logger=logger)
set_global_handler(handler)

from langchain_openai import ChatOpenAI


def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("What is the capital of France?")
    print(response.content)


main()
