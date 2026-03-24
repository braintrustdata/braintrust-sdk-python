"""Auto-instrument LangChain with Braintrust tracing.

Usage:
    export BRAINTRUST_API_KEY="your-api-key"
    export OPENAI_API_KEY="your-openai-api-key"
    python auto.py
"""

import braintrust


# Auto-instrument all supported libraries including LangChain
braintrust.auto_instrument()

from langchain_openai import ChatOpenAI


def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("What is the capital of France?")
    print(response.content)


main()
