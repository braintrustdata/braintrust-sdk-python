#!/usr/bin/env python
"""LangChain chain traced via braintrust.auto_instrument().

`auto_instrument()` installs a global Braintrust callback handler into
LangChain's configure hook, so chains are traced without having to pass a
handler explicitly via `config={"callbacks": [...]}`.
"""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-langchain")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = prompt | model
result = chain.invoke({"number": "2"})

print(result.content)
