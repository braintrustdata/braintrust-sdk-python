#!/usr/bin/env python
"""LlamaIndex query traced via braintrust.auto_instrument()."""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-llamaindex")

from llama_index.llms.openai import OpenAI


llm = OpenAI(model="gpt-4o-mini", temperature=0)
response = llm.complete("What is the capital of Australia?")
print(str(response))
