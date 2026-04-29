#!/usr/bin/env python
"""OpenRouter client traced via braintrust.auto_instrument()."""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-openrouter")

from openrouter import OpenRouter


client = OpenRouter()

response = client.chat.send(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of Australia?"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
