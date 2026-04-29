#!/usr/bin/env python
"""Cohere client traced via braintrust.auto_instrument()."""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-cohere")

import cohere


client = cohere.ClientV2()

response = client.chat(
    model="command-r-plus-08-2024",
    messages=[{"role": "user", "content": "What is the capital of Australia?"}],
)
print(response.message.content[0].text)
