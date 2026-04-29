#!/usr/bin/env python
"""Mistral client traced via braintrust.auto_instrument()."""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-mistral")

try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral


client = Mistral()

response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "What is the capital of Australia?"}],
)
print(response.choices[0].message.content)
