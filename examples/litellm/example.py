#!/usr/bin/env python
"""LiteLLM completion traced via braintrust.auto_instrument()."""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-litellm")

import litellm


response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of Australia?"}],
)
print(response.choices[0].message.content)
