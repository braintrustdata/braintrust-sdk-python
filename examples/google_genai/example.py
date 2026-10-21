#!/usr/bin/env python
"""Google GenAI client traced via braintrust.auto_instrument()."""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-google-genai")

from google import genai


client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What is the capital of Australia?",
)
print(response.text)
