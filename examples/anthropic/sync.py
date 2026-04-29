#!/usr/bin/env python
"""Sync Anthropic client traced via braintrust.auto_instrument()."""

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-anthropic-app")

import anthropic


client = anthropic.Anthropic()


@braintrust.traced
def ask_anthropic_sync(question, system=None):
    args = {
        "model": "claude-haiku-4-5",
        "max_tokens": 300,
        "temperature": 0.5,
        "messages": [{"role": "user", "content": question}],
    }
    if system:
        args["system"] = system
    msg = client.messages.create(**args)
    print(msg)


@braintrust.traced
def ask_anthropic_stream(question, system=None):
    args = {
        "max_tokens": 1024,
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": question}],
    }
    if system:
        args["system"] = system
    with client.messages.stream(**args) as stream:
        for msg in stream:
            pass
    message = stream.get_final_message()
    print(message)


@braintrust.traced
def ask_anthropic():
    print("asking questions")
    ask_anthropic_sync("What is the capital of Canada?")
    ask_anthropic_stream("What is the date tomorrow?", "today is 2025-03-26")


def main():
    ask_anthropic()


if __name__ == "__main__":
    main()
