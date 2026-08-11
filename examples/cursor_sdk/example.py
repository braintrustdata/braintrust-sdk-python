#!/usr/bin/env python
"""Cursor SDK agent traced with Braintrust auto-instrumentation."""

import os

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-cursor-sdk")

from cursor_sdk import Agent, LocalAgentOptions  # pylint: disable=import-error,wrong-import-position


with Agent.create(
    model="composer-2.5",
    local=LocalAgentOptions(cwd=os.getcwd()),
) as agent:
    result = agent.send("Summarize what this repository does").wait()
    print(result.result)
