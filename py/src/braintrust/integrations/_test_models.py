"""Recording defaults for current SDKs; older matrix entries replay their fixtures."""

import os


LATEST = os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION") == "latest"
OPENAI_CHAT_MODEL = "gpt-6-luna" if LATEST else "gpt-4o-mini"
# Disable reasoning for the short completions and Chat Completions tool loops.
OPENAI_CHAT_OPTIONS = {"reasoning_effort": "none"} if LATEST else {}
ANTHROPIC_CHAT_MODEL = "claude-haiku-5-5" if LATEST else "claude-haiku-4-5-20251001"
