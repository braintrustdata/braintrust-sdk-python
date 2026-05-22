"""Compatibility tests for Braintrust OpenAI wrapping with ddtrace."""

import importlib
import inspect

import openai
from braintrust import wrap_openai
from braintrust.integrations.openai.patchers import _wrap_chat_create
from wrapt import FunctionWrapper


def test_wrap_openai_wraps_instance_when_ddtrace_patched_class():
    ddtrace = importlib.import_module("ddtrace")

    ddtrace.patch(openai=True)

    class_create = inspect.getattr_static(openai.resources.chat.completions.Completions, "create")
    assert isinstance(class_create, FunctionWrapper)

    client = wrap_openai(openai.OpenAI())

    assert _wrap_chat_create.has_patch_marker(client.chat.completions)
