"""Observe real stdio streams without replacing the MCP transport or subprocess.

Replay runs the real server again and compares its sanitized wire transcript to
the checked-in recording. Nested HTTP calls use VCR. No protocol replies are
synthesized and no recorded stream stands in for the real transport.
"""

import contextlib
import json
import sys
from typing import Any

from wrapt import wrap_function_wrapper


TRANSCRIPT_ENV = "BRAINTRUST_MCP_TEST_TRANSCRIPT"


def observe_stdio(run: Any, path: str) -> None:
    messages: list[dict[str, Any]] = []

    def record(direction, message):
        data = message.message.model_dump(by_alias=True, mode="json", exclude_none=True)
        meta = data.get("params", {}).get("_meta", {})
        if "traceparent" in meta:
            meta["traceparent"] = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
        messages.append({"direction": direction, "message": data})
        with open(path, "w") as file:
            json.dump(messages, file, indent=2, sort_keys=True)

    @contextlib.asynccontextmanager
    async def stdio(wrapped, _instance, args, kwargs):
        async with wrapped(*args, **kwargs) as streams:
            read, write = streams

            async def receive(original, instance, call_args, call_kwargs):
                message = await original(*call_args, **call_kwargs)
                if instance is read:
                    record("client_to_server", message)
                return message

            async def send(original, instance, call_args, call_kwargs):
                result = await original(*call_args, **call_kwargs)
                if instance is write:
                    record("server_to_client", call_args[0])
                return result

            read_method, write_method = type(read).receive, type(write).send
            wrap_function_wrapper(type(read), "receive", receive)
            wrap_function_wrapper(type(write), "send", send)
            try:
                yield streams
            finally:
                type(read).receive, type(write).send = read_method, write_method

    wrap_function_wrapper(sys.modules[run.__module__], "stdio_server", stdio)
