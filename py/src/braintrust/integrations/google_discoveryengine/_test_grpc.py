"""Test-only recording at a single generated client's gRPC callable boundary.

The public GAPIC method still coerces/validates requests and dispatches normally.
Only the selected entry in this transport instance's ``_wrapped_methods`` is
replaced. No global gRPC patches, credentials, or auth metadata are recorded.
"""

import json
from contextlib import contextmanager

from google.api_core import exceptions
from grpc.aio import EOF
from wrapt import ObjectProxy


def _encode(message):
    return type(message).to_dict(message, preserving_proto_field_name=True)


def _raise_recorded_error(exchange):
    error = exchange.get("error")
    if error:
        error_type = {
            "InvalidArgument": exceptions.InvalidArgument,
            "InternalServerError": exceptions.InternalServerError,
        }[error["type"]]
        raise error_type(error["message"])


def _record_error(exchange, error):
    exchange["error"] = {"type": type(error).__name__, "message": error.message}


class _ReplayStream:
    def __init__(self, exchange, response_type):
        self.exchange = exchange
        self.responses = iter(exchange["responses"])
        self.response_type = response_type
        self._cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._cancelled:
            raise StopAsyncIteration
        try:
            response = next(self.responses)
        except StopIteration:
            _raise_recorded_error(self.exchange)
            raise StopAsyncIteration from None
        return self.response_type(response)

    async def read(self):
        try:
            return await self.__anext__()
        except StopAsyncIteration:
            return EOF

    def cancel(self):
        self._cancelled = True
        return True

    def cancelled(self):
        return self._cancelled


class _RecordingStream(ObjectProxy):
    def __init__(self, stream, exchange):
        super().__init__(stream)
        self._self_exchange = exchange
        self._self_iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self._self_iterator is None:
                self._self_iterator = self.__wrapped__.__aiter__()
            response = await self._self_iterator.__anext__()
        except exceptions.GoogleAPICallError as error:
            _record_error(self._self_exchange, error)
            raise
        self._self_exchange["responses"].append(_encode(response))
        return response

    async def read(self):
        try:
            response = await self.__wrapped__.read()
        except exceptions.GoogleAPICallError as error:
            _record_error(self._self_exchange, error)
            raise
        if response is not EOF:
            self._self_exchange["responses"].append(_encode(response))
        return response


@contextmanager
def grpc_cassette(client, method, response_type, path, *, record=False, streaming=False):
    transport = client.transport
    rpc = getattr(transport, method)
    original = transport._wrapped_methods[rpc]
    saved = None if record else json.loads(path.read_text())
    exchange = {"method": method, "requests": [], "responses": []}

    def observe_request(request):
        exchange["requests"].append(_encode(request))
        if saved is not None:
            index = len(exchange["requests"]) - 1
            assert exchange["requests"][index] == saved["requests"][index]
        return request

    async def invoke(request, **kwargs):
        observe_request(request)
        if saved is not None:
            assert saved["method"] == method
            if saved.get("error_at") == "call":
                _raise_recorded_error(saved)
            if streaming:
                return _ReplayStream(saved, response_type)
            _raise_recorded_error(saved)
            return response_type(saved["responses"][0])
        try:
            result = await original(request, **kwargs)
        except exceptions.GoogleAPICallError as error:
            _record_error(exchange, error)
            exchange["error_at"] = "call"
            raise
        if streaming:
            return _RecordingStream(result, exchange)
        exchange["responses"].append(_encode(result))
        return result

    transport._wrapped_methods[rpc] = invoke
    try:
        yield
    finally:
        transport._wrapped_methods[rpc] = original
        if record and (exchange["responses"] or "error" in exchange):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(exchange, indent=2) + "\n")
        elif saved is not None:
            assert len(exchange["requests"]) == len(saved["requests"])
