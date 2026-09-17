"""Tracing for Discovery Engine's direct v1 generation and ranking calls.

Requests are read without constructing or serializing a second GAPIC request.
Only selected protobuf submessages (citations, references, configuration) need
conversion; neither complete responses nor streams are serialized to JSON here.
"""

import logging
import time
import weakref
from collections.abc import Mapping
from itertools import islice

from braintrust.logger import start_span
from wrapt import ObjectProxy


_LOG = logging.getLogger(__name__)
_INSTRUMENTATION = "google-discoveryengine-auto"
_MAX_RANK_RESULTS = 100
_ANSWER_DETAILS = ("citations", "references", "grounding_supports", "related_questions", "answer_skipped_reasons")


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _message_dict(value):
    # Braintrust's serializer falls back to text for proto-plus messages.
    # Convert only selected protobuf fields, leaving ordinary values alone.
    if isinstance(value, Mapping):
        return value
    return type(value).to_dict(value, preserving_proto_field_name=True, always_print_fields_with_no_presence=False)


def _details(value, names):
    return {name: field for name in names if (field := _get(value, name))}


def _request(args, kwargs):
    request = args[0] if args else kwargs.get("request")
    return request if request is not None else kwargs


def _prepare(method, request):
    metadata = {"provider": "google"}
    if method == "rank":
        model = _get(request, "model")
        metadata.update(_details(request, ("ranking_config", "top_n", "ignore_record_details_in_response")))
        span_input = {
            "query": _get(request, "query", ""),
            "records": [_details(record, ("id", "title", "content")) for record in _get(request, "records", ())],
        }
    elif method == "check_grounding":
        model = None
        metadata.update(_details(request, ("grounding_config",)))
        grounding_spec = _get(request, "grounding_spec")
        if grounding_spec:
            metadata["grounding_spec"] = _message_dict(grounding_spec)
        span_input = {
            "answer_candidate": _get(request, "answer_candidate", ""),
            "facts": [_message_dict(fact) for fact in _get(request, "facts", ())],
        }
    else:
        converse = method == "converse_conversation"
        spec = _get(request, "summary_spec" if converse else "answer_generation_spec")
        model = _get(_get(spec, "model_spec"), "version" if converse else "model_version")
        query = _get(request, "query")
        span_input = [{"role": "user", "content": _get(query, "input" if converse else "text", "")}]
        preamble = _get(_get(spec, "model_prompt_spec" if converse else "prompt_spec"), "preamble")
        if preamble:
            span_input.insert(0, {"role": "system", "content": preamble})
        metadata.update(_details(request, ("serving_config", "session", "name")))
        metadata.update(_details(spec, ("include_citations", "answer_language_code", "summary_result_count")))
    if model:
        metadata["model"] = model
    return span_input, metadata


def _choice(text, **details):
    return {"index": 0, "message": {"role": "assistant", "content": text}, **details}


def _answer_details(answer):
    details = {}
    for name in _ANSWER_DETAILS:
        values = _get(answer, name)
        if values:
            details[name] = (
                [_message_dict(value) for value in values]
                if name in ("citations", "references", "grounding_supports")
                else list(values)
            )
    state = _get(answer, "state")
    if state:
        details["state"] = state.name if hasattr(state, "name") else state
    # An optional zero score is meaningful; don't drop it by testing truthiness.
    if answer is not None and "grounding_score" in answer:
        details["grounding_score"] = _get(answer, "grounding_score")
    return details


def _output(method, response):
    if method == "rank":
        return [
            {**_details(record, ("id", "title", "content")), "score": record.score}
            for record in islice(response.records, _MAX_RANK_RESULTS)
        ]
    if method == "check_grounding":
        return {
            "support_score": response.support_score,
            **{
                name: [_message_dict(item) for item in values]
                for name in ("cited_chunks", "cited_facts", "claims")
                if (values := getattr(response, name))
            },
        }
    if method == "converse_conversation":
        summary = response.reply.summary
        summary_metadata = summary.summary_with_metadata
        details = {}
        if summary_metadata.citation_metadata:
            details["citation_metadata"] = _message_dict(summary_metadata.citation_metadata)
        if summary_metadata.references:
            details["references"] = [_message_dict(reference) for reference in summary_metadata.references]
        if summary.summary_skipped_reasons:
            details["summary_skipped_reasons"] = list(summary.summary_skipped_reasons)
        return [_choice(summary.summary_text, **details)]
    return [_choice(response.answer.answer_text, **_answer_details(response.answer))]


def _safe_extract(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception:
        _LOG.warning("Could not extract Discovery Engine trace data", exc_info=True)
        return default


def _start(method, request):
    span_input, metadata = _safe_extract(_prepare, method, request, default=(None, {"provider": "google"}))
    return start_span(
        name=f"google_discoveryengine.{method}",
        type="task" if method in ("rank", "check_grounding") else "llm",
        input=span_input,
        metadata=metadata,
        internal={"instrumentation": _INSTRUMENTATION},
        set_current=method != "stream_answer_query",
    )


class _AnswerStreamState:
    def __init__(self, span):
        self.span = span
        self.started = time.monotonic()
        self.first_token = None
        self.text = []
        self.details = {}
        self.ended = False

    def add(self, response):
        answer = response.answer
        text = answer.answer_text
        from google.cloud.discoveryengine_v1 import Answer

        # SUCCEEDED is a complete snapshot, following the text/citation deltas.
        # Replace accumulated fields before consuming it to avoid duplication.
        if answer.state == Answer.State.SUCCEEDED:
            self.text.clear()
            self.details.clear()
        if text:
            if self.first_token is None:
                self.first_token = time.monotonic() - self.started
            self.text.append(text)
        # Keep protobuf leaves until final logging; no per-chunk serialization.
        for name in _ANSWER_DETAILS:
            values = getattr(answer, name)
            if values:
                self.details.setdefault(name, []).extend(values)
        for name in ("state", "grounding_score"):
            if name in answer:
                self.details[name] = getattr(answer, name)

    def finish(self, error=None):
        if self.ended:
            return
        self.ended = True
        details = _safe_extract(_answer_details, self.details, default={})
        output = [_choice("".join(self.text), **details)]
        metrics = {"time_to_first_token": self.first_token} if self.first_token is not None else {}
        self.span.log(output=output, metrics=metrics, **({"error": error} if error is not None else {}))
        self.span.end()


class _AnswerStream(ObjectProxy):
    def __init__(self, stream, state):
        super().__init__(stream)
        self._self_state = state
        # Retain only trace state, not the proxy/provider stream. GC can finalize
        # partial output while the provider handles its own transport cleanup.
        weakref.finalize(self, state.finish)
        self._self_iterator = iter(stream)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            response = next(self._self_iterator)
        except StopIteration:
            self._self_state.finish()
            raise
        except BaseException as error:
            self._self_state.finish(error)
            raise
        _safe_extract(self._self_state.add, response)
        return response

    def close(self):
        try:
            close = getattr(self.__wrapped__, "close", None)
            if close is not None:
                return close()
            return self.__wrapped__.cancel()
        finally:
            self._self_state.finish()

    def cancel(self):
        try:
            return self.__wrapped__.cancel()
        finally:
            self._self_state.finish()


class _AsyncAnswerStream(ObjectProxy):
    def __init__(self, stream, state):
        super().__init__(stream)
        self._self_state = state
        # Retain only trace state, not the proxy/provider stream. GC can finalize
        # partial output while the provider handles its own transport cleanup.
        weakref.finalize(self, state.finish)
        self._self_iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self._self_iterator is None:
                self._self_iterator = self.__wrapped__.__aiter__()
            response = await self._self_iterator.__anext__()
        except StopAsyncIteration:
            self._self_state.finish()
            raise
        except BaseException as error:
            self._self_state.finish(error)
            raise
        _safe_extract(self._self_state.add, response)
        return response

    async def read(self):
        from grpc.aio import EOF

        try:
            response = await self.__wrapped__.read()
        except BaseException as error:
            self._self_state.finish(error)
            raise
        if response is EOF:
            self._self_state.finish()
        else:
            _safe_extract(self._self_state.add, response)
        return response

    async def aclose(self):
        try:
            close = getattr(self.__wrapped__, "aclose", None)
            if close is not None:
                return await close()
            self.__wrapped__.cancel()
        finally:
            self._self_state.finish()

    def cancel(self):
        try:
            return self.__wrapped__.cancel()
        finally:
            self._self_state.finish()


def _call(method, wrapped, instance, args, kwargs):
    request = _request(args, kwargs)
    if method == "answer_query" and _get(request, "asynchronous_mode", False):
        return wrapped(*args, **kwargs)
    span = _start(method, request)
    if method == "stream_answer_query":
        state = _AnswerStreamState(span)
        try:
            return _AnswerStream(wrapped(*args, **kwargs), state)
        except BaseException as error:
            state.finish(error)
            raise
    with span:
        result = wrapped(*args, **kwargs)
        output = _safe_extract(_output, method, result)
        span.log(output=output)
        return result


async def _async_call(method, wrapped, instance, args, kwargs):
    request = _request(args, kwargs)
    if method == "answer_query" and _get(request, "asynchronous_mode", False):
        return await wrapped(*args, **kwargs)
    span = _start(method, request)
    if method == "stream_answer_query":
        state = _AnswerStreamState(span)
        try:
            return _AsyncAnswerStream(await wrapped(*args, **kwargs), state)
        except BaseException as error:
            state.finish(error)
            raise
    with span:
        result = await wrapped(*args, **kwargs)
        output = _safe_extract(_output, method, result)
        span.log(output=output)
        return result
