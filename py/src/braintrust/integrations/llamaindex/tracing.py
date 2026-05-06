"""Braintrust span handler for LlamaIndex instrumentation."""

import inspect
import time
from typing import Any

from braintrust.logger import NOOP_SPAN, Span, current_span, start_span
from braintrust.span_types import SpanTypeAttribute


def _extract_block_content(message: Any) -> str | None:
    if not hasattr(message, "blocks"):
        return None
    text_parts = [block.text for block in message.blocks if hasattr(block, "text") and block.text]
    if text_parts:
        return "\n".join(text_parts)
    return None


def _extract_message_content(message: Any) -> Any:
    content = getattr(message, "content", None)
    if content:
        return content
    return _extract_block_content(message)


def _extract_messages(messages: Any) -> list[dict[str, Any]] | None:
    if not messages:
        return None
    result = []
    for msg in messages:
        entry: dict[str, Any] = {}
        if hasattr(msg, "role"):
            entry["role"] = str(msg.role.value) if hasattr(msg.role, "value") else str(msg.role)
        content = _extract_message_content(msg)
        if content is not None:
            entry["content"] = content
        result.append(entry)
    return result


def _extract_response_output(result: Any) -> Any:
    if result is None:
        return None
    # Streaming/coroutine responses are consumed outside this span handler.
    # Do not log unstable object reprs such as "<generator object ...>".
    if inspect.isgenerator(result) or inspect.isasyncgen(result) or inspect.iscoroutine(result):
        return None
    # ChatResponse
    if hasattr(result, "message") and hasattr(result, "raw"):
        msg = result.message
        if not msg:
            return None
        output: dict[str, Any] = {}
        output["role"] = str(msg.role.value) if hasattr(msg.role, "value") else str(msg.role)
        content = _extract_message_content(msg)
        if content is not None:
            output["content"] = content
        return output
    # CompletionResponse
    if hasattr(result, "text") and hasattr(result, "raw"):
        return {"text": result.text}
    # Query response
    if hasattr(result, "response") and hasattr(result, "source_nodes"):
        output = {"response": result.response}
        if result.source_nodes:
            output["source_nodes"] = _extract_nodes(result.source_nodes)
        return output
    # List of NodeWithScore
    if isinstance(result, list) and result and hasattr(result[0], "node"):
        return _extract_nodes(result)
    if isinstance(result, str):
        return result
    return str(result)


def _extract_nodes(nodes: list[Any]) -> list[dict[str, Any]]:
    result = []
    for nws in nodes:
        entry: dict[str, Any] = {}
        if hasattr(nws, "score") and nws.score is not None:
            entry["score"] = nws.score
        node = nws.node if hasattr(nws, "node") else nws
        if hasattr(node, "text"):
            entry["text"] = node.text
        if hasattr(node, "id_"):
            entry["node_id"] = node.id_
        if hasattr(node, "metadata") and node.metadata:
            entry["metadata"] = node.metadata
        result.append(entry)
    return result


def _classify_instance(instance: Any) -> tuple[SpanTypeAttribute, str]:
    if instance is None:
        return SpanTypeAttribute.TASK, "llamaindex"

    cls_name = type(instance).__name__
    mro_names = {c.__name__ for c in type(instance).__mro__}

    if "BaseLLM" in mro_names or "LLM" in mro_names:
        return SpanTypeAttribute.LLM, cls_name

    if "BaseTool" in mro_names or "FunctionTool" in mro_names:
        return SpanTypeAttribute.TOOL, getattr(instance, "name", None) or cls_name

    if any(name in mro_names for name in ("BaseQueryEngine", "BaseAgent", "AgentRunner", "Workflow")):
        return SpanTypeAttribute.TASK, cls_name

    return SpanTypeAttribute.FUNCTION, cls_name


def _extract_input(bound_args: "inspect.BoundArguments") -> Any:
    args = {k: v for k, v in bound_args.arguments.items() if k != "self"}

    if "messages" in args:
        return _extract_messages(args["messages"])

    for key in ("str_or_query_bundle", "query_str", "query", "prompt"):
        if key in args:
            val = args[key]
            return val.query_str if hasattr(val, "query_str") else val

    if "nodes" in args and args["nodes"]:
        return _extract_nodes(args["nodes"])

    if len(args) == 1:
        return next(iter(args.values()))

    return None


class _SpanRecord:
    __slots__ = ("bt_span", "start_time")

    def __init__(self, bt_span: Span, start_time: float):
        self.bt_span = bt_span
        self.start_time = start_time


try:
    from llama_index.core.instrumentation.span import BaseSpan
    from llama_index_instrumentation.span_handlers.base import BaseSpanHandler

    class BraintrustSpanHandler(BaseSpanHandler["BaseSpan"]):
        _bt_spans: dict[str, _SpanRecord] = {}

        def model_post_init(self, __context: Any) -> None:
            super().model_post_init(__context)
            self._bt_spans = {}

        @classmethod
        def class_name(cls) -> str:
            return "BraintrustSpanHandler"

        def _find_parent_bt_span(self, parent_span_id: str | None) -> Span | None:
            if parent_span_id is None:
                cs = current_span()
                return cs if cs != NOOP_SPAN else None

            span_id = parent_span_id
            while span_id:
                record = self._bt_spans.get(span_id)
                if record is not None:
                    return record.bt_span

                parent_span = self.open_spans.get(span_id)
                if parent_span is None:
                    return None
                span_id = parent_span.parent_id

            return None

        def new_span(
            self,
            id_: str,
            bound_args: "inspect.BoundArguments",
            instance: Any | None = None,
            parent_span_id: str | None = None,
            tags: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> BaseSpan | None:
            start_time = time.time()
            span_type, span_name = _classify_instance(instance)
            input_data = _extract_input(bound_args)

            metadata: dict[str, Any] = {}
            if instance is not None:
                metadata["class"] = type(instance).__name__
                for attr in ("model", "model_name", "temperature", "max_tokens"):
                    val = getattr(instance, attr, None)
                    if val is not None:
                        metadata[attr] = val

            parent_bt_span = self._find_parent_bt_span(parent_span_id)

            event: dict[str, Any] = {}
            if metadata:
                event["metadata"] = metadata
            if input_data is not None:
                event["input"] = input_data

            if parent_bt_span is not None:
                bt_span = parent_bt_span.start_span(name=span_name, type=span_type, start_time=start_time, **event)
            else:
                bt_span = start_span(name=span_name, type=span_type, start_time=start_time, **event)

            bt_span.set_current()
            self._bt_spans[id_] = _SpanRecord(bt_span=bt_span, start_time=start_time)

            return BaseSpan(id_=id_, parent_id=parent_span_id)

        def prepare_to_exit_span(
            self,
            id_: str,
            bound_args: "inspect.BoundArguments",
            instance: Any | None = None,
            result: Any | None = None,
            **kwargs: Any,
        ) -> BaseSpan | None:
            record = self._bt_spans.pop(id_, None)
            if record is None:
                return None

            bt_span = record.bt_span
            output = _extract_response_output(result)

            # Token usage is intentionally not logged on LlamaIndex spans.
            # LlamaIndex is an orchestration layer; provider integrations own
            # token accounting. Emitting usage here would double-count when
            # provider spans are also present.
            log_kwargs: dict[str, Any] = {}
            if output is not None:
                log_kwargs["output"] = output
            if log_kwargs:
                bt_span.log(**log_kwargs)

            bt_span.unset_current()
            bt_span.end()
            return self.open_spans.get(id_)

        def prepare_to_drop_span(
            self,
            id_: str,
            bound_args: "inspect.BoundArguments",
            instance: Any | None = None,
            err: BaseException | None = None,
            **kwargs: Any,
        ) -> BaseSpan | None:
            record = self._bt_spans.pop(id_, None)
            if record is None:
                return None

            bt_span = record.bt_span
            bt_span.log(error=f"{type(err).__name__}: {err}" if err else "Unknown error")

            bt_span.unset_current()
            bt_span.end()
            return self.open_spans.get(id_)

except ImportError:
    BraintrustSpanHandler = None  # type: ignore[assignment,misc]
