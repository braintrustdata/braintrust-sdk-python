"""Braintrust span handler for LlamaIndex instrumentation.

Implements LlamaIndex's BaseSpanHandler to capture all span lifecycle events
and map them to Braintrust spans with rich metadata extraction.
"""

import inspect
import logging
import time
from typing import Any

from braintrust.logger import NOOP_SPAN, Span, current_span, init_logger, start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.version import VERSION as sdk_version


_logger = logging.getLogger("braintrust.integrations.llamaindex")

_INTEGRATION_NAME = "llamaindex-py"


def _safe_str(obj: Any) -> str | None:
    """Safely convert object to string representation."""
    if obj is None:
        return None
    try:
        return str(obj)
    except Exception:
        return None


def _extract_query_text(query: Any) -> str | None:
    """Extract query text from a string or QueryBundle."""
    if isinstance(query, str):
        return query
    if hasattr(query, "query_str"):
        return query.query_str
    return _safe_str(query)


def _extract_messages(messages: Any) -> list[dict[str, Any]] | None:
    """Extract message dicts from LlamaIndex ChatMessage list."""
    if not messages:
        return None
    result = []
    for msg in messages:
        entry: dict[str, Any] = {}
        if hasattr(msg, "role"):
            entry["role"] = str(msg.role.value) if hasattr(msg.role, "value") else str(msg.role)
        if hasattr(msg, "content"):
            entry["content"] = msg.content
        elif hasattr(msg, "blocks"):
            text_parts = []
            for block in msg.blocks:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            if text_parts:
                entry["content"] = "\n".join(text_parts)
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
            entry["additional_kwargs"] = msg.additional_kwargs
        result.append(entry)
    return result


def _extract_response_output(result: Any) -> Any:
    """Extract output from a LlamaIndex response object."""
    if result is None:
        return None

    # ChatResponse
    if hasattr(result, "message") and hasattr(result, "raw"):
        output: dict[str, Any] = {}
        msg = result.message
        if msg:
            output["role"] = str(msg.role.value) if hasattr(msg.role, "value") else str(msg.role)
            if hasattr(msg, "content"):
                output["content"] = msg.content
            elif hasattr(msg, "blocks"):
                text_parts = []
                for block in msg.blocks:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                if text_parts:
                    output["content"] = "\n".join(text_parts)
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                output["additional_kwargs"] = msg.additional_kwargs
        if hasattr(result, "logprobs") and result.logprobs:
            output["logprobs"] = result.logprobs
        return output

    # CompletionResponse
    if hasattr(result, "text") and hasattr(result, "raw"):
        output = {"text": result.text}
        if hasattr(result, "logprobs") and result.logprobs:
            output["logprobs"] = result.logprobs
        if hasattr(result, "additional_kwargs") and result.additional_kwargs:
            output["additional_kwargs"] = result.additional_kwargs
        return output

    # Query Response (has response attribute with text)
    if hasattr(result, "response") and hasattr(result, "source_nodes"):
        output = {"response": result.response}
        if result.source_nodes:
            output["source_nodes"] = _extract_nodes(result.source_nodes)
        if hasattr(result, "metadata") and result.metadata:
            output["metadata"] = result.metadata
        return output

    # List of NodeWithScore
    if isinstance(result, list) and result and hasattr(result[0], "node"):
        return _extract_nodes(result)

    # String
    if isinstance(result, str):
        return result

    return _safe_str(result)


def _extract_nodes(nodes: list[Any]) -> list[dict[str, Any]]:
    """Extract structured data from NodeWithScore list."""
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


def _extract_token_usage(raw: Any) -> dict[str, int | float]:
    """Extract token usage from the raw provider response.

    LlamaIndex stores the raw provider response in ChatResponse.raw / CompletionResponse.raw.
    This contains the actual token counts that OTEL-based integrations often miss.
    """
    metrics: dict[str, int | float] = {}
    if raw is None:
        return metrics

    # OpenAI-style: raw is a ChatCompletion or similar with .usage
    usage = getattr(raw, "usage", None)
    if usage is None and isinstance(raw, dict):
        usage = raw.get("usage")

    if usage is not None:
        if isinstance(usage, dict):
            metrics.update(
                {
                    k: v
                    for k, v in {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }.items()
                    if v is not None
                }
            )
            # Cache tokens (OpenAI)
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                cached = prompt_details.get("cached_tokens")
                if cached is not None:
                    metrics["prompt_cached_tokens"] = cached
        else:
            for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(usage, attr, None)
                if val is not None:
                    metrics[attr] = val
            # Cache tokens
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached = getattr(prompt_details, "cached_tokens", None)
                if cached is not None:
                    metrics["prompt_cached_tokens"] = cached

    # Anthropic-style: raw has input_tokens/output_tokens at top level
    if not metrics:
        for key_map in [
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ]:
            src, dst = key_map
            val = getattr(raw, src, None)
            if val is None and isinstance(raw, dict):
                val = raw.get(src)
            if val is not None:
                metrics[dst] = val
        if "prompt_tokens" in metrics and "completion_tokens" in metrics:
            metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]

    return metrics


def _classify_instance(instance: Any) -> tuple[SpanTypeAttribute, str]:
    """Determine the Braintrust span type and name from a LlamaIndex instance.

    Returns (span_type, span_name).
    """
    if instance is None:
        return SpanTypeAttribute.TASK, "llamaindex"

    cls_name = type(instance).__name__

    # Check class hierarchy using string names to avoid importing optional packages
    mro_names = {c.__name__ for c in type(instance).__mro__}

    if "BaseLLM" in mro_names or "LLM" in mro_names:
        model = getattr(instance, "model", None) or getattr(instance, "model_name", None) or ""
        name = f"{cls_name}" if not model else f"{cls_name} ({model})"
        return SpanTypeAttribute.LLM, name

    if "BaseEmbedding" in mro_names:
        model = getattr(instance, "model_name", None) or getattr(instance, "model", None) or ""
        name = f"{cls_name}" if not model else f"{cls_name} ({model})"
        return SpanTypeAttribute.FUNCTION, name

    if "BaseRetriever" in mro_names:
        return SpanTypeAttribute.FUNCTION, cls_name

    if "BaseSynthesizer" in mro_names:
        return SpanTypeAttribute.FUNCTION, cls_name

    if "BaseQueryEngine" in mro_names:
        return SpanTypeAttribute.TASK, cls_name

    if "BaseAgent" in mro_names or "AgentRunner" in mro_names:
        return SpanTypeAttribute.TASK, cls_name

    if "BaseTool" in mro_names or "FunctionTool" in mro_names:
        tool_name = getattr(instance, "name", None) or cls_name
        return SpanTypeAttribute.TOOL, tool_name

    if "BaseNodeParser" in mro_names or "NodeParser" in mro_names or "SentenceSplitter" in mro_names:
        return SpanTypeAttribute.FUNCTION, cls_name

    if "BaseNodePostprocessor" in mro_names:
        return SpanTypeAttribute.FUNCTION, cls_name

    if "Workflow" in mro_names:
        return SpanTypeAttribute.TASK, cls_name

    return SpanTypeAttribute.TASK, cls_name


def _extract_instance_metadata(instance: Any) -> dict[str, Any]:
    """Extract useful metadata from a LlamaIndex instance."""
    if instance is None:
        return {}

    metadata: dict[str, Any] = {"class": type(instance).__name__}
    mro_names = {c.__name__ for c in type(instance).__mro__}

    if "BaseLLM" in mro_names or "LLM" in mro_names:
        for attr in ("model", "model_name", "temperature", "max_tokens", "max_retries"):
            val = getattr(instance, attr, None)
            if val is not None:
                metadata[attr] = val

    elif "BaseEmbedding" in mro_names:
        for attr in ("model_name", "model", "embed_batch_size"):
            val = getattr(instance, attr, None)
            if val is not None:
                metadata[attr] = val

    elif "BaseRetriever" in mro_names:
        for attr in ("similarity_top_k",):
            val = getattr(instance, attr, None)
            if val is not None:
                metadata[attr] = val

    return metadata


def _extract_input_from_bound_args(bound_args: "inspect.BoundArguments", instance: Any) -> Any:
    """Extract meaningful input from function bound arguments."""
    args = bound_args.arguments

    # Remove 'self' if present
    args = {k: v for k, v in args.items() if k != "self"}

    # Common input patterns
    if "str_or_query_bundle" in args:
        return _extract_query_text(args["str_or_query_bundle"])
    if "query_str" in args:
        return args["query_str"]
    if "query" in args:
        return _extract_query_text(args["query"])
    if "prompt" in args:
        return args["prompt"]
    if "messages" in args:
        return _extract_messages(args["messages"])
    if "nodes" in args:
        return _extract_nodes(args["nodes"]) if args["nodes"] else None

    # For single-arg functions, return the arg directly
    if len(args) == 1:
        return next(iter(args.values()))

    # For multi-arg, return the dict (minus kwargs-style args)
    if args:
        return {k: _safe_str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for k, v in args.items()}

    return None


class _SpanRecord:
    """Internal record tracking a LlamaIndex span mapped to a Braintrust span."""

    __slots__ = ("bt_span", "start_time", "instance_type", "method_name")

    def __init__(self, bt_span: Span, start_time: float, instance_type: str, method_name: str):
        self.bt_span = bt_span
        self.start_time = start_time
        self.instance_type = instance_type
        self.method_name = method_name


try:
    from llama_index.core.instrumentation.span import BaseSpan
    from llama_index_instrumentation.span_handlers.base import BaseSpanHandler

    class BraintrustSpanHandler(BaseSpanHandler["BaseSpan"]):
        """Maps LlamaIndex spans to Braintrust spans.

        Registered on LlamaIndex's root dispatcher to capture all instrumented
        operations (LLM calls, retrieval, query engines, agents, etc.) and
        create corresponding Braintrust spans with rich metadata.
        """

        _bt_spans: dict[str, _SpanRecord] = {}

        def model_post_init(self, __context: Any) -> None:
            super().model_post_init(__context)
            self._bt_spans = {}

        @classmethod
        def class_name(cls) -> str:
            return "BraintrustSpanHandler"

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

            # Determine method name from span id (format: "ClassName.method-uuid")
            method_name = ""
            if id_ and "-" in id_:
                prefix = id_.rsplit("-", 5)[0] if id_.count("-") >= 5 else id_.split("-")[0]
                if "." in prefix:
                    method_name = prefix.split(".", 1)[1] if "." in prefix else prefix

            instance_metadata = _extract_instance_metadata(instance)
            input_data = _extract_input_from_bound_args(bound_args, instance)

            metadata: dict[str, Any] = {
                **instance_metadata,
                "braintrust": {
                    "integration_name": _INTEGRATION_NAME,
                    "sdk_version": sdk_version,
                    "language": "python",
                },
            }
            if method_name:
                metadata["method"] = method_name
            if tags:
                metadata["tags"] = tags

            # Find parent Braintrust span
            parent_bt_span = None
            if parent_span_id and parent_span_id in self._bt_spans:
                parent_bt_span = self._bt_spans[parent_span_id].bt_span
            else:
                cs = current_span()
                if cs != NOOP_SPAN:
                    parent_bt_span = cs

            event: dict[str, Any] = {"metadata": metadata}
            if input_data is not None:
                event["input"] = input_data

            try:
                if parent_bt_span is not None:
                    bt_span = parent_bt_span.start_span(
                        name=span_name,
                        type=span_type,
                        start_time=start_time,
                        **event,
                    )
                else:
                    bt_span = start_span(
                        name=span_name,
                        type=span_type,
                        start_time=start_time,
                        **event,
                    )

                if bt_span == NOOP_SPAN:
                    bt_span = init_logger().start_span(
                        name=span_name,
                        type=span_type,
                        start_time=start_time,
                        **event,
                    )

                bt_span.set_current()

                record = _SpanRecord(
                    bt_span=bt_span,
                    start_time=start_time,
                    instance_type=type(instance).__name__ if instance else "unknown",
                    method_name=method_name,
                )
                self._bt_spans[id_] = record

            except Exception:
                _logger.debug("Failed to create Braintrust span for %s", id_, exc_info=True)

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

            # Extract token metrics from raw provider response
            metrics: dict[str, int | float] = {}
            raw = getattr(result, "raw", None)
            if raw is not None:
                metrics.update(_extract_token_usage(raw))

            log_kwargs: dict[str, Any] = {}
            if output is not None:
                log_kwargs["output"] = output
            if metrics:
                log_kwargs["metrics"] = metrics

            if log_kwargs:
                bt_span.log(**log_kwargs)

            try:
                bt_span.unset_current()
            except ValueError as e:
                if "was created in a different Context" not in str(e):
                    raise

            bt_span.end()

            span_obj = self.open_spans.get(id_)
            return span_obj

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

            bt_span.log(error=str(err) if err else "Unknown error")

            try:
                bt_span.unset_current()
            except ValueError as e:
                if "was created in a different Context" not in str(e):
                    raise

            bt_span.end()

            span_obj = self.open_spans.get(id_)
            return span_obj

except ImportError:
    BraintrustSpanHandler = None  # type: ignore[assignment,misc]
