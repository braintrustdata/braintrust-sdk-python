"""Braintrust event handler for LlamaIndex instrumentation.

Enriches Braintrust spans with detailed event data that goes beyond what
OTEL-based integrations capture: LLM model configuration, token usage details,
retrieval scores, synthesis context chunks, reranking parameters, and more.
"""

import logging
from typing import Any

from braintrust.logger import current_span


_logger = logging.getLogger("braintrust.integrations.llamaindex")


def _safe_model_dict(model_dict: Any) -> dict[str, Any] | None:
    """Safely extract model configuration dict."""
    if not model_dict or not isinstance(model_dict, dict):
        return None
    safe = {}
    for k, v in model_dict.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
    return safe if safe else None


def _extract_node_details(nodes: Any) -> list[dict[str, Any]] | None:
    """Extract detailed node information including scores and metadata."""
    if not nodes:
        return None
    result = []
    for nws in nodes:
        entry: dict[str, Any] = {}
        if hasattr(nws, "score") and nws.score is not None:
            entry["score"] = nws.score
        node = nws.node if hasattr(nws, "node") else nws
        if hasattr(node, "text"):
            text = node.text
            entry["text"] = text[:500] + "..." if len(text) > 500 else text
        if hasattr(node, "id_"):
            entry["node_id"] = node.id_
        if hasattr(node, "metadata") and node.metadata:
            entry["metadata"] = node.metadata
        if hasattr(node, "start_char_idx"):
            entry["start_char_idx"] = node.start_char_idx
        if hasattr(node, "end_char_idx"):
            entry["end_char_idx"] = node.end_char_idx
        result.append(entry)
    return result if result else None


try:
    from llama_index_instrumentation.event_handlers.base import BaseEventHandler
    from llama_index.core.instrumentation.events import BaseEvent

    class BraintrustEventHandler(BaseEventHandler):
        """Enriches Braintrust spans with LlamaIndex event-level detail.

        Captures data that standard OTEL instrumentors miss:
        - LLM model configuration (model_dict) at call time
        - Detailed retrieval results with scores and node metadata
        - Synthesis context chunks
        - Reranking parameters and results
        - Embedding model configuration
        """

        @classmethod
        def class_name(cls) -> str:
            return "BraintrustEventHandler"

        def handle(self, event: "BaseEvent", **kwargs: Any) -> None:
            span = current_span()
            if not span or str(span) == "NOOP_SPAN":
                return

            event_cls_name = type(event).__name__

            try:
                self._handle_event(event, event_cls_name, span)
            except Exception:
                _logger.debug("Failed to handle event %s", event_cls_name, exc_info=True)

        def _handle_event(self, event: Any, cls_name: str, span: Any) -> None:
            # ── LLM events ──
            if cls_name == "LLMChatStartEvent":
                metadata: dict[str, Any] = {}
                model_config = _safe_model_dict(getattr(event, "model_dict", None))
                if model_config:
                    metadata["model_config"] = model_config
                additional = getattr(event, "additional_kwargs", None)
                if additional:
                    metadata["additional_kwargs"] = additional
                if metadata:
                    span.log(metadata=metadata)

            elif cls_name == "LLMChatEndEvent":
                response = getattr(event, "response", None)
                if response:
                    self._log_llm_response_metrics(response, span)

            elif cls_name == "LLMCompletionStartEvent":
                metadata = {}
                model_config = _safe_model_dict(getattr(event, "model_dict", None))
                if model_config:
                    metadata["model_config"] = model_config
                additional = getattr(event, "additional_kwargs", None)
                if additional:
                    metadata["additional_kwargs"] = additional
                if metadata:
                    span.log(metadata=metadata)

            elif cls_name == "LLMCompletionEndEvent":
                response = getattr(event, "response", None)
                if response:
                    self._log_llm_response_metrics(response, span)

            # ── Retrieval events ──
            elif cls_name == "RetrievalEndEvent":
                nodes = getattr(event, "nodes", None)
                if nodes:
                    node_details = _extract_node_details(nodes)
                    if node_details:
                        metadata = {
                            "retrieved_nodes": node_details,
                            "num_nodes_retrieved": len(nodes),
                        }
                        scores = [n.get("score") for n in node_details if n.get("score") is not None]
                        if scores:
                            metadata["retrieval_scores"] = {
                                "min": min(scores),
                                "max": max(scores),
                                "mean": sum(scores) / len(scores),
                            }
                        span.log(metadata=metadata)

            # ── Synthesis events ──
            elif cls_name == "GetResponseStartEvent":
                query_str = getattr(event, "query_str", None)
                text_chunks = getattr(event, "text_chunks", None)
                if text_chunks:
                    metadata = {"num_text_chunks": len(text_chunks)}
                    total_chars = sum(len(c) for c in text_chunks)
                    metadata["total_context_chars"] = total_chars
                    span.log(metadata=metadata)

            # ── Embedding events ──
            elif cls_name == "EmbeddingStartEvent":
                model_config = _safe_model_dict(getattr(event, "model_dict", None))
                if model_config:
                    span.log(metadata={"embedding_model_config": model_config})

            elif cls_name == "EmbeddingEndEvent":
                chunks = getattr(event, "chunks", None)
                embeddings = getattr(event, "embeddings", None)
                if chunks and embeddings:
                    span.log(
                        metadata={
                            "num_chunks_embedded": len(chunks),
                            "embedding_dimensions": len(embeddings[0]) if embeddings else None,
                        }
                    )

            # ── Rerank events ──
            elif cls_name == "ReRankStartEvent":
                metadata = {}
                top_n = getattr(event, "top_n", None)
                model_name = getattr(event, "model_name", None)
                nodes = getattr(event, "nodes", None)
                if top_n is not None:
                    metadata["rerank_top_n"] = top_n
                if model_name:
                    metadata["rerank_model"] = model_name
                if nodes:
                    metadata["rerank_input_count"] = len(nodes)
                if metadata:
                    span.log(metadata=metadata)

            elif cls_name == "ReRankEndEvent":
                nodes = getattr(event, "nodes", None)
                if nodes:
                    node_details = _extract_node_details(nodes)
                    if node_details:
                        span.log(
                            metadata={
                                "reranked_nodes": node_details,
                                "rerank_output_count": len(nodes),
                            }
                        )

        def _log_llm_response_metrics(self, response: Any, span: Any) -> None:
            """Extract and log metrics from an LLM response."""
            raw = getattr(response, "raw", None)
            if raw is None:
                return

            from braintrust.integrations.llamaindex.span_handler import _extract_token_usage

            metrics = _extract_token_usage(raw)
            if metrics:
                span.log(metrics=metrics)

            # Extract model name from raw response
            model = getattr(raw, "model", None)
            if model is None and isinstance(raw, dict):
                model = raw.get("model")
            if model:
                span.log(metadata={"model": model})

except ImportError:
    BraintrustEventHandler = None  # type: ignore[assignment,misc]
