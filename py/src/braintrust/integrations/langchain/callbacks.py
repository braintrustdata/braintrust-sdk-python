import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict
from uuid import UUID

from braintrust.generated_types import SpanAttributes
from braintrust.logger import NOOP_SPAN, Logger, Span, current_span, init_logger
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.version import VERSION as sdk_version
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs.llm_result import LLMResult
from tenacity import RetryCallState
from typing_extensions import NotRequired


_INSTRUMENTATION = "langchain-auto"
_INTEGRATION_NAME = "langchain-py"

_logger = logging.getLogger("braintrust.wrappers.langchain")


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


class LogEvent(TypedDict):
    input: NotRequired[Any]
    output: NotRequired[Any]
    expected: NotRequired[Any]
    error: NotRequired[str]
    tags: NotRequired[Sequence[str] | None]
    scores: NotRequired[Mapping[str, int | float]]
    metadata: NotRequired[Mapping[str, Any]]
    metrics: NotRequired[Mapping[str, int | float]]
    id: NotRequired[str]
    dataset_record_id: NotRequired[str]


# Only aliases where the package name doesn't strip cleanly to the provider.
_PROVIDER_ALIASES: dict[str, str] = {
    "langchain_google_vertexai": "google",
    "langchain_google_genai": "google",
    "langchain_google": "google",
    "langchain_aws": "aws",
    "langchain_bedrock": "aws",
    "langchain_azure_ai": "azure",
    "langchain_azure": "azure",
    "langchain_mistralai": "mistral",
}


def _provider_from_serialized(serialized: Mapping[str, Any] | None) -> str | None:
    if not serialized:
        return None
    for part in serialized.get("id") or []:
        if not isinstance(part, str):
            continue
        head = part.lower().split(".", 1)[0]
        if head in _PROVIDER_ALIASES:
            return _PROVIDER_ALIASES[head]
        if head.startswith("langchain_"):
            return head[len("langchain_") :]
    return None


def _resolve_name(name: str | None, serialized: Mapping[str, Any] | None, default: str) -> str:
    return name or (serialized or {}).get("name") or last_item((serialized or {}).get("id") or []) or default


_TOOL_KEYS = ("tools", "functions")
_TOOL_CHOICE_KEYS = ("tool_choice", "function_call")


def _split_tools(invocation_params: Mapping[str, Any] | None) -> tuple[dict[str, Any], Any, Any]:
    if not invocation_params:
        return {}, None, None
    tools: Any = None
    tool_choice: Any = None
    remaining: dict[str, Any] = {}
    for key, value in invocation_params.items():
        if key in _TOOL_KEYS and tools is None and value:
            tools = value
            continue
        if key in _TOOL_CHOICE_KEYS and tool_choice is None and value is not None:
            tool_choice = value
            continue
        remaining[key] = value
    return remaining, tools, tool_choice


class BraintrustCallbackHandler(BaseCallbackHandler):
    root_run_id: UUID | None = None

    def __init__(
        self,
        logger: Logger | Span | None = None,
        debug: bool = False,
    ):
        self.logger = logger
        self.spans: dict[UUID, Span] = {}
        self.debug = debug
        self.skipped_runs: set[UUID] = set()
        # Preserve memory logger context across async callbacks.
        self.run_inline = True

        self._start_times: dict[UUID, float] = {}
        self._first_token_times: dict[UUID, float] = {}
        self._ttft_ms: dict[UUID, float] = {}

    def _start_span(
        self,
        parent_run_id: UUID | None,
        run_id: UUID,
        name: str | None = None,
        type: SpanTypeAttribute | None = SpanTypeAttribute.TASK,
        span_attributes: SpanAttributes | Mapping[str, Any] | None = None,
        start_time: float | None = None,
        set_current: bool | None = None,
        parent: str | None = None,
        event: LogEvent | None = None,
    ) -> Span | None:
        if run_id in self.spans:
            _logger.warning(f"Span already exists for run_id {run_id} (this is likely a bug)")
            return

        if not parent_run_id:
            self.root_run_id = run_id

        current_parent = current_span()
        parent_span = None
        if parent_run_id and parent_run_id in self.spans:
            parent_span = self.spans[parent_run_id]
        elif current_parent != NOOP_SPAN:
            parent_span = current_parent
        elif self.logger is not None:
            parent_span = self.logger

        if event is None:
            event = {}

        tags = event.get("tags") or []
        event = {
            **event,
            "tags": None,
            "metadata": {
                "tags": tags,
                **(event.get("metadata") or {}),
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "braintrust": {
                    "integration_name": _INTEGRATION_NAME,
                    "sdk_version": sdk_version,
                    "language": "python",
                },
            },
        }

        if parent_span is None:
            span = start_span(
                name=name,
                type=type,
                span_attributes=span_attributes,
                start_time=start_time,
                set_current=set_current,
                parent=parent,
                **event,
            )
        else:
            span = parent_span.start_span(
                name=name,
                type=type,
                span_attributes=span_attributes,
                start_time=start_time,
                set_current=set_current,
                parent=parent,
                internal={"instrumentation": _INSTRUMENTATION},
                **event,
            )

        if self.logger != NOOP_SPAN and span == NOOP_SPAN:
            _logger.warning(
                "Braintrust logging not configured. Pass a `logger`, call `init_logger`, or run an experiment to configure Braintrust logging. Setting up a default."
            )
            span = init_logger().start_span(
                name=name,
                type=type,
                span_attributes=span_attributes,
                start_time=start_time,
                set_current=set_current,
                parent=parent,
                internal={"instrumentation": _INSTRUMENTATION},
                **event,
            )

        span.set_current()
        self.spans[run_id] = span
        return span

    def _end_span(
        self,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        input: Any | None = None,
        output: Any | None = None,
        expected: Any | None = None,
        error: str | None = None,
        tags: Sequence[str] | None = None,
        scores: Mapping[str, int | float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        metrics: Mapping[str, int | float] | None = None,
        dataset_record_id: str | None = None,
    ) -> None:
        if run_id not in self.spans:
            return

        if run_id in self.skipped_runs:
            self.skipped_runs.discard(run_id)
            return

        span = self.spans.pop(run_id)

        if self.root_run_id == run_id:
            self.root_run_id = None

        span.log(
            input=input,
            output=output,
            expected=expected,
            error=error,
            tags=None,
            scores=scores,
            metadata={
                **({"tags": tags} if tags else {}),
                **(metadata or {}),
            },
            metrics=metrics,
            dataset_record_id=dataset_record_id,
        )

        # Async callbacks may unset from a different context; span state is
        # tracked in self.spans, so this ValueError is benign.
        try:
            span.unset_current()
        except ValueError as e:
            if "was created in a different Context" in str(e):
                pass
            else:
                raise

        span.end()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id, error=str(error))
        self._start_times.pop(run_id, None)
        self._first_token_times.pop(run_id, None)
        self._ttft_ms.pop(run_id, None)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id, error=str(error))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id, error=str(error))

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id, error=str(error))

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_span(
            parent_run_id,
            run_id,
            type=SpanTypeAttribute.TOOL,
            name=action.tool,
            event={"input": action},
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id, output=finish)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tags = tags or []
        if "langsmith:hidden" in tags:
            self.skipped_runs.add(run_id)
            return

        metadata = metadata or {}
        resolved_name = (
            name
            or metadata.get("langgraph_node")
            or (serialized or {}).get("name")
            or last_item((serialized or {}).get("id") or [])
            or "Chain"
        )

        self._start_span(
            parent_run_id,
            run_id,
            name=resolved_name,
            event={
                "input": inputs,
                "tags": tags,
                "metadata": {"metadata": metadata},
            },
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id, output=outputs, tags=tags)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.perf_counter()
        self._first_token_times.pop(run_id, None)
        self._ttft_ms.pop(run_id, None)

        span_metadata: dict[str, Any] = {"metadata": metadata or {}}
        provider = _provider_from_serialized(serialized)
        if provider:
            span_metadata["provider"] = provider

        self._start_span(
            parent_run_id,
            run_id,
            name=_resolve_name(name, serialized, "LLM"),
            type=SpanTypeAttribute.LLM,
            event={"input": prompts, "tags": tags, "metadata": span_metadata},
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list["BaseMessage"]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.perf_counter()
        self._first_token_times.pop(run_id, None)
        self._ttft_ms.pop(run_id, None)

        remaining_params, tools, tool_choice = _split_tools(invocation_params)
        provider = _provider_from_serialized(serialized)

        span_metadata: dict[str, Any] = {
            "invocation_params": remaining_params,
            "metadata": metadata or {},
        }
        if tools:
            span_metadata["tools"] = tools
        if tool_choice is not None:
            span_metadata["tool_choice"] = tool_choice
        if provider:
            span_metadata["provider"] = provider

        self._start_span(
            parent_run_id,
            run_id,
            name=_resolve_name(name, serialized, "Chat Model"),
            type=SpanTypeAttribute.LLM,
            event={"input": messages, "tags": tags, "metadata": span_metadata},
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id not in self.spans:
            return

        metrics = _get_metrics_from_response(response)
        ttft = self._ttft_ms.pop(run_id, None)
        if ttft is not None:
            metrics["time_to_first_token"] = ttft

        self._start_times.pop(run_id, None)
        self._first_token_times.pop(run_id, None)

        model_name, provider = _model_and_provider_from_response(response)
        end_metadata: dict[str, Any] = {}
        if model_name:
            end_metadata["model"] = model_name
        if provider:
            end_metadata["provider"] = provider

        self._end_span(
            run_id,
            output=response,
            metrics=metrics,
            tags=tags,
            metadata=end_metadata,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_span(
            parent_run_id,
            run_id,
            name=_resolve_name(name, serialized, "Tool"),
            type=SpanTypeAttribute.TOOL,
            event={
                "input": inputs if inputs is not None else safe_parse_serialized_json(input_str),
                "tags": tags,
                "metadata": {"metadata": metadata or {}},
            },
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id, output=output)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_span(
            parent_run_id,
            run_id,
            name=_resolve_name(name, serialized, "Retriever"),
            type=SpanTypeAttribute.FUNCTION,
            event={
                "input": query,
                "tags": tags,
                "metadata": {"metadata": metadata or {}},
            },
        )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id, output=documents)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: "GenerationChunk | ChatGenerationChunk | None" = None,  # type: ignore
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id not in self._first_token_times:
            now = time.perf_counter()
            self._first_token_times[run_id] = now
            start = self._start_times.get(run_id)
            if start is not None:
                self._ttft_ms[run_id] = now - start

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pass


def clean_object(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in obj.items()
        if v is not None and not (isinstance(v, list) and not v) and not (isinstance(v, dict) and not v)
    }


def safe_parse_serialized_json(input_str: str) -> Any:
    try:
        return json.loads(input_str)
    except Exception:
        return input_str


def last_item(items: list[Any]) -> Any:
    return items[-1] if items else None


def _walk_generations(response: LLMResult):
    for generations in response.generations or []:
        yield from generations or []


def _model_and_provider_from_response(response: LLMResult) -> tuple[str | None, str | None]:
    model_name: str | None = None
    provider: str | None = None
    for generation in _walk_generations(response):
        message = getattr(generation, "message", None)
        if not message:
            continue
        rmeta = getattr(message, "response_metadata", None)
        if not isinstance(rmeta, dict):
            continue
        if not model_name:
            model_name = rmeta.get("model_name") or None
        if not provider:
            prov = rmeta.get("model_provider")
            if isinstance(prov, str) and prov:
                provider = prov.lower()
        if model_name and provider:
            break

    if not model_name:
        llm_output: dict[str, Any] = response.llm_output or {}
        model_name = llm_output.get("model_name") or llm_output.get("model") or None

    return model_name, provider


def _get_metrics_from_response(response: LLMResult):
    metrics: dict[str, Any] = {}

    for generation in _walk_generations(response):
        message = getattr(generation, "message", None)
        if not message:
            continue

        usage_metadata = getattr(message, "usage_metadata", None)
        if not (usage_metadata and isinstance(usage_metadata, dict)):
            continue

        metrics.update(
            clean_object(
                {
                    "total_tokens": usage_metadata.get("total_tokens"),
                    "prompt_tokens": usage_metadata.get("input_tokens"),
                    "completion_tokens": usage_metadata.get("output_tokens"),
                }
            )
        )

        input_token_details = usage_metadata.get("input_token_details")
        if not (input_token_details and isinstance(input_token_details, dict)):
            continue

        cache_read = input_token_details.get("cache_read")
        cache_creation = input_token_details.get("cache_creation")
        cache_creation_5m = input_token_details.get("ephemeral_5m_input_tokens")
        cache_creation_1h = input_token_details.get("ephemeral_1h_input_tokens")
        has_cache_creation_split = cache_creation_5m is not None or cache_creation_1h is not None

        if cache_read is not None:
            metrics["prompt_cached_tokens"] = cache_read
        if has_cache_creation_split:
            if cache_creation_5m is not None:
                metrics["prompt_cache_creation_5m_tokens"] = cache_creation_5m
            if cache_creation_1h is not None:
                metrics["prompt_cache_creation_1h_tokens"] = cache_creation_1h
            effective_cache_creation = (cache_creation_5m or 0) + (cache_creation_1h or 0)
        else:
            if cache_creation is not None:
                metrics["prompt_cache_creation_tokens"] = cache_creation
            effective_cache_creation = cache_creation or 0
        cache_tokens = (cache_read or 0) + effective_cache_creation

        prompt_tokens = metrics.get("prompt_tokens")
        completion_tokens = metrics.get("completion_tokens")
        total_tokens = metrics.get("total_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            # LangChain's input_token_details is a breakdown of input_tokens.
            # Fold cache tokens back into prompt/total only if the integration
            # reported uncached-input-only (cache tokens exceeding prompt total).
            if cache_tokens > prompt_tokens and total_tokens == prompt_tokens + completion_tokens:
                prompt_tokens += cache_tokens
                metrics["prompt_tokens"] = prompt_tokens
                metrics["total_tokens"] = total_tokens + cache_tokens
            metrics["tokens"] = prompt_tokens + completion_tokens

    if not metrics or not any(metrics.values()):
        llm_output: dict[str, Any] = response.llm_output or {}
        metrics = llm_output.get("token_usage") or llm_output.get("estimatedTokens") or {}

    return clean_object(metrics)
