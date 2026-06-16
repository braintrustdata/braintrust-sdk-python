"""Tracing helpers for the boto3 Bedrock Runtime integration."""

import io
import json
import time
from typing import Any

from braintrust.integrations.utils import (
    _log_and_end_span,
    _log_error_and_end_span,
    _materialize_attachment,
    _timing_metrics,
)
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import is_numeric


_PROVIDER = "bedrock"

_CONVERSE_METADATA_KEYS = (
    "guardrailConfig",
    "toolConfig",
    "additionalModelRequestFields",
    "additionalModelResponseFieldPaths",
    "performanceConfig",
    "requestMetadata",
)
_INFERENCE_CONFIG_KEYS = {
    "maxTokens": "max_tokens",
    "temperature": "temperature",
    "topP": "top_p",
    "stopSequences": "stop_sequences",
}


def _converse_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span = _start_span(
        "bedrock.converse",
        _converse_input(kwargs),
        _converse_metadata(kwargs, endpoint="converse"),
    )
    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    metadata = _converse_response_metadata(result)
    metrics = {
        **_timing_metrics(start_time, time.time()),
        **_converse_usage_metrics(result.get("usage") if isinstance(result, dict) else None),
        **_bedrock_latency_metrics(result.get("metrics") if isinstance(result, dict) else None),
    }
    _log_and_end_span(span, output=_converse_output(result), metrics=metrics, metadata=metadata or None)
    return result


def _converse_stream_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    span = _start_span(
        "bedrock.converse-stream",
        _converse_input(kwargs),
        _converse_metadata(kwargs, endpoint="converse-stream", stream=True),
    )
    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if isinstance(result, dict) and "stream" in result:
        result = dict(result)
        result["stream"] = _TracedConverseStream(result["stream"], span, start_time)
        return result

    _log_and_end_span(span, metrics=_timing_metrics(start_time, time.time()))
    return result


def _invoke_model_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    model_id = kwargs.get("modelId")
    span = _start_span(
        "bedrock.invoke_model",
        _invoke_model_input(kwargs),
        _invoke_model_metadata(model_id, endpoint="invoke_model"),
    )
    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    output = None
    if isinstance(result, dict) and "body" in result:
        body_bytes = _read_streaming_body(result["body"])
        if body_bytes is not None:
            output = _parse_json_bytes(body_bytes)
            result = dict(result)
            result["body"] = _replacement_streaming_body(body_bytes)
    output_usage = output.get("usage") if isinstance(output, dict) else None
    metrics = {
        **_timing_metrics(start_time, time.time()),
        **_converse_usage_metrics(output_usage),
        **_anthropic_invoke_usage_metrics(model_id, output),
    }
    _log_and_end_span(span, output=output, metrics=metrics)
    return result


def _invoke_model_stream_wrapper(wrapped, instance, args, kwargs):  # noqa: ARG001
    model_id = kwargs.get("modelId")
    span = _start_span(
        "bedrock.invoke_model_stream",
        _invoke_model_input(kwargs),
        _invoke_model_metadata(model_id, endpoint="invoke_model_stream", stream=True),
    )
    start_time = time.time()
    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        _log_error_and_end_span(span, error)
        raise

    if isinstance(result, dict) and "body" in result:
        result = dict(result)
        result["body"] = _TracedInvokeModelStream(result["body"], span, start_time, model_id)
        return result

    _log_and_end_span(span, metrics=_timing_metrics(start_time, time.time()))
    return result


def _start_span(name: str, span_input: Any, metadata: dict[str, Any]):
    return start_span(name=name, type=SpanTypeAttribute.LLM, input=span_input, metadata=metadata)


def _converse_metadata(kwargs: dict[str, Any], *, endpoint: str, stream: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": _PROVIDER, "endpoint": endpoint}
    model = kwargs.get("modelId")
    if model is not None:
        metadata["model"] = model
    inference_config = kwargs.get("inferenceConfig")
    if isinstance(inference_config, dict):
        for source_key, dest_key in _INFERENCE_CONFIG_KEYS.items():
            value = inference_config.get(source_key)
            if value is not None:
                metadata[dest_key] = value
    for key in _CONVERSE_METADATA_KEYS:
        value = kwargs.get(key)
        if value is not None:
            metadata[_camel_metadata_key(key)] = value
    if stream:
        metadata["stream"] = True
    return metadata


def _invoke_model_metadata(model_id: Any, *, endpoint: str, stream: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": _PROVIDER, "endpoint": endpoint}
    if model_id is not None:
        metadata["model"] = model_id
    if stream:
        metadata["stream"] = True
    return metadata


def _camel_metadata_key(key: str) -> str:
    return {
        "guardrailConfig": "guardrail_config",
        "toolConfig": "tool_config",
        "additionalModelRequestFields": "additional_model_request_fields",
        "additionalModelResponseFieldPaths": "additional_model_response_field_paths",
        "performanceConfig": "performance_config",
        "requestMetadata": "request_metadata",
    }.get(key, key)


def _converse_input(kwargs: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages: list[dict[str, Any]] = []
    system = _system_blocks_to_content(kwargs.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for message in kwargs.get("messages") or []:
        if not isinstance(message, dict):
            continue
        messages.append(_message_to_json(message))
    return messages or None


def _message_to_json(message: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    role = message.get("role")
    if role is not None:
        result["role"] = role
    result["content"] = _content_blocks_to_json(message.get("content") or [])
    return result


def _system_blocks_to_content(blocks: Any) -> list[Any]:
    if not isinstance(blocks, list):
        return []
    result = []
    for block in blocks:
        if isinstance(block, dict) and "text" in block:
            result.append({"type": "text", "text": block["text"]})
        else:
            result.append(block)
    return result


def _content_blocks_to_json(blocks: Any) -> list[Any]:
    if not isinstance(blocks, list):
        return []
    return [_content_block_to_json(block) for block in blocks]


def _content_block_to_json(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    if "text" in block:
        return {"type": "text", "text": block["text"]}
    if "toolUse" in block:
        return _tool_use_to_json(block.get("toolUse"))
    if "toolResult" in block:
        return _tool_result_to_json(block.get("toolResult"))
    if "image" in block:
        return _image_to_json(block.get("image"))
    if "document" in block:
        return _document_to_json(block.get("document"))
    if "reasoningContent" in block:
        return _reasoning_to_json(block.get("reasoningContent"))
    return block


def _tool_use_to_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "tool_use", "value": value}
    result: dict[str, Any] = {"type": "tool_use"}
    if value.get("toolUseId") is not None:
        result["id"] = value["toolUseId"]
    if value.get("name") is not None:
        result["name"] = value["name"]
    if value.get("input") is not None:
        result["input"] = value["input"]
    return result


def _tool_result_to_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "tool_result", "value": value}
    result: dict[str, Any] = {"type": "tool_result"}
    if value.get("toolUseId") is not None:
        result["tool_use_id"] = value["toolUseId"]
    if value.get("status") is not None:
        result["status"] = value["status"]
    content = value.get("content")
    if isinstance(content, list):
        result["content"] = [_tool_result_content_to_json(block) for block in content]
    return result


def _tool_result_content_to_json(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    if "text" in block:
        return {"type": "text", "text": block["text"]}
    if "json" in block:
        return {"type": "json", "json": block["json"]}
    return block


def _image_to_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "image", "value": value}
    result: dict[str, Any] = {"type": "image"}
    image_format = value.get("format")
    if image_format is not None:
        result["format"] = image_format
    source = value.get("source")
    if isinstance(source, dict):
        source_result = dict(source)
        image_bytes = source_result.pop("bytes", None)
        if image_bytes is not None:
            mime_type = f"image/{image_format or 'png'}"
            resolved = _materialize_attachment(image_bytes, mime_type=mime_type, prefix="image")
            if resolved is not None:
                result["image_url"] = {"url": resolved.attachment}
            else:
                source_result["bytes"] = image_bytes
        result["source"] = source_result
    return result


def _document_to_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "document", "value": value}
    result: dict[str, Any] = {"type": "document"}
    for key in ("format", "name", "citations"):
        if value.get(key) is not None:
            result[key] = value[key]
    source = value.get("source")
    if isinstance(source, dict):
        source_result = dict(source)
        document_bytes = source_result.pop("bytes", None)
        if document_bytes is not None:
            mime_type = _document_mime_type(value.get("format"))
            resolved = _materialize_attachment(document_bytes, mime_type=mime_type, prefix="document")
            if resolved is not None:
                result["file"] = {"file_data": resolved.attachment, "filename": resolved.filename}
            else:
                source_result["bytes"] = document_bytes
        result["source"] = source_result
    return result


def _document_mime_type(document_format: Any) -> str:
    return {
        "pdf": "application/pdf",
        "csv": "text/csv",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "html": "text/html",
        "md": "text/markdown",
        "txt": "text/plain",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(str(document_format), "application/octet-stream")


def _reasoning_to_json(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "reasoning"}
    if not isinstance(value, dict):
        result["value"] = value
        return result
    reasoning_text = value.get("reasoningText")
    if isinstance(reasoning_text, dict):
        if reasoning_text.get("text") is not None:
            result["text"] = reasoning_text["text"]
        if reasoning_text.get("signature") is not None:
            result["signature"] = reasoning_text["signature"]
    return result


def _converse_output(result: Any) -> Any:
    if not isinstance(result, dict):
        return None
    output = result.get("output")
    if isinstance(output, dict) and isinstance(output.get("message"), dict):
        return [_message_to_json(output["message"])]
    return output


def _converse_response_metadata(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    metadata: dict[str, Any] = {}
    if result.get("stopReason") is not None:
        metadata["stop_reason"] = result["stopReason"]
    for key in ("additionalModelResponseFields", "trace"):
        if result.get(key) is not None:
            metadata[_camel_metadata_key(key)] = result[key]
    return metadata


def _converse_usage_metrics(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    metrics: dict[str, Any] = {}
    input_tokens = _numeric_or_zero(usage.get("inputTokens"))
    cache_read = _numeric_or_zero(usage.get("cacheReadInputTokens"))
    cache_write = _numeric_or_zero(usage.get("cacheWriteInputTokens"))
    if input_tokens or "inputTokens" in usage or cache_read or cache_write:
        metrics["prompt_tokens"] = input_tokens + cache_read + cache_write
    if is_numeric(usage.get("outputTokens")):
        metrics["completion_tokens"] = usage["outputTokens"]
    if is_numeric(usage.get("cacheReadInputTokens")):
        metrics["prompt_cached_tokens"] = usage["cacheReadInputTokens"]
    if is_numeric(usage.get("cacheWriteInputTokens")):
        metrics["prompt_cache_creation_tokens"] = usage["cacheWriteInputTokens"]
    if is_numeric(usage.get("totalTokens")):
        metrics["tokens"] = usage["totalTokens"]
    elif "prompt_tokens" in metrics and "completion_tokens" in metrics:
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    return metrics


def _bedrock_latency_metrics(metrics_payload: Any) -> dict[str, Any]:
    if not isinstance(metrics_payload, dict):
        return {}
    latency = metrics_payload.get("latencyMs")
    return {"bedrock_latency_ms": latency} if is_numeric(latency) else {}


def _numeric_or_zero(value: Any) -> Any:
    return value if is_numeric(value) else 0


def _invoke_model_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    model_id = kwargs.get("modelId")
    if model_id is not None:
        result["modelId"] = model_id
    body = kwargs.get("body")
    parsed_body = _parse_json_body(body)
    if parsed_body is not None:
        result["body"] = parsed_body
    for key in (
        "contentType",
        "accept",
        "trace",
        "guardrailIdentifier",
        "guardrailVersion",
        "performanceConfigLatency",
    ):
        value = kwargs.get(key)
        if value is not None:
            result[key] = value
    return result


def _parse_json_body(body: Any) -> Any:
    if body is None:
        return None
    if isinstance(body, str):
        return _parse_json_text(body)
    if isinstance(body, (bytes, bytearray)):
        return _parse_json_bytes(bytes(body))
    return None


def _parse_json_bytes(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return data.decode("utf-8", errors="replace")


def _parse_json_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _read_streaming_body(body: Any) -> bytes | None:
    read = getattr(body, "read", None)
    if not callable(read):
        return None
    data = read()
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return None


def _replacement_streaming_body(data: bytes) -> Any:
    from botocore.response import StreamingBody

    return StreamingBody(io.BytesIO(data), len(data))


def _anthropic_invoke_usage_metrics(model_id: Any, output: Any) -> dict[str, Any]:
    if not isinstance(model_id, str) or "anthropic.claude" not in model_id.lower():
        return {}
    if not isinstance(output, dict):
        return {}
    usage = output.get("usage")
    if not isinstance(usage, dict):
        return {}
    metrics: dict[str, Any] = {}
    input_tokens = _numeric_or_zero(usage.get("input_tokens"))
    cache_read = _numeric_or_zero(usage.get("cache_read_input_tokens"))
    cache_create = _numeric_or_zero(usage.get("cache_creation_input_tokens"))
    if input_tokens or "input_tokens" in usage or cache_read or cache_create:
        metrics["prompt_tokens"] = input_tokens + cache_read + cache_create
    if is_numeric(usage.get("output_tokens")):
        metrics["completion_tokens"] = usage["output_tokens"]
    if is_numeric(usage.get("cache_read_input_tokens")):
        metrics["prompt_cached_tokens"] = usage["cache_read_input_tokens"]
    if is_numeric(usage.get("cache_creation_input_tokens")):
        metrics["prompt_cache_creation_tokens"] = usage["cache_creation_input_tokens"]
    if "prompt_tokens" in metrics and "completion_tokens" in metrics:
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    return metrics


class _TracedConverseStream:
    """Wrap a botocore EventStream and log the final span when it drains."""

    def __init__(self, stream: Any, span: Any, start_time: float):
        self._stream = stream
        self._span = span
        self._start_time = start_time
        self._first_token_time: float | None = None
        self._message_role = "assistant"
        self._content_blocks: dict[int, dict[str, Any]] = {}
        self._builders: dict[int, list[str]] = {}
        self._usage: dict[str, Any] | None = None
        self._metrics_payload: dict[str, Any] | None = None
        self._stop_reason: str | None = None
        self._finished = False

    def __iter__(self):
        try:
            for event in self._stream:
                self._observe(event)
                yield event
        except Exception as error:
            self._finish(error)
            raise
        else:
            self._finish(None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def close(self):
        close = getattr(self._stream, "close", None)
        try:
            return close() if callable(close) else None
        finally:
            self._finish(None)

    def _observe(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        if self._first_token_time is None:
            self._first_token_time = time.time()
        if "messageStart" in event:
            role = event.get("messageStart", {}).get("role")
            if role is not None:
                self._message_role = role
        elif "contentBlockStart" in event:
            data = event["contentBlockStart"]
            idx = int(data.get("contentBlockIndex", 0))
            start = data.get("start") or {}
            if "toolUse" in start:
                self._content_blocks[idx] = _tool_use_to_json(start["toolUse"])
            else:
                self._content_blocks.setdefault(idx, {})
        elif "contentBlockDelta" in event:
            data = event["contentBlockDelta"]
            idx = int(data.get("contentBlockIndex", 0))
            block = self._content_blocks.setdefault(idx, {})
            delta = data.get("delta") or {}
            if "text" in delta:
                block["type"] = "text"
                self._builders.setdefault(idx, []).append(delta["text"])
            elif "toolUse" in delta:
                block["type"] = "tool_use"
                tool_delta = delta["toolUse"] or {}
                if tool_delta.get("input") is not None:
                    self._builders.setdefault(idx, []).append(tool_delta["input"])
            elif "reasoningContent" in delta:
                block["type"] = "reasoning"
                reasoning = delta["reasoningContent"] or {}
                if reasoning.get("text") is not None:
                    self._builders.setdefault(idx, []).append(reasoning["text"])
                if reasoning.get("signature") is not None:
                    block["signature"] = reasoning["signature"]
            elif "citation" in delta:
                block.setdefault("citations", []).append(delta["citation"])
        elif "messageStop" in event:
            self._stop_reason = event.get("messageStop", {}).get("stopReason")
        elif "metadata" in event:
            metadata = event["metadata"] or {}
            if isinstance(metadata.get("usage"), dict):
                self._usage = metadata["usage"]
            if isinstance(metadata.get("metrics"), dict):
                self._metrics_payload = metadata["metrics"]

    def _finish(self, error: BaseException | None) -> None:
        if self._finished:
            return
        self._finished = True
        if error is not None:
            _log_error_and_end_span(self._span, error)
            return
        for idx, pieces in self._builders.items():
            block = self._content_blocks.setdefault(idx, {})
            text = "".join(pieces)
            if block.get("type") == "tool_use":
                block["input"] = text
            else:
                block["text"] = text
        content = [self._content_blocks[idx] for idx in sorted(self._content_blocks)]
        output = [{"role": self._message_role, "content": content}] if content else None
        metadata = {"stop_reason": self._stop_reason} if self._stop_reason else None
        metrics = {
            **_timing_metrics(self._start_time, time.time(), self._first_token_time),
            **_converse_usage_metrics(self._usage),
            **_bedrock_latency_metrics(self._metrics_payload),
        }
        _log_and_end_span(self._span, output=output, metrics=metrics, metadata=metadata)


class _TracedInvokeModelStream:
    """Wrap an InvokeModelWithResponseStream EventStream and log observed chunks."""

    def __init__(self, stream: Any, span: Any, start_time: float, model_id: Any):
        self._stream = stream
        self._span = span
        self._start_time = start_time
        self._model_id = model_id
        self._first_token_time: float | None = None
        self._chunks: list[Any] = []
        self._finished = False

    def __iter__(self):
        try:
            for event in self._stream:
                self._observe(event)
                yield event
        except Exception as error:
            self._finish(error)
            raise
        else:
            self._finish(None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def close(self):
        close = getattr(self._stream, "close", None)
        try:
            return close() if callable(close) else None
        finally:
            self._finish(None)

    def _observe(self, event: Any) -> None:
        if self._first_token_time is None:
            self._first_token_time = time.time()
        if not isinstance(event, dict):
            return
        chunk = event.get("chunk")
        if isinstance(chunk, dict) and isinstance(chunk.get("bytes"), (bytes, bytearray)):
            self._chunks.append(_parse_json_bytes(bytes(chunk["bytes"])))
        elif event:
            self._chunks.append(event)

    def _finish(self, error: BaseException | None) -> None:
        if self._finished:
            return
        self._finished = True
        if error is not None:
            _log_error_and_end_span(self._span, error)
            return
        output = self._chunks or None
        usage_metrics: dict[str, Any] = {}
        if output:
            for chunk in output:
                usage_metrics.update(_anthropic_invoke_usage_metrics(self._model_id, chunk))
        metrics = {**_timing_metrics(self._start_time, time.time(), self._first_token_time), **usage_metrics}
        _log_and_end_span(self._span, output=output, metrics=metrics)
