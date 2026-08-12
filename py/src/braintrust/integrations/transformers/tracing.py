"""Tracing helpers for local Hugging Face Transformers pipelines."""

from collections.abc import Callable, Iterator, Mapping

from braintrust.integrations.utils import _log_and_end_span, _log_error_and_end_span, _tensor_shape
from braintrust.logger import start_span as _bt_start_span
from braintrust.span_types import SpanTypeAttribute


_INSTRUMENTATION = "transformers-auto"
_PROVIDER = "huggingface"
_REQUEST_METADATA_KEYS = (
    "temperature",
    "top_p",
    "stop",
    "do_sample",
    "top_k",
    "repetition_penalty",
)


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)


def _operation_name(task: str) -> str:
    return f"huggingface.transformers.{task.replace('-', '_')}"


def _model_identifier(instance: object) -> str | None:
    model = getattr(instance, "model", None)
    config = getattr(model, "config", None)
    for value in (
        getattr(config, "_name_or_path", None),
        getattr(config, "name_or_path", None),
        getattr(config, "model_id", None),
        getattr(model, "name", None),
    ):
        if value:
            return value
    return None


def _metadata(instance: object, kwargs: dict[str, object], task: str) -> dict[str, object]:
    metadata: dict[str, object] = {"provider": _PROVIDER}
    model = _model_identifier(instance)
    if model is not None:
        metadata["model"] = model

    metadata["task"] = task
    device = getattr(instance, "device", None)
    if device is not None:
        metadata["device"] = device

    pipeline_dtype = getattr(getattr(instance, "model", None), "dtype", None)
    if pipeline_dtype is None:
        pipeline_dtype = getattr(instance, "torch_dtype", None)
    if pipeline_dtype is not None:
        metadata["torch_dtype"] = pipeline_dtype

    for key in _REQUEST_METADATA_KEYS:
        if key in kwargs and kwargs[key] is not None:
            metadata[key] = kwargs[key]
    if kwargs.get("max_new_tokens") is not None:
        metadata["max_tokens"] = kwargs["max_new_tokens"]
    return metadata


def _first_call_input(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    if args:
        return args[0]
    for key in ("inputs", "text_inputs"):
        if key in kwargs:
            return kwargs[key]
    return None


def _is_chat(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(
            isinstance(message, Mapping) and isinstance(message.get("role"), str) and "content" in message
            for message in value
        )
    )


def _message_input(value: object, *, allow_chat: bool = False) -> object:
    if allow_chat and _is_chat(value):
        return value
    if allow_chat and isinstance(value, list) and value and all(_is_chat(item) for item in value):
        return value
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [[{"role": "user", "content": item}] for item in value]
    return value


def _qa_message(question: object, context: object) -> dict[str, object]:
    return {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{question}",
    }


def _question_answering_input(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    question = kwargs.get("question")
    context = kwargs.get("context")

    if question is None and context is None and args:
        value = args[0]
        if isinstance(value, Mapping):
            question = value.get("question")
            context = value.get("context")
        elif isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return [[_qa_message(item.get("question"), item.get("context"))] for item in value]
        elif len(args) > 1:
            question, context = args[0], args[1]
        else:
            return value

    if isinstance(question, list) and isinstance(context, list):
        return [[_qa_message(item_question, item_context)] for item_question, item_context in zip(question, context)]
    if question is not None or context is not None:
        return [_qa_message(question, context)]
    return _first_call_input(args, kwargs)


def _input(task: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    if task == "feature-extraction":
        return _first_call_input(args, kwargs)
    if task == "question-answering":
        return _question_answering_input(args, kwargs)
    return _message_input(_first_call_input(args, kwargs), allow_chat=task == "text-generation")


def _result_records(result: object) -> Iterator[Mapping[str, object]]:
    if isinstance(result, Mapping):
        yield result
    elif isinstance(result, (list, tuple)):
        for item in result:
            yield from _result_records(item)


def _generated_content(record: Mapping[str, object], task: str) -> str | None:
    if task == "summarization":
        value = record.get("summary_text")
    elif task.startswith("translation"):
        value = record.get("translation_text")
    elif task == "question-answering":
        value = record.get("answer")
    else:
        value = record.get("generated_text")

    if isinstance(value, str):
        return value
    if _is_chat(value):
        content = value[-1].get("content")
        return content if isinstance(content, str) else None
    return None


def _choices(result: object, task: str) -> list[dict[str, object]] | None:
    choices = []
    for record in _result_records(result):
        content = _generated_content(record, task)
        if content is None:
            continue
        choices.append(
            {
                "index": len(choices),
                "message": {"role": "assistant", "content": content},
            }
        )
    return choices or None


def _output(task: str, result: object) -> object:
    if task == "feature-extraction":
        shape = _tensor_shape(result)
        return {"shape": shape} if shape is not None else None
    return _choices(result, task)


def _trace_pipeline_call(
    wrapped: Callable[..., object],
    instance: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    task: str,
) -> object:
    if kwargs.get("streamer") is not None:
        return wrapped(*args, **kwargs)

    span_input = _input(task, args, kwargs)
    metadata = _metadata(instance, kwargs, task)

    span = start_span(
        name=_operation_name(task),
        type=SpanTypeAttribute.LLM,
        input=span_input,
        metadata=metadata,
    )
    try:
        result = wrapped(*args, **kwargs)
    except BaseException as error:
        _log_error_and_end_span(span, error)
        raise

    output = _output(task, result)
    _log_and_end_span(span, output=output)
    return result


def _pipeline_call_for_task(
    wrapped: Callable[..., object],
    instance: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    expected_task: str,
) -> object:
    task = getattr(instance, "task", None)
    if task != expected_task:
        return wrapped(*args, **kwargs)
    return _trace_pipeline_call(wrapped, instance, args, kwargs, task)


def _text_generation_pipeline_call_wrapper(
    wrapped: Callable[..., object], instance: object, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    return _pipeline_call_for_task(wrapped, instance, args, kwargs, "text-generation")


def _text2text_generation_pipeline_call_wrapper(
    wrapped: Callable[..., object], instance: object, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    return _pipeline_call_for_task(wrapped, instance, args, kwargs, "text2text-generation")


def _summarization_pipeline_call_wrapper(
    wrapped: Callable[..., object], instance: object, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    return _pipeline_call_for_task(wrapped, instance, args, kwargs, "summarization")


def _translation_pipeline_call_wrapper(
    wrapped: Callable[..., object], instance: object, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    task = getattr(instance, "task", None)
    # In addition to ``translation``, Transformers exposes language-specific
    # task names such as ``translation_en_to_fr``.
    if task != "translation" and not task.startswith("translation_"):
        return wrapped(*args, **kwargs)
    return _trace_pipeline_call(wrapped, instance, args, kwargs, task)


def _feature_extraction_pipeline_call_wrapper(
    wrapped: Callable[..., object], instance: object, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    return _pipeline_call_for_task(wrapped, instance, args, kwargs, "feature-extraction")


def _question_answering_pipeline_call_wrapper(
    wrapped: Callable[..., object], instance: object, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    return _pipeline_call_for_task(wrapped, instance, args, kwargs, "question-answering")
