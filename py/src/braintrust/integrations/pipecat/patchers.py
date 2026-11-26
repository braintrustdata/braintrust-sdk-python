"""Pipecat integration patchers."""

import inspect
import json
from collections import deque
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from braintrust._audio.config import AudioRecordingOptions
from braintrust.bt_json import bt_safe_deep_copy
from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher
from braintrust.integrations.utils import _normalize_chat_messages

from .tracing import BraintrustPipecatObserver
from .tracing._common import LOG, MAX_PENDING, processor_kind
from .tracing._live import is_live_service
from .tracing._model import _ACTIVE_MODEL, _REALTIME_USAGE_ATTR, RealtimeUsage, client_covered, request_metadata


_DEFAULT_OBSERVER_OPTIONS: dict[str, Any] = {
    "capture_audio_attachments": None,
    "capture_user_audio_attachments": None,
    "capture_agent_audio_attachments": None,
    "audio_recording_options": None,
}


def set_default_observer_options(**options: Any) -> None:
    """Set options used when global patching injects a Braintrust observer."""
    _DEFAULT_OBSERVER_OPTIONS.update(options)


_OBSERVER_ATTR = "_braintrust_pipecat_observer"
_REALTIME_RECEIVE: ContextVar[RealtimeUsage | None] = ContextVar("braintrust_pipecat_realtime_receive", default=None)
_LLM_EXECUTION: ContextVar[Any] = ContextVar("braintrust_pipecat_llm_execution", default=None)
_ASSISTANT_TURN: ContextVar[Any] = ContextVar("braintrust_pipecat_assistant_turn", default=None)
# The observer of the running worker. Every processor task inherits it.
_ACTIVE_OBSERVER: ContextVar[BraintrustPipecatObserver | None] = ContextVar(
    "braintrust_pipecat_observer", default=None
)


@dataclass
class _ResponseExecution:
    """Request evidence can precede the observer's queued start notification."""

    processor: Any
    scope: ExitStack
    covered: bool | None = None
    operation: Any = None
    request: Any = None
    metrics: Any = field(default_factory=lambda: deque(maxlen=MAX_PENDING))
    error: Any = None

    def bind(self, observer: Any, operation: Any) -> None:
        if self.operation is not None:
            return
        self.operation = operation
        if operation is not None:
            if self.request is not None and operation.span is not None:
                messages, params = self.request
                operation.span.log(input=messages)
                operation.request(params)
            for frame in self.metrics:
                observer._execution_metrics(operation, frame)
            if self.error is not None:
                operation.error = self.error
        self.request = None
        self.metrics.clear()


def _with_braintrust_observer(observers: Any) -> tuple[list[Any], BraintrustPipecatObserver]:
    ret = list(observers or [])
    for observer in ret:
        if isinstance(observer, BraintrustPipecatObserver):
            return ret, observer
    observer = BraintrustPipecatObserver(**_DEFAULT_OBSERVER_OPTIONS)
    ret.append(observer)
    return ret, observer


def traced_pipeline_worker_init(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Inject the Braintrust Pipecat observer into PipelineWorker construction."""
    kwargs = dict(kwargs)
    kwargs["observers"], observer = _with_braintrust_observer(kwargs.get("observers"))
    result = wrapped(*args, **kwargs)
    setattr(instance, _OBSERVER_ATTR, observer)
    try:
        pipeline = instance.pipeline
        observer._terminal_processor = pipeline.processors[-1]
        _trace_processors(pipeline, observer)
    except Exception:
        LOG.warning("Unable to install Pipecat execution hooks", exc_info=True)
    return result


async def traced_pipeline_worker_run(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Run a worker with its session span current, so nested spans join it."""
    observer = getattr(instance, _OBSERVER_ATTR, None)
    if observer is None:
        return await wrapped(*args, **kwargs)
    token = _ACTIVE_OBSERVER.set(observer)
    try:
        with observer._current_session():
            return await wrapped(*args, **kwargs)
    finally:
        _ACTIVE_OBSERVER.reset(token)


async def traced_llm_push_frame(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Make an LLM service's llm_response span current while it makes the call.

    Cascade LLM services push their response start frame, send the provider
    request and push their response end frame from the same task. Spans the
    provider integration starts in between nest under llm_response.
    """
    observer = _ACTIVE_OBSERVER.get()
    frame = args[0] if args else kwargs.get("frame")
    kind = type(frame).__name__
    if observer is None or kind != "LLMFullResponseStartFrame" or is_live_service(instance):
        return await wrapped(*args, **kwargs)
    scope = _LLM_EXECUTION.get()
    if scope is not None and scope.processor is instance:
        scope.covered = client_covered(instance, "llm_response")
        observer._response_executions[frame.id] = scope
        if len(observer._response_executions) > MAX_PENDING:
            observer._response_executions.popitem(last=False)
    result, span = await observer._response_span(frame, lambda: wrapped(*args, **kwargs))
    response = observer._session.agent._response if observer._session is not None else None
    if scope is not None and scope.processor is instance and span is not None:
        operation = response.model if response is not None and response.span.id == span.id else None
        if operation is not None:
            scope.bind(observer, operation)
        scope.scope.enter_context(observer._model_scope(operation, parent=span))
    return result


async def traced_stt_process_generator(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Activate recognition in the service task that actually runs STT.

    STT services consume their run_stt generator here, including segmented
    services that execute it in a separate task. Merely constructing that
    generator, or activating a span in the observer, cannot scope its work.
    """
    observer = _ACTIVE_OBSERVER.get()
    if observer is None:
        return await wrapped(*args, **kwargs)
    with observer._current_recognition(instance):
        return await wrapped(*args, **kwargs)


def _controller_wait(instance):
    observer = _ACTIVE_OBSERVER.get()
    return observer._controller(instance) if observer is not None else None


def traced_observer_proxy(wrapped, instance, args, kwargs):
    proxy = wrapped(*args, **kwargs)
    if isinstance(proxy.observer, BraintrustPipecatObserver):
        proxy.observer._controller_queue = proxy.queue
    return proxy


class WorkerObserverProxyPatcher(FunctionWrapperPatcher):
    name = "worker_observer_proxy"
    target_module = "pipecat.pipeline.worker_observer"
    target_path = "WorkerObserver._create_proxy"
    wrapper = staticmethod(traced_observer_proxy)


async def traced_controller_frame(wrapped, instance, args, kwargs):
    from .tracing._eou import CONDITION, SPEECH_TIMEOUT_STRATEGY, stop_strategies

    condition = None
    try:
        wait = _controller_wait(instance)
        if wait is not None:
            frame = args[0] if args else kwargs["frame"]
            wait.observe(frame)
            frame_name = type(frame).__name__
            if frame_name == "TranscriptionFinalizedFrame" or (frame_name == "TranscriptionFrame" and frame.finalized):
                strategies = stop_strategies(instance)
                strategy = strategies[0] if len(strategies) == 1 else None
                if (
                    (type(strategy).__module__, type(strategy).__name__) == SPEECH_TIMEOUT_STRATEGY
                    and strategy._user_speech_wait_done
                    and not strategy._stt_wait_done
                ):
                    condition = (strategy, "transcription_finalized")
    except Exception:
        LOG.warning("Unable to observe Pipecat turn wait", exc_info=True)
    token = CONDITION.set(condition)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        CONDITION.reset(token)


async def traced_controller_event(wrapped, instance, args, kwargs):
    name = args[0] if args else kwargs.get("event_name")
    try:
        if name in ("on_user_turn_started", "on_user_turn_stopped"):
            wait = _controller_wait(instance)
            if wait is not None:
                wait.event(name, args[1:])
    except Exception:
        LOG.warning("Unable to observe Pipecat turn decision", exc_info=True)
    return await wrapped(*args, **kwargs)


async def traced_speech_timeout(wrapped, instance, args, kwargs):
    from .tracing._eou import CONDITION

    token = CONDITION.set((instance, "silence_timeout"))
    try:
        return await wrapped(*args, **kwargs)
    finally:
        CONDITION.reset(token)


class UserTurnControllerFramePatcher(FunctionWrapperPatcher):
    name = "user_turn_controller_frame"
    target_module = "pipecat.turns.user_turn_controller"
    target_path = "UserTurnController.process_frame"
    wrapper = staticmethod(traced_controller_frame)


class UserTurnControllerEventPatcher(FunctionWrapperPatcher):
    name = "user_turn_controller_event"
    target_module = "pipecat.turns.user_turn_controller"
    target_path = "UserTurnController._call_event_handler"
    wrapper = staticmethod(traced_controller_event)


class SpeechTimeoutPatcher(FunctionWrapperPatcher):
    name = "speech_timeout_eou_condition"
    target_module = "pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy"
    target_path = "SpeechTimeoutUserTurnStopStrategy._user_speech_timeout_handler"
    wrapper = staticmethod(traced_speech_timeout)


class EOUWaitPatcher(CompositeFunctionWrapperPatcher):
    """Install queue routing, lifecycle and causal observations together."""

    name = "eou_wait"
    sub_patchers = (
        WorkerObserverProxyPatcher,
        UserTurnControllerFramePatcher,
        UserTurnControllerEventPatcher,
        SpeechTimeoutPatcher,
    )

    @classmethod
    def applies(cls, module, version, *, target=None):
        for patcher, asynchronous, arguments in zip(
            cls.sub_patchers,
            (False, True, True, True),
            ({"observer": None}, {"frame": None}, {"event_name": "on_user_turn_stopped"}, {"timeout": 0}),
        ):
            method = patcher.resolve_target(module, version, target=target)
            if not callable(method) or inspect.iscoroutinefunction(method) != asynchronous:
                return False
            try:
                inspect.signature(method).bind(None, **arguments)
            except (TypeError, ValueError):
                return False
        return super().applies(module, version, target=target)


_AGGREGATION: ContextVar[Any] = ContextVar("braintrust_pipecat_aggregation", default=None)


def _has_base(obj: Any, name: str) -> bool:
    return any(base.__name__ == name for base in type(obj).__mro__)


def _trace_processors(pipeline: Any, observer: BraintrustPipecatObserver, *, native_turn: bool = False) -> None:
    processors = list(getattr(pipeline, "processors", ()))
    assistants = [
        _has_base(processor, "LLMAssistantAggregator") and AssistantTurnPatcher.applies(None, None, target=processor)
        for processor in processors
    ]
    assistant_indices = [index for index, assistant in enumerate(assistants) if assistant]
    for index, processor in enumerate(processors):
        member = native_turn or any(index < assistant for assistant in assistant_indices)
        if getattr(processor, "processors", ()):
            _trace_processors(processor, observer, native_turn=member)
        # The native assistant aggregator groups the full-response frames on
        # this downstream route. A bare model response is not a native turn.
        if member:
            observer._native_turn_processors.add(id(processor))
        if is_live_service(processor):
            continue
        kind = processor_kind(type(processor))
        patcher = {"llm": LLMExecutionPatcher, "stt": STTExecutionPatcher, "tts": TTSExecutionPatcher}.get(kind)
        # Composite patch() skips the composite's own applies() gate.
        if patcher is not None and patcher.applies(None, None, target=processor):
            patcher.patch(None, None, target=processor)
        # FunctionWrapperPatcher.patch() checks applies() itself.
        if kind == "llm":
            ModelRequestPatcher.patch(None, None, target=processor)
            if hasattr(processor, "get_llm_adapter"):
                adapter = processor.get_llm_adapter()
                if not ModelPromptPatcher.is_patched(None, None, target=adapter):
                    ModelPromptPatcher.patch(None, None, target=adapter)
        if _has_base(processor, "LLMUserAggregator") and UserAggregationPatcher.applies(None, None, target=processor):
            UserAggregationPatcher.patch(None, None, target=processor)
        if assistants[index]:
            AssistantTurnPatcher.patch(None, None, target=processor)
            observer._register_assistant(processor)


async def _traced_response(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
    """Keep scope through public frame processing, including late usage and exits."""
    active = _ACTIVE_OBSERVER.get()
    if active is None:
        return await wrapped(*args, **kwargs)
    with ExitStack() as scope:
        execution = _ResponseExecution(instance, scope)
        token = _LLM_EXECUTION.set(execution)
        try:
            return await wrapped(*args, **kwargs)
        except BaseException as exc:
            execution.error = exc
            operation = execution.operation
            if operation is not None:
                operation.error = exc
                if operation.parent is not None:
                    operation.parent.log(error=exc)
            raise
        finally:
            _LLM_EXECUTION.reset(token)


async def _traced_media(wrapped: Any, instance: Any, args: tuple, kwargs: dict, *, kind: str):
    active = _ACTIVE_OBSERVER.get()
    if active is None:
        async for frame in wrapped(*args, **kwargs):
            yield frame
        return
    with active._model_execution(instance, kind, args, kwargs, activate=False) as operation:
        stream = wrapped(*args, **kwargs)
        try:
            while True:
                try:
                    with active._model_scope(operation):
                        frame = await anext(stream)
                except StopAsyncIteration:
                    if operation is not None:
                        operation.complete = not operation.deferred
                    break
                active._model_frame(operation, frame, kind)
                yield frame
        finally:
            with active._model_scope(operation):
                await stream.aclose()


def _current_operation() -> tuple[Any, Any]:
    """The executing LLM request and its model operation, if either is known yet."""
    execution = _LLM_EXECUTION.get()
    return execution, _ACTIVE_MODEL.get() or (execution.operation if execution is not None else None)


def _traced_model_prompt(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
    def observe(params):
        execution, operation = _current_operation()
        if (operation is not None and operation.span is not None) or (execution is not None and operation is None):
            try:
                messages = params.get("messages")
                if not isinstance(messages, list):
                    return params
                system = params.get("system")
                if isinstance(system, str) and system:
                    messages = [{"role": "system", "content": system}, *messages]
                normalized = _normalize_chat_messages(messages)
                if operation is not None:
                    operation.request(params)
                else:
                    # Keep only what bind() logs, not the whole request.
                    execution.request = bt_safe_deep_copy((normalized, request_metadata(params)))
            except Exception:
                LOG.warning("Unable to trace Pipecat model prompt", exc_info=True)
                return params
            if operation is not None:
                operation.span.log(input=normalized)
        return params

    result = wrapped(*args, **kwargs)
    if inspect.isawaitable(result):

        async def call():
            return observe(await result)

        return call()
    return observe(result)


async def _traced_service_push(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    frame = args[0] if args else kwargs.get("frame")
    if observer is not None and type(frame).__name__ == "MetricsFrame":
        execution, operation = _current_operation()
        if execution is not None and execution.processor is instance and operation is None:
            execution.metrics.append(frame)
        elif operation is not None and operation.processor is instance:
            try:
                observer._execution_metrics(operation, frame)
            except Exception:
                LOG.warning("Unable to trace Pipecat model metrics", exc_info=True)
    return await wrapped(*args, **kwargs)


async def _traced_tts_frame(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    frame = args[0] if args else kwargs.get("frame")
    kind = type(frame).__name__
    if observer is not None and kind == "LLMFullResponseStartFrame":
        instance._braintrust_response = observer._response_frames.get(frame.id)
    elif kind == "TTSSpeakFrame":
        instance._braintrust_response = None
    try:
        return await wrapped(*args, **kwargs)
    finally:
        if kind in {"LLMFullResponseEndFrame", "InterruptionFrame", "CancelFrame", "EndFrame"}:
            instance._braintrust_response = None


async def _traced_aggregate_transcript(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
    try:
        parts = getattr(instance, "_aggregation", ())
        previous = parts[-1] if parts else None
    except Exception:
        return await wrapped(*args, **kwargs)
    result = await wrapped(*args, **kwargs)
    observer = _ACTIVE_OBSERVER.get()
    frame = args[0] if args else kwargs.get("frame")
    try:
        if (
            observer is not None
            and getattr(instance, "_aggregation", None)
            and instance._aggregation[-1] is not previous
        ):
            observer._aggregation_parts[id(instance._aggregation[-1])] = (instance._aggregation[-1], frame.id)
            if len(observer._aggregation_parts) > MAX_PENDING:
                observer._aggregation_parts.popitem(last=False)
    except Exception:
        LOG.warning("Unable to trace Pipecat transcript lineage", exc_info=True)
    return result


async def _traced_push_aggregation(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    frame_ids = []
    try:
        if observer is not None:
            for part in getattr(instance, "_aggregation", ()):
                evidence = observer._aggregation_parts.pop(id(part), None)
                if evidence is not None and evidence[0] is part:
                    frame_ids.append(evidence[1])
    except Exception:
        LOG.warning("Unable to trace Pipecat aggregation lineage", exc_info=True)
        frame_ids.clear()
    token = _AGGREGATION.set(tuple(frame_ids))
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _AGGREGATION.reset(token)


def traced_context_add_message(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    message = args[0] if args else kwargs.get("message")
    frame_ids = _AGGREGATION.get()
    if observer is not None and frame_ids:
        try:
            observer._associate_message(message, frame_ids)
        except Exception:
            LOG.warning("Unable to trace Pipecat context provenance", exc_info=True)
    return wrapped(*args, **kwargs)


async def _traced_assistant_frame(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    if observer is None:
        return await wrapped(*args, **kwargs)
    frame = args[0] if args else kwargs.get("frame")
    if type(frame).__name__ == "LLMFullResponseStartFrame":
        # Associate by the native response frame, never whichever turn happens
        # to be current when a delayed event handler eventually runs.
        observer._assistant_turns[id(instance)] = frame.id
    response_id = observer._assistant_turns.get(id(instance))
    token = _ASSISTANT_TURN.set((observer, response_id))
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _ASSISTANT_TURN.reset(token)


def traced_assistant_stopped(processor: Any, _message: Any, *_args: Any) -> None:
    # A shared callback captures no observer/turn. Pipecat 1.3 has no public
    # removal API, so it can remain registered safely if the processor is reused.
    binding = _ASSISTANT_TURN.get()
    if binding is None:
        return
    observer, response_id = binding
    try:
        timestamp = processor.get_clock().get_time()

        def complete():
            session = observer._session
            turn = observer._response_turns.get(response_id)
            if session is not None and turn is not None and session.agent._turn is turn:
                with session.clock.event(timestamp):
                    session.agent.request_completion()

        # Native callbacks can precede queued response observations. Preserve
        # their FIFO order and resolve the originating frame's turn afterward.
        observer._enqueue_controller(complete)
        if observer._assistant_turns.get(id(processor)) == response_id:
            observer._assistant_turns.pop(id(processor), None)
    except Exception:
        LOG.warning("Unable to trace Pipecat assistant completion", exc_info=True)


class ServiceMetricsPatcher(FunctionWrapperPatcher):
    name = "cascade_service_metrics"
    target_path = "push_frame"
    wrapper = staticmethod(_traced_service_push)


class ResponseExecutionPatcher(FunctionWrapperPatcher):
    name = "cascade_response_execution"
    target_path = "process_frame"
    wrapper = staticmethod(_traced_response)


class TTSExecutionFramePatcher(FunctionWrapperPatcher):
    name = "cascade_tts_frame"
    target_path = "process_frame"
    wrapper = staticmethod(_traced_tts_frame)


class TTSRunPatcher(FunctionWrapperPatcher):
    name = "cascade_tts_run"
    target_path = "run_tts"

    @staticmethod
    def wrapper(wrapped, instance, args, kwargs):
        return _traced_media(wrapped, instance, args, kwargs, kind="tts")


class STTRunPatcher(FunctionWrapperPatcher):
    name = "cascade_stt_run"
    target_path = "run_stt"

    @staticmethod
    def wrapper(wrapped, instance, args, kwargs):
        return _traced_media(wrapped, instance, args, kwargs, kind="stt")


def _accepts(method: Any, target: Any, arguments: tuple) -> bool:
    """Whether a hooked method accepts the call our wrapper forwards to it."""
    try:
        signature = inspect.signature(method)
        if next(iter(signature.parameters), None) in {"self", "cls"}:
            arguments = (target, *arguments)
        signature.bind(*arguments)
    except (TypeError, ValueError):
        return False
    return True


class _ExecutionPatcher(CompositeFunctionWrapperPatcher):
    """Require the complete execution/metrics surface before installing hooks."""

    @classmethod
    def applies(cls, module, version, *, target=None):
        if target is None or is_live_service(target):
            return False
        for patcher in cls.sub_patchers:
            method = patcher.resolve_target(module, version, target=target)
            if patcher in (TTSRunPatcher, STTRunPatcher):
                valid = inspect.isasyncgenfunction(inspect.unwrap(method))
            else:
                valid = inspect.iscoroutinefunction(method)
            arguments = (
                (None, None)
                if patcher
                in (
                    ServiceMetricsPatcher,
                    TTSExecutionFramePatcher,
                    ResponseExecutionPatcher,
                    AssistantFramePatcher,
                    TTSRunPatcher,
                )
                else (None,)
            )
            if not valid or not _accepts(method, target, arguments):
                return False
        return super().applies(module, version, target=target)


class LLMExecutionPatcher(_ExecutionPatcher):
    name = "cascade_llm_execution"
    sub_patchers = (ResponseExecutionPatcher, ServiceMetricsPatcher)


class TTSExecutionPatcher(_ExecutionPatcher):
    name = "cascade_tts_execution"
    sub_patchers = (TTSRunPatcher, TTSExecutionFramePatcher, ServiceMetricsPatcher)


class STTExecutionPatcher(_ExecutionPatcher):
    name = "cascade_stt_execution"
    sub_patchers = (STTRunPatcher, ServiceMetricsPatcher)


class ModelPromptPatcher(FunctionWrapperPatcher):
    name = "cascade_model_prompt"
    target_path = "get_llm_invocation_params"
    wrapper = staticmethod(_traced_model_prompt)


class ModelRequestPatcher(FunctionWrapperPatcher):
    # OpenAI-compatible services may override adapter fields in their final
    # request builder. Observe its result without changing provider parameters.
    name = "cascade_model_request"
    target_path = "build_chat_completion_params"
    wrapper = staticmethod(_traced_model_prompt)


class AggregateTranscriptPatcher(FunctionWrapperPatcher):
    name = "cascade_aggregate_transcript"
    target_path = "_handle_transcription"
    wrapper = staticmethod(_traced_aggregate_transcript)


class PushAggregationPatcher(FunctionWrapperPatcher):
    name = "cascade_push_aggregation"
    target_path = "_push_aggregation"
    wrapper = staticmethod(_traced_push_aggregation)


class LegacyPushAggregationPatcher(FunctionWrapperPatcher):
    name = "cascade_legacy_push_aggregation"
    target_path = "push_aggregation"
    wrapper = staticmethod(_traced_push_aggregation)
    superseded_by = (PushAggregationPatcher,)


class UserAggregationPatcher(CompositeFunctionWrapperPatcher):
    name = "cascade_user_aggregation"
    sub_patchers = (AggregateTranscriptPatcher, PushAggregationPatcher, LegacyPushAggregationPatcher)

    @classmethod
    def applies(cls, module, version, *, target=None):
        if not ContextAddMessagePatcher.applies(None, version):
            return False
        push = (
            PushAggregationPatcher
            if PushAggregationPatcher.applies(module, version, target=target)
            else LegacyPushAggregationPatcher
        )
        for patcher, arguments in ((AggregateTranscriptPatcher, (None,)), (push, ())):
            method = patcher.resolve_target(module, version, target=target)
            if not inspect.iscoroutinefunction(method) or not _accepts(method, target, arguments):
                return False
        return super().applies(module, version, target=target)


class AssistantFramePatcher(FunctionWrapperPatcher):
    name = "cascade_assistant_frame"
    target_path = "process_frame"
    wrapper = staticmethod(_traced_assistant_frame)


class AssistantTurnPatcher(_ExecutionPatcher):
    """Observe native turn events in their originating public frame scope."""

    name = "cascade_assistant_turn"
    sub_patchers = (AssistantFramePatcher,)

    @classmethod
    def applies(cls, module, version, *, target=None):
        for name in ("add_event_handler",):
            method = getattr(target, name, None)
            if not callable(method):
                return False
            try:
                inspect.signature(method).bind("on_assistant_turn_stopped", None)
            except (ValueError, TypeError):
                return False
        return super().applies(module, version, target=target)


class ContextAddMessagePatcher(FunctionWrapperPatcher):
    """Preserve native transcript-to-message lineage without text matching."""

    name = "context_add_message"
    target_module = "pipecat.processors.aggregators.llm_context"
    target_path = "LLMContext.add_message"
    wrapper = staticmethod(traced_context_add_message)

    @classmethod
    def applies(cls, module, version, *, target=None):
        method = cls.resolve_target(module, version, target=target)
        if not callable(method) or inspect.iscoroutinefunction(method):
            return False
        try:
            inspect.signature(method).bind(None, None)
        except (TypeError, ValueError):
            return False
        return super().applies(module, version, target=target)


class PipelineWorkerInitPatcher(FunctionWrapperPatcher):
    """Patch ``PipelineWorker.__init__`` to add a Braintrust observer."""

    name = "pipeline_worker_init"
    target_module = "pipecat.pipeline.worker"
    target_path = "PipelineWorker.__init__"
    wrapper = staticmethod(traced_pipeline_worker_init)


class PipelineWorkerRunPatcher(FunctionWrapperPatcher):
    """Patch ``PipelineWorker.run`` to make the session span current."""

    name = "pipeline_worker_run"
    target_module = "pipecat.pipeline.worker"
    target_path = "PipelineWorker.run"
    wrapper = staticmethod(traced_pipeline_worker_run)


class LLMServicePushFramePatcher(FunctionWrapperPatcher):
    """Patch ``LLMService.push_frame`` to nest provider spans under responses."""

    name = "llm_service_push_frame"
    target_module = "pipecat.services.llm_service"
    target_path = "LLMService.push_frame"
    wrapper = staticmethod(traced_llm_push_frame)


class STTServiceProcessGeneratorPatcher(FunctionWrapperPatcher):
    """Scope provider transcription spans under their Pipecat recognition."""

    name = "stt_service_process_generator"
    target_module = "pipecat.services.stt_service"
    target_path = "STTService.process_generator"
    wrapper = staticmethod(traced_stt_process_generator)


class PipelineWorkerPatcher(CompositeFunctionWrapperPatcher):
    """Instrument ``PipelineWorker`` construction and runs together."""

    name = "pipeline_worker"
    sub_patchers = (PipelineWorkerInitPatcher, PipelineWorkerRunPatcher)


async def traced_live_event(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    event = args[0] if args else kwargs.get("evt")
    if observer is not None:
        observer._live_event(instance, event)
    result = await wrapped(*args, **kwargs)
    if observer is not None:
        observer._live_event_processed(event)
    return result


async def traced_live_client_event(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    observer = _ACTIVE_OBSERVER.get()
    result = await wrapped(*args, **kwargs)
    if observer is not None:
        observer._live_client_event(args[0] if args else kwargs.get("event"))
    return result


class OpenAILiveEventPatcher(FunctionWrapperPatcher):
    name = "openai_live_event"
    target_module = "pipecat.services.openai.live.llm"
    target_path = "OpenAILiveLLMService._handle_server_event"
    wrapper = staticmethod(traced_live_event)


class OpenAILiveClientEventPatcher(FunctionWrapperPatcher):
    name = "openai_live_client_event"
    target_module = "pipecat.services.openai.live.llm"
    target_path = "OpenAILiveLLMService.send_client_event"
    wrapper = staticmethod(traced_live_client_event)


class OpenAILivePatcher(CompositeFunctionWrapperPatcher):
    name = "openai_live"
    sub_patchers = (OpenAILiveEventPatcher, OpenAILiveClientEventPatcher)

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        # Both hooks are needed to track a response through tool submission.
        # Let the composite own patch markers and idempotency after this gate.
        for patcher, argument in ((OpenAILiveEventPatcher, "evt"), (OpenAILiveClientEventPatcher, "event")):
            method = patcher.resolve_target(module, version, target=target)
            if not inspect.iscoroutinefunction(method):
                return False
            try:
                inspect.signature(method).bind(None, **{argument: None})
            except (TypeError, ValueError):
                return False
        return super().applies(module, version, target=target)


async def traced_realtime_receive(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if _ACTIVE_OBSERVER.get() is None:
        return await wrapped(*args, **kwargs)
    usage = RealtimeUsage()
    setattr(instance, _REALTIME_USAGE_ATTR, usage)
    token = _REALTIME_RECEIVE.set(usage)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _REALTIME_RECEIVE.reset(token)
        if getattr(instance, _REALTIME_USAGE_ATTR, None) is usage:
            delattr(instance, _REALTIME_USAGE_ATTR)


def traced_realtime_server_event(wrapped: Any, _instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    event = wrapped(*args, **kwargs)
    observation = _REALTIME_RECEIVE.get()
    if observation is None:
        return event
    try:
        kind = getattr(event, "type", None)
        if kind == "session.created":
            observation.session_metadata = {
                key: value
                for key, value in (
                    ("contrib.pipecat.session_id", event.session.id),
                    ("model", event.session.model),
                )
                if value is not None
            }
        elif kind == "response.created":
            observation.response_id = event.response.id
        elif kind == "response.done":
            # Pipecat's model can drop native breakdowns or default missing
            # counts to zero. Read usage from the terminal wire event.
            raw = json.loads(args[0] if args else kwargs["str"])
            observation.response_done(event.response.id, raw["response"].get("usage"))
    except Exception:
        LOG.warning("Unable to trace Pipecat Realtime server event", exc_info=True)
    return event


class OpenAIRealtimeReceivePatcher(FunctionWrapperPatcher):
    name = "openai_realtime_receive"
    target_module = "pipecat.services.openai.realtime.llm"
    target_path = "OpenAIRealtimeLLMService._receive_task_handler"
    wrapper = staticmethod(traced_realtime_receive)


class OpenAIRealtimeServerEventPatcher(FunctionWrapperPatcher):
    name = "openai_realtime_server_event"
    target_module = "pipecat.services.openai.realtime.events"
    target_path = "parse_server_event"
    wrapper = staticmethod(traced_realtime_server_event)


class OpenAIRealtimePatcher(CompositeFunctionWrapperPatcher):
    name = "openai_realtime"
    sub_patchers = (OpenAIRealtimeReceivePatcher, OpenAIRealtimeServerEventPatcher)

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        receive = OpenAIRealtimeReceivePatcher.resolve_target(module, version, target=target)
        parse = OpenAIRealtimeServerEventPatcher.resolve_target(module, version, target=target)
        return inspect.iscoroutinefunction(receive) and callable(parse) and not inspect.iscoroutinefunction(parse)


def wrap_pipeline_worker(
    PipelineWorker: Any,
    *,
    capture_audio_attachments: bool | None = None,
    capture_user_audio_attachments: bool | None = None,
    capture_agent_audio_attachments: bool | None = None,
    audio_recording_options: AudioRecordingOptions | None = None,
) -> Any:
    """Instrument a Pipecat ``PipelineWorker`` class directly.

    Every worker the class constructs afterwards is traced. Options match
    ``setup_pipecat()``. Options omitted here keep their current values, so
    wrapping again never discards earlier configuration; unset options fall
    back to the generic audio capture environment variables.
    """
    set_default_observer_options(
        **{
            name: value
            for name, value in {
                "capture_audio_attachments": capture_audio_attachments,
                "capture_user_audio_attachments": capture_user_audio_attachments,
                "capture_agent_audio_attachments": capture_agent_audio_attachments,
                "audio_recording_options": audio_recording_options,
            }.items()
            if value is not None
        }
    )
    from pipecat.processors.aggregators.llm_context import LLMContext  # pylint: disable=import-error
    from pipecat.services.llm_service import LLMService  # pylint: disable=import-error
    from pipecat.services.stt_service import STTService  # pylint: disable=import-error

    # The same gate and composite own global markers for every entrypoint.
    if EOUWaitPatcher.applies(None, None):
        EOUWaitPatcher.patch(None, None)
    ContextAddMessagePatcher.wrap_target(LLMContext)
    LLMServicePushFramePatcher.wrap_target(LLMService)
    STTServiceProcessGeneratorPatcher.wrap_target(STTService)
    # Optional on older Pipecat versions; the patchers feature-detect the module.
    if OpenAILivePatcher.applies(None, None):
        OpenAILivePatcher.patch(None, None)
    if OpenAIRealtimePatcher.applies(None, None):
        OpenAIRealtimePatcher.patch(None, None)
    return PipelineWorkerPatcher.wrap_target(PipelineWorker)
