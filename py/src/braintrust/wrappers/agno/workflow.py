import time
from typing import Any

from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from wrapt import wrap_function_wrapper

from .utils import (
    _aggregate_workflow_chunks,
    _try_to_dict,
    extract_metadata,
    extract_metrics,
    extract_streaming_metrics,
    is_async_iterator,
    is_patched,
    is_sync_iterator,
    mark_patched,
)


def _extract_workflow_input(
    args: Any,
    kwargs: Any,
    *,
    execution_input_index: int,
    workflow_run_response_index: int,
) -> dict[str, Any]:
    """Extract workflow input from execution method parameters."""
    execution_input = (
        args[execution_input_index] if len(args) > execution_input_index else kwargs.get("execution_input")
    )
    workflow_run_response = (
        args[workflow_run_response_index]
        if len(args) > workflow_run_response_index
        else kwargs.get("workflow_run_response")
    )

    result: dict[str, Any] = {}

    if execution_input:
        if hasattr(execution_input, "input"):
            result["input"] = execution_input.input
        result["execution_input"] = _try_to_dict(execution_input)

    if workflow_run_response:
        result["run_response"] = _try_to_dict(workflow_run_response)

    return result


def wrap_workflow(Workflow: Any) -> Any:
    if is_patched(Workflow):
        return Workflow

    def _workflow_span_config(instance: Any, suffix: str) -> tuple[str, dict[str, Any]]:
        workflow_name = getattr(instance, "name", None) or "Workflow"
        return f"{workflow_name}.{suffix}", extract_metadata(instance, "workflow")

    def _extract_workflow_agent_input(args: Any, kwargs: Any) -> dict[str, Any]:
        user_input = args[0] if len(args) > 0 else kwargs.get("user_input")
        execution_input = args[2] if len(args) > 2 else kwargs.get("execution_input")

        result: dict[str, Any] = {"input": user_input}
        if execution_input:
            result["execution_input"] = _try_to_dict(execution_input)
        return result

    def execute_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
        workflow_name = getattr(instance, "name", None) or "Workflow"
        span_name = f"{workflow_name}.run"

        input_data = _extract_workflow_input(args, kwargs, execution_input_index=1, workflow_run_response_index=2)
        workflow_metadata = extract_metadata(instance, "workflow")

        with start_span(
            name=span_name,
            type=SpanTypeAttribute.TASK,
            input=input_data,
            metadata=workflow_metadata,
            propagated_event={"metadata": workflow_metadata},
        ) as span:
            result = wrapped(*args, **kwargs)
            span.log(
                output=result,
                metrics=extract_metrics(result),
            )
            return result

    if hasattr(Workflow, "_execute"):
        wrap_function_wrapper(Workflow, "_execute", execute_wrapper)

    def execute_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
        workflow_name = getattr(instance, "name", None) or "Workflow"
        span_name = f"{workflow_name}.run_stream"

        input_data = _extract_workflow_input(args, kwargs, execution_input_index=1, workflow_run_response_index=2)
        workflow_metadata = extract_metadata(instance, "workflow")
        workflow_run_response = args[2] if len(args) > 2 else kwargs.get("workflow_run_response")

        def _trace_stream():
            start = time.time()
            span = start_span(
                name=span_name,
                type=SpanTypeAttribute.TASK,
                input=input_data,
                metadata=workflow_metadata,
                propagated_event={"metadata": workflow_metadata},
            )
            span.set_current()

            should_unset = True
            try:
                first = True
                all_chunks = []

                for chunk in wrapped(*args, **kwargs):
                    if first:
                        span.log(
                            metrics={
                                "time_to_first_token": time.time() - start,
                            }
                        )
                        first = False
                    all_chunks.append(chunk)
                    yield chunk

                aggregated = _aggregate_workflow_chunks(all_chunks, workflow_run_response)

                span.log(
                    output=aggregated,
                    metrics=extract_streaming_metrics(aggregated, start),
                )
            except GeneratorExit:
                should_unset = False
                raise
            except Exception as e:
                span.log(
                    error=str(e),
                )
                raise
            finally:
                if should_unset:
                    span.unset_current()
                span.end()

        return _trace_stream()

    if hasattr(Workflow, "_execute_stream"):
        wrap_function_wrapper(Workflow, "_execute_stream", execute_stream_wrapper)

    async def aexecute_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
        workflow_name = getattr(instance, "name", None) or "Workflow"
        span_name = f"{workflow_name}.arun"

        input_data = _extract_workflow_input(args, kwargs, execution_input_index=2, workflow_run_response_index=3)
        workflow_metadata = extract_metadata(instance, "workflow")

        with start_span(
            name=span_name,
            type=SpanTypeAttribute.TASK,
            input=input_data,
            metadata=workflow_metadata,
            propagated_event={"metadata": workflow_metadata},
        ) as span:
            result = await wrapped(*args, **kwargs)
            span.log(
                output=result,
                metrics=extract_metrics(result),
            )
            return result

    if hasattr(Workflow, "_aexecute"):
        wrap_function_wrapper(Workflow, "_aexecute", aexecute_wrapper)

    def aexecute_stream_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
        workflow_name = getattr(instance, "name", None) or "Workflow"
        span_name = f"{workflow_name}.arun_stream"

        input_data = _extract_workflow_input(args, kwargs, execution_input_index=2, workflow_run_response_index=3)
        workflow_metadata = extract_metadata(instance, "workflow")
        workflow_run_response = args[3] if len(args) > 3 else kwargs.get("workflow_run_response")

        async def _trace_stream():
            start = time.time()
            span = start_span(
                name=span_name,
                type=SpanTypeAttribute.TASK,
                input=input_data,
                metadata=workflow_metadata,
                propagated_event={"metadata": workflow_metadata},
            )
            span.set_current()

            should_unset = True
            try:
                first = True
                all_chunks = []

                async for chunk in wrapped(*args, **kwargs):
                    if first:
                        span.log(
                            metrics={
                                "time_to_first_token": time.time() - start,
                            }
                        )
                        first = False
                    all_chunks.append(chunk)
                    yield chunk

                aggregated = _aggregate_workflow_chunks(all_chunks, workflow_run_response)

                span.log(
                    output=aggregated,
                    metrics=extract_streaming_metrics(aggregated, start),
                )
            except GeneratorExit:
                should_unset = False
                raise
            except Exception as e:
                span.log(
                    error=str(e),
                )
                raise
            finally:
                if should_unset:
                    span.unset_current()
                span.end()

        return _trace_stream()

    if hasattr(Workflow, "_aexecute_stream"):
        wrap_function_wrapper(Workflow, "_aexecute_stream", aexecute_stream_wrapper)

    def execute_workflow_agent_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
        stream = kwargs.get("stream", False)
        span_suffix = "run_stream" if stream else "run"
        span_name, workflow_metadata = _workflow_span_config(instance, span_suffix)
        input_data = _extract_workflow_agent_input(args, kwargs)

        span = start_span(
            name=span_name,
            type=SpanTypeAttribute.TASK,
            input=input_data,
            metadata=workflow_metadata,
            propagated_event={"metadata": workflow_metadata},
        )
        span.set_current()
        start = time.time()
        try:
            result = wrapped(*args, **kwargs)
            if stream and is_sync_iterator(result):

                def _trace_stream():
                    should_unset = True
                    try:
                        first = True
                        all_chunks = []
                        for chunk in result:
                            if first:
                                span.log(metrics={"time_to_first_token": time.time() - start})
                                first = False
                            all_chunks.append(chunk)
                            yield chunk

                        aggregated = _aggregate_workflow_chunks(all_chunks)
                        span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
                    except GeneratorExit:
                        should_unset = False
                        raise
                    except Exception as e:
                        span.log(error=str(e))
                        raise
                    finally:
                        if should_unset:
                            span.unset_current()
                        span.end()

                return _trace_stream()

            span.log(output=result, metrics=extract_metrics(result))
            span.unset_current()
            span.end()
            return result
        except Exception as e:
            span.log(error=str(e))
            span.unset_current()
            span.end()
            raise

    if hasattr(Workflow, "_execute_workflow_agent"):
        wrap_function_wrapper(Workflow, "_execute_workflow_agent", execute_workflow_agent_wrapper)

    async def aexecute_workflow_agent_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any):
        stream = kwargs.get("stream", False)
        span_suffix = "arun_stream" if stream else "arun"
        span_name, workflow_metadata = _workflow_span_config(instance, span_suffix)
        input_data = _extract_workflow_agent_input(args, kwargs)

        span = start_span(
            name=span_name,
            type=SpanTypeAttribute.TASK,
            input=input_data,
            metadata=workflow_metadata,
            propagated_event={"metadata": workflow_metadata},
        )
        span.set_current()
        start = time.time()
        try:
            result = await wrapped(*args, **kwargs)
            if stream and is_async_iterator(result):

                async def _trace_stream():
                    should_unset = True
                    try:
                        first = True
                        all_chunks = []
                        async for chunk in result:
                            if first:
                                span.log(metrics={"time_to_first_token": time.time() - start})
                                first = False
                            all_chunks.append(chunk)
                            yield chunk

                        aggregated = _aggregate_workflow_chunks(all_chunks)
                        span.log(output=aggregated, metrics=extract_streaming_metrics(aggregated, start))
                    except GeneratorExit:
                        should_unset = False
                        raise
                    except Exception as e:
                        span.log(error=str(e))
                        raise
                    finally:
                        if should_unset:
                            span.unset_current()
                        span.end()

                return _trace_stream()

            span.log(output=result, metrics=extract_metrics(result))
            span.unset_current()
            span.end()
            return result
        except Exception as e:
            span.log(error=str(e))
            span.unset_current()
            span.end()
            raise

    if hasattr(Workflow, "_aexecute_workflow_agent"):
        wrap_function_wrapper(Workflow, "_aexecute_workflow_agent", aexecute_workflow_agent_wrapper)

    mark_patched(Workflow)
    return Workflow
