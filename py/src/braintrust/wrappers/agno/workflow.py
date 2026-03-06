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
    is_patched,
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

                aggregated = _aggregate_workflow_chunks(all_chunks)

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

                aggregated = _aggregate_workflow_chunks(all_chunks)

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

    mark_patched(Workflow)
    return Workflow
