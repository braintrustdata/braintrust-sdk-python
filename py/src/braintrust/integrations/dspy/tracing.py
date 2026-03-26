"""DSPy-specific callback, span creation, and metadata extraction."""

from typing import Any

from braintrust.logger import current_span, start_span
from braintrust.span_types import SpanTypeAttribute


try:
    from dspy.utils.callback import BaseCallback

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False
    BaseCallback = object  # type: ignore[assignment,misc]


class BraintrustDSpyCallback(BaseCallback):
    """Callback handler that logs DSPy execution traces to Braintrust.

    This callback integrates DSPy with Braintrust's observability platform, automatically
    logging language model calls, module executions, tool invocations, and evaluations.

    Logged information includes:
    - Input parameters and output results
    - Execution latency
    - Error information when exceptions occur
    - Hierarchical span relationships for nested operations

    Basic Example:
        ```python
        import dspy
        from braintrust import init_logger
        from braintrust.integrations.dspy import BraintrustDSpyCallback

        # Initialize Braintrust
        init_logger(project="dspy-example")

        # Configure DSPy with callback
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm, callbacks=[BraintrustDSpyCallback()])

        # Use DSPy - execution is automatically logged
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="What is 2+2?")
        ```

    Advanced Example with LiteLLM Patching:
        For additional detailed token metrics from LiteLLM's wrapper, patch before importing DSPy
        and disable DSPy's disk cache:

        ```python
        from braintrust.integrations.litellm import patch_litellm
        patch_litellm()

        import dspy
        from braintrust import init_logger
        from braintrust.integrations.dspy import BraintrustDSpyCallback

        init_logger(project="dspy-example")

        # Disable disk cache to ensure LiteLLM calls are traced
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)

        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm, callbacks=[BraintrustDSpyCallback()])
        ```

    The callback creates Braintrust spans for:
    - DSPy module executions (Predict, ChainOfThought, ReAct, etc.)
    - LLM calls with latency metrics
    - Tool calls
    - Evaluation runs

    For detailed token usage and cost metrics, use LiteLLM patching (see Advanced Example above).
    The patched LiteLLM wrapper will create additional "Completion" spans with comprehensive metrics.

    Spans are automatically nested based on the execution hierarchy.
    """

    def __init__(self):
        """Initialize the Braintrust DSPy callback handler."""
        if not _HAS_DSPY:
            raise ImportError("DSPy is not installed. Please install it with: pip install dspy")
        super().__init__()
        # Map call_id to span objects for proper nesting
        self._spans: dict[str, Any] = {}

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """Log the start of a language model call.

        Args:
            call_id: Unique identifier for this call
            instance: The LM instance being called
            inputs: Input parameters to the LM
        """
        metadata = {}
        if hasattr(instance, "model"):
            metadata["model"] = instance.model
        if hasattr(instance, "provider"):
            metadata["provider"] = str(instance.provider)

        for key in ["temperature", "max_tokens", "top_p", "top_k", "stop"]:
            if key in inputs:
                metadata[key] = inputs[key]

        parent = current_span()
        parent_export = parent.export() if parent else None

        span = start_span(
            name="dspy.lm",
            input=inputs,
            metadata=metadata,
            parent=parent_export,
        )
        span.set_current()
        self._spans[call_id] = span

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """Log the end of a language model call.

        Args:
            call_id: Unique identifier for this call
            outputs: Output from the LM, or None if there was an exception
            exception: Exception raised during execution, if any
        """
        span = self._spans.pop(call_id, None)
        if not span:
            return

        try:
            log_data = {}
            if exception:
                log_data["error"] = exception
            if outputs is not None:
                log_data["output"] = outputs

            if log_data:
                span.log(**log_data)
        finally:
            span.unset_current()
            span.end()

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """Log the start of a DSPy module execution.

        Args:
            call_id: Unique identifier for this call
            instance: The Module instance being called
            inputs: Input parameters to the module's forward() method
        """
        cls = instance.__class__
        cls_name = cls.__name__
        module_name = f"{cls.__module__}.{cls_name}"

        parent = current_span()
        parent_export = parent.export() if parent else None

        span = start_span(
            name=f"dspy.module.{cls_name}",
            input=inputs,
            metadata={"module_class": module_name},
            parent=parent_export,
        )
        span.set_current()
        self._spans[call_id] = span

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """Log the end of a DSPy module execution.

        Args:
            call_id: Unique identifier for this call
            outputs: Output from the module, or None if there was an exception
            exception: Exception raised during execution, if any
        """
        span = self._spans.pop(call_id, None)
        if not span:
            return

        try:
            log_data = {}
            if exception:
                log_data["error"] = exception
            if outputs is not None:
                if hasattr(outputs, "toDict"):
                    output_dict = outputs.toDict()
                elif hasattr(outputs, "__dict__"):
                    output_dict = outputs.__dict__
                else:
                    output_dict = outputs
                log_data["output"] = output_dict

            if log_data:
                span.log(**log_data)
        finally:
            span.unset_current()
            span.end()

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """Log the start of a tool invocation.

        Args:
            call_id: Unique identifier for this call
            instance: The Tool instance being called
            inputs: Input parameters to the tool
        """
        tool_name = "unknown"
        if hasattr(instance, "name"):
            tool_name = instance.name
        elif hasattr(instance, "__name__"):
            tool_name = instance.__name__
        elif hasattr(instance, "func") and hasattr(instance.func, "__name__"):
            tool_name = instance.func.__name__

        parent = current_span()
        parent_export = parent.export() if parent else None

        span = start_span(
            name=tool_name,
            span_attributes={"type": SpanTypeAttribute.TOOL},
            input=inputs,
            parent=parent_export,
        )
        span.set_current()
        self._spans[call_id] = span

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """Log the end of a tool invocation.

        Args:
            call_id: Unique identifier for this call
            outputs: Output from the tool, or None if there was an exception
            exception: Exception raised during execution, if any
        """
        span = self._spans.pop(call_id, None)
        if not span:
            return

        try:
            log_data = {}
            if exception:
                log_data["error"] = exception
            if outputs is not None:
                log_data["output"] = outputs

            if log_data:
                span.log(**log_data)
        finally:
            span.unset_current()
            span.end()

    def on_evaluate_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """Log the start of an evaluation run.

        Args:
            call_id: Unique identifier for this call
            instance: The Evaluate instance
            inputs: Input parameters to the evaluation
        """
        metadata = {}
        if hasattr(instance, "metric") and instance.metric:
            if hasattr(instance.metric, "__name__"):
                metadata["metric"] = instance.metric.__name__
        if hasattr(instance, "num_threads"):
            metadata["num_threads"] = instance.num_threads

        parent = current_span()
        parent_export = parent.export() if parent else None

        span = start_span(
            name="dspy.evaluate",
            input=inputs,
            metadata=metadata,
            parent=parent_export,
        )
        span.set_current()
        self._spans[call_id] = span

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """Log the end of an evaluation run.

        Args:
            call_id: Unique identifier for this call
            outputs: Output from the evaluation, or None if there was an exception
            exception: Exception raised during execution, if any
        """
        span = self._spans.pop(call_id, None)
        if not span:
            return

        try:
            log_data = {}
            if exception:
                log_data["error"] = exception
            if outputs is not None:
                log_data["output"] = outputs
                if isinstance(outputs, dict):
                    metrics = {}
                    for key in ["accuracy", "score", "total", "correct"]:
                        if key in outputs:
                            try:
                                metrics[key] = float(outputs[key])
                            except (ValueError, TypeError):
                                pass
                    if metrics:
                        log_data["metrics"] = metrics

            if log_data:
                span.log(**log_data)
        finally:
            span.unset_current()
            span.end()


def _configure_wrapper(wrapped, instance, args, kwargs):
    """Wrapper for dspy.configure that auto-adds BraintrustDSpyCallback."""
    callbacks = kwargs.get("callbacks")
    if callbacks is None:
        callbacks = []
    else:
        callbacks = list(callbacks)

    if not any(isinstance(cb, BraintrustDSpyCallback) for cb in callbacks):
        callbacks.append(BraintrustDSpyCallback())

    kwargs["callbacks"] = callbacks
    return wrapped(*args, **kwargs)
