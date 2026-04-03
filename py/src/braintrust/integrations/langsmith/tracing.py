"""LangSmith integration helpers, tracing wrappers, and LangSmith-to-Braintrust adapters."""

import inspect
import logging
import os
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, ParamSpec, TypeVar
from uuid import UUID

from braintrust.framework import EvalCase
from braintrust.logger import current_logger, traced


logger = logging.getLogger(__name__)

_LANGSMITH_STANDALONE: bool | None = None

P = ParamSpec("P")
R = TypeVar("R")


def _langsmith_standalone_from_env() -> bool:
    value = os.environ.get("BRAINTRUST_LANGSMITH_STANDALONE")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _set_langsmith_standalone_override(standalone: bool | None = None) -> bool:
    """Set an explicit standalone override for patched LangSmith surfaces."""
    global _LANGSMITH_STANDALONE
    _LANGSMITH_STANDALONE = standalone
    return get_langsmith_standalone()


def get_langsmith_standalone() -> bool:
    """Return whether LangSmith should be bypassed in favor of Braintrust-only behavior."""
    if _LANGSMITH_STANDALONE is not None:
        return _LANGSMITH_STANDALONE
    return _langsmith_standalone_from_env()


def reset_langsmith_state() -> None:
    """Reset standalone overrides and collected Braintrust eval results."""
    global _LANGSMITH_STANDALONE
    _LANGSMITH_STANDALONE = None


def _resolve_eval_project() -> tuple[str, str | None]:
    project_name = None
    project_id = None

    active_logger = current_logger()
    if active_logger is not None:
        try:
            project = active_logger.project
            project_name = getattr(project, "name", None)
            candidate_project_id = getattr(project, "id", None)
            if isinstance(candidate_project_id, str):
                try:
                    UUID(candidate_project_id)
                    project_id = candidate_project_id
                except ValueError:
                    project_id = None
        except Exception:
            pass

    if project_name is None:
        project_name = os.environ.get("LANGCHAIN_PROJECT")

    return project_name or "langsmith-migration", project_id


# =============================================================================
# Raw wrapt wrappers used by integration patchers
# =============================================================================


def _traceable_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _make_traceable_wrapper(wrapped, standalone=get_langsmith_standalone())(*args, **kwargs)


def _client_evaluate_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_evaluate_with_fallback(
        wrapped,
        args,
        kwargs,
        standalone=get_langsmith_standalone(),
        client=instance,
    )


async def _client_aevaluate_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return await _run_aevaluate_with_fallback(
        wrapped,
        args,
        kwargs,
        standalone=get_langsmith_standalone(),
        client=instance,
    )


def _evaluate_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return _run_evaluate_with_fallback(
        wrapped,
        args,
        kwargs,
        standalone=get_langsmith_standalone(),
    )


async def _aevaluate_wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    return await _run_aevaluate_with_fallback(
        wrapped,
        args,
        kwargs,
        standalone=get_langsmith_standalone(),
    )


# =============================================================================
# Wrapper factories / execution helpers
# =============================================================================


def _make_traceable_wrapper(traceable: Callable[..., Any], *, standalone: bool) -> Callable[..., Any]:
    def traceable_wrapper(*args: Any, **kwargs: Any) -> Any:
        func = args[0] if args and callable(args[0]) else None

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            span_name = kwargs.get("name") or fn.__name__
            traced_fn: Callable[..., Any] = fn

            if not standalone:
                traced_fn = traceable(fn, **kwargs)

            return traced(name=span_name)(traced_fn)  # type: ignore[return-value]

        if func is not None:
            return decorator(func)
        return decorator

    return traceable_wrapper


def _run_evaluate_with_fallback(
    wrapped: Callable[..., Any],
    args: Any,
    kwargs: Any,
    *,
    standalone: bool,
    client: Any | None = None,
) -> Any:
    result = None
    if not standalone:
        result = wrapped(*args, **kwargs)

    try:
        result = _run_braintrust_eval(args, kwargs, client=client)
    except Exception as exc:
        if standalone:
            raise
        logger.warning("Braintrust evaluate failed: %s", exc)

    return result


async def _run_aevaluate_with_fallback(
    wrapped: Callable[..., Any],
    args: Any,
    kwargs: Any,
    *,
    standalone: bool,
    client: Any | None = None,
) -> Any:
    result = None
    if not standalone:
        result = await wrapped(*args, **kwargs)

    try:
        result = await _run_braintrust_eval_async(args, kwargs, client=client)
    except Exception as exc:
        if standalone:
            raise
        logger.warning("Braintrust aevaluate failed: %s", exc)

    return result


# =============================================================================
# Braintrust evaluation logic
# =============================================================================


def _run_braintrust_eval(
    args: Any,
    kwargs: Any,
    client: Any | None = None,
) -> Any:
    """Run Braintrust Eval with LangSmith-style arguments."""
    from braintrust.framework import Eval

    target = args[0] if args else kwargs.get("target")
    data = args[1] if len(args) > 1 else kwargs.get("data")
    evaluators = kwargs.get("evaluators")
    experiment_prefix = kwargs.get("experiment_prefix")
    description = kwargs.get("description")
    metadata = kwargs.get("metadata")
    max_concurrency = kwargs.get("max_concurrency")
    num_repetitions = kwargs.get("num_repetitions", 1)
    project_name, project_id = _resolve_eval_project()

    scorers = []
    if evaluators:
        for evaluator in evaluators:
            scorers.append(_make_braintrust_scorer(evaluator))

    return Eval(
        name=project_name,
        data=_convert_langsmith_data(data, client=client),
        task=_make_braintrust_task(target),
        scores=scorers,
        experiment_name=experiment_prefix,
        project_id=project_id,
        description=description,
        metadata=metadata,
        max_concurrency=max_concurrency,
        trial_count=num_repetitions,
    )


async def _run_braintrust_eval_async(
    args: Any,
    kwargs: Any,
    client: Any | None = None,
) -> Any:
    """Run Braintrust EvalAsync with LangSmith-style arguments."""
    from braintrust.framework import EvalAsync

    target = args[0] if args else kwargs.get("target")
    data = args[1] if len(args) > 1 else kwargs.get("data")
    evaluators = kwargs.get("evaluators")
    experiment_prefix = kwargs.get("experiment_prefix")
    description = kwargs.get("description")
    metadata = kwargs.get("metadata")
    max_concurrency = kwargs.get("max_concurrency")
    num_repetitions = kwargs.get("num_repetitions", 1)
    project_name, project_id = _resolve_eval_project()

    scorers = []
    if evaluators:
        for evaluator in evaluators:
            scorers.append(_make_braintrust_scorer(evaluator))

    return await EvalAsync(
        name=project_name,
        data=_convert_langsmith_data(data, client=client),
        task=_make_braintrust_task(target),
        scores=scorers,
        experiment_name=experiment_prefix,
        project_id=project_id,
        description=description,
        metadata=metadata,
        max_concurrency=max_concurrency,
        trial_count=num_repetitions,
    )


# =============================================================================
# Data conversion helpers
# =============================================================================


def _wrap_output(output: Any) -> Dict[str, Any]:
    """Wrap non-dict outputs the same way LangSmith does."""
    if not isinstance(output, dict):
        return {"output": output}
    return output


def _make_braintrust_scorer(
    evaluator: Callable[..., Any],
) -> Callable[..., Any]:
    """Create a Braintrust scorer from a LangSmith evaluator."""
    evaluator_name = getattr(evaluator, "__name__", "score")

    def braintrust_scorer(input: Any, output: Any, expected: Optional[Any] = None, **kwargs: Any) -> Any:
        from braintrust.score import Score

        outputs = _wrap_output(output)
        reference_outputs = expected.outputs if hasattr(expected, "outputs") else expected
        result = evaluator(input, outputs, reference_outputs)

        return Score(
            name=result.get("key", evaluator_name),
            score=result.get("score"),
            metadata=result.get("metadata", {}),
        )

    braintrust_scorer.__name__ = evaluator_name
    return braintrust_scorer


def _resolve_data_client(client: Any | None) -> Any:
    if client is not None and hasattr(client, "list_examples"):
        return client

    from langsmith import Client  # pylint: disable=import-error

    return Client()


def _convert_langsmith_data(
    data: Any,
    *,
    client: Any | None = None,
) -> Callable[[], Iterator[EvalCase[Any, Any]]]:
    """Convert LangSmith data format to Braintrust data format."""

    def load_data() -> Iterator[EvalCase[Any, Any]]:
        source: Iterable[Any]
        if callable(data):
            source = data()  # type: ignore[misc]
        elif isinstance(data, str):
            try:
                source = _resolve_data_client(client).list_examples(dataset_name=data)
            except Exception as exc:
                logger.warning("Failed to load LangSmith dataset %r: %s", data, exc)
                return
        elif hasattr(data, "__iter__"):
            source = data
        else:
            source = [data]

        for item in source:
            if hasattr(item, "inputs"):
                yield EvalCase(
                    input=item.inputs,
                    expected=item,
                    metadata=getattr(item, "metadata", None),
                )
            elif isinstance(item, dict):
                if "inputs" in item:
                    yield EvalCase(
                        input=item["inputs"],
                        expected=item,
                        metadata=item.get("metadata"),
                    )
                elif "input" in item:
                    yield EvalCase(
                        input=item["input"],
                        expected=item.get("expected"),
                        metadata=item.get("metadata"),
                    )
                else:
                    yield EvalCase(input=item)
            else:
                yield EvalCase(input=item)

    return load_data


def _make_braintrust_task(target: Callable[..., Any]) -> Callable[..., Any]:
    """Convert a LangSmith target function to Braintrust task format."""

    def task_fn(task_input: Any, hooks: Any) -> Any:
        if isinstance(task_input, dict):
            unwrapped = inspect.unwrap(target)

            try:
                sig = inspect.signature(unwrapped)
                params = list(sig.parameters.keys())
                if len(params) == 1:
                    return target(task_input)
                if all(param in task_input for param in params):
                    return target(**task_input)
                return target(task_input)
            except (ValueError, TypeError):
                try:
                    return target(**task_input)
                except TypeError:
                    return target(task_input)
        return target(task_input)

    return task_fn
