"""Optional Pydantic AI Harness patchers."""

import inspect
from typing import Any

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import _current_agent_span


async def _step_persistence_for_run_wrapper(wrapped: Any, _instance: Any, args: Any, kwargs: Any) -> Any:
    span = _current_agent_span()
    capability = await wrapped(*args, **kwargs)
    if span is not None:
        run_id = getattr(capability, "run_id", None)
        if isinstance(run_id, str):
            lineage = {"run_id": run_id}
            parent_run_id = getattr(capability, "parent_run_id", None)
            if isinstance(parent_run_id, str):
                lineage["parent_run_id"] = parent_run_id
            span.log(metadata={"step_persistence": lineage})
    return capability


class StepPersistencePatcher(FunctionWrapperPatcher):
    name = "pydantic_ai.harness.step_persistence.for_run"
    target_module = "pydantic_ai_harness.step_persistence"
    target_path = "StepPersistence.for_run"
    wrapper = _step_persistence_for_run_wrapper

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        return super().applies(module, version, target=target) and inspect.iscoroutinefunction(
            cls.resolve_target(module, version, target=target)
        )
