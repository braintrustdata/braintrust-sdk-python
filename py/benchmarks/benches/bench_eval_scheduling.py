"""Benchmarks for ordinary eval task execution and trial scheduling."""

import asyncio
import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.framework import EvalCase, Evaluator, await_or_run, run_evaluator
from braintrust.logger import BraintrustState

from benchmarks._utils import disable_pyperf_psutil


async def _async_task(input_value):
    return input_value


def _sync_task(input_value):
    return input_value


_state = BraintrustState()
_evaluator = Evaluator(
    project_name="benchmark",
    eval_name="eval-scheduling",
    data=[EvalCase(input=i) for i in range(25)],
    task=_async_task,
    scores=[],
    experiment_name=None,
    metadata=None,
    max_concurrency=5,
)


async def _bench_sync_task():
    await await_or_run(asyncio.get_running_loop(), _sync_task, 1)


async def _bench_eval_scheduler():
    await run_evaluator(None, _evaluator, None, [], state=_state, enable_cache=False)


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_async_func("eval_scheduling.sync_task", _bench_sync_task)
    runner.bench_async_func("eval_scheduling.25_async_trials", _bench_eval_scheduler)


if __name__ == "__main__":
    main()
