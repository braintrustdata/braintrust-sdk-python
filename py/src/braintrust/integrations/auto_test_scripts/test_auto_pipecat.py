import asyncio
import importlib
import inspect
import os
import tempfile
from pathlib import Path

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


def _ensure_nltk_punkt_tab():
    data_dir = Path(tempfile.gettempdir()) / "braintrust-pipecat-nltk-data"
    punkt_tab = data_dir / "tokenizers" / "punkt_tab"
    punkt_tab.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NLTK_DATA", str(data_dir))


def _import(path):
    _ensure_nltk_punkt_tab()
    module_name, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def _worker_kwargs(**overrides):
    PipelineWorker = _import("pipecat.pipeline.worker.PipelineWorker")
    signature = inspect.signature(PipelineWorker)
    kwargs = {"idle_timeout_secs": None}
    for name, value in {
        "enable_turn_tracking": False,
        "enable_rtvi": False,
        "check_dangling_tasks": False,
    }.items():
        if name in signature.parameters:
            kwargs[name] = value
    kwargs.update(overrides)
    return kwargs


def _runner_kwargs(**overrides):
    WorkerRunner = _import("pipecat.workers.runner.WorkerRunner")
    signature = inspect.signature(WorkerRunner)
    kwargs = {"handle_sigint": False}
    if "check_dangling_tasks" in signature.parameters:
        kwargs["check_dangling_tasks"] = False
    kwargs.update(overrides)
    return kwargs


async def main():
    with autoinstrument_test_context("test_auto_pipecat", integration="pipecat") as memory_logger:
        _ensure_nltk_punkt_tab()
        results = auto_instrument()
        assert results.get("pipecat") is True

        EndFrame = _import("pipecat.frames.frames.EndFrame")
        LLMContextFrame = _import("pipecat.frames.frames.LLMContextFrame")
        Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
        PipelineParams = _import("pipecat.pipeline.worker.PipelineParams")
        PipelineWorker = _import("pipecat.pipeline.worker.PipelineWorker")
        LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
        OpenAILLMService = _import("pipecat.services.openai.llm.OpenAILLMService")
        WorkerRunner = _import("pipecat.workers.runner.WorkerRunner")

        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(
                model="gpt-4o-mini",
                temperature=0.0,
                max_completion_tokens=20,
            ),
        )
        worker = PipelineWorker(
            Pipeline([llm]),
            **_worker_kwargs(
                name="bt-auto-pipecat-worker",
                params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            ),
        )
        context = LLMContext(
            messages=[
                {"role": "developer", "content": "Answer with exactly the requested text and no punctuation."},
                {"role": "user", "content": "Say: braintrust auto pipecat"},
            ]
        )

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(_worker, _frame):
            await worker.queue_frames([LLMContextFrame(context), EndFrame()])

        runner = WorkerRunner(**_runner_kwargs())
        await runner.add_workers(worker)
        await asyncio.wait_for(runner.run(), timeout=20)

        logs = memory_logger.pop()
        names = {log.get("span_attributes", {}).get("name") for log in logs}
        assert "pipecat_pipeline" in names
        assert "pipecat_llm_response" in names


if __name__ == "__main__":
    asyncio.run(main())
