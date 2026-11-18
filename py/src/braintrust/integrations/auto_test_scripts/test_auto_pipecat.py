import asyncio
import importlib
import inspect
import os
import tempfile
from pathlib import Path

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context


TEST_MODEL = "gpt-6-luna"


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
        if os.environ.get("BRAINTRUST_TEST_PIPECAT_ENTRYPOINT") == "wrap":
            from braintrust.integrations.pipecat import wrap_pipeline_worker

            wrap_pipeline_worker(_import("pipecat.pipeline.worker.PipelineWorker"))
        else:
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
                model=TEST_MODEL,
                temperature=1.0,
                extra={"reasoning_effort": "none"},
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
        assert "session" in names
        assert "llm_response" in names
        await check_local_turn_wait(memory_logger)


async def check_local_turn_wait(memory_logger):
    """Real pipeline and timers with synthetic speech-boundary frames, no IO."""
    frames = importlib.import_module("pipecat.frames.frames")
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    Worker = _import("pipecat.pipeline.worker.PipelineWorker")
    Runner = _import("pipecat.workers.runner.WorkerRunner")
    Identity = _import("pipecat.processors.filters.identity_filter.IdentityFilter")
    Context = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    Pair = _import("pipecat.processors.aggregators.llm_response_universal.LLMContextAggregatorPair")
    Params = _import("pipecat.processors.aggregators.llm_response_universal.LLMUserAggregatorParams")
    Strategies = _import("pipecat.turns.user_turn_strategies.UserTurnStrategies")
    VADStart = _import("pipecat.turns.user_start.VADUserTurnStartStrategy")
    Timeout = _import("pipecat.turns.user_stop.SpeechTimeoutUserTurnStopStrategy")
    pair = Pair(
        Context(),
        user_params=Params(
            user_turn_strategies=Strategies(
                start=[VADStart(enable_user_speaking_frames=False)],
                stop=[Timeout(user_speech_timeout=0.03, enable_user_speaking_frames=False)],
            )
        ),
    )
    worker = Worker(Pipeline([Identity(), pair.user()]), **_worker_kwargs())
    ready, committed = asyncio.Event(), asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def started(*_args):
        ready.set()

    @pair.user().event_handler("on_user_turn_stopped")
    async def stopped(*_args):
        committed.set()

    runner = Runner(**_runner_kwargs())
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(ready.wait(), 5)
        await worker.queue_frame(frames.VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.03)
        await worker.queue_frames(
            [
                frames.VADUserStoppedSpeakingFrame(stop_secs=0.01),
                frames.TranscriptionFrame("Hello", "caller", "0", finalized=True),
            ]
        )
        await asyncio.wait_for(committed.wait(), 5)
        await worker.queue_frame(frames.EndFrame())
        await asyncio.wait_for(running, 10)
    finally:
        if not running.done():
            await worker.cancel()
            await running
    logs = memory_logger.pop()
    waits = [row for row in logs if row.get("span_attributes", {}).get("name") == "eou_wait"]
    assert len(waits) == 1
    assert waits[0]["metadata"]["eou.outcome"] == "committed"
    assert waits[0]["span_parents"] == [waits[0]["metadata"]["turn.id"]]
    assert all("end" in row["metrics"] for row in logs)


if __name__ == "__main__":
    asyncio.run(main())
    print("SUCCESS")
