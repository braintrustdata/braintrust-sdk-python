# pylint: disable=protected-access,too-few-public-methods

import asyncio
import importlib
import inspect
import os
import tempfile
from pathlib import Path

import pytest
from braintrust import logger
from braintrust.integrations.pipecat import (
    BraintrustPipecatObserver,
    PipecatIntegration,
    setup_pipecat,
    wrap_pipeline_worker,
)
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.logger import Attachment
from braintrust.test_helpers import init_test_logger


@pytest.fixture
def memory_logger():
    init_test_logger("test-project-pipecat-py-tracing")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _ensure_nltk_punkt_tab():
    data_dir = Path(tempfile.gettempdir()) / "braintrust-pipecat-nltk-data"
    punkt_tab = data_dir / "tokenizers" / "punkt_tab"
    punkt_tab.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NLTK_DATA", str(data_dir))


def _import(path):
    _ensure_nltk_punkt_tab()
    module_name, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def _span_name(log):
    return log.get("span_attributes", {}).get("name")


def _span_type(log):
    return log.get("span_attributes", {}).get("type")


def _spans_named(logs, name):
    return [log for log in logs if _span_name(log) == name]


def _single_span(logs, name):
    matches = _spans_named(logs, name)
    assert len(matches) == 1, (name, matches)
    return matches[0]


def _pipeline_worker_kwargs(**overrides):
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


def _make_worker(pipeline, **overrides):
    PipelineWorker = _import("pipecat.pipeline.worker.PipelineWorker")
    return PipelineWorker(pipeline, **_pipeline_worker_kwargs(**overrides))


def _worker_runner_kwargs(**overrides):
    WorkerRunner = _import("pipecat.workers.runner.WorkerRunner")
    signature = inspect.signature(WorkerRunner)
    kwargs = {"handle_sigint": False}
    if "check_dangling_tasks" in signature.parameters:
        kwargs["check_dangling_tasks"] = False
    kwargs.update(overrides)
    return kwargs


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_pipecat_observer_capture_audio_attachments_adds_tts_and_user_audio(memory_logger):
    TTSStartedFrame = _import("pipecat.frames.frames.TTSStartedFrame")
    TTSTextFrame = _import("pipecat.frames.frames.TTSTextFrame")
    TTSAudioRawFrame = _import("pipecat.frames.frames.TTSAudioRawFrame")
    TTSStoppedFrame = _import("pipecat.frames.frames.TTSStoppedFrame")
    UserStartedSpeakingFrame = _import("pipecat.frames.frames.UserStartedSpeakingFrame")
    UserAudioRawFrame = _import("pipecat.frames.frames.UserAudioRawFrame")
    UserStoppedSpeakingFrame = _import("pipecat.frames.frames.UserStoppedSpeakingFrame")

    observer = BraintrustPipecatObserver(capture_audio_attachments=True)
    first_chunk = b"\x00\x00\x01\x00" * 20
    second_chunk = b"\x02\x00\x03\x00" * 10

    await observer.on_pipeline_started()
    await observer._handle_frame(TTSStartedFrame(context_id="ctx"))
    await observer._handle_frame(TTSTextFrame("hello from tts", aggregated_by="sentence", context_id="ctx"))
    await observer._handle_frame(TTSAudioRawFrame(first_chunk, sample_rate=16000, num_channels=1, context_id="ctx"))
    await observer._handle_frame(TTSAudioRawFrame(second_chunk, sample_rate=16000, num_channels=1, context_id="ctx"))
    await observer._handle_frame(TTSStoppedFrame(context_id="ctx"))
    await observer._handle_frame(UserStartedSpeakingFrame())
    await observer._handle_frame(UserAudioRawFrame(first_chunk, sample_rate=16000, num_channels=1, user_id="user-1"))
    await observer._handle_frame(UserAudioRawFrame(second_chunk, sample_rate=16000, num_channels=1, user_id="user-1"))
    await observer._handle_frame(UserStoppedSpeakingFrame())
    await observer.cleanup()

    logs = memory_logger.pop()
    tts_span = _single_span(logs, "tts_response")
    tts_audio = tts_span["output"]["audio"]
    assert isinstance(tts_audio, Attachment)
    assert tts_audio.reference["content_type"] == "audio/wav"
    assert tts_audio.data.startswith(b"RIFF")
    assert tts_span["output"]["audio_size_bytes"] == len(first_chunk) + len(second_chunk)
    assert tts_span["output"]["sample_rate"] == 16000
    assert tts_span["output"]["num_channels"] == 1
    assert tts_span["output"]["num_frames"] == 60

    user_span = _single_span(logs, "user_speaking")
    user_audio = user_span["input"]["audio"]
    assert isinstance(user_audio, Attachment)
    assert user_audio.reference["content_type"] == "audio/wav"
    assert user_audio.data.startswith(b"RIFF")
    assert user_span["input"]["audio_size_bytes"] == len(first_chunk) + len(second_chunk)
    assert user_span["input"]["num_frames"] == 60
    assert user_span["input"]["user_id"] == "user-1"

    observer_without_start = BraintrustPipecatObserver(capture_audio_attachments=True)
    await observer_without_start.on_pipeline_started()
    await observer_without_start._handle_frame(
        UserAudioRawFrame(first_chunk, sample_rate=16000, num_channels=1, user_id="user-1")
    )
    await observer_without_start._handle_frame(
        UserAudioRawFrame(second_chunk, sample_rate=16000, num_channels=1, user_id="user-1")
    )
    await observer_without_start._handle_frame(UserStoppedSpeakingFrame())
    await observer_without_start.cleanup()

    logs = memory_logger.pop()
    auto_started_user_span = _single_span(logs, "user_speaking")
    assert auto_started_user_span["input"]["num_frames"] == 60


@pytest.mark.asyncio
async def test_pipecat_observer_audio_capture_env_vars_are_independent(monkeypatch, memory_logger):
    TTSStartedFrame = _import("pipecat.frames.frames.TTSStartedFrame")
    TTSAudioRawFrame = _import("pipecat.frames.frames.TTSAudioRawFrame")
    TTSStoppedFrame = _import("pipecat.frames.frames.TTSStoppedFrame")
    UserStartedSpeakingFrame = _import("pipecat.frames.frames.UserStartedSpeakingFrame")
    UserAudioRawFrame = _import("pipecat.frames.frames.UserAudioRawFrame")
    UserStoppedSpeakingFrame = _import("pipecat.frames.frames.UserStoppedSpeakingFrame")
    audio = b"\x00\x00\x01\x00" * 20

    monkeypatch.setenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", "true")
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", "false")
    observer = BraintrustPipecatObserver()
    assert observer.capture_user_audio_attachments is True
    assert observer.capture_agent_audio_attachments is False

    await observer.on_pipeline_started()
    await observer._handle_frame(TTSStartedFrame(context_id="ctx"))
    await observer._handle_frame(TTSAudioRawFrame(audio, sample_rate=16000, num_channels=1, context_id="ctx"))
    await observer._handle_frame(TTSStoppedFrame(context_id="ctx"))
    await observer._handle_frame(UserStartedSpeakingFrame())
    await observer._handle_frame(UserAudioRawFrame(audio, sample_rate=16000, num_channels=1, user_id="user-1"))
    await observer._handle_frame(UserStoppedSpeakingFrame())
    await observer.cleanup()

    logs = memory_logger.pop()
    assert "audio" not in _single_span(logs, "tts_response").get("output", {})
    assert isinstance(_single_span(logs, "user_speaking")["input"]["audio"], Attachment)

    monkeypatch.setenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", "false")
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", "true")
    observer = BraintrustPipecatObserver()
    assert observer.capture_user_audio_attachments is False
    assert observer.capture_agent_audio_attachments is True

    await observer.on_pipeline_started()
    await observer._handle_frame(TTSStartedFrame(context_id="ctx"))
    await observer._handle_frame(TTSAudioRawFrame(audio, sample_rate=16000, num_channels=1, context_id="ctx"))
    await observer._handle_frame(TTSStoppedFrame(context_id="ctx"))
    await observer._handle_frame(UserStartedSpeakingFrame())
    await observer._handle_frame(UserAudioRawFrame(audio, sample_rate=16000, num_channels=1, user_id="user-1"))
    await observer._handle_frame(UserStoppedSpeakingFrame())
    await observer.cleanup()

    logs = memory_logger.pop()
    assert isinstance(_single_span(logs, "tts_response")["output"]["audio"], Attachment)
    assert not _spans_named(logs, "user_speaking")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_setup_pipecat_traces_real_pipeline_frames(memory_logger):
    EndFrame = _import("pipecat.frames.frames.EndFrame")
    LLMContextFrame = _import("pipecat.frames.frames.LLMContextFrame")
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    OpenAILLMService = _import("pipecat.services.openai.llm.OpenAILLMService")
    WorkerRunner = _import("pipecat.workers.runner.WorkerRunner")
    PipelineParams = _import("pipecat.pipeline.worker.PipelineParams")

    assert setup_pipecat(project_name="test-project-pipecat-py-tracing")
    init_test_logger("test-project-pipecat-py-tracing")
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=20,
        ),
    )
    worker = _make_worker(
        Pipeline([llm]),
        name="bt-pipecat-test-worker",
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    context = LLMContext(
        messages=[
            {"role": "developer", "content": "Answer with exactly the requested text and no punctuation."},
            {"role": "user", "content": "Say: braintrust pipecat integration"},
        ]
    )

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(_worker, _frame):
        await worker.queue_frames([LLMContextFrame(context), EndFrame()])

    runner = WorkerRunner(**_worker_runner_kwargs())
    await runner.add_workers(worker)
    await asyncio.wait_for(runner.run(), timeout=20)

    logs = memory_logger.pop()
    pipeline_span = _single_span(logs, "pipecat_pipeline")
    assert _span_type(pipeline_span) == "task"
    assert pipeline_span.get("metrics", {}).get("end") is not None

    llm_span = _single_span(logs, "pipecat_llm_response")
    assert _span_type(llm_span) == "task"
    assert llm_span["input"] == context.messages
    assert llm_span["output"][0]["finish_reason"] == "stop"
    assert "braintrust pipecat integration" in llm_span["output"][0]["message"]["content"].lower()
    assert llm_span["metadata"]["provider"] == "openai"
    assert llm_span["metadata"]["model"] == "gpt-4o-mini"
    assert set(llm_span.get("metrics", {})) <= {
        "time_to_first_token",
        "prompt_tokens",
        "completion_tokens",
        "tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
        "start",
        "end",
    }
    assert llm_span["metrics"]["prompt_tokens"] > 0
    assert llm_span["metrics"]["completion_tokens"] > 0
    assert llm_span["metrics"]["tokens"] >= llm_span["metrics"]["completion_tokens"]


@pytest.mark.vcr
def test_setup_and_wrap_pipeline_worker_are_idempotent():
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    PipelineWorker = _import("pipecat.pipeline.worker.PipelineWorker")
    IdentityFilter = _import("pipecat.processors.filters.identity_filter.IdentityFilter")

    assert PipecatIntegration.min_version == "1.3.0"
    assert setup_pipecat(project_name="test-project-pipecat-py-tracing")
    assert setup_pipecat(project_name="test-project-pipecat-py-tracing")
    assert setup_pipecat(project_name="test-project-pipecat-py-tracing", capture_audio_attachments=True)
    assert wrap_pipeline_worker(PipelineWorker) is PipelineWorker

    capturing_worker = _make_worker(Pipeline([IdentityFilter()]))
    capturing_observers = getattr(getattr(capturing_worker, "_observer"), "_observers")
    capturing_bt_observer = next(
        observer for observer in capturing_observers if isinstance(observer, BraintrustPipecatObserver)
    )
    assert capturing_bt_observer.capture_audio_attachments is True

    assert setup_pipecat(project_name="test-project-pipecat-py-tracing", capture_audio_attachments=False)
    explicit_observer = BraintrustPipecatObserver()
    worker = _make_worker(Pipeline([IdentityFilter()]), observers=[explicit_observer])
    worker_observer = getattr(worker, "_observer")
    observers = getattr(worker_observer, "_observers")
    braintrust_observers = [observer for observer in observers if isinstance(observer, BraintrustPipecatObserver)]
    assert braintrust_observers == [explicit_observer]


@pytest.mark.vcr
@pytest.mark.skipif(__import__("sys").version_info < (3, 11), reason="Pipecat AI 1.x requires Python 3.11+")
def test_auto_instrument_pipecat_subprocess():
    pytest.importorskip("pipecat")
    verify_autoinstrument_script("test_auto_pipecat.py")
