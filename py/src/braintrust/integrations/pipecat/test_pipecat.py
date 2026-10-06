# pylint: disable=protected-access,too-few-public-methods

import asyncio
import importlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from braintrust import SpanCustomizer, logger, set_span_customizers
from braintrust.integrations.pipecat import (
    BraintrustPipecatObserver,
    setup_pipecat,
    wrap_pipeline_worker,
)
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.integrations.versioning import detect_module_version, version_satisfies
from braintrust.logger import Attachment
from braintrust.test_helpers import init_test_logger


@pytest.fixture
def memory_logger():
    init_test_logger("test-project-pipecat-py-tracing")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture
def vcr_cassette_name(request):
    return request.node.originalname or request.node.name


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


@pytest.mark.asyncio
async def test_span_customizer_redacts_incremental_tts_input(memory_logger):
    TTSStartedFrame = _import("pipecat.frames.frames.TTSStartedFrame")
    TTSTextFrame = _import("pipecat.frames.frames.TTSTextFrame")
    TTSStoppedFrame = _import("pipecat.frames.frames.TTSStoppedFrame")

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            if "input" in data:
                data["input"] = "[redacted]"
            return data

    observer = BraintrustPipecatObserver()
    set_span_customizers([Redact()])
    try:
        await observer.on_pipeline_started()
        await observer._handle_frame(TTSStartedFrame(context_id="ctx"))
        first_frame = TTSTextFrame("private first", aggregated_by="sentence", context_id="ctx")
        await observer._handle_frame(first_frame)
        initial = memory_logger.pop()
        pipeline = _single_span(initial, "pipecat_pipeline")
        tts = _single_span(initial, "tts_response")
        assert tts["input"] == "[redacted]"
        assert tts["span_parents"] == [pipeline["span_id"]]
        assert first_frame.text == "private first"

        second_frame = TTSTextFrame("private second", aggregated_by="sentence", context_id="ctx")
        await observer._handle_frame(second_frame)
        await observer._handle_frame(TTSStoppedFrame(context_id="ctx"))
        await observer.cleanup()
        updates = memory_logger.pop()
        updated_tts = next(row for row in updates if row["id"] == tts["id"])
        assert updated_tts["input"] == "[redacted]"
        assert updated_tts["span_id"] == tts["span_id"]
        assert second_frame.text == "private second"
    finally:
        set_span_customizers(None)


def _make_worker(pipeline, **overrides):
    PipelineWorker = _import("pipecat.pipeline.worker.PipelineWorker")
    return PipelineWorker(pipeline, **overrides)


def _worker_runner_kwargs(**overrides):
    # pytest owns process signal handling; retain all pipeline/lifecycle defaults.
    return {"handle_sigint": False, **overrides}


@pytest.mark.parametrize("capture_user,capture_agent", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.asyncio
async def test_legacy_audio_capture_policy(monkeypatch, memory_logger, capture_user, capture_agent):
    # Local PCM handling and opt-in policy have no HTTP behavior to record.
    import io
    import wave

    frames = importlib.import_module("pipecat.frames.frames")
    monkeypatch.setenv("BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS", str(capture_user).lower())
    monkeypatch.setenv("BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS", str(capture_agent).lower())
    observer = BraintrustPipecatObserver(trace_turns=False)
    chunks = [b"\x01\x00" * 40, b"\x02\x00" * 20]
    await observer.on_pipeline_started()
    await observer._handle_frame(frames.TTSStartedFrame(context_id="ctx"))
    for chunk in chunks:
        await observer._handle_frame(frames.TTSAudioRawFrame(chunk, 16000, 1, context_id="ctx"))
    await observer._handle_frame(frames.TTSStoppedFrame(context_id="ctx"))
    # Cover both explicit speech start and the implicit start on first audio.
    if not capture_agent:
        await observer._handle_frame(frames.UserStartedSpeakingFrame())
    for chunk in chunks:
        await observer._handle_frame(frames.UserAudioRawFrame(chunk, 16000, 1, user_id="user-1"))
    await observer._handle_frame(frames.UserStoppedSpeakingFrame())
    await observer.cleanup()
    logs = memory_logger.pop()
    tts = _single_span(logs, "tts_response").get("output", {})
    user = _single_span(logs, "user_speaking")["input"] if capture_user else {}
    assert ("audio" in tts) is capture_agent
    assert bool(_spans_named(logs, "user_speaking")) is capture_user
    for payload in (tts, user):
        if "audio" not in payload:
            continue
        assert isinstance(payload["audio"], Attachment)
        assert payload["audio"].reference["content_type"] == "audio/wav"
        with wave.open(io.BytesIO(payload["audio"].data)) as audio:
            assert audio.getframerate() == 16000
            assert audio.getnchannels() == 1
            assert audio.readframes(audio.getnframes()) == b"".join(chunks)
        assert payload["audio_size_bytes"] == 120
        assert payload["num_frames"] == 60


@pytest.mark.vcr
@pytest.mark.parametrize("native", [False, True], ids=["legacy", "native"])
@pytest.mark.asyncio
async def test_setup_pipecat_traces_real_pipeline_frames(memory_logger, native):
    EndFrame = _import("pipecat.frames.frames.EndFrame")
    LLMContextFrame = _import("pipecat.frames.frames.LLMContextFrame")
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    LLMContext = _import("pipecat.processors.aggregators.llm_context.LLMContext")
    OpenAILLMService = _import("pipecat.services.openai.llm.OpenAILLMService")
    WorkerRunner = _import("pipecat.workers.runner.WorkerRunner")
    PipelineParams = _import("pipecat.pipeline.worker.PipelineParams")

    if native and not version_satisfies(
        detect_module_version(importlib.import_module("pipecat"), ("pipecat",)), ">=1.12.0"
    ):
        pytest.skip("Native voice hooks require Pipecat 1.12")
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
    processors = [llm]
    if native:
        pair = _import("pipecat.processors.aggregators.llm_response_universal.LLMContextAggregatorPair")(LLMContext())
        params = _import("pipecat.transports.base_transport.TransportParams")()
        processors = [
            _import("pipecat.transports.base_input.BaseInputTransport")(params),
            _import("pipecat.services.openai.stt.OpenAISTTService")(api_key="unused"),
            pair.user(),
            llm,
            _import("pipecat.transports.base_output.BaseOutputTransport")(params),
            pair.assistant(),
        ]
    worker = _make_worker(
        Pipeline(processors),
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
        await worker.queue_frames([LLMContextFrame(context), EndFrame(reason="pipeline complete")])

    runner = WorkerRunner(**_worker_runner_kwargs())
    await runner.add_workers(worker)
    await asyncio.wait_for(runner.run(), timeout=20)

    observer = next(o for o in getattr(worker, "_observer")._observers if isinstance(o, BraintrustPipecatObserver))
    assert (observer._voice is not None) is native

    logs = memory_logger.pop()
    if native:
        pipeline_span = _single_span(logs, "pipecat.pipeline")
        turn = _single_span(logs, "assistant_turn")
        llm_span = _single_span(logs, "llm_response")
        assert turn["span_parents"] == [pipeline_span["span_id"]]
        assert llm_span["span_parents"] == [turn["span_id"]]
        assert llm_span["metadata"]["turn.id"] == turn["span_id"]
        assert llm_span["span_attributes"]["type"] == "llm"
        assert "braintrust pipecat integration" in llm_span["output"][0]["content"].lower()
        assert llm_span["metadata"]["contrib.pipecat.usage"]["value"]["total_tokens"] == llm_span["metrics"]["tokens"]
        assert llm_span["metadata"]["model"] == "gpt-4o-mini"
        assert llm_span["metadata"]["provider"] == "openai"
        assert llm_span["metrics"]["prompt_tokens"] > 0
        assert llm_span["metrics"]["completion_tokens"] > 0
        assert llm_span["metrics"]["time_to_first_token"] >= 0
        assert llm_span["metadata"]["contrib.pipecat.ttfb"]
        return
    pipeline_span = _single_span(logs, "pipecat_pipeline")
    assert _span_type(pipeline_span) == "task"
    assert pipeline_span.get("metrics", {}).get("end") is not None
    assert pipeline_span["metadata"]["terminal_frame"] == "EndFrame"
    assert pipeline_span["metadata"]["reason"] == "pipeline complete"

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


def test_setup_and_wrap_pipeline_worker_are_idempotent():
    Pipeline = _import("pipecat.pipeline.pipeline.Pipeline")
    PipelineWorker = _import("pipecat.pipeline.worker.PipelineWorker")
    IdentityFilter = _import("pipecat.processors.filters.identity_filter.IdentityFilter")

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


@pytest.mark.parametrize("metric_name,capture_audio", [("TurnMetricsData", False), ("SmartTurnMetricsData", True)])
@pytest.mark.asyncio
async def test_legacy_turn_metrics_follow_speech_or_pipeline(memory_logger, metric_name, capture_audio):
    frames = importlib.import_module("pipecat.frames.frames")
    metric = _import(f"pipecat.metrics.metrics.{metric_name}")(
        processor="BaseSmartTurn", is_complete=True, probability=0.97, e2e_processing_time_ms=82.4
    )
    observer = BraintrustPipecatObserver(trace_turns=False, capture_audio_attachments=capture_audio)
    await observer.on_pipeline_started()
    await observer._handle_frame(frames.UserStartedSpeakingFrame())
    await observer._handle_frame(frames.MetricsFrame(data=[metric]))
    await observer.cleanup()
    span = _single_span(memory_logger.pop(), "user_speaking" if capture_audio else "pipecat_pipeline")
    assert span["metadata"]["contrib.pipecat.turn_metrics"] == [
        dict(
            type=metric_name,
            processor="BaseSmartTurn",
            is_complete=True,
            probability=0.97,
            e2e_processing_time_ms=82.4,
        )
    ]


@pytest.mark.asyncio
async def test_ttfb_routes_by_processor_and_retains_unmatched(memory_logger):
    observer = BraintrustPipecatObserver(trace_turns=False, capture_audio_attachments=False)
    frames = importlib.import_module("pipecat.frames.frames")
    metric_class = _import("pipecat.metrics.metrics.TTFBMetricsData")
    processor_class = _import("pipecat.processors.frame_processor.FrameProcessor")
    llm, tts, stt = [processor_class(name=name) for name in ("llm", "tts", "stt")]
    await observer._handle_frame(frames.LLMFullResponseStartFrame(), processor=llm)
    await observer._handle_frame(frames.TTSStartedFrame(), processor=tts)
    for processor, value in ((tts, 0.09), (stt, 0.12), (llm, 0.24)):
        await observer._handle_frame(
            frames.MetricsFrame(data=[metric_class(processor=processor.name, value=value)]), processor=processor
        )
    usage_class = _import("pipecat.metrics.metrics.LLMUsageMetricsData")
    tokens_class = _import("pipecat.metrics.metrics.LLMTokenUsage")
    await observer._handle_frame(
        frames.MetricsFrame(
            data=[
                usage_class(
                    processor="llm", value=tokens_class(prompt_tokens=10, completion_tokens=5, total_tokens=15)
                ),
                usage_class(
                    processor="unrelated-classifier",
                    value=tokens_class(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                ),
            ]
        ),
        processor=llm,
    )
    await observer._handle_frame(frames.TTSStoppedFrame(), processor=tts)
    observer._close_all_open_spans()
    logs = memory_logger.pop()
    assert _single_span(logs, "tts_response")["metadata"]["contrib.pipecat.ttfb"][0]["value"] == 0.09
    assert _single_span(logs, "pipecat_llm_response")["metrics"]["time_to_first_token"] == 0.24
    assert _single_span(logs, "pipecat_llm_response")["metrics"]["tokens"] == 15
    assert _single_span(logs, "pipecat_pipeline")["metadata"]["contrib.pipecat.ttfb"][0]["processor"] == "stt"


def test_ttfb_router_bounds_and_rejects_mismatched_sources():
    from braintrust.integrations.pipecat.ttfb import TTFBRouter

    metric_class = _import("pipecat.metrics.metrics.TTFBMetricsData")
    root, operation = [], []
    router = TTFBRouter(lambda **row: root.append(row))
    source = SimpleNamespace(name="tts")
    router.start("a", source, lambda **row: operation.append(row))
    metric = metric_class(processor="tts", value=0.1)
    router.capture(metric, SimpleNamespace(name="tts"))
    assert not operation  # Same name is not the same processor instance.
    for _ in range(35):
        router.capture(metric, source)
    assert operation[-1]["metadata"]["braintrust.ttfb.omitted"] == 3
    measurement = _import("pipecat.metrics.metrics.ProcessingMetricsData")(processor="tts", value=0.5)
    router.capture_measurement(measurement, SimpleNamespace(name="tts"))
    assert root[-1]["metadata"]["contrib.pipecat.measurements"][0]["type"] == "ProcessingMetricsData"
    for _ in range(35):
        router.capture_measurement(measurement, source)
    assert operation[-1]["metadata"]["braintrust.measurements.omitted"] == 3
    retained = [
        row["metadata"]["contrib.pipecat.measurements"]
        for row in operation
        if "contrib.pipecat.measurements" in row["metadata"]
    ]
    assert len(retained[-1]) == 32
    router.start("a", source, lambda **row: operation.append(row))
    assert router.capture(metric, source) is None
    router.clear()
    assert not router.active


def test_request_metrics_keep_identity_across_overlap_shutdown_and_limits():
    from braintrust.integrations.pipecat.ttfb import TTFBRouter

    usage = _import("pipecat.metrics.metrics.TTSUsageMetricsData")
    root, first, second = [], [], []
    source = SimpleNamespace(name="tts")
    router = TTFBRouter(lambda **row: root.append(row))
    a, b = ("tts", "a"), ("tts", "b")
    router.start(b, source, lambda **row: second.append(row))
    router.capture_request(usage(processor="tts", value=11), source, a)
    assert not root and not second  # An active sibling is not the request's owner.
    router.start(a, source, lambda **row: first.append(row))
    router.capture_request(usage(processor="tts", value=22), source, b)
    router.end(a)
    router.capture_request(usage(processor="tts", value=33), source, a)
    assert [m["value"] for m in first[-1]["metadata"]["contrib.pipecat.measurements"]] == [11, 33]
    assert [m["value"] for m in second[-1]["metadata"]["contrib.pipecat.measurements"]] == [22]
    router.capture_request(usage(processor="tts", value=44), SimpleNamespace(name="tts"), b)
    assert root[-1]["metadata"]["contrib.pipecat.measurements"][0]["value"] == 44
    # Requests that never produce audio, including cancellation, cannot grow
    # retention without bound or be silently dropped at shutdown.
    for index in range(300):
        router.capture_request(usage(processor="tts", value=index), source, ("tts", str(index)))
    assert len(router.pending) == 256
    for index in range(100):
        key = ("tts", f"completed-{index}")
        router.start(key, source, lambda **row: None)
        router.end(key)
    assert len(router.completed) == 64
    router.clear()
    assert not router.pending and not router.completed and not router.active
    assert root[-1]["metadata"]["braintrust.measurements.omitted"] == 269
