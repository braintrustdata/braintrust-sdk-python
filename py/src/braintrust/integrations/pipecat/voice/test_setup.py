# pylint: disable=import-error
# Pipecat is installed in the integration test session, not the shared lint environment.
"""Public setup against real Pipecat processors; no provider traffic required."""

import pytest
from braintrust import SpanCustomizer, set_span_customizers
from braintrust.integrations.pipecat import BraintrustPipecatObserver, setup_pipecat, wrap_pipeline_worker
from braintrust.integrations.pipecat.test_pipecat import _make_worker, memory_logger  # noqa: F401
from braintrust.test_helpers import init_test_logger
from pipecat.pipeline.pipeline import Pipeline  # pylint: disable=import-error
from pipecat.processors.aggregators.llm_context import LLMContext  # pylint: disable=import-error
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,  # pylint: disable=import-error
)
from pipecat.services.openai.stt import OpenAISTTService  # pylint: disable=import-error
from pipecat.transports.base_input import BaseInputTransport  # pylint: disable=import-error
from pipecat.transports.base_output import BaseOutputTransport  # pylint: disable=import-error
from pipecat.transports.base_transport import TransportParams  # pylint: disable=import-error


@pytest.mark.asyncio
@pytest.mark.parametrize("user,agent", [(False, False), (True, False), (False, True), (True, True)])
async def test_existing_setup_installs_one_observer_and_restores_hooks(memory_logger, user, agent):
    from pipecat.pipeline.worker import PipelineWorker

    for _ in range(2):
        assert setup_pipecat(capture_user_audio_attachments=user, capture_agent_audio_attachments=agent)
    assert wrap_pipeline_worker(PipelineWorker) is PipelineWorker
    logger = init_test_logger("test-project-pipecat-py-tracing")
    params = TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    source, destination = BaseInputTransport(params), BaseOutputTransport(params)
    aggregators = LLMContextAggregatorPair(LLMContext())
    stt = OpenAISTTService(api_key="test-key")
    original_run, original_audio, original_write = stt.run_stt, stt.process_audio_frame, destination.write_audio_frame
    with logger.start_span(name="customer-session") as parent:
        explicit = (
            BraintrustPipecatObserver(capture_user_audio_attachments=user, capture_agent_audio_attachments=agent)
            if user and agent
            else None
        )
        worker = _make_worker(
            Pipeline([source, stt, aggregators.user(), destination, aggregators.assistant()]),
            observers=[explicit] if explicit else [],
        )
    observers = [o for o in worker._observer._observers if isinstance(o, BraintrustPipecatObserver)]
    assert len(observers) == 1
    observer = observers[0]
    if explicit:
        assert observer is explicit
    voice = observer._voice
    assert voice is not None
    assert voice.capture_user_audio is user
    assert voice.capture_agent_audio is agent
    assert (stt.process_audio_frame != original_audio) is user
    assert (destination.write_audio_frame != original_write) is agent
    assert stt.run_stt != original_run
    await observer.cleanup()
    assert stt.run_stt == original_run
    assert stt.process_audio_frame == original_audio
    assert destination.write_audio_frame == original_write
    rows = memory_logger.pop()
    root = next(row for row in rows if row.get("span_attributes", {}).get("name") == "pipecat.pipeline")
    assert root["span_parents"] == [parent.span_id]
    assert not any(row.get("span_attributes", {}).get("name") == "pipecat_pipeline" for row in rows)


@pytest.mark.asyncio
async def test_native_spans_honor_export_customizer(memory_logger):
    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            data.pop("input", None)
            data.pop("output", None)
            data.pop("metadata", None)
            return data

    observer = BraintrustPipecatObserver()
    pair = LLMContextAggregatorPair(LLMContext())
    pipeline = Pipeline(
        [
            BaseInputTransport(TransportParams()),
            OpenAISTTService(api_key="test"),
            pair.user(),
            BaseOutputTransport(TransportParams()),
            pair.assistant(),
        ]
    )
    set_span_customizers([Redact()])
    try:
        observer._bind_pipeline(pipeline)
        turn = observer._voice.turns.start("user", "test")
        observer._voice.turns.log_message(turn, "private transcript")
        await observer.cleanup()
        assert all("metadata" not in row and "input" not in row and "output" not in row for row in memory_logger.pop())
    finally:
        set_span_customizers(None)
