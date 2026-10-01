# pylint: disable=unsubscriptable-object
# Turn roles are assigned through setattr in the native lifecycle dispatcher.
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.frames.frames import (  # pylint: disable=import-error
    BotSpeakingFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext  # pylint: disable=import-error
from pipecat.processors.frame_processor import FrameDirection  # pylint: disable=import-error

from .instrumentation import NativeObserver
from .test_instrumentation import Span


class TurnTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = NativeObserver(Span(), retain_audio=False)
        self.user = SimpleNamespace(name="user aggregator")
        self.assistant = SimpleNamespace(name="assistant aggregator")
        self.observer.user_aggregator = self.user
        self.observer.assistant_aggregator = self.assistant

    async def push(self, frame, source=None):
        await self.observer.on_push_frame(
            SimpleNamespace(
                frame=frame,
                first_push=True,
                source=source or SimpleNamespace(name="source"),
                destination=SimpleNamespace(name="sink"),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=1234,
            )
        )

    async def test_tool_continuation_preserves_origin_after_next_user_starts(self):
        observer = self.observer
        await self.push(UserStartedSpeakingFrame())
        first_user = observer.turns.user
        assert first_user is not None
        observer.turns.confirm("user")
        context = LLMContext([{"role": "user", "content": "order?"}])
        await self.push(LLMContextFrame(context), self.user)
        observer.turns.stop("user", SimpleNamespace(content="order?", timestamp="t1", user_id="u"))
        await self.push(LLMFullResponseStartFrame())
        first_assistant = observer.turns.assistant
        assert first_assistant is not None
        observer.turns.confirm("assistant")
        first_model = observer.llm
        await self.push(
            FunctionCallsStartedFrame(
                [
                    FunctionCallFromLLM(
                        function_name="lookup_order",
                        tool_call_id="call-1",
                        arguments={},
                        context=context,
                    )
                ]
            )
        )
        await self.push(
            FunctionCallInProgressFrame(
                function_name="lookup_order",
                tool_call_id="call-1",
                arguments={},
                group_id="group-1",
            )
        )
        tool = observer.tools["call-1"]
        await self.push(LLMFullResponseEndFrame())
        observer.turns.stop("assistant", SimpleNamespace(content="", timestamp="t2", interrupted=False))
        await self.push(UserStartedSpeakingFrame())
        await self.push(
            FunctionCallResultFrame(
                function_name="lookup_order",
                tool_call_id="call-1",
                arguments={},
                result={"status": "sent"},
            )
        )
        context.add_message({"role": "tool", "tool_call_id": "call-1", "content": "sent"})
        await self.push(LLMContextFrame(context), self.assistant)
        await self.push(LLMFullResponseStartFrame())
        continuation = observer.turns.assistant
        assert continuation is not None
        self.assertIsNot(continuation, first_assistant)
        self.assertEqual(
            continuation["metadata"]["braintrust.turn.reply_to"],
            first_user["span"].span_id,
        )
        self.assertIn(tool, first_model.children)
        self.assertEqual(
            tool.rows[0]["metadata"]["braintrust.turn.id"],
            first_assistant["span"].span_id,
        )
        self.assertEqual(
            observer.llm.rows[0]["metadata"]["braintrust.continuation.tool_call_ids"],
            ["call-1"],
        )
        await self.push(TTSStartedFrame(context_id="tts-1"))
        self.assertIn(observer.tts["tts-1"]["span"], continuation["span"].children)
        # A historical tool result does not relink a later context push.
        await self.push(LLMContextFrame(context), self.assistant)
        self.assertIsNone(observer.context_reply_to)
        self.assertEqual(observer.context_tool_results, [])
        await observer.finish()

    async def test_low_value_events_skip_serialization_and_keep_later_interruption(
        self,
    ):
        with patch(
            "braintrust.integrations.pipecat.voice.instrumentation.native_value",
            side_effect=AssertionError("serialized"),
        ):
            for _ in range(500):
                await self.push(BotSpeakingFrame())
        first, second = InterruptionFrame(), InterruptionFrame()
        first.broadcast_sibling_id = second.id
        second.broadcast_sibling_id = first.id
        await self.push(first)
        await self.push(second)
        self.assertEqual(len(self.observer.events), 1)
        self.assertEqual(self.observer.filtered_events["BotSpeakingFrame"], 500)
        self.assertEqual(self.observer.duplicate_events, 1)
        self.assertEqual(self.observer.omitted_events, 0)
        await self.observer.finish()

    async def test_delayed_callback_ends_its_turn_not_new_response(self):
        await self.push(LLMFullResponseStartFrame())
        first = self.observer.turns.assistant
        assert first is not None
        self.observer.turns.confirm("assistant")
        await self.push(LLMFullResponseEndFrame())
        await self.push(LLMFullResponseStartFrame())
        second = self.observer.turns.assistant
        assert second is not None
        self.observer.turns.stop("assistant", SimpleNamespace(content="", timestamp="old", interrupted=True))
        self.assertTrue(first["ended"])
        self.assertIs(self.observer.turns.assistant, second)
        self.observer.turns.confirm("assistant")
        await self.observer.finish()
        self.assertTrue(second["ended"])
        self.assertTrue(second["span"].rows[-1]["metadata"]["braintrust.turn.incomplete"])

    async def test_interruption_ends_tts_and_transcription_does_not_guess_a_turn(self):
        await self.push(UserStartedSpeakingFrame())
        self.observer.turns.stop("user", SimpleNamespace(content="old", timestamp="t1"))
        await self.push(TranscriptionFrame(text="new question", user_id="", timestamp="t2"))
        transcription = self.observer.root.children[-1]
        self.assertEqual(transcription.rows[0]["name"], "pipecat.stt_transcription")
        self.assertNotIn("braintrust.turn.id", transcription.rows[0]["metadata"])
        await self.push(TTSStartedFrame(context_id="interrupted"))
        state = self.observer.tts["interrupted"]
        await self.push(InterruptionFrame())
        self.assertEqual(self.observer.tts, {})
        self.assertIn(state, self.observer.recordings)
        self.assertEqual(state["span"].rows[-1]["metadata"]["pipecat.end_frame"], "InterruptionFrame")
        await self.observer.finish()

    async def test_native_turn_metrics_ownership_and_limits(self):
        from pipecat.frames.frames import MetricsFrame  # pylint: disable=import-error
        from pipecat.metrics.metrics import SmartTurnMetricsData, TurnMetricsData  # pylint: disable=import-error

        observer = self.observer
        await self.push(UserStartedSpeakingFrame())
        turn = observer.turns.user
        for metric_class in (TurnMetricsData, SmartTurnMetricsData):
            for index in range(20):
                frame = MetricsFrame(
                    data=[
                        metric_class(
                            processor="BaseSmartTurn",
                            is_complete=index == 19,
                            probability=0.97,
                            e2e_processing_time_ms=82.4,
                        )
                    ]
                )
                await self.push(frame, self.user)
                await self.push(frame, self.user)  # broadcast/repeated observations do not duplicate predictions
        metadata = {key: value for row in turn["span"].rows for key, value in row.get("metadata", {}).items()}
        assert len(metadata["pipecat.turn_metrics"]) == 32
        assert metadata["braintrust.turn_metrics.omitted"] == 8
        assert metadata["pipecat.turn_metrics"][0]["processor"] == "BaseSmartTurn"
        assert metadata["pipecat.turn_metrics"][20]["type"] == "SmartTurnMetricsData"
        before = len(turn["span"].rows)
        await self.push(
            MetricsFrame(
                data=[TurnMetricsData(processor="other", is_complete=True, probability=1, e2e_processing_time_ms=3)]
            )
        )
        assert len(turn["span"].rows) == before
        observer.turns.stop("user", SimpleNamespace(content="hello", timestamp="t", user_id="u"))
        await self.push(UserStartedSpeakingFrame())
        assert not any("pipecat.turn_metrics" in row.get("metadata", {}) for row in observer.turns.user["span"].rows)
        await observer.finish()
