import inspect
from typing import Any

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from . import eou, realtime, tracing


def _patcher(name, module, path, wrapper):
    return type(
        name,
        (FunctionWrapperPatcher,),
        {
            "name": name,
            "target_module": f"livekit.agents.{module}",
            "target_path": path,
            "wrapper": staticmethod(wrapper),
        },
    )


class EOUWaitPatcher(CompositeFunctionWrapperPatcher):
    name = "eou_wait"
    sub_patchers = (
        _patcher("eou_ensure", "voice.audio_recognition", "AudioRecognition._ensure_eou_wait_span", eou.ensure),
        _patcher("eou_end", "voice.audio_recognition", "AudioRecognition._end_eou_wait_span", eou.end),
        _patcher("eou_snapshot", "voice.audio_recognition", "AudioRecognition._take_turn_snapshot", eou.snapshot),
        _patcher(
            "eou_run_detection", "voice.audio_recognition", "AudioRecognition._run_eou_detection", eou.run_detection
        ),
        _patcher("eou_vad", "voice.audio_recognition", "AudioRecognition._on_vad_event", eou.vad_event),
        _patcher(
            "eou_detection", "voice.audio_recognition", "AudioRecognition._track_eou_detection_span", eou.detection
        ),
    )

    @classmethod
    def applies(cls, module, version, *, target=None):
        end = cls.sub_patchers[1].resolve_target(module, version, target=target)
        try:
            snapshot_required = "turn_snapshot" in inspect.signature(end).parameters
        except (TypeError, ValueError):
            return False
        contracts = (
            (False, {"user_turn_span": None, "trigger": "vad", "last_speaking_time": None, "endpointing_delay": 0}),
            (
                False,
                {"outcome": "committed", "end_time": None, **({"turn_snapshot": None} if snapshot_required else {})},
            ),
            (False, {}),
            (False, {"chat_ctx": None, "trigger": "vad", "skip_reply": False}),
            (True, {"ev": None}),
            (False, {"span": None}),
        )
        for index, (patcher, (asynchronous, arguments)) in enumerate(zip(cls.sub_patchers, contracts)):
            method = patcher.resolve_target(module, version, target=target)
            if index == 2 and method is None and not snapshot_required:
                continue  # LiveKit 1.8.9 predates manual snapshots.
            if not callable(method) or inspect.iscoroutinefunction(method) != asynchronous:
                return False
            try:
                inspect.signature(method).bind(None, **arguments)
            except (TypeError, ValueError):
                return False
        return super().applies(module, version, target=target)


class AgentSessionPatcher(CompositeFunctionWrapperPatcher):
    name = "livekit_cascade_session"
    sub_patchers = (
        _patcher("cascade_user_turn_end", "voice.agent_activity", "_end_user_turn_span", tracing.end_user_turn),
        _patcher(
            "cascade_recognition_end",
            "voice.audio_recognition",
            "AudioRecognition._end_user_turn_span",
            tracing.recognition_end,
        ),
        _patcher("cascade_start", "voice.agent_session", "AgentSession.start", tracing.session_start),
        _patcher("cascade_close", "voice.agent_session", "AgentSession._aclose_impl", tracing.session_close),
        _patcher("cascade_run", "voice.agent_session", "AgentSession.run", tracing.session_run),
        _patcher("cascade_activity_start", "voice.agent_activity", "AgentActivity.start", tracing.activity_start),
        _patcher(
            "cascade_participant_linked",
            "voice.agent_session",
            "AgentSession._on_room_io_participant_linked",
            tracing.participant_linked,
        ),
        _patcher("cascade_input", "voice.agent_activity", "AgentActivity.push_audio", tracing.push_audio),
        _patcher(
            "cascade_speech_task",
            "voice.agent_activity",
            "AgentActivity._create_speech_task",
            tracing.create_speech_task,
        ),
        _patcher("cascade_agent_turn", "voice.agent_activity", "_agent_turn", tracing.agent_turn),
        _patcher("cascade_speech_done", "voice.speech_handle", "SpeechHandle._mark_done", tracing.speech_done),
        _patcher("cascade_continue", "voice.agent_activity", "_continue_discarded_turn", tracing.continue_turn),
        _patcher(
            "cascade_user_turn",
            "voice.audio_recognition",
            "AudioRecognition._ensure_user_turn_span",
            tracing.ensure_user,
        ),
        _patcher(
            "cascade_user_speech_start",
            "voice.agent_activity",
            "AgentActivity.on_start_of_speech",
            tracing.user_speech_start,
        ),
        _patcher(
            "cascade_user_speech_end",
            "voice.agent_activity",
            "AgentActivity.on_end_of_speech",
            tracing.user_speech_end,
        ),
        _patcher("cascade_stt", "voice.audio_recognition", "AudioRecognition._process_stt_event", tracing.process_stt),
        _patcher("cascade_stt_recognize", "stt.stt", "STT.recognize", tracing.stt_recognize),
        _patcher("cascade_end_of_turn", "voice.agent_activity", "AgentActivity.on_end_of_turn", tracing.end_of_turn),
        _patcher(
            "cascade_preemptive", "voice.agent_activity", "AgentActivity.on_preemptive_generation", tracing.preemptive
        ),
        _patcher(
            "cascade_user_completed",
            "voice.agent_activity",
            "AgentActivity._user_turn_completed_impl",
            tracing.user_completed,
        ),
        _patcher("cascade_llm", "voice.generation", "_llm_inference_task", tracing.llm_operation),
        _patcher("cascade_tts", "voice.generation", "_tts_inference_task", tracing.tts_operation),
        _patcher("cascade_tts_created", "voice.agent_activity", "perform_tts_inference", tracing.tts_created),
        _patcher(
            "realtime_generation",
            "voice.agent_activity",
            "AgentActivity._realtime_generation_task_impl",
            realtime.generate,
        ),
        _patcher("cascade_stt_pump", "voice.audio_recognition", "_STTPipeline._stt_pump", tracing.stt_pump),
        _patcher("cascade_stt_rebind", "voice.audio_recognition", "_STTPipeline._rebind_node", tracing.rebind_stt),
        _patcher("cascade_audio_forward", "voice.generation", "_audio_forwarding_task", tracing.audio_forward),
        _patcher("cascade_tool", "voice.tool_executor", "_ToolExecutor.execute", tracing.execute_tool),
    )


def wrap_livekit_agents(target: Any) -> Any:
    # Framework operation hooks are module functions; instance-only wrapping
    # cannot observe the complete cascade. Use the same idempotent setup path.
    from .integration import LiveKitAgentsIntegration

    LiveKitAgentsIntegration.setup()
    return target
