"""Direct LiveKit Agents patchers."""

from typing import Any

from braintrust.integrations.base import (
    CompositeFunctionWrapperPatcher,
    FunctionWrapperPatcher,
    InstanceSetupFunctionPatcher,
)

from .tracing import (
    attach_metrics_handler,
    attach_playback_handler,
    traced_audio_output_capture_frame,
    traced_execute_tools_task,
    traced_llm_chat,
    traced_llm_stream_init,
    traced_llm_stream_run,
    traced_session_aexit,
    traced_session_audio_output_changed,
    traced_session_close,
    traced_session_run,
    traced_session_say,
    traced_session_start,
    traced_stt_recognize,
    traced_tts_synthesize,
    traced_update_user_state,
    traced_vad_stream,
    traced_vad_stream_anext,
)


class AgentSessionStartPatcher(InstanceSetupFunctionPatcher):
    name = "agent_session_start"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession.start"
    wrapper = staticmethod(traced_session_start)
    instance_setup = staticmethod(attach_metrics_handler)


class AgentSessionRunPatcher(InstanceSetupFunctionPatcher):
    name = "agent_session_run"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession.run"
    wrapper = staticmethod(traced_session_run)
    instance_setup = staticmethod(attach_metrics_handler)


class AgentSessionSayPatcher(InstanceSetupFunctionPatcher):
    name = "agent_session_say"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession.say"
    wrapper = staticmethod(traced_session_say)
    instance_setup = staticmethod(attach_metrics_handler)


class AgentSessionAexitPatcher(FunctionWrapperPatcher):
    name = "agent_session_aexit"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession.__aexit__"
    wrapper = staticmethod(traced_session_aexit)


class AgentSessionAclosePatcher(FunctionWrapperPatcher):
    name = "agent_session_aclose"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession.aclose"
    wrapper = staticmethod(traced_session_close)


class AgentSessionAcloseImplPatcher(FunctionWrapperPatcher):
    name = "agent_session_aclose_impl"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession._aclose_impl"
    wrapper = staticmethod(traced_session_close)


class AgentSessionClosePatcher(FunctionWrapperPatcher):
    name = "agent_session_close"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession.close"
    wrapper = staticmethod(traced_session_close)


class AgentSessionUserStatePatcher(FunctionWrapperPatcher):
    name = "agent_session_user_state"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession._update_user_state"
    wrapper = staticmethod(traced_update_user_state)


class AgentSessionAudioOutputChangedPatcher(FunctionWrapperPatcher):
    name = "agent_session_audio_output_changed"
    target_module = "livekit.agents.voice.agent_session"
    target_path = "AgentSession._on_audio_output_changed"
    wrapper = staticmethod(traced_session_audio_output_changed)


class AgentSessionPatcher(CompositeFunctionWrapperPatcher):
    name = "agent_session"
    sub_patchers = (
        AgentSessionStartPatcher,
        AgentSessionRunPatcher,
        AgentSessionSayPatcher,
        AgentSessionAexitPatcher,
        AgentSessionAclosePatcher,
        AgentSessionAcloseImplPatcher,
        AgentSessionClosePatcher,
        AgentSessionUserStatePatcher,
        AgentSessionAudioOutputChangedPatcher,
    )


class STTRecognizePatcher(InstanceSetupFunctionPatcher):
    name = "stt_recognize"
    target_module = "livekit.agents.stt.stt"
    target_path = "STT.recognize"
    wrapper = staticmethod(traced_stt_recognize)
    instance_setup = staticmethod(attach_metrics_handler)


class TTSSynthesizePatcher(InstanceSetupFunctionPatcher):
    name = "tts_synthesize"
    target_module = "livekit.agents.tts.tts"
    target_path = "TTS.synthesize"
    wrapper = staticmethod(traced_tts_synthesize)
    instance_setup = staticmethod(attach_metrics_handler)


class OpenAITTSSynthesizePatcher(InstanceSetupFunctionPatcher):
    name = "openai_tts_synthesize"
    target_module = "livekit.plugins.openai.tts"
    target_path = "TTS.synthesize"
    wrapper = staticmethod(traced_tts_synthesize)
    instance_setup = staticmethod(attach_metrics_handler)


class VADStreamPatcher(InstanceSetupFunctionPatcher):
    name = "vad_stream"
    target_module = "livekit.agents.vad"
    target_path = "VAD.stream"
    wrapper = staticmethod(traced_vad_stream)
    instance_setup = staticmethod(attach_metrics_handler)


class VADStreamAnextPatcher(FunctionWrapperPatcher):
    name = "vad_stream_anext"
    target_module = "livekit.agents.vad"
    target_path = "VADStream.__anext__"
    wrapper = staticmethod(traced_vad_stream_anext)


class FunctionToolPatcher(FunctionWrapperPatcher):
    name = "function_tool"
    target_module = "livekit.agents.voice.generation"
    target_path = "_execute_tools_task"
    wrapper = staticmethod(traced_execute_tools_task)


class InferenceLLMChatPatcher(FunctionWrapperPatcher):
    name = "inference_llm_chat"
    target_module = "livekit.agents.inference.llm"
    target_path = "LLM.chat"
    wrapper = staticmethod(traced_llm_chat)


class OpenAILLMChatPatcher(FunctionWrapperPatcher):
    name = "openai_llm_chat"
    target_module = "livekit.plugins.openai.llm"
    target_path = "LLM.chat"
    wrapper = staticmethod(traced_llm_chat)


class InferenceLLMStreamInitPatcher(FunctionWrapperPatcher):
    name = "inference_llm_stream_init"
    target_module = "livekit.agents.inference.llm"
    target_path = "LLMStream.__init__"
    wrapper = staticmethod(traced_llm_stream_init)


class InferenceLLMStreamRunPatcher(FunctionWrapperPatcher):
    name = "inference_llm_stream_run"
    target_module = "livekit.agents.inference.llm"
    target_path = "LLMStream._run"
    wrapper = staticmethod(traced_llm_stream_run)


class AudioOutputCaptureFramePatcher(InstanceSetupFunctionPatcher):
    name = "audio_output_capture_frame"
    target_module = "livekit.agents.voice.io"
    target_path = "AudioOutput.capture_frame"
    wrapper = staticmethod(traced_audio_output_capture_frame)
    instance_setup = staticmethod(attach_playback_handler)


class MetricsEmitterPatcher(CompositeFunctionWrapperPatcher):
    name = "metrics_emitters"
    sub_patchers = (
        STTRecognizePatcher,
        TTSSynthesizePatcher,
        OpenAITTSSynthesizePatcher,
        VADStreamPatcher,
        VADStreamAnextPatcher,
        FunctionToolPatcher,
        InferenceLLMChatPatcher,
        OpenAILLMChatPatcher,
        InferenceLLMStreamInitPatcher,
        InferenceLLMStreamRunPatcher,
        AudioOutputCaptureFramePatcher,
    )


def wrap_livekit_agents(target: Any) -> Any:
    """Instrument a LiveKit Agents class or instance directly.

    This helper is useful when global setup is not desired. It applies any
    LiveKit Agents patchers whose target method exists on ``target`` and returns
    ``target`` for convenient chaining.
    """
    AgentSessionPatcher.wrap_target(target)
    MetricsEmitterPatcher.wrap_target(target)
    return target
