from braintrust._audio.config import AudioRecordingOptions

from .integration import LiveKitAgentsIntegration
from .patchers import wrap_livekit_agents


def setup_livekit_agents(*, audio: AudioRecordingOptions | None = None) -> bool:
    """Instrument LiveKit 1.8.9–1.8.10 cascade sessions using audio trace schema v3.

    Recording requires dedicated USER/AGENT audio attachment environment flags
    and LiveKit privacy permission. Audio uploads are direct; trace customizers
    cannot veto or undo them. Repeated setup retains explicit audio defaults.
    """
    from .tracing import configure

    configure(audio)
    return LiveKitAgentsIntegration.setup()


__all__ = ["LiveKitAgentsIntegration", "setup_livekit_agents", "wrap_livekit_agents"]
