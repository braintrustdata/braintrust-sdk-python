# Braintrust audio

Install `braintrust-audio` alongside `braintrust` to encode enabled session
recordings as Opus in Ogg by default. The package uses PyAV's bundled FFmpeg
codecs. Each segment is an independent file with exact playable duration.

Installing this extension does not enable recording. LiveKit recording requires
`BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS=true` and/or
`BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true`, and permission under LiveKit's
effective privacy policy. Without the extension, default recording warns once
and uses raw PCM. Explicit Opus configuration requires this extension.

```python
from braintrust.audio import AudioEncoding, AudioRecordingOptions
from braintrust.integrations.livekit_agents import setup_livekit_agents

setup_livekit_agents(audio=AudioRecordingOptions(
    encoding=AudioEncoding(codec="opus", container="ogg"),
))
```

Audio uploads use `Attachment.upload()` directly. Span customizers and masking
apply to the trace records published afterward; removing an attachment reference
does not prevent or undo its upload. Control audio egress using the dedicated
capture flags and LiveKit privacy settings.
