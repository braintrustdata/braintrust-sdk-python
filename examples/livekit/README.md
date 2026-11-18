# LiveKit voice tracing and recording

Run this example with Python 3.12–3.14. Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
`LIVEKIT_API_SECRET`, `OPENAI_API_KEY`, and `BRAINTRUST_API_KEY`, then run from
this directory:

```sh
uv sync
uv run voice_agent.py dev
```

Connect a caller to the same LiveKit room to talk to the agent.

## Choose a pipeline

| `LIVEKIT_MODE` | Pipeline |
| --- | --- |
| `cascade` (default) | Speech-to-text → text model → text-to-speech |
| `half-cascade` | Realtime text model → text-to-speech |
| `realtime` | Native speech-to-speech |

For example:

```sh
LIVEKIT_MODE=realtime uv run voice_agent.py dev
```

## What to expect

Traces appear in the `livekit-cascade` Braintrust project, with user and agent
turns, transcripts, model requests, tool calls, and available usage and timing.
Turn timing also shows the wait after the user finishes speaking.

Audio recording is off by default. To include input and generated audio, set
these flags before starting the worker:

```sh
export BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS=true
export BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true
```

Transcripts are still logged when audio recording is off. LiveKit already
installs PyAV; recordings use Ogg/Opus when available and raw PCM otherwise.

## Customize recordings

Replace the example's `setup_livekit_agents()` call to change recording options.
For example, use 30-second WAV segments:

```python
from braintrust.audio import AudioEncoding, AudioRecordingOptions
from braintrust.integrations.livekit_agents import setup_livekit_agents

setup_livekit_agents(
    audio=AudioRecordingOptions(
        segment_duration_seconds=30,
        encoding=AudioEncoding(codec="pcm", container="wav"),
    )
)
```

The capture flags above still control whether audio is recorded.
