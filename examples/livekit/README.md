# LiveKit cascade tracing and recording

This example qualifies `livekit-agents==1.8.10`. Set the usual LiveKit and OpenAI
credentials, plus `BRAINTRUST_API_KEY`. Install with `uv sync` from this directory.

```sh
export BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS=true
export BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true
uv run cascade.py dev
```

See [recording configuration and privacy](../../docs/livekit-cascade.md). The
default is Opus in Ogg when `braintrust-audio` is installed. Neither installing
the extension nor the generic `BRAINTRUST_CAPTURE_ATTACHMENTS` flag enables audio.
