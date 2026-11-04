# LiveKit cascade tracing and recording

This example qualifies `livekit-agents==1.8.10`. Set the usual LiveKit and OpenAI
credentials, plus `BRAINTRUST_API_KEY`. Use Python 3.12–3.14 and install with
`uv sync` from this directory.

```sh
export BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS=true
export BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true
uv run cascade.py dev
```

With `braintrust[audio]`, enabled recordings default to Opus in Ogg. Installing
the extra or setting the generic `BRAINTRUST_CAPTURE_ATTACHMENTS` flag does not
enable audio capture.
