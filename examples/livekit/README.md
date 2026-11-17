# LiveKit voice tracing and recording

The integration supports LiveKit 1.8.9+; this example pins 1.8.10. Set the usual LiveKit and OpenAI
credentials, plus `BRAINTRUST_API_KEY`. Use Python 3.12–3.14 and install with
`uv sync` from this directory.

```sh
export BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS=true
export BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true
uv run voice_agent.py dev
```

Set `LIVEKIT_MODE` to choose the pipeline:

```sh
LIVEKIT_MODE=realtime uv run voice_agent.py dev
```

| `LIVEKIT_MODE`      | Pipeline                          | Models used by this example                                   |
| ------------------- | --------------------------------- | ------------------------------------------------------------- |
| `cascade` (default) | STT → text LLM → TTS              | `gpt-4o-mini-transcribe` → `gpt-4.1-mini` → `gpt-4o-mini-tts` |
| `half-cascade`      | Audio → Realtime text model → TTS | `gpt-realtime` with text output → `tts-1`                     |
| `realtime`          | Native speech-to-speech           | `gpt-realtime`                                                |

Agent instructions appear in each `llm_response` span's input as a `system` message,
including any instructions specific to that response.

With `braintrust[audio]`, enabled recordings default to Opus in Ogg. Installing
the extra or setting the generic `BRAINTRUST_CAPTURE_ATTACHMENTS` flag does not
enable audio capture.

The example enables both LiveKit and OpenAI instrumentation:

```python
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.openai import setup_openai

setup_livekit_agents()
setup_openai()
```

LiveKit records turns and speech; OpenAI records model requests and usage under
those operations. `auto_instrument()` also enables both integrations.

Both audio capture flags default to false. OpenAI instrumentation also honors
these flags: `USER` enables transcription/translation input audio, and `AGENT`
enables generated output audio. This is a direction convention; OpenAI does not
identify who is speaking. With capture disabled, transcripts are still logged,
but provider audio attachments are omitted. Configure the flags before starting
the worker; changing them does not retract already queued or uploaded audio.

LiveKit redaction is also rechecked when queued records are exported. Redacted
exports omit participant/speaker identities and provider request metadata such
as tool descriptions and schemas. Automatic provider spans retain only model
and provider attribution in metadata, alongside usage and timing metrics.

```text
session
├── user_turn
│   └── stt
│       └── Transcription
└── agent_turn
    ├── llm_response (task)
    │   └── Chat Completion (llm, token usage)
    ├── tool
    ├── tts
    └── agent_speech
```

For adapters without known provider instrumentation, LiveKit emits an LLM child
from the model data exposed by the framework. It does not infer provider support
from whether a span appears. OpenAI Realtime uses a WebSocket API that the OpenAI
integration does not instrument, so LiveKit supplies its `livekit.llm_response` LLM
child. Agent instructions and model usage are available on that child.


STT, LLM response, and TTS task wrappers record their own start/end times.
Provider integrations own model metrics. When a client has no known integration,
LiveKit supplies a `livekit.stt`, `livekit.llm_response`, or `livekit.tts` model
child instead. OpenAI HTTP recognition/synthesis defer to OpenAI instrumentation
regardless of whether it is enabled. Native streaming STT and Realtime use
LiveKit's own clients. A native STT stream can span several turns and is traced
as a session-level operation; transcript delivery alone does not create a model
operation. With user audio capture enabled, a native STT model child includes a
WAV of the input consumed by its node, limited to the first 4 MiB of PCM per
stream. `contrib.livekit.input_audio_truncated=true` marks a partial recording;
recognition continues normally after the capture limit. This model input is
separate from the segmented session recording. Duration-only STT usage reports
do not create token counts. TTS model children record the text after LiveKit's
configured text transforms. Media model inputs identify their `operation` as
`transcribe` or `speech`.


LiveKit turn diagnostics are included without an OpenTelemetry exporter. User
turns expose `contrib.livekit.end_of_turn_delay`,
`contrib.livekit.transcription_delay`, and
`contrib.livekit.on_user_turn_completed_delay` in seconds when available.
`livekit.eou_wait` shows the observed interval from speech end to an accepted
turn decision. User speech spans retain native `contrib.livekit.vad_silence_duration`
and `contrib.livekit.vad_inference_duration`, also in seconds. Missing native
timing stays absent. Model timing and usage continue to belong to model children.
